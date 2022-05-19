import itertools
from typing import List, Union
from collections import OrderedDict
from functools import partial
from symengine import log, S, Symbol
from tinydb import where
from pycalphad.model import _MAX_PARAM_NESTING
import pycalphad.variables as v
from pycalphad.core.utils import unpack_components, wrap_symbol
from pycalphad import Model
from pycalphad.core.errors import DofError


def _get_species(i, j, k, l) -> v.Species:
    """Return a Species for quadruplet given by constituents

    Canonicalizes the Species by sorting among the cation and anion sublattices.

    Parameters
    ----------
    constituents : List[Union[str, v.Species]]

    """
    constituents = [i, j, k, l]
    if all(isinstance(c, v.Species) for c in constituents):
        # if everything is a species, get the string names as constituents
        constituents = [c.name for c in constituents]
    if len(constituents) == 4:
        constituents = sorted(constituents[0:2]) + sorted(constituents[2:4])
    name = "".join(constituents)
    constituent_dict = {}
    # using get() will increment c instead of overriding if c is already defined
    for c in constituents:
        constituent_dict[c] = constituent_dict.get(c, 0.0) + 1.0
    return v.Species(name, constituents=constituent_dict)


class ModelMQMQA(Model):
    """
    Symbolic implementation of the modified quasichemical model in the
    quadruplet approximation developed by Pelton _et al._ [1]_. The formulation
    here largely follows the derivation by Poschmann _et al._ [2]_.

    Following pycalphad convention, ``ModelMQMQA.G`` defines the energy for one
    "formula unit" of this phase. The MQMQA formulations by Pelton _et al._ and
    Poschmann _et al._ do not formally define the energy per formula unit in
    the same way as is done for the CEF. However, it is convienient to define
    one formula unit of an MQMQA phase to be the energy corresponding to one
    mole of quadruplets. In terms of the equations from Poschmann _et al._,
    which are largely followed in this class, that means
    :math:`\\sum_{ab/xy} n_{ab/xy} = 1` and therefore
    :math:`n_{ab/xy} = X_{ij/kl}` by definition.

    This class is only semantically a subclass of ``Model``. It implements the
    API expected for a Model in a self-contained way without any need to rely
    on the Model superclass, but it is created as a subclass to satisfy various
    ``isinstance`` checks in the codebase. The subclassing on Model should be
    removed once a suitable abstract base class or protocol is defined.

    References
    ----------
    .. [1] Pelton, Chartrand, and Eriksson, The Modified Quasi-chemical Model: Part IV. Two-Sublattice Quadruplet Approximation, Metallurgical and Materials Transactions A, 32(6) (2001) 1409-1416 doi: `10.1007/s11661-001-0230-7 <https://doi.org/10.1007/s11661-001-0230-7>`_
    .. [2] Poschmann, Bajpai, Fitzpatrick, and Piro, Recent developments for molten salt systems in Thermochimica, CALPHAD 75 (2021) 102341 doi: `10.1016/j.calphad.2021.102341 <https://doi.org/10.1016/j.calphad.2021.102341>`_

    """

    contributions = [
        ("ref", "reference_energy"),
        ("idmix", "ideal_mixing_energy"),
        ("xsmix", "excess_mixing_energy"),
    ]

    def __init__(self, dbe, comps, phase_name, parameters=None):
        self._dbe = dbe
        self._reference_model = None
        self.components = set()
        self.constituents = []
        self.phase_name = phase_name.upper()
        phase = dbe.phases[self.phase_name]
        self.site_ratios = tuple(list(phase.sublattices))

        active_species = unpack_components(dbe, comps)
        constituents = []
        for sublattice in dbe.phases[phase_name].constituents:
            sublattice_comps = set(sublattice).intersection(active_species)
            self.components |= sublattice_comps
            constituents.append(sublattice_comps)
        self.components = sorted(self.components)
        # create self.cations and self.anions properties to use instead of constituents
        if len(constituents) == 2:
            self.cations = sorted(constituents[0])
            self.anions = sorted(constituents[1])
        else:
            raise ValueError(f"Exactly two sublattices are required for the MQMQA. Got {len(constituents)} sublattices: {constituents}.")

        # In several places we use the assumption that the cation lattice and
        # anion lattice have no common species; we validate that assumption here
        shared_species = set(self.cations).intersection(set(self.anions))
        assert len(shared_species) == 0, f"No species can be shared between the two MQMQA lattices, got {shared_species}"

        # Verify that this phase is still possible to build
        if len(self.cations) == 0:
            raise DofError(f"{self.phase_name}: Cation sublattice of {phase.constituents[0]} has no active species in {self.components}")
        if len(self.anions) == 0:
            raise DofError(f"{self.phase_name}: Anion sublattice of {phase.constituents[1]} has no active species in {self.components}")

        self._pairs = list(itertools.product(self.cations, self.anions))
        quads = list(itertools.product(
            itertools.combinations_with_replacement(self.cations, 2),
            itertools.combinations_with_replacement(self.anions, 2),
        ))
        self._quadruplets = [(A, B, X, Y) for (A, B), (X, Y) in quads]
        quad_species = [_get_species(A, B, X, Y) for (A, B), (X, Y) in quads]
        self.constituents = [sorted(quad_species)]

        # Convert string symbol names to symengine Symbol objects
        # This makes xreplace work with the symbols dict
        symbols = {Symbol(s): val for s, val in dbe.symbols.items()}

        if parameters is not None:
            self._parameters_arg = parameters
            if isinstance(parameters, dict):
                symbols.update([(wrap_symbol(s), val) for s, val in parameters.items()])
            else:
                # Lists of symbols that should remain symbolic
                for s in parameters:
                    symbols.pop(wrap_symbol(s))
        else:
            self._parameters_arg = None

        self._symbols = {wrap_symbol(key): value for key, value in symbols.items()}

        self.models = OrderedDict()
        self.build_phase(dbe)

        for name, value in self.models.items():
            self.models[name] = self.symbol_replace(value, symbols)

    def __eq__(self, other):
        if self is other:
            return True
        elif type(self) != type(other):
            return False
        else:
            return self.__dict__ == other.__dict__

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(repr(self))

    # Default methods, don't need to be overridden unless you want to change the behavior
    @property
    def state_variables(self) -> List[v.StateVariable]:
        """Return a sorted list of state variables used in the ast which are not site fractions."""
        return sorted((x for x in self.ast.free_symbols if not isinstance(x, v.SiteFraction) and isinstance(x, v.StateVariable)), key=str)

    @property
    def site_fractions(self) -> List[v.SiteFraction]:
        """Return a sorted list of site fractions used in the ast."""
        return sorted((x for x in self.ast.free_symbols if isinstance(x, v.SiteFraction)), key=str)

    @property
    def ast(self):
        "Return the full abstract syntax tree of the model."
        return sum(self.models.values())

    def _pair_test(self, constituent_array):
        """Return True if the constituent array represents a pair.

        Pairs have only one species in each sublattice.
        """
        for subl in constituent_array:
            if len(subl) > 1:
                return False
            constituent = subl[0]
            if constituent not in self.components:
                return False
        return True

    def _array_validity(self, constituent_array):
        """Return True if the constituent array is satisfies all components."""
        for subl in constituent_array:
            for constituent in subl:
                if constituent not in self.components:
                    return False
        return True

    def _zeta_ik(self, i, k):
        pair_query = (
            (where("phase_name") == self.phase_name) & \
            (where("parameter_type") == "MQMG") & \
            (where("constituent_array").test(lambda x: x == ((i,), (k,))))
        )
        params = list(self._dbe._parameters.search(pair_query))
        assert len(params) == 1, f"Expected exactly one pair parameter for ({i, k}), got {len(params)}: {params}"
        param = params[0]
        return param["zeta"]

    def _X_ijkl(self, i, j, k, l) -> v.SiteFraction:
        """
        Shorthand for creating a site fraction object v.Y for a quadruplet (ij/kl)
        """
        return v.Y(self.phase_name, 0, _get_species(i, j, k, l))

    def _n_ik(self, i, k):
        """
        Return the amount of pair, n_i/k, for the pair i/k.

        Follows Poschmann Eq. 5 except that including zeta here (as by
        Poschmann) seems to break the tests when the coordination numbers
        vary across quadruplets.
        """
        n_ik = S.Zero
        for a, b, x, y in self._quadruplets:
            n_ik += self._X_ijkl(a, b, x, y) * ((a == i) + (b == i)) * ((x == k) + (y == k))
        return n_ik

    def _n_ik_star(self, i, k):
        """
        Return the amount of pair, n^*_i/k, for the pair i/k following Lambotte JCT 2011 Eq. A.3

        This term is only used in the pair part of the entropy.
        """
        n_ik = S.Zero
        for a, b, x, y in self._quadruplets:
            n_ik += self._X_ijkl(a, b, x, y) * ((a == i) + (b == i)) * ((x == k) + (y == k))
        n_ik /= self._zeta_ik(i, k)
        return n_ik

    def _X_ik(self, i: v.Species, k: v.Species):
        """
        Return the endmember fraction, X_i/k, for a the pair i/k following Poschmann Eq. 6
        """
        X_ik = self._n_ik(i, k) / sum(self._n_ik(a, x) for a, x in self._pairs)
        return X_ik

    def _X_ik_star(self, i: v.Species, k: v.Species):
        """
        Return the endmember fraction, X^*_i/k, for a the pair i/k.

        Follows Lambotte JCT 2011 Eq. A.5, but this is simply Poschmann Eq. 6
        with n^*_ik instead of n_ik.

        This term is only used in the pair part of the entropy and as a
        sub-expression of F_i (also part of the pair part of the entropy).
        """
        X_ik = self._n_ik_star(i, k) / sum(self._n_ik_star(a, x) for a, x in self._pairs)
        return X_ik

    def _n_i(self, dbe, species):
        """
        Return the mass of the species following Poschmann Eq. 7 and 8.
        """
        Z = partial(self.Z, dbe)  # alias for notation
        n_i = S.Zero
        if species in self.cations:
            i = species
            for a, b, x, y in self._quadruplets:
                n_i += self._X_ijkl(a, b, x, y) * ((a == i) / Z(a, a, b, x, y) + (b == i) / Z(b, a, b, x, y))
        else:
            assert species in self.anions
            k = species
            for a, b, x, y in self._quadruplets:
                n_i += self._X_ijkl(a, b, x, y) * ((x == k) / Z(x, a, b, x, y) + (y == k) / Z(y, a, b, x, y))
        return n_i

    def _X_i(self, dbe, species: v.Species):
        """
        Return the site fraction of species on it's sublattice. Poschmann Eq. 9 and 10.
        """
        if species in self.cations:
            return self._n_i(dbe, species) / sum(self._n_i(dbe, a) for a in self.cations)
        else:
            assert species in self.anions
            return self._n_i(dbe, species) / sum(self._n_i(dbe, x) for x in self.anions)

    def _Y_i(self, species: v.Species):
        """
        Return the site equivalent fraction of species following Poschmann Eq. 11 and 12.
        """
        Y_i = S.Zero
        if species in self.cations:
            i = species
            for a, b, x, y in self._quadruplets:
                Y_i += self._X_ijkl(a, b, x, y) * ((a == i) + (b == i)) / 2
        else:
            assert species in self.anions
            k = species
            for a, b, x, y in self._quadruplets:
                Y_i += self._X_ijkl(a, b, x, y) * ((x == k) + (y == k)) / 2
        return Y_i

    def _F_i(self, species: v.Species):
        """
        Return the coordination-equivalent fraction of species, Poschmann Eq. 13 and 14.
        """
        soln_type = self._dbe.phases[self.phase_name].model_hints["mqmqa"]["type"]
        F_i = S.Zero
        if species in self.cations:
            i = species
            for a, x in self._pairs:
                if soln_type == "SUBG":
                    F_i += int(a == i) * self._X_ik(a, x)
                elif soln_type == "SUBQ":
                    F_i += int(a == i) * self._X_ik_star(a, x)
        else:
            assert species in self.anions
            k = species
            for a, x in self._pairs:
                if soln_type == "SUBG":
                    F_i += int(x == k) * self._X_ik(a, x)
                elif soln_type == "SUBQ":
                    F_i += int(x == k) * self._X_ik_star(a, x)
        return F_i


    def _chemical_group_filter(self, dbe, symmetric_species, asymmetric_species, sublattice):
        """
        Return a function ``f(m)`` that returns ``True`` if m is symmetric with
        the symmetric_species and asymmetric with the asymmetric_species.
        """
        # sublattice should be "cations" or "anions"
        chem_group_dict = dbe.phases[self.phase_name].model_hints["mqmqa"]["chemical_groups"][sublattice]
        def _f(species):
            if species == symmetric_species:
                return False
            elif species == asymmetric_species:
                return False
            elif chem_group_dict[species] == chem_group_dict[symmetric_species] and chem_group_dict[species] != chem_group_dict[asymmetric_species]:
                return True  # This chemical group should be mixed
            else:
                return False
        return _f

    def _Chi_mix(self, dbe, i, j, k, l):
        """
        (:math:`\\chi_{ij/k}`) following Poschmann Eq. 21 (SUBG-type model) or Eq. 22 (SUBQ-type models), respectively.
        """
        cations = self.cations
        anions = self.anions
        X_ijkl = self._X_ijkl
        soln_type = dbe.phases[self.phase_name].model_hints["mqmqa"]["type"]

        mixing_term_numerator = S.Zero
        mixing_term_denominator = S.Zero

        if i == j and k == l:
            raise ValueError(f"Excess energies for pairs are not defined. Got quadruplet {(i, j, k, l)}")
        elif i != j and k == l:  # Mixing on first sublattice
            # Chi_{ij/kk} type
            nu = list(filter(self._chemical_group_filter(dbe, i, j, "cations"), cations))
            gamma = list(filter(self._chemical_group_filter(dbe, j, i, "cations"), cations))
            for idx, a in enumerate([i] + nu):  # enumerate to avoid double counting
                for b in ([i] + nu)[idx:]:
                    if soln_type == "SUBG":
                        mixing_term_numerator += X_ijkl(a, b, k, k)
                    elif soln_type == "SUBQ":
                        # Eq. 22 loop over anions
                        for x in anions:
                            for y in anions:
                                mixing_term_numerator += X_ijkl(a, b, x, y) * ((k == x) + (k == y)) / 2
                    else:
                        raise ValueError(f"Unknown solution type: {soln_type}")
            for idx, a in enumerate([i, j] + nu + gamma):  # enumerate to avoid double counting
                for b in ([i, j] + nu + gamma)[idx:]:
                    if soln_type == "SUBG":
                        mixing_term_denominator += X_ijkl(a, b, k, k)
                    elif soln_type == "SUBQ":
                        # Eq. 22 loop over anions
                        for x in anions:
                            for y in anions:
                                mixing_term_denominator += X_ijkl(a, b, x, y) * ((k == x) + (k == y)) / 2
                    else:
                        raise ValueError(f"Unknown solution type: {soln_type}")
            return mixing_term_numerator / mixing_term_denominator
        elif i == j and k != l: # Mixing on second sublattice
            # Chi_{ii/kl} type
            nu = list(filter(self._chemical_group_filter(dbe, k, l, "anions"), anions))
            gamma = list(filter(self._chemical_group_filter(dbe, l, k, "anions"), anions))
            for idx, x in enumerate([k] + nu):  # enumerate to avoid double counting
                for y in ([k] + nu)[idx:]:
                    if soln_type == "SUBG":
                        mixing_term_numerator += X_ijkl(i, i, x, y)
                    elif soln_type == "SUBQ":
                        # Eq. 22 loop over cations
                        for a in cations:
                            for b in cations:
                                mixing_term_numerator += X_ijkl(a, b, x, y) * ((i == a) + (i == b)) / 2
                    else:
                        raise ValueError(f"Unknown solution type: {soln_type}")
            for idx, x in enumerate([k, l] + nu + gamma):  # enumerate to avoid double counting
                for y in ([k, l] + nu + gamma)[idx:]:
                    if soln_type == "SUBG":
                        mixing_term_denominator += X_ijkl(i, i, x, y)
                    elif soln_type == "SUBQ":
                        # Eq. 22 loop over cations
                        for a in cations:
                            for b in cations:
                                mixing_term_denominator += X_ijkl(a, b, x, y) * ((i == a) + (i == b)) / 2
                    else:
                        raise ValueError(f"Unknown solution type: {soln_type}")
            return mixing_term_numerator / mixing_term_denominator
        else:
            raise ValueError(f"Computing Chi_mix is not supported for reciprocal quadruplets. Got quadruplet {(i, j, k, l)}.")

    def _Y_ik(self, i, k):
        """
        Poschmann Eq. 20
        """
        term = S.Zero
        for a, b, x, y in self._quadruplets:
            term += self._X_ijkl(a,b,x,y) * ((a == i) + (b == i)) * ((x == k) + (y == k)) / 4
        return term

    def _Xi_mix(self, dbe, i, j, k, l):
        """
        (:math:`\\xi_{ij/k}`) following Poschmann Eq. 19
        """
        # For mixing in cations (i != j, k == l), nu are cations (nu != i, nu != j) where i and nu have the same
        # chemical group and j has a different chemical group.
        mixing_term = S.Zero
        if i == j and k == l:
            raise ValueError(f"Computing Xi_mix is not supported for pair quadruplets (there must be mixing among cations or anions). Got quadruplet {(i, j, k, l)}.")
        elif i != j and k == l:  # Mixing on first sublattice
            nu = list(filter(self._chemical_group_filter(dbe, i, j, "cations"), self.cations))
            for a in [i] + nu:
                mixing_term += self._Y_ik(a, k)
            return mixing_term
        elif i == j and k != l:  # Mixing on second sublattice
            nu = list(filter(self._chemical_group_filter(dbe, k, l, "anions"), self.anions))
            for x in [k] + nu:
                mixing_term += self._Y_ik(i, x)
            return mixing_term
        else:
            raise ValueError(f"Computing Xi_mix is not supported for reciprocal quadruplets (there can only be mixing among cations _or_ anions). Got quadruplet {(i, j, k, l)}.")

    def _calc_Z(self, dbe, species, A, B, X, Y):
        # In derivations of the MQMQA, charges are written as if they have the same sign. The absolute values of the charges are used here.
        Z = partial(self.Z, dbe)
        if (species == A) or (species == B):
            species_is_cation = True
        elif (species == X) or (species == Y):
            species_is_cation = False
        else:
            raise ValueError(f"{species} is not A ({A}), B ({B}), X ({X}) or Y ({Y}).")

        if A == B and X == Y:
            raise ValueError(f"Z({species}, {A}{B}/{X}{Y}) is a pure pair and must be defined explictly")
        elif A != B and X != Y:
            # This is a reciprocal AB/XY quadruplet and needs to be calculated by eq 23 and 24 in Pelton et al. Met Trans B (2001)
            F = 1/8 * (  # eq. 24
                  abs(A.charge) / Z(A, A,A,X,Y)
                + abs(B.charge) / Z(B, B,B,X,Y)
                + abs(X.charge) / Z(X, A,B,X,X)
                + abs(Y.charge) / Z(Y, A,B,Y,Y)
                )
            if species_is_cation:
                inv_Z = F * (
                    Z(X, A,B,X,X) / (abs(X.charge) * Z(species, A,B,X,X))
                    + Z(Y, A,B,Y,Y) / (abs(Y.charge) * Z(species, A,B,Y,Y))
                            )

            else:
                inv_Z = F * (
                    Z(A, A,A,X,Y) / (abs(A.charge) * Z(species, A,A,X,Y))
                    + Z(B, B,B,X,Y) / (abs(B.charge) * Z(species, B,B,X,Y))
                            )

            return 1 / inv_Z
        elif A != B:  # X == Y
            # Need to calculate Z^i_AB/XX (Y = X).
            # We assume Z^A_ABXX = Z^A_AAXX = Z^A_AAYY
            # and Z^X_ABXX = (q_X + q_Y)/(q_A/Z^A_AAXX + q_B/Z^B_BBXX)  # note: q_X = q_Y, etc. since Y = X
            # We don't know if these are correct, but that's what's implemented in Thermochimica
            if species_is_cation:
                return Z(species, species, species, X, X)
            else:
                return 2*abs(species.charge) / (
                    abs(A.charge) / Z(A, A,A,species,species) + abs(B.charge) / Z(B, B,B,species,species)
                    )
        elif X != Y:  # A == B
            # These use the same equations as A != B case with the same assumptions
            if species_is_cation:
                # similarly, Z^A_AAXY = (q_A + q_B)/(q_X/Z^X_AAXX + q_Y/Z^Y_AAYY)
                return 2*abs(species.charge)/(abs(X.charge)/Z(X, species, species, X, X) + abs(Y.charge)/Z(Y, species, species, Y, Y))
            else:
                return Z(species, A, A, species, species)
        raise ValueError("This should be unreachable")


    def Z(self, dbe, species: v.Species, A: v.Species, B: v.Species, X: v.Species, Y: v.Species):
        # Canonicalize the order of cations and anions in alphabetical order
        # Note that the coordination number parameters (the constituents and
        # the values) _must_ obey this canonical sorted order.
        A, B = sorted((A, B))
        X, Y = sorted((X, Y))
        Zs = dbe._parameters.search(
            (where("phase_name") == self.phase_name) & \
            (where("parameter_type") == "MQMZ") & \
            (where("constituent_array").test(lambda x: x == ((A, B), (X, Y))))
        )
        if len(Zs) == 0:
            return self._calc_Z(dbe, species, A, B, X, Y)
        elif len(Zs) == 1:
            sp_idx = [A, B, X, Y].index(species)
            return Zs[0]["coordinations"][sp_idx]
        else:
            raise ValueError(f"Expected exactly one Z for {species} of {((A, B), (X, Y))}, got {len(Zs)}")

    def get_internal_constraints(self):
        constraints = []
        # Site fraction constraint
        site_frac_sum = S.Zero
        for a, b, x, y in self._quadruplets:
            site_frac_sum += self._X_ijkl(a, b, x, y)
        constraints.append(site_frac_sum - 1)  # = 0
        return constraints

    @property
    def normalization(self):
        """
        Total number of moles of this phase.

        Divide by this normalization factor to convert from J/mole-formula to J/mole-atoms
        """
        return sum(self._n_i(self._dbe, i) for i in self.cations + self.anions if "VA" not in i.constituents)

    def moles(self, species, per_formula_unit=False):
        "Number of moles of species or elements."
        species = v.Species(species)
        result = S.Zero
        for i in self.cations + self.anions:
            if list(species.constituents.keys())[0] in i.constituents:
                result += self._n_i(self._dbe, i)
        if per_formula_unit:
            return result
        else:
            return result / self.normalization

    degree_of_ordering = DOO = S.Zero
    curie_temperature = TC = S.Zero
    beta = BMAG = S.Zero
    neel_temperature = NT = S.Zero

    # pylint: disable=C0103
    # These are standard abbreviations from Thermo-Calc for these quantities
    GM = property(lambda self: self.ast / self.normalization)
    G = property(lambda self: self.ast)
    energy = GM
    entropy = SM = property(lambda self: -self.GM.diff(v.T))
    enthalpy = HM = property(lambda self: self.GM - v.T * self.GM.diff(v.T))
    heat_capacity = CPM = property(lambda self: -v.T * self.GM.diff(v.T, v.T))
    # pylint: enable=C0103
    mixing_energy = GM_MIX = property(lambda self: self.GM - self.reference_model.GM)
    mixing_enthalpy = HM_MIX = property(lambda self: self.GM_MIX - v.T * self.GM_MIX.diff(v.T))
    mixing_entropy = SM_MIX = property(lambda self: -self.GM_MIX.diff(v.T))
    mixing_heat_capacity = CPM_MIX = property(lambda self: -v.T * self.GM_MIX.diff(v.T, v.T))

    def reference_energy(self, dbe):
        """
        Returns energies contributed by pairs.
        """
        pair_query = (
            (where("phase_name") == self.phase_name) & \
            (where("parameter_type") == "MQMG") & \
            (where("constituent_array").test(self._pair_test))
        )
        params = dbe._parameters.search(pair_query)
        terms = S.Zero
        for param in params:
            a = i = param["constituent_array"][0][0]
            x = k = param["constituent_array"][1][0]
            X_ax = S.Zero
            for b in self.cations:
                for y in self.anions:
                    X_ax += self._X_ijkl(a,b,x,y) * ((a == i) + (b == i)) * ((x == k) + (y == k)) / (2 * self.Z(dbe, a, a,b,x,y))
            G_ax = param["parameter"]
            terms += X_ax * G_ax / param["stoichiometry"][0]
        return terms

    def ideal_mixing_energy(self, dbe):
        # notational niceties
        n_i = partial(self._n_i, dbe)
        X_i = partial(self._X_i, dbe)
        X_ik = self._X_ik
        Y_i = self._Y_i
        F_i = self._F_i
        X_ijkl = self._X_ijkl
        soln_type = dbe.phases[self.phase_name].model_hints["mqmqa"]["type"]
        if soln_type == "SUBG":
            pow_X_ik = 1.0
            pow_Y_i = 1.0
        elif soln_type == "SUBQ":
            pow_X_ik = 0.75
            pow_Y_i = 0.5

        Sid = S.Zero
        # Individual constituents
        for A in self.cations:
            Sid += n_i(A) * log(X_i(A))
        for X in self.anions:
            Sid += n_i(X) * log(X_i(X))
        # Pairs
        if soln_type == "SUBG":
            for i, k in self._pairs:
                Sid += self._n_ik(i, k) * log(X_ik(i, k) / (F_i(i) * F_i(k)))
        elif soln_type == "SUBQ":
            for i, k in self._pairs:
                Sid += self._n_ik_star(i, k) * log(self._X_ik_star(i, k) / (F_i(i) * F_i(k)))
        # Quadruplets
        for i, j, k, l in self._quadruplets:
            C_ijkl = (2 - (i == j)) * (2 - (k == l))
            Sid += X_ijkl(i, j, k, l) * log(X_ijkl(i, j, k, l) / (C_ijkl * (X_ik(i, k) * X_ik(i, l) * X_ik(j, k) * X_ik(j, l))**pow_X_ik / (Y_i(i) * Y_i(j) * Y_i(k) * Y_i(l))**pow_Y_i))
        return Sid * v.T * v.R

    def excess_mixing_energy(self, dbe):
        params = dbe._parameters.search(
            (where("phase_name") == self.phase_name) &
            (where("parameter_type") == "MQMX") &
            (where("constituent_array").test(self._array_validity))
        )

        cations = self.cations
        anions = self.anions

        X_ijkl = self._X_ijkl
        Z = partial(self.Z, dbe)

        energy = S.Zero
        for param in params:
            (A, B), (X, Y) = param["constituent_array"]
            exponents = param["exponents"]
            mixing_code = param["mixing_code"]
            m = param["additional_mixing_constituent"]
            p_alpha = exponents[0]
            q_alpha = exponents[1]
            r_alpha = param["additional_mixing_exponent"]
            # Poschmann Eq. 23-26
            mixing_term = S.Zero
            if A != B and X == Y:
                if mixing_code == "G":
                    # Poschmann Eq. 23 (cations mixing)
                    mixing_term += self._Chi_mix(dbe, A, B, X, X)**p_alpha * self._Chi_mix(dbe, B, A, X, X)**q_alpha
                elif mixing_code == "Q":
                    # Poschmann Eq. 19 and 20 (cations mixing)
                    Xi_ijk = self._Xi_mix(dbe, A, B, X, X)
                    Xi_jik = self._Xi_mix(dbe, B, A, X, X)
                    # Poschmann Eq. 24
                    mixing_term += Xi_ijk**p_alpha * Xi_jik**q_alpha / (Xi_ijk + Xi_jik)**(p_alpha + q_alpha)
                else:
                    raise ValueError(f"Unknown mixing code {mixing_code} for parameter {param}")
                if m != v.Species(None):
                    # Poschmann Eq. 25 and 26 ternary term (same for both mixing codes)
                    Xi_ijk = self._Xi_mix(dbe, A, B, X, X)
                    Xi_jik = self._Xi_mix(dbe, B, A, X, X)
                    Y_mk = self._Y_ik(m, X)
                    nu = list(filter(self._chemical_group_filter(dbe, A, B, "cations"), cations))
                    gamma = list(filter(self._chemical_group_filter(dbe, B, A, "cations"), cations))
                    if m in gamma:
                        mixing_term *= Y_mk / Xi_jik * (1 - self._Y_ik(B, X) / Xi_jik)**(r_alpha - 1)
                    elif m in nu:
                        mixing_term *= Y_mk / Xi_ijk * (1 - self._Y_ik(A, X) / Xi_ijk)**(r_alpha - 1)
                    else:  # not in nu or gamma
                        mixing_term *= Y_mk * (1 - Xi_ijk - Xi_jik)**(r_alpha - 1)
            elif A == B and X != Y:
                if mixing_code == "G":
                    # Poschmann Eq. 23 (anions mixing)
                    mixing_term += self._Chi_mix(dbe, A, A, X, Y)**p_alpha * self._Chi_mix(dbe, A, A, Y, X)**q_alpha
                elif mixing_code == "Q":
                    # Poschmann Eq. 19 and 20 (anions mixing)
                    Xi_ikl = self._Xi_mix(dbe, A, A, X, Y)
                    Xi_ilk = self._Xi_mix(dbe, A, A, Y, X)
                    # Poschmann Eq. 24
                    mixing_term += Xi_ikl**p_alpha * Xi_ilk**q_alpha / (Xi_ikl + Xi_ilk)**(p_alpha + q_alpha)
                else:
                    raise ValueError(f"Unknown mixing code {mixing_code} for parameter {param}")
                if m != v.Species(None):
                    # Poschmann Eq. 25 and 26 ternary term (same for both mixing codes)
                    Xi_ikl = self._Xi_mix(dbe, A, A, X, Y)
                    Xi_ilk = self._Xi_mix(dbe, A, A, Y, X)
                    Y_im = self._Y_ik(A, m)
                    nu = list(filter(self._chemical_group_filter(dbe, X, Y, "anions"), anions))
                    gamma = list(filter(self._chemical_group_filter(dbe, Y, X, "anions"), anions))
                    if m in gamma:
                        mixing_term *= Y_im / Xi_ilk * (1 - self._Y_ik(A, Y) / Xi_ilk)**(r_alpha - 1)
                    elif m in nu:
                        mixing_term *= Y_im / Xi_ikl * (1 - self._Y_ik(A, X) / Xi_ikl)**(r_alpha - 1)
                    else:  # not in nu or gamma
                        mixing_term *= Y_im * (1 - Xi_ikl - Xi_ilk)**(r_alpha - 1)
            else:
                mixing_term = S.One  # No mixing, this is a modification to the formation energy of this quadruplet
            g = param["parameter"] * mixing_term

            # Poschmann Eq. 17
            cation_factor = S.Zero
            if A == B:
                for m in cations:
                    if m != A:
                        cation_factor += X_ijkl(A,m,X,Y) / Z(A, A,m,X,Y)
                cation_factor *= Z(A, A,B,X,Y) / 2
            anion_factor = S.Zero
            if X == Y:
                for m in anions:
                    if m != X:
                        anion_factor += X_ijkl(A,B,X,m) / Z(X, A,B,X,m)
                anion_factor *= Z(X, A,B,X,Y) / 2
            energy += 0.5 * g * (X_ijkl(A,B,X,Y) + cation_factor + anion_factor)

        return energy

    def shift_reference_state(self, reference_states, dbe, contrib_mods=None, output=("GM", "HM", "SM", "CPM"), fmt_str="{}R"):
        raise NotImplementedError()

    def build_phase(self, dbe):
        """
        Generate the symbolic form of all the contributions to this phase.

        Parameters
        ----------
        dbe : 'pycalphad.io.Database'
        """
        self.models.clear()
        for key, value in self.__class__.contributions:
            self.models[key] = S(getattr(self, value)(dbe))

    @staticmethod
    def symbol_replace(obj, symbols):
        """
        Substitute values of symbols into 'obj'.

        Parameters
        ----------
        obj : symengine object
        symbols : dict mapping symengine.Symbol to symengine object

        Returns
        -------
        symengine object
        """
        try:
            # Need to do more substitutions to catch symbols that are functions
            # of other symbols
            for iteration in range(_MAX_PARAM_NESTING):
                obj = obj.xreplace(symbols)
                undefs = [x for x in obj.free_symbols if not isinstance(x, v.StateVariable)]
                if len(undefs) == 0:
                    break
        except AttributeError:
            # Can't use xreplace on a float
            pass
        return obj

    def _build_reference_model(self, preserve_ideal=True):
        raise NotImplementedError()

    @property
    def reference_model(self):
        raise NotImplementedError("Endmember reference models do not have a physical meaning for MQMQA models")
