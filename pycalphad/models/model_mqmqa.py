import copy
import itertools
from typing import List, Tuple
from collections import Counter, OrderedDict
from functools import partial
from sympy import log, S, Symbol
from tinydb import where
from pycalphad.model import _MAX_PARAM_NESTING
import pycalphad.variables as v
from pycalphad.core.utils import unpack_components, wrap_symbol
from pycalphad import Model


def get_species(*constituents) -> v.Species:
    """Return a Species for a pair or quadruplet given by constituents

    Canonicalizes the Species by sorting among the A and X sublattices.

    Parameters
    ----------
    constituents : List[Tuple[str, v.Species]]

    Examples
    --------
        get_species('A_1.0', 'B_1.0')
    """
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


# TODO: cleanup this class (style)
# TODO: remove dead code (e.g. duplicate reference_energy)
# TODO: document the model contributions with the mathematics
class ModelMQMQA(Model):
    """

    One peculiarity about the ModelMQMQA is that the charges in the way the
    model are written are assumed to be positive. We take the absolute value
    whenever there is a charge.

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

        # build `constituents` here so we can build the pairs and quadruplets
        # *before* `super().__init__` calls `self.build_phase`. We leave it to
        # the Model to build self.constituents and do the error checking.
        active_species = unpack_components(dbe, comps)
        constituents = []
        for sublattice in dbe.phases[phase_name].constituents:
            sublattice_comps = set(sublattice).intersection(active_species)
            self.components |= sublattice_comps
            constituents.append(sublattice_comps)
        self.components = sorted(self.components)
        # create self.cations and self.anions properties to use instead of constituents
        if len(constituents) == 1:
            self.ele = constituents
        else:
            self.cations = sorted(constituents[0])
            self.anions = sorted(constituents[1])

        # In several places we use the assumption that the cation lattice and
        # anion lattice have no common species; we validate that assumption here
        shared_species = set(self.cations).intersection(set(self.anions))
        assert len(shared_species) == 0, f"No species can be shared between the two MQMQA lattices, got {shared_species}"

        quads = itertools.product(
            itertools.combinations_with_replacement(self.cations, 2),
            itertools.combinations_with_replacement(self.anions, 2),
        )
        quad_species = [get_species(A, B, X, Y) for (A, B), (X, Y) in quads]
        self.constituents = [sorted(quad_species)]

        # Verify that this phase is still possible to build
        if len(self.cations) == 0:
            raise DofError(f"{self.phase_name}: Cation sublattice of {phase.constituents[0]} has no active species in {self.components}")
        if len(self.anions) == 0:
            raise DofError(f"{self.phase_name}: Anion sublattice of {phase.constituents[1]} has no active species in {self.components}")

        # Set self.nonvacant_elements, only used by get_multiphase_constraint
        # TODO: can we remove this? or re-work it?
        desired_active_pure_elements = [list(x.constituents.keys()) for x in self.components]
        desired_active_pure_elements = [el.upper() for constituents in desired_active_pure_elements for el in constituents]
        pure_elements = sorted(set(desired_active_pure_elements))
        self.nonvacant_elements = [x for x in pure_elements if x != "VA"]

        # Convert string symbol names to sympy Symbol objects
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

    def _p(self, *ABXYs: v.Species) -> v.SiteFraction:
        """Shorthand for creating a site fraction object v.Y for a quadruplet.

        The name `p` is intended to mirror construction of `p(A,B,X,Y)`
        quadruplets, following Sundman's notation.
        """
        return v.Y(self.phase_name, 0, get_species(*ABXYs))

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

    def _mixing_test(self, constituent_array):
        """Return True if the constituent array is satisfies all components."""
        for subl in constituent_array:
            for constituent in subl:
                if constituent not in self.components:
                    return False
        return True

    def M(self, dbe, species):
        """Return the mass of the species.

        The returned expression is composed only of v.Y objects for
        quadruplets, p(A,B,X,Y) in Sundman's notation. The expression
        constructed here follows equation (8) of Sundman's notes.

        This is the same as X_A in Pelton's notation.
        """
        cations = self.cations
        anions = self.anions

        # aliases for notation
        Z = partial(self.Z, dbe)
        p = self._p
        M = S.Zero

        if species in cations:
            A = species
            for i, X in enumerate(anions):
                for Y in anions[i:]:
                    M += p(A,A,X,Y)/Z(A,A,A,X,Y)
                    for B in cations:
                        M += p(A, B, X, Y)/Z(A, A, B, X, Y)
        else:
            assert species in anions
            X = species
            for i, A in enumerate(cations):
                for B in cations[i:]:
                    M += p(A,B,X,X) / Z(X,A,B,X,X)
                    for Y in anions:
                        M += p(A,B,X,Y) / Z(X,A,B,X,Y)
        return M

    def ξ(self, A: v.Species, X: v.Species):
        """Return the endmember fraction, ξ_A:X, for a pair Species A:X

        The returned expression is composed only of v.Y objects for
        quadruplets, p(A,B,X,Y) in Sundman's notation. The expression
        constructed here follow equation (12) of Sundman's notes.

        This is the same as X_A/X in Pelton's notation.
        """
        cations = self.cations
        anions = self.anions

        p = self._p  # alias to keep the notation close to the math

        # Sundman notes equation (12)
        return 0.25 * (
            p(A,A,X,X)
            + sum(p(A,A,X,Y) for Y in anions)
            + sum(p(A,B,X,X) for B in cations)
            + sum(p(A,B,X,Y) for B, Y in itertools.product(cations, anions))
        )

    def w(self, species: v.Species):
        """Return the coordination equivalent site fraction of species.

        The returned expression is composed only of v.Y objects for
        quadruplets, p(A,B,X,Y) in Sundman's notation. The expression
        constructed here follow equation (15) of Sundman's notes.


        This is the same as Y_i in Pelton's notation.
        """
        p = self._p
        cations = self.cations
        anions = self.anions

        w = S.Zero
        if species in cations:
            A = species
            for i, X in enumerate(anions):
                for Y in anions[i:]:
                    w += p(A, A, X, Y)
                    for B in cations:
                        w += p(A, B, X, Y)
        else:
            assert species in anions
            X = species
            for i, A in enumerate(cations):
                for B in cations[i:]:
                    w += p(A, B, X, X)
                    for Y in anions:
                        w += p(A, B, X, Y)
        return 0.5 * w

    def ϑ(self, dbe, species: v.Species):
        """Return the site fraction of species on it's sublattice.

        The returned expression is composed only of v.Y objects for
        quadruplets, p(A,B,X,Y) in Sundman's notation, and (constant)
        coordination numbers. The expression constructed here follow equation
        (10) of Sundman's notes.

        This is the same as X_i in Pelton's notation.
        """
        cations = self.cations
        anions = self.anions

        if species in cations:
            return self.M(dbe, species) / sum(self.M(dbe, sp) for sp in cations)
        else:
            assert species in anions
            return self.M(dbe, species) / sum(self.M(dbe, sp) for sp in anions)

    def _calc_Z(self, dbe: "pycalphad.io.Database", species, A, B, X, Y):
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


    def Z(self, dbe: "pycalphad.io.Database", species: v.Species, A: v.Species, B: v.Species, X: v.Species, Y: v.Species):
        Z_cat = sorted((A,B))
        Z_an = sorted((X,Y))
        Z_quad = (tuple(Z_cat),tuple(Z_an))
        Zs = dbe._parameters.search(
            (where("phase_name") == self.phase_name) & \
            (where("parameter_type") == "Z") & \
            (where("diffusing_species").test(lambda sp: sp.name == species.name)) & \
            (where("constituent_array").test(lambda x: x == Z_quad)))
            # quadruplet needs to be in 1 sublattice constituent array `[[q]]`, in tuples
        if len(Zs) == 0:
            # TODO: add this to the database so we don't need to recalculate? where should that happen?
            return self._calc_Z(dbe, species, A, B, X, Y)
        elif len(Zs) == 1:
            return Zs[0]["parameter"]
        else:
            raise ValueError(f"Expected exactly one Z for {species} of {((A, B), (X, Y))}, got {len(Zs)}")

    def get_internal_constraints(self):
        constraints = []
        p = self._p
        total_quad = -1
        cations = self.cations
        anions = self.anions
        for i, A in enumerate(cations):
            for B in cations[i:]:
                for j, X in enumerate(anions):
                    for Y in anions[j:]:
                        total_quad += p(A,B,X,Y)
        constraints.append(total_quad)
        return constraints

    @property
    def normalization(self):
        """Divide by this normalization factor to convert from J/mole-quadruplets to J/mole-atoms"""
        # No_Vac is to fix the normalization so that it does not take vacancy into consideration
        no_vac = [j for j in self.components for o in j.constituents if o != "VA"]
        const_spe = [k for i in no_vac for j, k in i.constituents.items()]
        return sum(self.M(self._dbe, c) * const_spe[count] for count, c in enumerate(no_vac))

    def moles(self, species, per_formula_unit=False):
        "Number of moles of species or elements."
        species = v.Species(species)
        result = S.Zero
        for i in itertools.chain(self.cations, self.anions):
            if list(species.constituents.keys())[0] in i.constituents:
                count = list(i.constituents.values())[0]
                result += self.M(self._dbe, i) * count
        # moles is supposed to compute the moles of a pure element, but with a caveat that pycalphad assumes sum(moles(c) for c in comps) == 1
        # The correct solution is to make the changes where pycalphad assumes n=1. But I think it would be easier to change how we implement the model so that the model has n=1 and the energies are normalized to per-mole-atoms.
        # Since normalizing to moles of quadruplets is allowing us to easily compare with thermochimica, I'm thinking that we might be able to fake pycalphad into thinking we have N=1 by normalizing "moles" to n=1
        # The energies will not be normalized to moles of atoms (and so you cannot yet use this Model to compare to other phases), but internally it should be correct and in agreement with thermochimica
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
        # This considers the pair contributions to the energy, the first sum terms in Eq. 37 in Pelton2001.
        """
        Returns the weighted average of the endmember energies
        in symbolic form.
        """
        pair_query = (
            (where("phase_name") == self.phase_name) & \
            (where("parameter_order") == 0) & \
            (where("parameter_type") == "G") & \
            (where("constituent_array").test(self._pair_test))
        )
        p = self._p
        anions = self.anions
        cations = self.cations
        params = dbe._parameters.search(pair_query)
        terms = S.Zero
        for param in params:
            A = param["constituent_array"][0][0]
            X = param["constituent_array"][1][0]
            ξ_AX = S.Zero
            for B in cations:
                for Y in anions:
                    factor = 1
                    if B == A: factor *= 2  # Double count (for symmetry?)
                    if Y == X: factor *= 2  # Double count
                    ξ_AX += factor * p(A,B,X,Y) / (2 * self.Z(dbe, A, A,B,X,Y))
            G_AX = param["parameter"]
            terms += ξ_AX * G_AX
        return terms

    def ideal_mixing_energy(self, dbe):
        # notational niceties
        M = partial(self.M, dbe)
        ϑ = partial(self.ϑ, dbe)
        ξ = self.ξ
        w = self.w
        p = self._p
        soln_type = dbe.phases[self.phase_name].model_hints["mqmqa"]["type"]
        cations = self.cations
        anions = self.anions
        if soln_type == "SUBQ":
            exp1 = 0.75
            exp2 = 0.5
        elif soln_type == "SUBG":
            exp1 = 1.0
            exp2 = 1.0
        Sid = S.Zero

        ζ = 2.4  # hardcoded, but we can get it from the model_hints (SUBQ) or the pairs (SUBG)
        for A in cations:
            Sid += M(A) * log(ϑ(A))  # term 1
        for X in anions:
            Sid += M(X) * log(ϑ(X))  # term 2
        for A in cations:
            for X in anions:
                ξ_AX = ξ(A, X)
                p_AAXX = p(A,A,X,X)
                w_A = w(A)
                w_X = w(X)
                Sid += 4 / ζ * ξ_AX * log(ξ_AX / (w_A * w_X))  # term 3

        # flatter loop over all quadruplets:
        # for A, B, X, Y in ((A, B, X, Y) for i, A in enumerate(cations) for B in cations[i:] for j, X in enumerate(anions) for Y in anions[j:]):
        # Count last 4 terms in the sum
        for i, A in enumerate(cations):
            for B in cations[i:]:
                for j, X in enumerate(anions):
                    for Y in anions[j:]:
                        factor = 1
                        if A != B: factor *= 2
                        if X != Y: factor *= 2
                        Sid += p(A,B,X,Y)*log(p(A,B,X,Y)/(factor * (ξ(A,X)**(exp1))*(ξ(A,Y)**(exp1))*(ξ(B,X)**(exp1))*(ξ(B,Y)**(exp1)) / ((w(A)**(exp2))*(w(B)**(exp2))*(w(X)**(exp2))*(w(Y)**(exp2)))))
        return Sid * v.T * v.R

    def excess_mixing_t1(self, dbe, constituent_array):
        Z = partial(self.Z, dbe)
        cations = self.cations
        anions = self.anions
        p = self._p
        subl_1 = constituent_array[0]
        subl_2 = constituent_array[1]
        A = subl_1[0]
        B = subl_1[1]
        X = subl_2[0]
        Y = subl_2[1]
        # TODO: Figure out how to connect this. Below is the correct expression. Maybe this can be its own function separately
        # And it can be called in the other final function
        # TODO: can this be merged with ξ?
        return 0.5 * (p(A,B,X,Y)
            + sum(0.5 * Z(j, A,B,j,j) * sum(p(A,B,i,Y) / Z(Y, A,B,i,Y) for i in anions if i != Y) for j in anions if j == X == Y)
            + sum(0.5 * Z(q, q,q,X,Y) * sum(p(r,B,X,Y) / Z(B, r,B,X,Y) for r in cations if r != B) for q in cations if q == A == B)
        )

    def X_1_2(self, dbe, constituent_array, diffusing_species):
        chem_groups_cat = dbe.phases[self.phase_name].model_hints["mqmqa"]["chemical_groups"]["cations"]
        chem_groups_an = dbe.phases[self.phase_name].model_hints["mqmqa"]["chemical_groups"]["anions"]
        soln_type = dbe.phases[self.phase_name].model_hints["mqmqa"]["type"]

        cations = self.cations
        anions = self.anions
        p = self._p
        res1 = S.Zero
        res2 = S.Zero

        subl_1 = constituent_array[0]
        subl_2 = constituent_array[1]
        A = subl_1[0]
        B = subl_1[1]
        X = subl_2[0]
        Y = subl_2[1]
        # non_diff_spe is either the first element of the subl1/2 or is the remaining element of subl1/2 when diffusing species is one of the elements in the sublattice
        if diffusing_species in subl_1:
            non_diff_spe = [i for i in subl_1 if i != diffusing_species][0]
            k_As_cat = [i for i in cations if chem_groups_cat[i] != chem_groups_cat[diffusing_species] if diffusing_species in cations and i not in subl_1]
            l_As_cat = [i for i in cations if chem_groups_cat[i] != chem_groups_cat[non_diff_spe] if diffusing_species and i not in subl_1]
        elif diffusing_species in subl_2:
            non_diff_spe = [i for i in subl_2 if i != diffusing_species][0]
            k_As_an = [i for i in anions if chem_groups_an[i] != chem_groups_an[diffusing_species] if diffusing_species in anions and i not in subl_1]
            l_As_an = [i for i in anions if chem_groups_an[i] != chem_groups_an[non_diff_spe] if diffusing_species in anions and i not in subl_1]

        # This is all assuming that there will be only two groups for symmetrical and asymmetrical
        if X == Y and diffusing_species in subl_1:
            As_diff = [diffusing_species]
            if Counter(k_As_cat) != Counter(l_As_cat):
                As_diff.extend(l_As_cat)
            for count, a in enumerate(As_diff):
                for b in As_diff[count:]:
                    res1 += p(a,b,X,Y)
                    if soln_type == "SUBQ":
                        res1 += 0.5 * sum(p(a,b,X,Y) for Y in anions if Y != X)

            if Counter(k_As_cat) != Counter(l_As_cat):
                subl_1 = list(subl_1)
                subl_1.extend(k_As_cat)
                subl_1.extend(l_As_cat)

            for count, a in enumerate(subl_1):
                for b in subl_1[count:]:
                    res2 += p(a,b,X,Y)
                    if soln_type == "SUBQ":
                        res2 += 0.5 * sum(p(a,b,X,Y) for Y in anions if Y != X)
        elif A == B and diffusing_species in subl_2:
            As_diff = [diffusing_species]
            if Counter(k_As_an) != Counter(l_As_an):
                As_diff.extend(l_As_an)
            for count, x in enumerate(As_diff):
                for y in As_diff[count:]:
                    res1 += p(A,B,x,y)
                    if soln_type == "SUBQ":
                        res1 += 0.5 * sum(p(A,B,x,y) for B in cations if A != B)

            if Counter(k_As_an) != Counter(l_As_an):
                subl_2 = list(subl_2)
                subl_2.extend(k_As_an)
                subl_2.extend(l_As_an)

            for count, a in enumerate(subl_2):
                for b in subl_2[count:]:
                    res2 += p(A,B,x,y)
                    if soln_type == "SUBQ":
                        res2 += 0.5 * sum(p(A,B,x,y) for B in cations if A != B)
        return res1 / res2

    def id_symm(self, dbe, A, B, C):
        cations = self.cations
        anions = self.anions
        chem_groups_cat = dbe.phases[self.phase_name].model_hints["mqmqa"]["chemical_groups"]["cations"]
        chem_groups_an = dbe.phases[self.phase_name].model_hints["mqmqa"]["chemical_groups"]["anions"]
        in_lis = [A, B, C]
        if set(in_lis).issubset(cations) is True:
            chem_lis = [i for i in in_lis if chem_groups_cat[i] != chem_groups_cat[A]]
        if set(in_lis).issubset(anions) is True:
            chem_lis = [i for i in in_lis if chem_groups_an[i] != chem_groups_an[A]]
        if len(chem_lis) == 1:
            symm_check = chem_lis[0]
        elif len(chem_lis) > 1:
            symm_check_2 = list(set(in_lis) - set(chem_lis))
            symm_check = symm_check_2[0]
        else:
            symm_check = 0
        return symm_check

    def K_1_2(self, dbe, A, B):
        w = self.w
        num_res = S.Zero
        chem_groups_cat = dbe.phases[self.phase_name].model_hints["mqmqa"]["chemical_groups"]["cations"]
        chem_groups_an = dbe.phases[self.phase_name].model_hints["mqmqa"]["chemical_groups"]["anions"]
        cations = self.cations
        anions = self.anions
        result_cons = [A, B]
        the_rest = set(cations) - set(result_cons)
        result_2 = [i for i in the_rest if chem_groups_cat[i] == chem_groups_cat[result_cons[0]] and chem_groups_cat[i] != chem_groups_cat[result_cons[1]]]
        num_res += w(A)

        for i in result_2:
            num_res += w(i)

        return num_res

    def _chemical_group_filter(self, dbe, symmetric_species, asymmetric_species, sublattice):
        # TODO: another flavor of this would be to check if species == symmetric_species and return true, then we wouldn't have to explictly construct [A] + nu and [A, B] + gamma
        # sublattice should be "cations" or "anions"
        chem_group_dict = dbe.phases[self.phase_name].model_hints["mqmqa"]["chemical_groups"][sublattice]
        def _f(species):
            if chem_group_dict[species] == chem_group_dict[symmetric_species] and chem_group_dict[species] != chem_group_dict[asymmetric_species]:
                return True  # This chemical group should be mixed
            else:
                return False
        return _f

    def _Chi_mix(self, dbe, A, B, X, Y):
        # Compute Poschmann Eq. 21 (SUBG-type model) or Eq. 22 (SUBQ-type model)
        cations = self.cations
        anions = self.anions
        p = self._p

        mixing_term_numerator = S.Zero
        mixing_term_denominator = S.Zero

        if A == B and X == Y:
            raise ValueError(f"Excess energies for pairs are not defined. Got quadruplet {(A, B, X, Y)}")
        elif A != B and X == Y:  # Mixing on first sublattice
            # TODO: add support for SUBQ type where there is a loop over the anions
            nu = list(filter(self._chemical_group_filter(dbe, A, B, "cations"), cations))
            gamma = list(filter(lambda sp: self._chemical_group_filter(dbe, A, B, "cations")(sp) or self._chemical_group_filter(dbe, B, A, "cations")(sp), cations))
            for idx, i in enumerate([A] + nu):  # enumerate to avoid double counting
                for j in ([A] + nu)[idx:]:
                    mixing_term_numerator += p(i, j, X, Y)
            for idx, i in enumerate([A, B] + gamma):  # enumerate to avoid double counting
                for j in ([A, B] + gamma)[idx:]:
                    mixing_term_denominator += p(i, j, X, Y)
            return mixing_term_numerator / mixing_term_denominator
        elif A == B and X != Y:
            # Mixing on second sublattice
            raise NotImplementedError()
        else:
            raise ValueError(f"Excess energies for reciprocal quadruplets are not implemented. Got quadruplet {(A, B, X, Y)}")

    def _Xi_mix(self, dbe, A, B, X, Y):
        # For mixing in cations,  A != B, X == Y, this term is follows Poschmann Eq. 20
        # for any terms nu (nu in cations) for all systems where A and nu have the same
        # chemical group and B has a different chemical group
        raise NotImplementedError()

    def excess_mixing_energy(self, dbe):
        params = [
            # {
            #     "parameter": -50000,
            #     "exponents": [[0, 0], [0, 0]],
            #     "constituent_array": ((v.Species('CU+2.0', {'CU': 1.0}, charge=2.0), v.Species('NI+2.0', {'NI': 1.0}, charge=2.0)), (v.Species('VA-1.0', {'VA': 1.0}, charge=-1.0), v.Species('VA-1.0', {'VA': 1.0}, charge=-1.0))),
            #     "parameter_type": "MQMX",
            #     "phase_name": "REGLIQ",
            #     "mixing_code": "G",
            #     "additional_mixing_constituent": None,
            #     "additional_mixing_exponent": 0
            # },
            {
                "parameter": -10000,
                "exponents": [1, 0, 0, 0],
                "constituent_array": ((v.Species('CU+2.0', {'CU': 1.0}, charge=2.0), v.Species('NI+2.0', {'NI': 1.0}, charge=2.0)), (v.Species('VA-1.0', {'VA': 1.0}, charge=-1.0), v.Species('VA-1.0', {'VA': 1.0}, charge=-1.0))),
                "parameter_type": "MQMX",
                "phase_name": "REGLIQ",
                "mixing_code": "G",  # special types, "G" is one equation, "Q" is another. Thermochimica makes mention of "R" and "B", but I don't know if they are implemented and there's no equations in the paper. It could make sense just to have these as different `parameter_type`
                "additional_mixing_constituent": None,  # Can be a v.Species() that is not in the constituent array. This for ternary mixing. With the way the equations are laid out, it probably makes sense to keep it separate from the constituent array
                "additional_mixing_exponent": 0
            },
        ]

        cations = self.cations
        anions = self.anions

        p = self._p
        Z = partial(self.Z, dbe)

        energy = S.Zero
        for param in params:
            (A, B), (X, Y) = param["constituent_array"]
            exponents = param["exponents"]
            mixing_code = param["mixing_code"]

            # TODO: handle (Chi and Zeta mixing) x (binary and ternary mixing)
            # Poschmann Eq. 23-26
            if A != B and X == Y:
                if param["mixing_code"] == "G":
                    # Poschmann Eq. 23 (cations mixing)
                    mixing_term = self._Chi_mix(dbe, A, B, X, X)**exponents[0] * self._Chi_mix(dbe, B, A, X, X)**exponents[1]
                else:
                    raise ValueError(f"Unknown mixing type code {mixing_code}")
            elif A == B and X != Y:
                raise NotImplementedError()
            else:
                raise ValueError(f"Unsupported mixing configuration for quadruplet {(A, B, X, Y)}")
            g = param["parameter"] * mixing_term


            # Poschmann Eq. 17
            cation_factor = S.Zero
            if A == B:
                for m in cations:
                    if m != A:
                        cation_factor += p(A,m,X,Y) / Z(A, A,m,X,Y)
                cation_factor *= Z(A, A,B,X,Y) / 2
            anion_factor = S.Zero
            if X == Y:
                for m in anions:
                    if m != X:
                        anion_factor += p(A,B,X,m) / Z(X, A,B,X,m)
                anion_factor *= Z(X, A,B,X,Y) / 2
            energy += 0.5 * g * (p(A,B,X,Y) + cation_factor + anion_factor)

        return energy

    def jorge_excess_mixing_energy(self, dbe):
    # def excess_mixing_energy(self, dbe):

        w = self.w
        ξ = self.ξ
        cations = self.cations
        anions = self.anions
        pair_query_1 = dbe._parameters.search(
            (where("phase_name") == self.phase_name) &
            (where("parameter_type") == "EXMG") &
            (where("constituent_array").test(self._mixing_test))
        )
        pair_query_3 = dbe._parameters.search(
            (where("phase_name") == self.phase_name) &
            (where("parameter_type") == "EXMQ") &
            (where("diffusing_species") != v.Species(None)) &
            (where("constituent_array").test(self._mixing_test))
        )

        indi_que_3 = [i["parameter_order"] for i in pair_query_3]
        X_ex = S.Zero
        for param in pair_query_1:
            index = param["parameter_order"]
            coeff = param["parameter"]
            diff = [i for i in indi_que_3 if 0 < (i - index) <= 4]
            X_ex_1 = S.One
            X_a_Xb_tern = S.One
            X_ex_2 = S.Zero

            if len(diff) == 0:
                X_ex_0 = 1
            for parse in pair_query_3:
                expon = parse["parameter"]
                diff_spe = parse["diffusing_species"]
                cons_arr = parse["constituent_array"]

                cons_cat = cons_arr[0]
                cons_an = cons_arr[1]
                A = cons_cat[0]
                B = cons_cat[1]
                X = cons_an[0]
                Y = cons_an[1]
                Sub_ex_1 = S.Zero
                Sub_ex_2 = S.Zero
                if A != B and X == Y:
                    Sub_ex_1 += 1
                elif A == B and X != Y:
                    Sub_ex_2 += 1

                if 0 < (parse["parameter_order"] - index) <= 4 and diff_spe in cons_cat:
                    X_ex_1 *= (self.X_1_2(dbe, cons_arr, diff_spe)) ** expon
                    if X_ex_1 == 1:
                        X_ex_0 = 0
                    else:
                        X_ex_0 = 1
                # elif diff_spe in cations and diff_spe not in cons_cat and \
                # 0<(parse['parameter_order']-index)<=2:
                #     X_tern_diff_spe=parse['parameter_order']-index
                #     X_a_Xb_tern*=self.X_1_2(dbe,cons_arr,cons_cat[X_tern_diff_spe-1])**expon

                # TODO: IMPORTANT!!!! MIGHT NEED TO ADD SOMETHING HERE TO MAKE SURE ORDER OF ELEMENTS IN QUAD IS NOT AFFECTING

                elif (
                    diff_spe in cations
                    and diff_spe not in cons_cat
                    and 0 < (parse["parameter_order"] - index) <= 4
                    and self.id_symm(dbe, A, B, diff_spe) == diff_spe
                ):
                    if 0 < (parse["parameter_order"] - index) <= 2:
                        X_tern_diff_spe = parse["parameter_order"] - index
                        X_a_Xb_tern *= self.X_1_2(dbe, cons_arr, cons_cat[X_tern_diff_spe - 1]) ** expon
                    # IMPORTANT! expon is not the best way to fix this for parameteres higher than 1
                    X_ex_2 += expon * (X_a_Xb_tern * (ξ(diff_spe, X) / w(X)) * ((1 - self.K_1_2(dbe, A, B) - self.K_1_2(dbe, B, A)) ** (expon - 1)))

                elif (
                    diff_spe in cations
                    and diff_spe not in cons_cat
                    and 2 < (parse["parameter_order"] - index) <= 4
                    and self.id_symm(dbe, A, B, diff_spe) == 0
                ):
                    # This is for when they're all in the same species group
                    if 0 < (parse["parameter_order"] - index) <= 2:
                        X_tern_diff_spe = parse["parameter_order"] - index
                        X_a_Xb_tern *= self.X_1_2(dbe, cons_arr, cons_cat[X_tern_diff_spe - 1]) ** expon

                    X_ex_2 += expon * (X_a_Xb_tern * (ξ(diff_spe, X) / w(X)) * ((1 - self.K_1_2(dbe, A, B) - self.K_1_2(dbe, B, A)) ** (expon - 1)))

                elif (
                    diff_spe in cations
                    and diff_spe not in cons_cat
                    and 2 < (parse["parameter_order"] - index) <= 4
                    and self.id_symm(dbe, A, B, diff_spe) == B
                ):
                    if 0 < (parse["parameter_order"] - index) <= 2:
                        X_tern_diff_spe = parse["parameter_order"] - index
                        X_a_Xb_tern *= self.X_1_2(dbe, cons_arr, cons_cat[X_tern_diff_spe - 1]) ** expon
                    X_ex_2 += expon * (X_a_Xb_tern * (ξ(diff_spe, X) / (w(X) * self.K_1_2(dbe, A, B))) * (1 - (ξ(A, X) / (w(X) * self.K_1_2(dbe, A, B)))) ** (expon - 1))

                elif (
                    diff_spe in cations
                    and diff_spe not in cons_cat
                    and 0 < (parse["parameter_order"] - index) <= 4
                    and self.id_symm(dbe, A, B, diff_spe) == A
                ):
                    if 0 < (parse["parameter_order"] - index) <= 2:
                        X_tern_diff_spe = parse["parameter_order"] - index
                        X_a_Xb_tern *= self.X_1_2(dbe, cons_arr, cons_cat[X_tern_diff_spe - 1]) ** expon
                    X_ex_2 += expon * (X_a_Xb_tern * (ξ(diff_spe, X) / (w(X) * self.K_1_2(dbe, B, A))) * (1 - (ξ(B, X) / (w(X) * self.K_1_2(dbe, B, A)))) ** (expon - 1))
                # LAST TERM IN THE 17.52 EQUATION
                elif diff_spe in anions and Sub_ex_1 == 1 and 0 < (parse["parameter_order"] - index) <= 4:
                    if 0 < (parse["parameter_order"] - index) <= 2:
                        X_tern_diff_spe = parse["parameter_order"] - index
                        X_a_Xb_tern *= self.X_1_2(dbe, cons_arr, cons_cat[X_tern_diff_spe - 1]) ** expon
                    X_ex_2 += expon * X_a_Xb_tern * w(diff_spe) * w(X) ** (expon - 1)

                elif (
                    diff_spe in anions
                    and diff_spe not in cons_an
                    and 0 < (parse["parameter_order"] - index) <= 4
                    and self.id_symm(dbe, X, Y, diff_spe) == diff_spe
                ):
                    if 0 < (parse["parameter_order"] - index) <= 2:
                        X_tern_diff_spe = parse["parameter_order"] - index
                        X_a_Xb_tern *= self.X_1_2(dbe, cons_arr, cons_an[X_tern_diff_spe - 1]) ** expon

                    X_ex_2 += expon * (X_a_Xb_tern * (ξ(A, diff_spe) / w(A)) * ((1 - self.K_1_2(dbe, X, Y) - self.K_1_2(dbe, Y, X)) ** (expon - 1)))

                elif (
                    diff_spe in anions
                    and diff_spe not in cons_an
                    and 0 < (parse["parameter_order"] - index) <= 4
                    and self.id_symm(dbe, X, Y, diff_spe) == 0
                ):
                    if 0 < (parse["parameter_order"] - index) <= 2:
                        X_tern_diff_spe = parse["parameter_order"] - index
                        X_a_Xb_tern *= self.X_1_2(dbe, cons_arr, cons_an[X_tern_diff_spe - 1]) ** expon

                    X_ex_2 += expon * (X_a_Xb_tern * (ξ(A, diff_spe) / w(A)) * ((1 - self.K_1_2(dbe, X, Y) - self.K_1_2(dbe, Y, X)) ** (expon - 1)))

                    X_ex_2 += expon * (X_a_Xb_tern * (ξ(A, diff_spe) / w(A)) * ((1 - self.K_1_2(dbe, X, Y) - self.K_1_2(dbe, Y, X)) ** (expon - 1)))

                elif (
                    diff_spe in anions
                    and diff_spe not in cons_an
                    and 2 < (parse["parameter_order"] - index) <= 4
                    and self.id_symm(dbe, X, Y, diff_spe) == X
                ):
                    if 0 < (parse["parameter_order"] - index) <= 2:
                        X_tern_diff_spe = parse["parameter_order"] - index
                        X_a_Xb_tern *= self.X_1_2(dbe, cons_arr, cons_an[X_tern_diff_spe - 1]) ** expon

                    X_ex_2 += expon * (X_a_Xb_tern * (ξ(A, diff_spe) / (w(A) * self.K_1_2(dbe, X, Y))) * (1 - (ξ(A, Y) / (w(A) * self.K_1_2(dbe, X, Y)))) ** (expon - 1))

                elif (
                    diff_spe in anions
                    and diff_spe not in cons_an
                    and 2 < (parse["parameter_order"] - index) <= 4
                    and self.id_symm(dbe, X, Y, diff_spe) == Y
                ):
                    if 0 < (parse["parameter_order"] - index) <= 2:
                        X_tern_diff_spe = parse["parameter_order"] - index
                        X_a_Xb_tern *= self.X_1_2(dbe, cons_arr, cons_an[X_tern_diff_spe - 1]) ** expon

                    X_ex_2 += expon * (X_a_Xb_tern * (ξ(A, diff_spe) / (w(A) * self.K_1_2(dbe, X, Y))) * (1 - (ξ(A, X) / (w(A) * self.K_1_2(dbe, X, Y)))) ** (expon - 1))
            # This is assuming that one wouldn't have both interaction parameters for anions and cations at the same time
            if X_ex_2 != 0 and expon == 0:
                expon += 1
            if X_ex_2 == 0:
                X_ex_2 += 1
                expon += 1
            X_ex += self.excess_mixing_t1(dbe, param["constituent_array"]) * coeff * X_ex_1 * (X_ex_2 / expon) * X_ex_0
        # used to multiplt X_ex_0 paramter. But I don't think it does anything anymore
        return X_ex

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
        obj : SymPy object
        symbols : dict mapping sympy.Symbol to SymPy object

        Returns
        -------
        SymPy object
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

    def _array_validity(self, constituent_array):
        """
        Return True if the constituent_array contains only active species of the current Model instance.
        """
        if len(constituent_array) != len(self.constituents):
            return False
        for param_sublattice, model_sublattice in zip(constituent_array, self.constituents):
            if not (set(param_sublattice).issubset(model_sublattice) or (param_sublattice[0] == v.Species("*"))):
                return False
        return True

    def _interaction_test(self, constituent_array):
        """
        Return True if the constituent_array is valid and has more than one
        species in at least one sublattice.
        """
        if not self._array_validity(constituent_array):
            return False
        return any([len(sublattice) > 1 for sublattice in constituent_array])

    def _build_reference_model(self, preserve_ideal=True):
        raise NotImplementedError()

    @property
    def reference_model(self):
        raise NotImplementedError()
