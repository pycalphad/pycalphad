import copy
import itertools
from typing import List, Tuple
from collections import Counter, OrderedDict
from functools import partial
from sympy import exp, log, Abs, Add, And, Float, Mul, Piecewise, Pow, S, sin, StrictGreaterThan, Symbol, zoo, oo, nan
from tinydb import where
from pycalphad.model import _MAX_PARAM_NESTING
import pycalphad.variables as v
from pycalphad.core.utils import unpack_components, wrap_symbol, get_pure_elements
from pycalphad import Model
import numpy as np
from fractions import Fraction

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
    name = ''.join(constituents)
    constituent_dict = {}
    # using get() will increment c instead of overriding if c is already defined
    for c in constituents:
        constituent_dict[c] = constituent_dict.get(c, 0.0) + 1.0
    return v.Species(name, constituents=constituent_dict)


# TODO: cleanup this class (style)
# TODO: remove dead code (e.g. duplicate reference_energy)
# TODO: document the model contributions with the mathematics
class ModelMQMQA:
    """

    One peculiarity about the ModelMQMQA is that the charges in the way the
    model are written are assumed to be positive. We take the absolute value
    whenever there is a charge.

    """
    contributions = [
        ('ref', 'traditional_reference_energy'),
        ('idmix', 'ideal_mixing_energy'),
        ('xsmix', 'excess_mixing_energy'),
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
        if len(constituents)==1:
            self.ele=constituents
        else:
            self.cations = sorted(constituents[0])
            self.anions = sorted(constituents[1])

        # In several places we use the assumption that the cation lattice and
        # anion lattice have no common species; we validate that assumption here
        shared_species = set(self.cations).intersection(set(self.anions))
        assert len(shared_species) == 0, f"No species can be shared between the two MQMQA lattices, got {shared_species}"

        quads = itertools.product(itertools.combinations_with_replacement(self.cations, 2), itertools.combinations_with_replacement(self.anions, 2))
        quad_species = [get_species(A,B,X,Y) for (A, B), (X, Y) in quads]
        self.constituents = [sorted(quad_species)]

        # Verify that this phase is still possible to build
        if len(self.cations) == 0:
            raise DofError(f'{self.phase_name}: Cation sublattice of {phase.constituents[0]} has no active species in {self.components}')
        if len(self.anions) == 0:
            raise DofError(f'{self.phase_name}: Anion sublattice of {phase.constituents[1]} has no active species in {self.components}')

        # Set self.nonvacant_elements, only used by get_multiphase_constraint
        # TODO: can we remove this? or re-work it?
        desired_active_pure_elements = [list(x.constituents.keys()) for x in self.components]
        desired_active_pure_elements = [el.upper() for constituents in desired_active_pure_elements
                                        for el in constituents]
        pure_elements = sorted(set(desired_active_pure_elements))
        self.nonvacant_elements = [x for x in pure_elements if x != 'VA']

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
        """Return True if the constituent array is satisfies all components.
                """
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
#        print(species.constituents, species.charge)
        # aliases for notation
        Z = partial(self.Z, dbe)
        p = self._p
        M = S.Zero
#        print(species,species.constituents)
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
                    M += p(A,B,X,X)/Z(X,A,B,X,X)
                    for Y in anions:
                        M += p(A, B, X, Y)/Z(X, A, B, X, Y)
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
            p(A,A,X,X) +
        sum(p(A,A,X,Y) for Y in anions) +
        sum(p(A,B,X,X) for B in cations) +
        sum(p(A,B,X,Y) for B, Y in itertools.product(cations, anions))
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
        return 0.5*w

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
            return self.M(dbe, species)/sum(self.M(dbe, sp) for sp in cations)
        else:
            assert species in anions
            return self.M(dbe, species)/sum(self.M(dbe, sp) for sp in anions)

    def _calc_Z(self, dbe: 'pycalphad.io.Database', species, A, B, X, Y):
        Z = partial(self.Z,  dbe)
#         print(f'calculating $Z^{{{species}}}_{{{A} {B} {X} {Y}}}$')
        if (species == A) or (species == B):
            species_is_cation = True
        elif (species == X) or (species == Y):
            species_is_cation = False
        else:
            raise ValueError(f"{species} is not A ({A}), B ({B}), X ({X}) or Y ({Y}).")

        if A == B and X == Y:
            raise ValueError(f'Z({species}, {A}{B}/{X}{Y}) is a pure pair and must be defined explictly')
        elif A != B and X != Y:
            # This is a reciprocal AB/XY quadruplet and needs to be calculated by eq 23 and 24 in Pelton et al. Met Trans B (2001)
            F = 1/8 * (  # eq. 24
                  abs(A.charge)/Z(A, A, A, X, Y)
                + abs(B.charge)/Z(B, B, B, X, Y)
                + abs(X.charge)/Z(X, A, B, X, X)
                + abs(Y.charge)/Z(Y, A, B, Y, Y)
                )
            if species_is_cation:
                inv_Z = F * (
                              Z(X, A, B, X, X)/(abs(X.charge) * Z(species, A, B, X, X))
                            + Z(Y, A, B, Y, Y)/(abs(Y.charge) * Z(species, A, B, Y, Y))
                            )

            else:
                inv_Z = F * (
                              Z(A, A, A, X, Y)/(abs(A.charge) * Z(species, A, A, X, Y))
                            + Z(B, B, B, X, Y)/(abs(B.charge) * Z(species, B, B, X, Y))
                            )

            return 1/inv_Z
        elif A != B:  # X == Y
            # Need to calculate Z^i_AB/XX (Y = X).
            # We assume Z^A_ABXX = Z^A_AAXX = Z^A_AAYY
            # and Z^X_ABXX = (q_X + q_Y)/(q_A/Z^A_AAXX + q_B/Z^B_BBXX)  # note: q_X = q_Y, etc. since Y = X
            # We don't know if these are correct, but that's what's implemented in Thermochimica
            if species_is_cation:
                return Z(species, species, species, X, X)
            else:
#                 print(f'calculating bad $Z^{{{species}}}_{{{A} {B} {X} {Y}}}$')
                return 2*abs(species.charge)/(abs(A.charge)/Z(A, A, A, species, species) + abs(B.charge)/Z(B, B, B, species, species))
        elif X != Y:  # A == B
            # These use the same equations as A != B case with the same assumptions
            if species_is_cation:
                # similarly, Z^A_AAXY = (q_A + q_B)/(q_X/Z^X_AAXX + q_Y/Z^Y_AAYY)
#                 print(f'calculating bad $Z^{{{species}}}_{{{A} {B} {X} {Y}}}$')
                return 2*abs(species.charge)/(abs(X.charge)/Z(X, species, species, X, X) + abs(Y.charge)/Z(Y, species, species, Y, Y))
            else:
                return Z(species, A, A, species, species)
        raise ValueError("This should be unreachable")


    def Z(self, dbe: 'pycalphad.io.Database', species: v.Species, A: v.Species, B: v.Species, X: v.Species, Y: v.Species):
        Z_cat=sorted((A,B))
        Z_an=sorted((X,Y))
        Z_quad=(tuple(Z_cat),tuple(Z_an))
        Zs = dbe._parameters.search(
            (where('phase_name') == self.phase_name) & \
            (where('parameter_type') == "Z") & \
            (where('diffusing_species').test(lambda sp: sp.name == species.name)) & \
            (where('constituent_array').test(lambda x: x == Z_quad)))
            # quadruplet needs to be in 1 sublattice constituent array `[[q]]`, in tuples
        if len(Zs) == 0:
            # TODO: add this to the database so we don't need to recalculate? where should that happen?
            return self._calc_Z(dbe, species, A, B, X, Y)
        elif len(Zs) == 1:
            return Zs[0]['parameter']
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
        #No_Vac is to fix the normalization so that it does not take vacancy into consideration 
        no_vac=[j for j in self.components for o in j.constituents if o!='VA']
        const_spe=[k for i in no_vac for j,k in i.constituents.items()]        
        return sum(self.M(self._dbe, c)*const_spe[count] for count,c in enumerate(no_vac))    
    
    def moles(self, species, per_formula_unit=False):
        "Number of moles of species or elements."
        species = v.Species(species)
        result = S.Zero
        for i in itertools.chain(self.cations, self.anions):
            if list(species.constituents.keys())[0] in i.constituents:
                count=list(i.constituents.values())[0]
                result += self.M(self._dbe, i)*count
        # moles is supposed to compute the moles of a pure element, but with a caveat that pycalphad assumes sum(moles(c) for c in comps) == 1
        # The correct solution is to make the changes where pycalphad assumes n=1. But I think it would be easier to change how we implement the model so that the model has n=1 and the energies are normalized to per-mole-atoms.
        # Since normalizing to moles of quadruplets is allowing us to easily compare with thermochimica, I'm thinking that we might be able to fake pycalphad into thinking we have N=1 by normalizing "moles" to n=1
        # The energies will not be normalized to moles of atoms (and so you cannot yet use this Model to compare to other phases), but internally it should be correct and in agreement with thermochimica
        if per_formula_unit:
            return result
        else:
            return result/self.normalization

    def moles_(self, species):
        "Number of moles of species or elements."
        species = v.Species(species)
        result = S.Zero
        for i in itertools.chain(self.cations, self.anions):
            if list(species.constituents.keys())[0] in i.constituents:
                result += self.M(self._dbe, i)
        # moles is supposed to compute the moles of a pure element, but with a caveat that pycalphad assumes sum(moles(c) for c in comps) == 1
        # The correct solution is to make the changes where pycalphad assumes n=1. But I think it would be easier to change how we implement the model so that the model has n=1 and the energies are normalized to per-mole-atoms.
        # Since normalizing to moles of quadruplets is allowing us to easily compare with thermochimica, I'm thinking that we might be able to fake pycalphad into thinking we have N=1 by normalizing "moles" to n=1
        # The energies will not be normalized to moles of atoms (and so you cannot yet use this Model to compare to other phases), but internally it should be correct and in agreement with thermochimica
        return result
    
    def Coax(self,dbe,A,B,X,Y):
    #Taking nomenclature from thermochimica. Only going to calculate the one for the cation
    #This value is important when calculating the surface energies 
        Z = partial(self.Z, dbe)
        Coa=Z(A,A,B,X,Y)
        Cox=Z(X,A,B,X,Y)
#        print(Coa,Cox)
        ratio_C=Coa/Cox
        if ratio_C==ratio_C:
            fin_Coa=1
        else:
            frac=Fraction(ratio_C).limit_denominator()
            low_com_mul=np.lcm(frac.numerator,frac.denominator)
            fin_Coa=low_com_mul/frac.numerator
        return fin_Coa
        

    @property
    def degree_of_ordering(self):
        result = S.Zero
        site_ratio_normalization = S.Zero
        # Calculate normalization factor
        for idx, sublattice in enumerate(self.constituents):
            active = set(sublattice).intersection(self.components)
            subl_content = sum(int(spec.number_of_atoms > 0) * v.SiteFraction(self.phase_name, idx, spec) for spec in active)
            site_ratio_normalization += self.site_ratios[idx] * subl_content

        site_ratios = [c/site_ratio_normalization for c in self.site_ratios]
        for comp in self.components:
            if comp.number_of_atoms == 0:
                continue
            comp_result = S.Zero
            for idx, sublattice in enumerate(self.constituents):
                active = set(sublattice).intersection(set(self.components))
                if comp in active:
                    comp_result += site_ratios[idx] * Abs(v.SiteFraction(self.phase_name, idx, comp) - self.moles(comp)) / self.moles(comp)
            result += comp_result
        return result / sum(int(spec.number_of_atoms > 0) for spec in self.components)
    DOO = degree_of_ordering

    # Can be defined as a list of pre-computed first derivatives
    gradient = None

    # Note: In order-disorder phases, TC will always be the *disordered* value of TC
    curie_temperature = TC = S.Zero
    beta = BMAG = S.Zero
    neel_temperature = NT = S.Zero

    #pylint: disable=C0103
    # These are standard abbreviations from Thermo-Calc for these quantities
    GM = property(lambda self: self.ast)
    G = property(lambda self: self.ast*self.normalization)
    energy = GM
    entropy = SM = property(lambda self: -self.GM.diff(v.T))
    enthalpy = HM = property(lambda self: self.GM - v.T*self.GM.diff(v.T))
    heat_capacity = CPM = property(lambda self: -v.T*self.GM.diff(v.T, v.T))
    #pylint: enable=C0103
    mixing_energy = GM_MIX = property(lambda self: self.GM - self.reference_model.GM)
    mixing_enthalpy = HM_MIX = property(lambda self: self.GM_MIX - v.T*self.GM_MIX.diff(v.T))
    mixing_entropy = SM_MIX = property(lambda self: -self.GM_MIX.diff(v.T))
    mixing_heat_capacity = CPM_MIX = property(lambda self: -v.T*self.GM_MIX.diff(v.T, v.T))

    def traditional_reference_energy(self,dbe):
        Gibbs={}
        pair_query = (
            (where('phase_name') == self.phase_name) & \
            (where('parameter_order') == 0) & \
            (where('parameter_type') == "G") & \
            (where('constituent_array').test(self._pair_test))
        )
        cations = self.cations
        anions = self.anions
        params = dbe._parameters.search(pair_query)
        p = self._p
        surf=S.Zero
        for param in params:
            subl_1 = param['constituent_array'][0]
            subl_2 = param['constituent_array'][1]

            A=subl_1[0]
            B=subl_1[0]
            X=subl_2[0]
            Y=subl_2[0]
            Gibbs[A,B,X,Y]=param['parameter']
        
#        NA=[i for i in cations for name,count in i.constituents.items() if name=='NA'][0]
#        CL=[i for i in anions for name,count in i.constituents.items() if name=='CL'][0]
#        VA=[i for i in anions for name,count in i.constituents.items() if name=='VA'][0]
        
#        AL1=[i for i in cations for name,count in i.constituents.items() if name=='AL' and count==1][0]
#        AL2=[i for i in cations for name,count in i.constituents.items() if name=='AL' and count==2][0]

        for i, A in enumerate(cations):
            for B in cations[i:]:
                for j, X in enumerate(anions):
                    for Y in anions[j:]:
                        term1=((abs(X.charge)/self.Z(dbe,X,A,B,X,Y))+(abs(Y.charge)/self.Z(dbe,Y,A,B,X,Y)))**(-1)
#                        print('term1',self.Z(dbe,X,A,B,X,Y),self.Z(dbe,Y,A,B,X,Y),A,B,X,Y)
                        term2=(abs(X.charge)*self.Z(dbe,A,A,A,X,X)/(2*self.Z(dbe,A,A,B,X,Y)*self.Z(dbe,X,A,B,X,Y)))*(Gibbs[A,A,X,X]*2/(self.Z(dbe,A,A,A,X,X)*self.Coax(dbe,A,A,X,X)))
#                        print('second term',Gibbs[A,A,X,X],A,B,X,Y)
                        term3=(abs(X.charge)*self.Z(dbe,B,B,B,X,X)/(2*self.Z(dbe,B,A,B,X,Y)*self.Z(dbe,X,A,B,X,Y)))*(Gibbs[B,B,X,X]*2/(self.Z(dbe,B,B,B,X,X)*self.Coax(dbe,B,B,X,X)))#*self.Coax(dbe,B,B,X,X)
#                        print('third term',term3,A,B,X,Y)
                        term4=(abs(Y.charge)*self.Z(dbe,A,A,A,Y,Y)/(2*self.Z(dbe,A,A,B,X,Y)*self.Z(dbe,Y,A,B,X,Y)))*(Gibbs[A,A,Y,Y]*2/(self.Z(dbe,A,A,A,Y,Y)*self.Coax(dbe,A,A,Y,Y)))#*self.Coax(dbe,A,A,Y,Y)
#                        print('fourth term',term4,A,B,X,Y)
                        term5=(abs(Y.charge)*self.Z(dbe,B,B,B,Y,Y)/(2*self.Z(dbe,B,A,B,X,Y)*self.Z(dbe,Y,A,B,X,Y)))*(Gibbs[B,B,Y,Y]*2/(self.Z(dbe,B,B,B,Y,Y)*self.Coax(dbe,B,B,Y,Y)))#*self.Coax(dbe,B,B,Y,Y)
#                        print('fifth term',term5,A,B,X,Y)
                        final_term=term1*(term2+term3+term4+term5)
                        surf+=p(A,B,X,Y)*final_term
        return surf/self.normalization


    def reference_energy(self, dbe):
        """
        Returns the weighted average of the endmember energies
        in symbolic form.
        """
        pair_query = (
            (where('phase_name') == self.phase_name) & \
            (where('parameter_order') == 0) & \
            (where('parameter_type') == "G") & \
            (where('constituent_array').test(self._pair_test))
        )
        self._ξ = S.Zero
        params = dbe._parameters.search(pair_query)
        terms = S.Zero
        for param in params:
            A = param['constituent_array'][0][0]
            X = param['constituent_array'][1][0]
            ξ_AX = self.ξ(A, X)
            self._ξ += ξ_AX
            G_AX = param['parameter']
            Z = self.Z(dbe, A, A, A, X, X)
            terms += (ξ_AX * G_AX)*2/Z
        return terms/self.normalization

    def ideal_mixing_energy(self, dbe):
        # notational niceties
        M = partial(self.M, dbe)
        ϑ = partial(self.ϑ, dbe)
        ξ = self.ξ
        w = self.w
        p = self._p
        soln_type=dbe.phases[self.phase_name].model_hints['mqmqa']['type']
        cations = self.cations
        anions = self.anions
        if soln_type=='SUBQ':
            exp1=0.75
            exp2=0.5
        elif soln_type=='SUBG':
            exp1=1.0
            exp2=1.0       
        Sid = S.Zero
        self.t1 = S.Zero
        self.t2 = S.Zero
        self.t3 = S.Zero
        self.t4 = S.Zero
        ζ = 2.4  # hardcoded, but we can get it from the model_hints (SUBQ) or the pairs (SUBG)
        for A in cations:
            Sid += M(A)*log(ϑ(A))  # term 1
            self.t1 += M(A)*log(ϑ(A))
        for X in anions:
            Sid += M(X)*log(ϑ(X))  # term 2
            self.t2 += M(X)*log(ϑ(X))
        for A in cations:
            for X in anions:
                ξ_AX = ξ(A,X)
                p_AAXX = p(A,A,X,X)
                w_A = w(A)
                w_X = w(X)
#                Sid += 4/ζ*ξ_AX*log(ξ_AX/(w_A*w_X))  # term 3
                Sid += 4/ζ*ξ_AX*log(ξ_AX/(w_A*w_X))
                self.t3 += 4/ζ*ξ_AX*log(ξ_AX/(w_A*w_X))
#                self.t3 += ξ_AX*log(ξ_AX/(w_A*w_X))
        
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
#                        print(factor,A,B,X,Y)
                        Sid += p(A,B,X,Y)*log(p(A,B,X,Y)/(factor * (ξ(A,X)**(exp1))*(ξ(A,Y)**(exp1))*(ξ(B,X)**(exp1))*(ξ(B,Y)**(exp1)) / ((w(A)**(exp2))*(w(B)**(exp2))*(w(X)**(exp2))*(w(Y)**(exp2)))))
                        self.t4 = p(A,B,X,Y)*log(p(A,B,X,Y)/(factor * (ξ(A,X)**(1))*(ξ(A,Y)**(1))*(ξ(B,X)**(1))*(ξ(B,Y)**(1)) / ((w(A)**(0.5))*(w(B)**(0.5))*(w(X)**(0.5))*(w(Y)**(0.5)))))
    #                        self.t4 += p(A,B,X,Y)*log(p(A,B,X,Y)/(factor*(ξ(A,X)*ξ(A,Y)*ξ(B,X)*ξ(B,Y))**(0.75) / (w(A)*w(B)*w(X)*w(Y))**(0.5)))
#                        self.t4 += factor * ξ(A,X)*ξ(A,Y)*ξ(B,X)*ξ(B,Y) / (w(A)*w(B)*w(X)*w(Y))
        return Sid*v.T*v.R/self.normalization#(self.t1+self.t2+self.t3+self.t4)


    def excess_mixing_t1(self,dbe,constituent_array):
#        Exid=S.Zero
        Z = partial(self.Z, dbe)
        cations=self.cations
        anions=self.anions
        p = self._p
        subl_1 = constituent_array[0]
        subl_2 = constituent_array[1]
        A=subl_1[0]
        B=subl_1[1]
        X=subl_2[0]
        Y=subl_2[1]
        test=[p(A,B,i,Y) for i in anions if i!=Y]
##Figure out how to connect this. Below is the correct expression. Maybe this can be its own function separately
#And it can be called in the other final function
        return 0.5*(p(A,B,X,Y)
                    +sum(0.5*Z(j,A,B,j,j)*sum(p(A,B,i,Y)/Z(Y,A,B,i,Y) for i in anions if i!=Y) for j in anions if j==X==Y)
                    +sum(0.5*Z(q,q,q,X,Y)*sum(p(r,B,X,Y)/Z(B,r,B,X,Y) for r in cations if r!=B) for q in cations if q==A==B))


    def X_1_2(self,dbe, constituent_array, diffusing_species):
        chem_groups_cat=dbe.phases[self.phase_name].model_hints['mqmqa']['chemical_groups']['cations']
        chem_groups_an=dbe.phases[self.phase_name].model_hints['mqmqa']['chemical_groups']['anions']
        soln_type=dbe.phases[self.phase_name].model_hints['mqmqa']['type']

        cations=self.cations
        anions=self.anions
        p = self._p
        res1=S.Zero
        res2=S.Zero
        res1_rec_quad=S.Zero

        subl_1 = constituent_array[0]
        subl_2 = constituent_array[1]
        A=subl_1[0]
        B=subl_1[1]
        X=subl_2[0]
        Y=subl_2[1]
        #non_diff_spe is either the first element of the subl1/2 or is the remaining element of subl1/2 when diffusing species is one of the elements in the sublattice                                  
        if diffusing_species in subl_1:
            non_diff_spe=[i for i in subl_1 if i !=diffusing_species][0]
            k_As_cat=[i for i in cations if chem_groups_cat[i]!=chem_groups_cat[diffusing_species] if diffusing_species in cations and i not in subl_1]
            l_As_cat=[i for i in cations if chem_groups_cat[i]!=chem_groups_cat[non_diff_spe] if diffusing_species and i not in subl_1]
        elif diffusing_species in subl_2:
            non_diff_spe=[i for i in subl_2 if i !=diffusing_species][0]
            k_As_an=[i for i in anions if chem_groups_an[i]!=chem_groups_an[diffusing_species] if diffusing_species in anions and i not in subl_1]
            l_As_an=[i for i in anions if chem_groups_an[i]!=chem_groups_an[non_diff_spe] if diffusing_species in anions and i not in subl_1]
####This is all assuming that there will be only two groups for symmetrical and asymmetrical

        if X==Y and diffusing_species in subl_1:
            As_diff=[diffusing_species]
            if Counter(k_As_cat)!=Counter(l_As_cat):
                As_diff.extend(l_As_cat)
            for count,a in enumerate(As_diff):
                for b in As_diff[count:]:
                    res1+=p(a,b,X,Y)
                    if soln_type=='SUBQ':
                        res1+=0.5*sum(p(a,b,X,Y) for Y in anions if Y!=X)
#            res1+=0.5*sum(p(A,B,X,Y) for Y in anions if Y!=X)

            if Counter(k_As_cat)!=Counter(l_As_cat):
                subl_1=list(subl_1)
                subl_1.extend(k_As_cat)
                subl_1.extend(l_As_cat)

            for count,a in enumerate(subl_1):
                for b in subl_1[count:]:
                    res2+=p(a,b,X,Y)
                    if soln_type=='SUBQ':
                        res2+=0.5*sum(p(a,b,X,Y) for Y in anions if Y!=X)
        elif A==B and diffusing_species in subl_2:
            As_diff=[diffusing_species]
            if Counter(k_As_an)!=Counter(l_As_an):
                As_diff.extend(l_As_an)
            for count,x in enumerate(As_diff):
                for y in As_diff[count:]:
                    res1+=p(A,B,x,y)
                    if soln_type=='SUBQ':
                        res1+=0.5*sum(p(A,B,x,y) for B in cations if A!=B)
#            res1+=0.5*sum(p(A,B,X,Y) for Y in anions if Y!=X)

            if Counter(k_As_an)!=Counter(l_As_an):
                subl_2=list(subl_2)
                subl_2.extend(k_As_an)
                subl_2.extend(l_As_an)

            for count,a in enumerate(subl_2):
                for b in subl_2[count:]:
                    res2+=p(A,B,x,y)
                    if soln_type=='SUBQ':                    
                        res2+=0.5*sum(p(A,B,x,y) for B in cations if A!=B)
        return res1/res2

    def id_symm(self,dbe,A,B,C):
        cations=self.cations
        anions=self.anions
        chem_groups_cat=dbe.phases[self.phase_name].model_hints['mqmqa']['chemical_groups']['cations']
        chem_groups_an=dbe.phases[self.phase_name].model_hints['mqmqa']['chemical_groups']['anions']
        in_lis=[A,B,C]
        if set(in_lis).issubset(cations) is True:
            chem_lis=[i for i in in_lis if chem_groups_cat[i]!=chem_groups_cat[A]]
        if set(in_lis).issubset(anions) is True:
            chem_lis=[i for i in in_lis if chem_groups_an[i]!=chem_groups_an[A]]
        if len(chem_lis)==1:
            symm_check=chem_lis[0]
        elif len(chem_lis)>1:
            symm_check_2=list(set(in_lis)-set(chem_lis))
            symm_check=symm_check_2[0]
        else:
            symm_check=0
        return symm_check


    def K_1_2(self,dbe,A,B):
        w = self.w
        num_res=S.Zero
        chem_groups_cat=dbe.phases[self.phase_name].model_hints['mqmqa']['chemical_groups']['cations']
        chem_groups_an=dbe.phases[self.phase_name].model_hints['mqmqa']['chemical_groups']['anions']
        cations=self.cations
        anions=self.anions
        result_cons=[A,B]
        the_rest=set(cations)-set(result_cons)
        result_2=[i for i in the_rest if chem_groups_cat[i]==chem_groups_cat[result_cons[0]] and chem_groups_cat[i]!=chem_groups_cat[result_cons[1]]]
        num_res+=w(A)

        for i in result_2:
            num_res+=w(i)

        return num_res

    def excess_mixing_energy(self,dbe):

        w = self.w
        ξ = self.ξ
        cations=self.cations
        anions=self.anions
        pair_query_1 = dbe._parameters.search(
            (where('phase_name') == self.phase_name) & \
            (where('parameter_type') == "EXMG") & \
            (where('constituent_array').test(self._mixing_test))
        )
        pair_query_3 = dbe._parameters.search(
            (where('phase_name') == self.phase_name) & \
            (where('parameter_type') == "EXMQ") & \
            (where('diffusing_species') != v.Species(None)) & \
            (where('constituent_array').test(self._mixing_test))
        )
        test=[i['parameter'] for i in pair_query_3]
#        print(pair_query_1,pair_query_3)
        indi_que_3=[i['parameter_order'] for i in pair_query_3 ]
        chem_groups=dbe.phases[self.phase_name].model_hints['mqmqa']['chemical_groups']
        X_ex=S.Zero
        X_tern_original=S.One
        for param in pair_query_1:
#            print('parameter in query 1',param)
            index=param['parameter_order']
            coeff=param['parameter']
            diff=[i for i in indi_que_3 if 0<(i-index)<=4]
            X_ex_1=S.One
            X_a_Xb_tern=S.One
            X_ex_2=S.Zero

            if len(diff)==0:
                X_ex_0=1
            for parse in pair_query_3:
                exp=parse['parameter']
                diff_spe=parse['diffusing_species']
                cons_arr=parse['constituent_array']
#                print('exponent',exp,coeff,parse['parameter_order'])
                cons_cat=cons_arr[0]
                cons_an=cons_arr[1]
                A=cons_cat[0]
                B=cons_cat[1]
                X=cons_an[0]
                Y=cons_an[1]
                Sub_ex_1=S.Zero
                Sub_ex_2=S.Zero
                if A!=B and X==Y:
                    Sub_ex_1+=1
                elif A==B and X!=Y:
                    Sub_ex_2+=1   
#                print('JORGE TEST!',A,B,X,Y,Sub_ex_1,Sub_ex_2)
                if 0<(parse['parameter_order']-index)<=4 and diff_spe in cons_cat:
                    X_ex_1*=(self.X_1_2(dbe,cons_arr,diff_spe))**exp
#                    print(coeff)
#                    print(exp)
#                    print('This is the cons_array mayn',cons_arr, 'ANNND diffusing species',diff_spe)
#                    print(self.X_1_2(dbe,cons_arr,diff_spe))
                    if X_ex_1==1:
                        X_ex_0=0
                    else:
                        X_ex_0=1                               
#                elif diff_spe in cations and diff_spe not in cons_cat and \
#                0<(parse['parameter_order']-index)<=2:
#                    X_tern_diff_spe=parse['parameter_order']-index
#                    X_a_Xb_tern*=self.X_1_2(dbe,cons_arr,cons_cat[X_tern_diff_spe-1])**exp


####IMPORTANT!!!! MIGHT NEED TO ADD SOMETHING HERE TO MAKE SURE ORDER OF ELEMENTS IN QUAD IS NOT AFFECTING


                elif diff_spe in cations and diff_spe not in cons_cat \
                and 0<(parse['parameter_order']-index)<=4 \
                and self.id_symm(dbe,A,B,diff_spe)==diff_spe:
                    if 0<(parse['parameter_order']-index)<=2:
                        X_tern_diff_spe=parse['parameter_order']-index
                        X_a_Xb_tern*=self.X_1_2(dbe,cons_arr,cons_cat[X_tern_diff_spe-1])**exp
#IMPORTANT! exp is not the best way to fix this for parameteres higher than 1                        
                    X_ex_2+=exp*(X_a_Xb_tern*(ξ(diff_spe,X)/w(X))\
                    *((1-self.K_1_2(dbe,A,B)-self.K_1_2(dbe,B,A))**(exp-1)))
    
                elif diff_spe in cations and diff_spe not in cons_cat \
                and 2<(parse['parameter_order']-index)<=4 \
                and self.id_symm(dbe,A,B,diff_spe)==0:
                 #This is for when they're all in the same species group
                    if 0<(parse['parameter_order']-index)<=2:
                        X_tern_diff_spe=parse['parameter_order']-index
                        X_a_Xb_tern*=self.X_1_2(dbe,cons_arr,cons_cat[X_tern_diff_spe-1])**exp

                    X_ex_2+=exp*(X_a_Xb_tern*(ξ(diff_spe,X)/w(X))\
                    *((1-self.K_1_2(dbe,A,B)-self.K_1_2(dbe,B,A))**(exp-1)))
#                    print(X_ex_2,coeff,self.excess_mixing_t1(dbe,param['constituent_array']))

                elif diff_spe in cations and diff_spe not in cons_cat \
                and 2<(parse['parameter_order']-index)<=4 \
                and self.id_symm(dbe,A,B,diff_spe)==B:
                    if 0<(parse['parameter_order']-index)<=2:
                        X_tern_diff_spe=parse['parameter_order']-index
                        X_a_Xb_tern*=self.X_1_2(dbe,cons_arr,cons_cat[X_tern_diff_spe-1])**exp        
                    X_ex_2+=exp*(X_a_Xb_tern*(ξ(diff_spe,X)/(w(X)*self.K_1_2(dbe,A,B)))\
                    *(1-(ξ(A,X)/(w(X)*self.K_1_2(dbe,A,B))))**(exp-1))

                elif diff_spe in cations and diff_spe not in cons_cat \
                and 0<(parse['parameter_order']-index)<=4 \
                and self.id_symm(dbe,A,B,diff_spe)==A:
                    if 0<(parse['parameter_order']-index)<=2:
                        X_tern_diff_spe=parse['parameter_order']-index
                        X_a_Xb_tern*=self.X_1_2(dbe,cons_arr,cons_cat[X_tern_diff_spe-1])**exp        
                    X_ex_2+=exp*(X_a_Xb_tern*(ξ(diff_spe,X)/(w(X)*self.K_1_2(dbe,B,A)))\
                    *(1-(ξ(B,X)/(w(X)*self.K_1_2(dbe,B,A))))**(exp-1))
####LAST TERM IN THE 17.52 EQUATION                    
                elif diff_spe in anions and Sub_ex_1==1 \
                and 0<(parse['parameter_order']-index)<=4:
                    if 0<(parse['parameter_order']-index)<=2:
                        X_tern_diff_spe=parse['parameter_order']-index
                        X_a_Xb_tern*=self.X_1_2(dbe,cons_arr,cons_cat[X_tern_diff_spe-1])**exp        
                    X_ex_2+=exp*X_a_Xb_tern*w(diff_spe)*w(X)**(exp-1)
                    
                elif diff_spe in anions and diff_spe not in cons_an \
                and 0<(parse['parameter_order']-index)<=4 \
                and self.id_symm(dbe,X,Y,diff_spe)==diff_spe:
                    if 0<(parse['parameter_order']-index)<=2:
                        X_tern_diff_spe=parse['parameter_order']-index
                        X_a_Xb_tern*=self.X_1_2(dbe,cons_arr,cons_an[X_tern_diff_spe-1])**exp
                        
                    X_ex_2+=exp*(X_a_Xb_tern*(ξ(A,diff_spe)/w(A))\
                    *((1-self.K_1_2(dbe,X,Y)-self.K_1_2(dbe,Y,X))**(exp-1)))                    

                elif diff_spe in anions and diff_spe not in cons_an \
                and 0<(parse['parameter_order']-index)<=4 \
                and self.id_symm(dbe,X,Y,diff_spe)==0:
                    if 0<(parse['parameter_order']-index)<=2:
                        X_tern_diff_spe=parse['parameter_order']-index
                        X_a_Xb_tern*=self.X_1_2(dbe,cons_arr,cons_an[X_tern_diff_spe-1])**exp
                        
                    X_ex_2+=exp*(X_a_Xb_tern*(ξ(A,diff_spe)/w(A))\
                    *((1-self.K_1_2(dbe,X,Y)-self.K_1_2(dbe,Y,X))**(exp-1)))  

                        
                    X_ex_2+=exp*(X_a_Xb_tern*(ξ(A,diff_spe)/w(A))\
                    *((1-self.K_1_2(dbe,X,Y)-self.K_1_2(dbe,Y,X))**(exp-1)))  

                    
                elif diff_spe in anions and diff_spe not in cons_an \
                and 2<(parse['parameter_order']-index)<=4 \
                and self.id_symm(dbe,X,Y,diff_spe)==X:
                    if 0<(parse['parameter_order']-index)<=2:
                        X_tern_diff_spe=parse['parameter_order']-index
                        X_a_Xb_tern*=self.X_1_2(dbe,cons_arr,cons_an[X_tern_diff_spe-1])**exp
                        
                    X_ex_2+=exp*(X_a_Xb_tern*(ξ(A,diff_spe)/(w(A)*self.K_1_2(dbe,X,Y)))\
                    *(1-(ξ(A,Y)/(w(A)*self.K_1_2(dbe,X,Y))))**(exp-1))
                    
                elif diff_spe in anions and diff_spe not in cons_an \
                and 2<(parse['parameter_order']-index)<=4 \
                and self.id_symm(dbe,X,Y,diff_spe)==Y:
                    if 0<(parse['parameter_order']-index)<=2:
                        X_tern_diff_spe=parse['parameter_order']-index
                        X_a_Xb_tern*=self.X_1_2(dbe,cons_arr,cons_an[X_tern_diff_spe-1])**exp
                        
                    X_ex_2+=exp*(X_a_Xb_tern*(ξ(A,diff_spe)/(w(A)*self.K_1_2(dbe,X,Y)))\
                    *(1-(ξ(A,X)/(w(A)*self.K_1_2(dbe,X,Y))))**(exp-1))                    
#This is assuming that one wouldn't have both interaction parameters for anions and cations at the same time                                                      
            if X_ex_2!=0 and exp==0:
                exp+=1     
#            print('CHECK HERE',coeff,X_ex_1,X_ex_0,X_ex_2,exp)
            if X_ex_2==0:
                X_ex_2+=1
                exp+=1
#            print(param['constituent_array'])
#            print('part 1',self.excess_mixing_t1(dbe,param['constituent_array']),coeff)
            X_ex+=self.excess_mixing_t1(dbe,param['constituent_array'])*coeff*X_ex_1*(X_ex_2/exp)*X_ex_0
#used to multiplt X_ex_0 paramter. But I don't think it does anything anymore
        return X_ex/self.normalization

    def shift_reference_state(self, reference_states, dbe, contrib_mods=None, output=('GM', 'HM', 'SM', 'CPM'), fmt_str="{}R"):
        """
        Add new attributes for calculating properties w.r.t. an arbitrary pure element reference state.

        Parameters
        ----------
        reference_states : Iterable of ReferenceState
            Pure element ReferenceState objects. Must include all the pure
            elements defined in the current model.
        dbe : Database
            Database containing the relevant parameters.
        output : Iterable, optional
            Parameters to subtract the ReferenceState from, defaults to ('GM', 'HM', 'SM', 'CPM').
        contrib_mods : Mapping, optional
            Map of {model contribution: new value}. Used to adjust the pure
            reference model contributions at the time this is called, since
            the `models` attribute of the pure element references are
            effectively static after calling this method.
        fmt_str : str, optional
            String that will be formatted with the `output` parameter name.
            Defaults to "{}R", e.g. the transformation of 'GM' -> 'GMR'

        """
        # Error checking
        # We ignore the case that the ref states are overspecified (same ref states can be used in different models w/ different active pure elements)
#        active_species = unpack_components(dbe, comps)
        model_pure_elements = set(get_pure_elements(dbe, self.components))
        refstate_pure_elements_list = get_pure_elements(dbe, [r.species for r in reference_states])
        refstate_pure_elements = set(refstate_pure_elements_list)
        if len(refstate_pure_elements_list) != len(refstate_pure_elements):
            raise DofError("Multiple ReferenceState objects exist for at least one pure element: {}".format(refstate_pure_elements_list))
        if not refstate_pure_elements.issuperset(model_pure_elements):
            raise DofError("Non-existent ReferenceState for pure components {} in {} for {}".format(model_pure_elements.difference(refstate_pure_elements), self, self.phase_name))

        contrib_mods = contrib_mods or {}

        def _pure_element_test(constituent_array):
            all_comps = set()
            for sublattice in constituent_array:
                if len(sublattice) != 1:
                    return False
                all_comps.add(sublattice[0].name)
            pure_els = all_comps.intersection(model_pure_elements)
            return len(pure_els) == 1
        # Remove interactions from a copy of the Database, avoids any element/VA interactions.
        endmember_only_dbe = copy.deepcopy(dbe)
        endmember_only_dbe._parameters.remove(~where('constituent_array').test(_pure_element_test))
        reference_dict = {out: [] for out in output}  # output: terms list
        pure_components=[i.constituents for i in self.components]
        comps=[j for i in self.components for j in i.constituents.keys()]

        for ref_state in reference_states:
            mod_hints=[i for i in dbe.phases[ref_state.phase_name].model_hints.keys()]
            if ref_state.species.constituents not in pure_components:
                continue
            if 'mqmqa' not in mod_hints:
                mod_Mod=Model(dbe,comps,ref_state.phase_name)
                mod_pure= mod_Mod.__class__(endmember_only_dbe, [ref_state.species, v.Species('VA')], ref_state.phase_name, parameters=self._parameters_arg)
            else:
                mod_pure =self.__class__(endmember_only_dbe, [ref_state.species, v.Species('VA')], ref_state.phase_name, parameters=self._parameters_arg)
#            print('DOES MOD WORK?',mod)
#THIS IS TAKING INTO CONSIDERATION THE OTHER MODEL! WILL NEED TO ADD AN IF STATEMENT TO DISTINGUISH

#            mod_pure = self.__class__(endmember_only_dbe, [ref_state.species, v.Species('VA')], ref_state.phase_name, parameters=self._parameters_arg)
            # apply the modifications to the Models
            for contrib, new_val in contrib_mods.items():
                mod_pure.models[contrib] = new_val
            # set all the free site fractions to one, this should effectively delete any mixing terms spuriously added, e.g. idmix
            site_frac_subs = {sf: 1 for sf in mod_pure.ast.free_symbols if isinstance(sf, v.SiteFraction)}
            for mod_key, mod_val in mod_pure.models.items():
                mod_pure.models[mod_key] = mod_val.subs(site_frac_subs)
                
            moles = self.moles(ref_state.species)                
            # get the output property of interest, substitute the fixed state variables (e.g. T=298.15) and add the pure element moles weighted term to the list of terms
            # substitution of fixed state variables has to happen after getting the attribute in case there are any derivatives involving that state variable
            for out in reference_dict.keys():
                mod_out = getattr(mod_pure, out).subs(ref_state.fixed_statevars)
                reference_dict[out].append(mod_out*moles)
                
        # set the attribute on the class
        for out, terms in reference_dict.items():
            reference_contrib = Add(*terms)
            referenced_value = getattr(self, out) - reference_contrib

            setattr(self, fmt_str.format(out), referenced_value)
            
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
            if not (set(param_sublattice).issubset(model_sublattice) or (param_sublattice[0] == v.Species('*'))):
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
        """
        Build a reference_model for the current model, referenced to the endmembers.

        Parameters
        ----------
        dbe : Database
        preserve_ideal : bool, optional
            If True, the default, the ideal mixing energy will not be subtracted out.


        See Also
        --------
        Model.reference_model

        Notes
        -----
        Requires that self.build_phase has already been called.

        """
        endmember_only_dbe = copy.deepcopy(self._dbe)
        endmember_only_dbe._parameters.remove(where('constituent_array').test(self._interaction_test))
        mod_endmember_only = self.__class__(endmember_only_dbe, self.components, self.phase_name, parameters=self._parameters_arg)
        if preserve_ideal:
            mod_endmember_only.models['idmix'] = 0
        self._reference_model = mod_endmember_only
        if self.models.get('ord', S.Zero) != S.Zero:
                for k in self.reference_model.models.keys():
                    self._reference_model.models[k] = nan

    @property
    def reference_model(self):
        """
        Return a Model containing only energy contributions from endmembers.

        Returns
        -------
        Model

        Notes
        -----
        The reference_model is defined such that subtracting it from the model
        will set the energy of the endmembers for the _MIX properties of this
        class to zero. The _MIX properties generated here allow users to see
        mixing energies on the internal degrees of freedom of this phase.

        The reference_model AST can be modified in the same way as the current Model.

        Ideal mixing is always added to the AST, we need to set it to zero here
        so that it's not subtracted out of the reference. However, we have this
        option so users can just see the mixing properties in terms of the
        parameters.

        If the current model has an ordering energy as part of a partitioned
        model, then this special reference state is not well defined because
        the endmembers in the model have energetic contributions from
        the ordered endmember energies and the disordered mixing energies.
        Therefore, this reference state cannot be used sensibly for partitioned
        models and the energies of all reference_model.models are set to nan.

        Since build_reference_model requires that Database instances are copied
        and new instances of Model are created, it can be computationally
        expensive to build the reference Model by default. This property delays
        building the reference_model until it is used.

        """
        if self._reference_model is None:
            self._build_reference_model()
        return self._reference_model
