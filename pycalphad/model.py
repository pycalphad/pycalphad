"""
The model module provides support for using a Database to perform
calculations under specified conditions.
"""
import copy
from sympy import exp, log, Abs, Add, And, Float, Mul, Piecewise, Pow, S, sin, StrictGreaterThan, Symbol, zoo, oo, nan
from tinydb import where
import pycalphad.variables as v
from pycalphad.core.errors import DofError
from pycalphad.core.constants import MIN_SITE_FRACTION
from pycalphad.core.utils import unpack_components, get_pure_elements, wrap_symbol
from pycalphad.core.constraints import is_multiphase_constraint
import numpy as np
from collections import OrderedDict, defaultdict

# Maximum number of levels deep we check for symbols that are functions of
# other symbols
_MAX_PARAM_NESTING = 32


class ReferenceState():
    """
    Define the phase and any fixed state variables as a reference state for a component.

    Parameters
    ----------

    Attributes
    ----------
    fixed_statevars : dict
        Dictionary of {StateVariable: value} that will be fixed, e.g. {v.T: 298.15, v.P: 101325}
    phase_name : str
        Name of phase
    species : Species
        pycalphad Species variable

    """
    def __init__(self, species, reference_phase, fixed_statevars=None):
        """
        Parameters
        ----------
        species : str or Species
            Species to define the reference state for. Only pure elements supported.
        reference_phase : str
            Name of phase
        fixed_statevars : None, optional
            Dictionary of {StateVariable: value} that will be fixed, e.g. {v.T: 298.15, v.P: 101325}
            If None (the default), an empty dict will be created.

        """
        if isinstance(species, v.Species):
            self.species = species
        else:
            self.species = v.Species(species)
        self.phase_name = reference_phase
        self.fixed_statevars = fixed_statevars if fixed_statevars is not None else {}

    def __repr__(self):
        if len(self.fixed_statevars.keys()) > 0:
            s = "ReferenceState('{}', '{}', {})".format(self.species.name, self.phase_name, self.fixed_statevars)
        else:
            s = "ReferenceState('{}', '{}')".format(self.species.name, self.phase_name)
        return s


class Model(object):
    """
    Models use an abstract representation of the function
    for calculation of values under specified conditions.

    Parameters
    ----------
    dbe : Database
        Database containing the relevant parameters.
    comps : list
        Names of components to consider in the calculation.
    phase_name : str
        Name of phase model to build.
    parameters : dict or list
        Optional dictionary of parameters to be substituted in the model.
        A list of parameters will cause those symbols to remain symbolic.
        This will overwrite parameters specified in the database

    Methods
    -------
    None yet.

    Examples
    --------
    None yet.
    """
    # We only use the contributions attribute in build_phase.
    # Users should not access it later since subclasses can override build_phase
    # and make self.models inconsistent with contributions.
    # Note that we include atomic ordering last since it uses self.models
    # to figure out its contribution.
    contributions = [('ref', 'reference_energy'), ('idmix', 'ideal_mixing_energy'),
                     ('xsmix', 'excess_mixing_energy'), ('mag', 'magnetic_energy'),
                     ('2st', 'twostate_energy'), ('ein', 'einstein_energy'),
                     ('ord', 'atomic_ordering_energy')]
    def __init__(self, dbe, comps, phase_name, parameters=None):
        self._dbe = dbe
        self._reference_model = None
        self.components = set()
        self.constituents = []
        self.phase_name = phase_name.upper()
        phase = dbe.phases[self.phase_name]
        self.site_ratios = list(phase.sublattices)
        active_species = unpack_components(dbe, comps)
        for idx, sublattice in enumerate(phase.constituents):
            subl_comps = set(sublattice).intersection(active_species)
            self.components |= subl_comps
            # Support for variable site ratios in ionic liquid model
            if phase.model_hints.get('ionic_liquid_2SL', False):
                if idx == 0:
                    subl_idx = 1
                elif idx == 1:
                    subl_idx = 0
                else:
                    raise ValueError('Two-sublattice ionic liquid specified with more than two sublattices')
                self.site_ratios[subl_idx] = Add(*[v.SiteFraction(self.phase_name, idx, spec) * abs(spec.charge) for spec in subl_comps])
        if phase.model_hints.get('ionic_liquid_2SL', False):
            # Special treatment of "neutral" vacancies in 2SL ionic liquid
            # These are treated as having variable valence
            for idx, sublattice in enumerate(phase.constituents):
                subl_comps = set(sublattice).intersection(active_species)
                if v.Species('VA') in subl_comps:
                    if idx == 0:
                        subl_idx = 1
                    elif idx == 1:
                        subl_idx = 0
                    else:
                        raise ValueError('Two-sublattice ionic liquid specified with more than two sublattices')
                    self.site_ratios[subl_idx] += self.site_ratios[idx] * v.SiteFraction(self.phase_name, idx, v.Species('VA'))
        self.site_ratios = tuple(self.site_ratios)

        # Verify that this phase is still possible to build
        is_pure_VA = set()
        for sublattice in phase.constituents:
            sublattice_comps = set(sublattice).intersection(self.components)
            if len(sublattice_comps) == 0:
                # None of the components in a sublattice are active
                # We cannot build a model of this phase
                raise DofError(
                    '{0}: Sublattice {1} of {2} has no components in {3}' \
                    .format(self.phase_name, sublattice,
                            phase.constituents,
                            self.components))
            is_pure_VA.add(sum(set(map(lambda s : getattr(s, 'number_of_atoms'),sublattice_comps))))
            self.constituents.append(sublattice_comps)
        if sum(is_pure_VA) == 0:
            #The only possible component in a sublattice is vacancy
            #We cannot build a model of this phase
            raise DofError(
                '{0}: Sublattices of {1} contains only VA (VACUUM) constituents' \
                .format(self.phase_name, phase.constituents))
        self.components = sorted(self.components)
        desired_active_pure_elements = [list(x.constituents.keys()) for x in self.components]
        desired_active_pure_elements = [el.upper() for constituents in desired_active_pure_elements
                                        for el in constituents]
        self.pure_elements = sorted(set(desired_active_pure_elements))
        self.nonvacant_elements = [x for x in self.pure_elements if x != 'VA']

        # Defines the mixing model to use between any two elements
        self.binary_excess_models = defaultdict(lambda: 'REDLICH-KISTER')
        user_excess_model = phase.model_hints.get('excess_model', None)
        if (user_excess_model is not None) and (not isinstance(user_excess_model, dict)):
            self.binary_excess_models = defaultdict(lambda: user_excess_model)
        if isinstance(user_excess_model, dict):
            for pair, excess_model in user_excess_model.items():
                p1, p2 = pair
                # Convert string of species name into the Species object
                p1 = [i for i in dbe.species if i.name == p1][0]
                p2 = [i for i in dbe.species if i.name == p2][0]
                pair = frozenset([p1, p2])
                if not pair.issubset(self.components):
                    continue
                self.binary_excess_models[pair] = excess_model

        coord_equiv_y_fractions = {}
        if phase.model_hints.get('quasichem_fact00', False):
            # Species constituent amounts are m/z, where m is the multiplicity and z is the coordination number
            # This is consistent with the idea that, formally, quasichemical models behave like associate models
            # with associates defined with molar ratios of m/z (but the entropy differs)
            for el in self.nonvacant_elements:
                coord_number = self._get_coordination_number(dbe, el)
                coord_equiv_y_fractions[el] = sum(
                    [((coord_number * v.Species(c).constituents.get(el, 0) / 2) * self.moles(c))
                     for c in self.components])
        self.coordination_equivalent_fractions = coord_equiv_y_fractions

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

        self.site_fractions = sorted([x for x in self.variables if isinstance(x, v.SiteFraction)], key=str)
        self.state_variables = sorted([x for x in self.variables if not isinstance(x, v.SiteFraction)], key=str)

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

    def moles(self, species):
        "Number of moles of species or elements."
        species = v.Species(species)
        is_pure_element = (len(species.constituents.keys()) == 1 and
                           list(species.constituents.keys())[0] == species.name)
        result = S.Zero
        normalization = S.Zero
        if is_pure_element:
            element = list(species.constituents.keys())[0]
            for idx, sublattice in enumerate(self.constituents):
                active = set(sublattice).intersection(self.components)
                result += self.site_ratios[idx] * \
                    sum(int(spec.number_of_atoms > 0) * spec.constituents.get(element, 0) * v.SiteFraction(self.phase_name, idx, spec)
                        for spec in active)
                normalization += self.site_ratios[idx] * \
                    sum(spec.number_of_atoms * v.SiteFraction(self.phase_name, idx, spec)
                        for spec in active)
        else:
            for idx, sublattice in enumerate(self.constituents):
                active = set(sublattice).intersection(self.components)
                if len(active) == 0:
                    continue
                result += self.site_ratios[idx] * v.SiteFraction(self.phase_name, idx, species)
                normalization += self.site_ratios[idx] * \
                    sum(int(spec.number_of_atoms > 0) * v.SiteFraction(self.phase_name, idx, spec)
                        for spec in active)
        return result / normalization

    @property
    def ast(self):
        "Return the full abstract syntax tree of the model."
        return Add(*list(self.models.values()))

    @property
    def variables(self):
        "Return state variables in the model."
        return sorted([x for x in self.ast.free_symbols if isinstance(x, v.StateVariable)], key=str)

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
    neel_temperature = NT = S.Zero

    #pylint: disable=C0103
    # These are standard abbreviations from Thermo-Calc for these quantities
    energy = GM = property(lambda self: self.ast)
    entropy = SM = property(lambda self: -self.GM.diff(v.T))
    enthalpy = HM = property(lambda self: self.GM - v.T*self.GM.diff(v.T))
    heat_capacity = CPM = property(lambda self: -v.T*self.GM.diff(v.T, v.T))
    #pylint: enable=C0103
    mixing_energy = GM_MIX = property(lambda self: self.GM - self.reference_model.GM)
    mixing_enthalpy = HM_MIX = property(lambda self: self.GM_MIX - v.T*self.GM_MIX.diff(v.T))
    mixing_entropy = SM_MIX = property(lambda self: -self.GM_MIX.diff(v.T))
    mixing_heat_capacity = CPM_MIX = property(lambda self: -v.T*self.GM_MIX.diff(v.T, v.T))

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

    def get_internal_constraints(self):
        constraints = []
        for idx, sublattice in enumerate(self.constituents):
            constraints.append(sum(v.SiteFraction(self.phase_name, idx, spec) for spec in sublattice) - 1)
        return constraints

    def get_multiphase_constraints(self, conds):
        fixed_chempots = [cond for cond in conds.keys() if isinstance(cond, v.ChemicalPotential)]
        multiphase_constraints = []
        for statevar in sorted(conds.keys(), key=str):
            if not is_multiphase_constraint(statevar):
                continue
            if isinstance(statevar, v.Composition):
                multiphase_constraints.append(Symbol('NP') * self.moles(statevar.species))
            elif statevar == v.N:
                multiphase_constraints.append(Symbol('NP') * (sum(self.moles(spec) for spec in self.nonvacant_elements)))
            elif statevar in [v.T, v.P]:
                return multiphase_constraints.append(S.Zero)
            else:
                raise NotImplementedError
        return multiphase_constraints

    def build_phase(self, dbe):
        """
        Generate the symbolic form of all the contributions to this phase.

        Parameters
        ----------
        dbe : Database
        """
        contrib_vals = list(OrderedDict(self.__class__.contributions).values())
        if 'atomic_ordering_energy' in contrib_vals:
            if contrib_vals.index('atomic_ordering_energy') != (len(contrib_vals) - 1):
                # Check for a common mistake in custom models
                # Users that need to override this behavior should override build_phase
                raise ValueError('\'atomic_ordering_energy\' must be the final contribution')
        self.models.clear()
        for key, value in self.__class__.contributions:
            self.models[key] = S(getattr(self, value)(dbe))

    def _purity_test(self, constituent_array):
        """
        Check if constituent array only has one species in its array
        This species must also be an active species
        """
        if len(constituent_array) != len(self.constituents):
            return False
        for sublattice in constituent_array:
            if len(sublattice) != 1:
                return False
            if (sublattice[0] not in self.components) and \
                (sublattice[0] != v.Species('*')):
                return False
        return True

    def _array_validity(self, constituent_array):
        """
        Check that the current array contains only active species.
        """
        if len(constituent_array) != len(self.constituents):
            return False
        for sublattice in constituent_array:
            valid = set(sublattice).issubset(self.components) \
                or sublattice[0] == v.Species('*')
            if not valid:
                return False
        return True

    def _interaction_test(self, constituent_array):
        """
        Check if constituent array has more than one active species in
        its array for at least one sublattice.
        """
        result = False
        if len(constituent_array) != len(self.constituents):
            return False
        for sublattice in constituent_array:
            # check if all elements involved are also active
            valid = set(sublattice).issubset(self.components) \
                or sublattice[0] == v.Species('*')
            if len(sublattice) > 1 and valid:
                result = True
            if not valid:
                result = False
                break
        return result

    @property
    def _site_ratio_normalization(self):
        """
        Calculates the normalization factor based on the number of sites
        in each sublattice.
        """
        site_ratio_normalization = S.Zero
        # Calculate normalization factor
        for idx, sublattice in enumerate(self.constituents):
            active = set(sublattice).intersection(self.components)
            subl_content = sum(spec.number_of_atoms * v.SiteFraction(self.phase_name, idx, spec) for spec in active)
            site_ratio_normalization += self.site_ratios[idx] * subl_content
        return site_ratio_normalization

    @staticmethod
    def _Muggianu_correction_dict(comps): #pylint: disable=C0103
        """
        Replace y_i -> y_i + (1 - sum(y involved in parameter)) / m,
        where m is the arity of the interaction parameter.
        Returns a dict converting the list of Symbols (comps) to this.
        m is assumed equal to the length of comps.

        When incorporating binary, ternary or n-ary interaction parameters
        into systems with more than n components, the sum of site fractions
        involved in the interaction parameter may no longer be unity. This
        breaks the symmetry of the parameter. The solution suggested by
        Muggianu, 1975, is to renormalize the site fractions by replacing them
        with a term that will sum to unity even in higher-order systems.
        There are other solutions that involve retaining the asymmetry for
        physical reasons, but this solution works well for components that
        are physically similar.

        This procedure is based on an analysis by Hillert, 1980,
        published in the Calphad journal.
        """
        arity = len(comps)
        return_dict = {}
        correction_term = (S.One - Add(*comps)) / arity
        for comp in comps:
            return_dict[comp] = comp + correction_term
        return return_dict

    def redlich_kister_sum(self, phase, param_search, param_query):
        """
        Construct parameter in Redlich-Kister polynomial basis, using
        the Muggianu ternary parameter extension.
        """
        rk_terms = []

        # search for desired parameters
        params = param_search(param_query)
        for param in params:
            # iterate over every sublattice
            mixing_term = S.One
            for subl_index, comps in enumerate(param['constituent_array']):
                comp_symbols = None
                # convert strings to symbols
                if comps[0] == v.Species('*'):
                    # Handle wildcards in constituent array
                    comp_symbols = \
                        [
                            v.SiteFraction(phase.name, subl_index, comp)
                            for comp in sorted(set(phase.constituents[subl_index])\
                                .intersection(self.components))
                        ]
                    mixing_term *= Add(*comp_symbols)
                else:
                    comp_symbols = \
                        [
                            v.SiteFraction(phase.name, subl_index, comp)
                            for comp in comps
                        ]
                    if phase.model_hints.get('quasichem_fact00', False):
                        # TODO: Only pair model supported here
                        if len(comps) == 2:
                            pair_comp_symbols = \
                                [
                                    v.SiteFraction(phase.name, subl_index, comp)
                                    for comp in comps if len(comp.constituents.keys()) == 2
                                ]
                            if len(pair_comp_symbols) > 1:
                                raise ValueError('Unsupported quasichemical interaction parameter: ' + str(param))
                            mixing_term *= pair_comp_symbols[0] / 2
                        else:
                            mixing_term *= Mul(*comp_symbols)
                    else:
                        mixing_term *= Mul(*comp_symbols)
                # is this a higher-order interaction parameter?
                if len(comps) == 2 and param['parameter_order'] > 0:
                    # interacting sublattice, add the interaction polynomial
                    mixing_species = frozenset((comp_symbols[0].species, comp_symbols[1].species))
                    excess_model = self.binary_excess_models[mixing_species]
                    if excess_model == 'REDLICH-KISTER':
                        mixing_term *= Pow(comp_symbols[0] - \
                            comp_symbols[1], param['parameter_order'])
                    elif excess_model == 'POLYNOM':
                        if phase.model_hints.get('quasichem_fact00', False):
                            # TODO: Only pair model supported here
                            power_term = S.Zero
                            for comp in comp_symbols:
                                power_term += comp * 2 / len(comp.species.constituents.keys())
                            power_term /= 2
                        else:
                            # TODO: Use independent component in the order specified by the mixed excess model typedef
                            power_term = comp_symbols[1]
                        mixing_term *= Pow(power_term, param['parameter_order'])
                    else:
                        raise ValueError('Unknown binary excess model: ' + str(self.binary_excess_models[mixing_species]))
                if len(comps) == 3:
                    # 'parameter_order' is an index to a variable when
                    # we are in the ternary interaction parameter case

                    # NOTE: The commercial software packages seem to have
                    # a "feature" where, if only the zeroth
                    # parameter_order term of a ternary parameter is specified,
                    # the other two terms are automatically generated in order
                    # to make the parameter symmetric.
                    # In other words, specifying only this parameter:
                    # PARAMETER G(FCC_A1,AL,CR,NI;0) 298.15  +30300; 6000 N !
                    # Actually implies:
                    # PARAMETER G(FCC_A1,AL,CR,NI;0) 298.15  +30300; 6000 N !
                    # PARAMETER G(FCC_A1,AL,CR,NI;1) 298.15  +30300; 6000 N !
                    # PARAMETER G(FCC_A1,AL,CR,NI;2) 298.15  +30300; 6000 N !
                    #
                    # If either 1 or 2 is specified, no implicit parameters are
                    # generated.
                    # We need to handle this case.
                    if param['parameter_order'] == 0:
                        # are _any_ of the other parameter_orders specified?
                        ternary_param_query = (
                            (where('phase_name') == param['phase_name']) & \
                            (where('parameter_type') == \
                                param['parameter_type']) & \
                            (where('constituent_array') == \
                                param['constituent_array'])
                        )
                        other_tern_params = param_search(ternary_param_query)
                        if len(other_tern_params) == 1 and \
                            other_tern_params[0] == param:
                            # only the current parameter is specified
                            # We need to generate the other two parameters.
                            order_one = copy.deepcopy(param)
                            order_one['parameter_order'] = 1
                            order_two = copy.deepcopy(param)
                            order_two['parameter_order'] = 2
                            # Add these parameters to our iteration.
                            params.extend((order_one, order_two))
                    # Include variable indicated by parameter order index
                    # Perform Muggianu adjustment to site fractions
                    mixing_term *= comp_symbols[param['parameter_order']].subs(
                        self._Muggianu_correction_dict(comp_symbols),
                        simultaneous=True)
            if phase.model_hints.get('ionic_liquid_2SL', False):
                # Special normalization rules for parameters apply under this model
                # Reference: Bo Sundman, "Modification of the two-sublattice model for liquids",
                # Calphad, Volume 15, Issue 2, 1991, Pages 109-119, ISSN 0364-5916
                if not any([m.species.charge < 0 for m in mixing_term.free_symbols]):
                    pair_rule = {}
                    # Cation site fractions must always appear with vacancy site fractions
                    va_subls = [(v.Species('VA') in phase.constituents[idx]) for idx in range(len(phase.constituents))]
                    va_subl_idx = (len(phase.constituents) - 1) - va_subls[::-1].index(True)
                    va_present = any((v.Species('VA') in c) for c in param['constituent_array'])
                    if va_present and (max(len(c) for c in param['constituent_array']) == 1):
                        # No need to apply pair rule for VA-containing endmember
                        pass
                    elif va_subl_idx > -1:
                        for sym in mixing_term.free_symbols:
                            if sym.species.charge > 0:
                                pair_rule[sym] = sym * v.SiteFraction(sym.phase_name, va_subl_idx, v.Species('VA'))
                    mixing_term = mixing_term.xreplace(pair_rule)
                    # This parameter is normalized differently due to the variable charge valence of vacancies
                    mixing_term *= self.site_ratios[va_subl_idx]
            rk_terms.append(mixing_term * param['parameter'])
        return Add(*rk_terms)

    def reference_energy(self, dbe):
        """
        Returns the weighted average of the endmember energies
        in symbolic form.
        """
        pure_param_query = (
            (where('phase_name') == self.phase_name) & \
            (where('parameter_order') == 0) & \
            (where('parameter_type') == "G") & \
            (where('constituent_array').test(self._purity_test))
        )
        phase = dbe.phases[self.phase_name]
        param_search = dbe.search
        pure_energy_term = self.redlich_kister_sum(phase, param_search,
                                                   pure_param_query)
        return pure_energy_term / self._site_ratio_normalization

    def _get_coordination_number(self, dbe, el):
        """
        Returns the coordination number (Z) of an element in symbolic form.
        For quasichemical models.
        """
        # TC TDBs store the coordination number in the "VK" parameter type
        z_param_query = (
            (where('phase_name') == self.phase_name) &
            (where('parameter_order') == 0) &
            (where('parameter_type') == "VK") &
            (where('constituent_array').test(self._purity_test))
        )
        phase = dbe.phases[self.phase_name]
        param_search = dbe.search
        z_term = self.redlich_kister_sum(phase, param_search, z_param_query)
        replace_dict = {}
        # Only return terms for the desired element
        for subl_index, sublattice in enumerate(phase.constituents):
            active_comps = set(sublattice).intersection(self.components)
            for comp in active_comps:
                sitefrac = \
                    v.SiteFraction(phase.name, subl_index, comp)
                if comp.constituents.get(el, 0) != 0:
                    replace_dict[sitefrac] = 1
                else:
                    replace_dict[sitefrac] = 0
        return z_term.xreplace(replace_dict)

    def ideal_mixing_energy(self, dbe):
        #pylint: disable=W0613
        """
        Returns the ideal mixing energy in symbolic form.
        """
        phase = dbe.phases[self.phase_name]
        # Normalize site ratios
        site_ratio_normalization = self._site_ratio_normalization
        site_ratios = self.site_ratios
        site_ratios = [c/site_ratio_normalization for c in site_ratios]
        ideal_mixing_term = S.Zero
        sitefrac_limit = Float(MIN_SITE_FRACTION/10.)
        for subl_index, sublattice in enumerate(phase.constituents):
            active_comps = set(sublattice).intersection(self.components)
            ratio = site_ratios[subl_index]
            for comp in active_comps:
                sitefrac = \
                    v.SiteFraction(phase.name, subl_index, comp)
                # We lose some precision here, but this makes the limit behave nicely
                # We're okay until fractions of about 1e-12 (platform-dependent)
                mixing_term = Piecewise((sitefrac*log(sitefrac),
                                         StrictGreaterThan(sitefrac, sitefrac_limit, evaluate=False)), (0, True),
                                        evaluate=False)
                if phase.model_hints.get('quasichem_fact00', False):
                    constituents = comp.constituents.keys()
                    if len(constituents) == 1:
                        # This is an A-A pair
                        constit_element = list(constituents)[0]
                        mixing_term -= Piecewise((2*sitefrac*log(self.coordination_equivalent_fractions[constit_element]),
                                                 StrictGreaterThan(sitefrac, sitefrac_limit, evaluate=False)), (0, True),
                                                 evaluate=False)
                    elif len(constituents) == 2:
                        # This is an A-B pair
                        constit_elements = list(constituents)
                        mul_y = Mul(*[self.coordination_equivalent_fractions[el] for el in constit_elements])
                        mixing_term -= Piecewise((sitefrac*log(2*mul_y),
                                                 StrictGreaterThan(sitefrac, sitefrac_limit, evaluate=False)), (0, True),
                                                 evaluate=False)
                    else:
                        raise ValueError('Only pairwise quasichemical models are supported')
                ideal_mixing_term += (mixing_term*ratio)
        if phase.model_hints.get('quasichem_fact00', False):
            for el in self.nonvacant_elements:
                el_moles = self.moles(el)
                mixing_term = Piecewise((el_moles*log(el_moles),
                                         StrictGreaterThan(el_moles, sitefrac_limit, evaluate=False)), (0, True),
                                        evaluate=False)
                ideal_mixing_term += mixing_term
        ideal_mixing_term *= (v.R * v.T)
        return ideal_mixing_term

    def excess_mixing_energy(self, dbe):
        """
        Build the binary, ternary and higher order interaction term
        Here we use Redlich-Kister polynomial basis by default
        Here we use the Muggianu ternary extension by default
        Replace y_i -> y_i + (1 - sum(y involved in parameter)) / m,
        where m is the arity of the interaction parameter
        """
        phase = dbe.phases[self.phase_name]
        param_search = dbe.search
        param_query = (
            (where('phase_name') == self.phase_name) & \
                ((where('parameter_type') == 'G') |
                 (where('parameter_type') == 'L')) & \
                (where('constituent_array').test(self._interaction_test))
            )
        excess_term = self.redlich_kister_sum(phase, param_search, param_query)
        return excess_term / self._site_ratio_normalization

    def magnetic_energy(self, dbe):
        #pylint: disable=C0103, R0914
        """
        Return the energy from magnetic ordering in symbolic form.
        The implemented model is the Inden-Hillert-Jarl formulation.
        The approach follows from the background of W. Xiong et al, Calphad, 2012.
        """
        phase = dbe.phases[self.phase_name]
        param_search = dbe.search
        self.TC = self.curie_temperature = S.Zero
        self.BMAG = self.beta = S.Zero
        if 'ihj_magnetic_structure_factor' not in phase.model_hints:
            return S.Zero
        if 'ihj_magnetic_afm_factor' not in phase.model_hints:
            return S.Zero

        site_ratio_normalization = self._site_ratio_normalization
        # define basic variables
        afm_factor = phase.model_hints['ihj_magnetic_afm_factor']

        if afm_factor == 0:
            # Apply improved magnetic model which does not use AFM / Weiss factor
            return self.xiong_magnetic_energy(dbe)

        bm_param_query = (
            (where('phase_name') == phase.name) & \
            (where('parameter_type') == 'BMAGN') & \
            (where('constituent_array').test(self._array_validity))
        )
        tc_param_query = (
            (where('phase_name') == phase.name) & \
            (where('parameter_type') == 'TC') & \
            (where('constituent_array').test(self._array_validity))
        )

        mean_magnetic_moment = \
            self.redlich_kister_sum(phase, param_search, bm_param_query)
        beta = mean_magnetic_moment / Piecewise(
            (afm_factor, mean_magnetic_moment <= 0),
            (1., True),
            evaluate=False
            )
        self.BMAG = self.beta = beta

        curie_temp = \
            self.redlich_kister_sum(phase, param_search, tc_param_query)
        tc = curie_temp / Piecewise(
            (afm_factor, curie_temp <= 0),
            (1., True),
            evaluate=False
            )
        self.TC = self.curie_temperature = tc

        # Used to prevent singularity
        tau_positive_tc = v.T / (curie_temp + 1e-9)
        tau_negative_tc = v.T / ((curie_temp/afm_factor) + 1e-9)

        # define model parameters
        p = phase.model_hints['ihj_magnetic_structure_factor']
        A = 518/1125 + (11692/15975)*(1/p - 1)
        # factor when tau < 1 and tc < 0
        sub_tau_neg_tc = 1 - (1/A) * ((79/(140*p))*(tau_negative_tc**(-1)) + (474/497)*(1/p - 1) \
            * ((tau_negative_tc**3)/6 + (tau_negative_tc**9)/135 + (tau_negative_tc**15)/600)
                              )
        # factor when tau < 1 and tc > 0
        sub_tau_pos_tc = 1 - (1/A) * ((79/(140*p))*(tau_positive_tc**(-1)) + (474/497)*(1/p - 1) \
            * ((tau_positive_tc**3)/6 + (tau_positive_tc**9)/135 + (tau_positive_tc**15)/600)
                              )
        # factor when tau >= 1 and tc > 0
        super_tau_pos_tc = -(1/A) * ((tau_positive_tc**-5)/10 + (tau_positive_tc**-15)/315 + (tau_positive_tc**-25)/1500)
        # factor when tau >= 1 and tc < 0
        super_tau_neg_tc = -(1/A) * ((tau_negative_tc**-5)/10 + (tau_negative_tc**-15)/315 + (tau_negative_tc**-25)/1500)

        # This is an optimization to reduce the complexity of the compile-time expression
        expr_cond_pairs = [(sub_tau_neg_tc, curie_temp/afm_factor > v.T),
                           (sub_tau_pos_tc, curie_temp > v.T),
                           (super_tau_pos_tc, And(curie_temp < v.T, curie_temp > 0)),
                           (super_tau_neg_tc, And(curie_temp/afm_factor < v.T, curie_temp < 0)),
                           (0, True)
                           ]
        g_term = Piecewise(*expr_cond_pairs, evaluate=False)

        return v.R * v.T * log(beta+1) * \
            g_term / site_ratio_normalization

    def xiong_magnetic_energy(self, dbe):
        """
        Return the energy from magnetic ordering in symbolic form.
        The approach follows W. Xiong et al, Calphad, 2012.
        """
        phase = dbe.phases[self.phase_name]
        param_search = dbe.search
        self.TC = self.curie_temperature = S.Zero
        if 'ihj_magnetic_structure_factor' not in phase.model_hints:
            return S.Zero
        if 'ihj_magnetic_afm_factor' not in phase.model_hints:
            return S.Zero

        site_ratio_normalization = self._site_ratio_normalization
        # define basic variables
        afm_factor = phase.model_hints['ihj_magnetic_afm_factor']

        if afm_factor != 0:
            raise ValueError('Xiong model called with nonzero AFM / Weiss factor')

        nt_param_query = (
            (where('phase_name') == phase.name) & \
            (where('parameter_type') == 'NT') & \
            (where('constituent_array').test(self._array_validity))
        )

        bm_param_query = (
            (where('phase_name') == phase.name) & \
            (where('parameter_type') == 'BMAGN') & \
            (where('constituent_array').test(self._array_validity))
        )
        tc_param_query = (
            (where('phase_name') == phase.name) & \
            (where('parameter_type') == 'TC') & \
            (where('constituent_array').test(self._array_validity))
        )

        mean_magnetic_moment = \
            self.redlich_kister_sum(phase, param_search, bm_param_query)
        beta = mean_magnetic_moment

        curie_temp = \
            self.redlich_kister_sum(phase, param_search, tc_param_query)
        neel_temp = \
            self.redlich_kister_sum(phase, param_search, nt_param_query)

        self.TC = self.curie_temperature = curie_temp.subs(self._symbols)
        self.NT = self.neel_temperature = neel_temp.subs(self._symbols)

        tau_curie = v.T / curie_temp
        tau_curie = tau_curie.xreplace({zoo: 1.0e10})
        tau_neel = v.T / neel_temp
        tau_neel = tau_neel.xreplace({zoo: 1.0e10})

        # define model parameters
        p = phase.model_hints['ihj_magnetic_structure_factor']
        D = 0.33471979 + 0.49649686*(1/p - 1)
        sub_tau_curie = 1 - (1/D) * ((0.38438376/p)*(tau_curie**(-1)) + 0.63570895*(1/p - 1) \
            * ((tau_curie**3)/6 + (tau_curie**9)/135 + (tau_curie**15)/600) + (tau_curie**21)/1617
                              )
        sub_tau_neel = 1 - (1/D) * ((0.38438376/p)*(tau_neel**(-1)) + 0.63570895*(1/p - 1) \
            * ((tau_neel**3)/6 + (tau_neel**9)/135 + (tau_neel**15)/600) + (tau_neel**21)/1617
                              )
        super_tau_curie = -(1/D) * ((tau_curie**-7)/21 + (tau_curie**-21)/630 + (tau_curie**-35)/2975 + (tau_curie**-49)/8232)
        super_tau_neel = -(1/D) * ((tau_neel**-7)/21 + (tau_neel**-21)/630 + (tau_neel**-35)/2975 + (tau_neel**-49)/8232)

        expr_cond_pairs_curie = [(0, tau_curie <= 0),
                                 (super_tau_curie, tau_curie > 1),
                                 (sub_tau_curie, True)
                                ]
        expr_cond_pairs_neel = [(0, tau_neel <= 0),
                                (super_tau_neel, tau_neel > 1),
                                (sub_tau_neel, True)
                               ]
        g_term = Piecewise(*expr_cond_pairs_curie, evaluate=False) + Piecewise(*expr_cond_pairs_neel, evaluate=False)

        return v.R * v.T * log(beta+1) * \
            g_term / site_ratio_normalization

    def twostate_energy(self, dbe):
        """
        Return the energy from liquid-amorphous two-state model.
        """
        phase = dbe.phases[self.phase_name]
        param_search = dbe.search
        site_ratio_normalization = self._site_ratio_normalization
        gd_param_query = (
            (where('phase_name') == phase.name) & \
            (where('parameter_type') == 'GD') & \
            (where('constituent_array').test(self._array_validity))
        )
        gd = self.redlich_kister_sum(phase, param_search, gd_param_query)
        if gd == S.Zero:
            return S.Zero
        return -v.R * v.T * log(1 + exp(-gd / (v.R * v.T))) / site_ratio_normalization

    def einstein_energy(self, dbe):
        """
        Return the energy based on the Einstein model.
        Note that THETA parameters are actually LN(THETA).
        All Redlich-Kister summation is done in log-space,
        then exp() is called on the result.
        """
        phase = dbe.phases[self.phase_name]
        param_search = dbe.search
        theta_param_query = (
            (where('phase_name') == phase.name) & \
            (where('parameter_type') == 'THETA') & \
            (where('constituent_array').test(self._array_validity))
        )
        lntheta = self.redlich_kister_sum(phase, param_search, theta_param_query)
        theta = exp(lntheta)
        if lntheta != 0:
            result = 1.5*v.R*theta + 3*v.R*v.T*log(1-exp(-theta/v.T))
        else:
            result = 0
        return result / self._site_ratio_normalization

    @staticmethod
    def mole_fraction(species_name, phase_name, constituent_array,
                      site_ratios):
        """
        Return an abstract syntax tree of the mole fraction of the
        given species as a function of its constituent site fractions.

        Note that this will treat vacancies the same as any other component,
        i.e., this will not give the correct _overall_ composition for
        sublattices containing vacancies with other components by normalizing
        by a factor of 1 - y_{VA}. This is because we use this routine in the
        order-disorder model to calculate the disordered site fractions from
        the ordered site fractions, so we need _all_ site fractions, including
        VA, to sum to unity.
        """

        # Normalize site ratios
        site_ratio_normalization = 0
        numerator = S.Zero
        for idx, sublattice in enumerate(constituent_array):
            # sublattices with only vacancies don't count
            if sum(spec.number_of_atoms for spec in list(sublattice)) == 0:
                continue
            if species_name in list(sublattice):
                site_ratio_normalization += site_ratios[idx]
                numerator += site_ratios[idx] * \
                    v.SiteFraction(phase_name, idx, species_name)

        if site_ratio_normalization == 0 and species_name.name == 'VA':
            return 1

        if site_ratio_normalization == 0:
            raise ValueError('Couldn\'t find ' + species_name + ' in ' + \
                str(constituent_array))

        return numerator / site_ratio_normalization

    def atomic_ordering_energy(self, dbe):
        """
        Return the atomic ordering contribution in symbolic form.
        Description follows Servant and Ansara, Calphad, 2001.
        """
        phase = dbe.phases[self.phase_name]
        ordered_phase_name = phase.model_hints.get('ordered_phase', None)
        disordered_phase_name = phase.model_hints.get('disordered_phase', None)
        if phase.name != ordered_phase_name:
            return S.Zero
        disordered_model = self.__class__(dbe, sorted(self.components),
                                          disordered_phase_name)
        constituents = [sorted(set(c).intersection(self.components)) \
                for c in dbe.phases[ordered_phase_name].constituents]

        # Fix variable names
        variable_rename_dict = {}
        disordered_sitefracs = [x for x in disordered_model.energy.free_symbols if isinstance(x, v.SiteFraction)]
        for atom in disordered_sitefracs:
            # Replace disordered phase site fractions with mole fractions of
            # ordered phase site fractions.
            # Special case: Pure vacancy sublattices
            all_species_in_sublattice = \
                dbe.phases[disordered_phase_name].constituents[
                    atom.sublattice_index]
            if atom.species.name == 'VA' and len(all_species_in_sublattice) == 1:
                # Assume: Pure vacancy sublattices are always last
                vacancy_subl_index = \
                    len(dbe.phases[ordered_phase_name].constituents)-1
                variable_rename_dict[atom] = \
                    v.SiteFraction(
                        ordered_phase_name, vacancy_subl_index, atom.species)
            else:
                # All other cases: replace site fraction with mole fraction
                variable_rename_dict[atom] = \
                    self.mole_fraction(
                        atom.species,
                        ordered_phase_name,
                        constituents,
                        dbe.phases[ordered_phase_name].sublattices
                        )
        # Save all of the ordered energy contributions
        # This step is why this routine must be called _last_ in build_phase
        ordered_energy = Add(*list(self.models.values()))
        self.models.clear()
        # Copy the disordered energy contributions into the correct bins
        for name, value in disordered_model.models.items():
            self.models[name] = value.xreplace(variable_rename_dict)
        # All magnetic parameters will be defined in the disordered model
        self.TC = self.curie_temperature = disordered_model.TC
        self.TC = self.curie_temperature = self.TC.xreplace(variable_rename_dict)

        molefraction_dict = {}

        # Construct a dictionary that replaces every site fraction with its
        # corresponding mole fraction in the disordered state
        ordered_sitefracs = [x for x in ordered_energy.free_symbols if isinstance(x, v.SiteFraction)]
        for sitefrac in ordered_sitefracs:
            all_species_in_sublattice = \
                dbe.phases[ordered_phase_name].constituents[
                    sitefrac.sublattice_index]
            if sitefrac.species.name == 'VA' and len(all_species_in_sublattice) == 1:
                # pure-vacancy sublattices should not be replaced
                # this handles cases like AL,NI,VA:AL,NI,VA:VA and
                # ensures the VA's don't get mixed up
                continue
            molefraction_dict[sitefrac] = \
                self.mole_fraction(sitefrac.species,
                                   ordered_phase_name, constituents,
                                   dbe.phases[ordered_phase_name].sublattices)

        return ordered_energy - ordered_energy.xreplace(molefraction_dict)


    # TODO: fix case for VA interactions: L(PHASE,A,VA:VA;0)-type parameters
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
        for ref_state in reference_states:
            if ref_state.species not in self.components:
                continue
            mod_pure = self.__class__(endmember_only_dbe, [ref_state.species, v.Species('VA')], ref_state.phase_name, parameters=self._parameters_arg)
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


class TestModel(Model):
    """
    Test Model object for global minimization.

    Equation 15.2 in:
    P.M. Pardalos, H.E. Romeijn (Eds.), Handbook of Global Optimization,
    vol. 2. Kluwer Academic Publishers, Boston/Dordrecht/London (2002)

    Parameters
    ----------
    dbf : Database
        Ignored by TestModel but retained for API compatibility.
    comps : sequence
        Names of components to consider in the calculation.
    phase : str
        Name of phase model to build.
    solution : sequence, optional
        Float array locating the true minimum. Same length as 'comps'.
        If not specified, randomly generated and saved to self.solution

    Methods
    -------
    None yet.

    Examples
    --------
    None yet.
    """
    def __init__(self, dbf, comps, phase, solution=None, kmax=None):
        self.components = set(comps)
        if 'VA' in self.components:
            raise ValueError('Vacancies are unsupported in TestModel')
        self.models = dict()
        variables = [v.SiteFraction(phase.upper(), 0, x) for x in sorted(self.components)]
        if solution is None:
            solution = np.random.dirichlet(np.ones_like(variables, dtype=np.int))
        self.solution = dict(list(zip(variables, solution)))
        kmax = kmax if kmax is not None else 2
        scale_factor = 1e4 * len(self.components)
        ampl_scale = 1e3 * np.ones(kmax, dtype=np.float)
        freq_scale = 10 * np.ones(kmax, dtype=np.float)
        polys = Add(*[ampl_scale[i] * sin(freq_scale[i] * Add(*[Add(*[(varname - sol)**(j+1)
                                                                      for varname, sol in self.solution.items()])
                                                                for j in range(kmax)]))**2
                      for i in range(kmax)])
        self.models['test'] = scale_factor * Add(*[(varname - sol)**2 for varname, sol in self.solution.items()]) + polys
