"""
The model module provides support for using a Database to perform
calculations under specified conditions.
"""
from __future__ import division
import copy
from sympy import exp, log, Abs, Add, Mul, Piecewise, Pow, S, sin, Symbol, zoo, oo
from tinydb import where
import pycalphad.variables as v
from pycalphad.core.constants import MIN_SITE_FRACTION
import numpy as np
from collections import OrderedDict

# Maximum number of levels deep we check for symbols that are functions of
# other symbols
_MAX_PARAM_NESTING = 32

class DofError(Exception):
    "Error due to missing degrees of freedom."
    pass

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
    parameters : dict
        Optional dictionary of parameters to be substituted in the model.
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
        # Constrain possible components to those within phase's d.o.f
        possible_comps = {x.upper() for x in comps}
        comps = sorted(comps, key=str)
        self.components = set()
        self.constituents = []
        self.phase_name = phase_name.upper()
        phase = dbe.phases[self.phase_name]
        self.site_ratios = phase.sublattices
        for sublattice in phase.constituents:
            self.components |= set(sublattice).intersection(possible_comps)
        # Verify that this phase is still possible to build
        for sublattice in phase.constituents:
            if len(set(sublattice).intersection(self.components)) == 0:
                # None of the components in a sublattice are active
                # We cannot build a model of this phase
                raise DofError(
                    '{0}: Sublattice {1} of {2} has no components in {3}' \
                    .format(self.phase_name, sublattice,
                            phase.constituents,
                            self.components))
            self.constituents.append(set(sublattice).intersection(self.components))

        # Convert string symbol names to sympy Symbol objects
        # This makes xreplace work with the symbols dict
        symbols = {Symbol(s): val for s, val in dbe.symbols.items()}

        def wrap_symbol(obj):
            if isinstance(obj, Symbol):
                return obj
            else:
                return Symbol(obj)
        if parameters is not None:
            symbols.update([(wrap_symbol(s), val) for s, val in parameters.items()])
        self._symbols = {wrap_symbol(key): value for key, value in symbols.items()}

        self.models = OrderedDict()
        self.build_phase(dbe)
        self.site_fractions = sorted(self.ast.atoms(v.SiteFraction), key=str)

        for name, value in self.models.items():
            self.models[name] = self.symbol_replace(value, symbols)

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
                undefs = obj.atoms(Symbol) - obj.atoms(v.StateVariable)
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

    def standard_mole_fraction(self, species):
        "Mole fraction which correctly normalizes for vacancies."
        result = S.Zero
        site_ratio_normalization = S.Zero
        # Calculate normalization factor
        for idx, sublattice in enumerate(self.constituents):
            active = set(sublattice).intersection(self.components)
            if 'VA' in active:
                site_ratio_normalization += self.site_ratios[idx] * \
                    (1.0 - v.SiteFraction(self.phase_name, idx, 'VA'))
            else:
                site_ratio_normalization += self.site_ratios[idx]
        site_ratios = [c/site_ratio_normalization for c in self.site_ratios]
        # Sum up site fraction contributions from each component sublattice
        for idx, sublattice in enumerate(self.constituents):
            active = set(sublattice).intersection(set(self.components))
            if species in active:
                result += site_ratios[idx] * \
                    v.SiteFraction(self.phase_name, idx, species)
        return result

    @property
    def ast(self):
        "Return the full abstract syntax tree of the model."
        return Add(*list(self.models.values()))

    @property
    def variables(self):
        "Return state variables in the model."
        return sorted(self.ast.atoms(v.StateVariable), key=str)

    @property
    def degree_of_ordering(self):
        result = S.Zero
        site_ratio_normalization = S.Zero
        # Calculate normalization factor
        for idx, sublattice in enumerate(self.constituents):
            active = set(sublattice).intersection(self.components)
            if 'VA' in active:
                site_ratio_normalization += self.site_ratios[idx] * \
                    (1.0 - v.SiteFraction(self.phase_name, idx, 'VA'))
            else:
                site_ratio_normalization += self.site_ratios[idx]
        site_ratios = [c/site_ratio_normalization for c in self.site_ratios]
        for comp in sorted([c for c in self.components if c != 'VA']):
            comp_result = S.Zero
            for idx, sublattice in enumerate(self.constituents):
                active = set(sublattice).intersection(set(self.components))
                if comp in active:
                    comp_result += site_ratios[idx] * Abs(v.SiteFraction(self.phase_name, idx, comp) - self.standard_mole_fraction(comp)) / self.standard_mole_fraction(comp)
            result += comp_result
        return result / len([c for c in self.components if c != 'VA'])
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
    mixing_energy = GM_MIX = property(lambda self: self.GM - self.models['ref'])
    mixing_enthalpy = HM_MIX = \
        property(lambda self: self.GM_MIX - v.T*self.GM_MIX.diff(v.T))
    mixing_entropy = SM_MIX = property(lambda self: -self.GM_MIX.diff(v.T))
    mixing_heat_capacity = CPM_MIX = \
        property(lambda self: -v.T*self.GM_MIX.diff(v.T, v.T))

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
        for sublattice in constituent_array:
            if len(sublattice) != 1:
                return False
            if (sublattice[0] not in self.components) and \
                (sublattice[0] != '*'):
                return False
        return True

    def _array_validity(self, constituent_array):
        """
        Check that the current array contains only active species.
        """
        for sublattice in constituent_array:
            valid = set(sublattice).issubset(self.components) \
                or sublattice[0] == '*'
            if not valid:
                return False
        return True

    def _interaction_test(self, constituent_array):
        """
        Check if constituent array has more than one active species in
        its array for at least one sublattice.
        """
        result = False
        for sublattice in constituent_array:
            # check if all elements involved are also active
            valid = set(sublattice).issubset(self.components) \
                or sublattice[0] == '*'
            if len(sublattice) > 1 and valid:
                result = True
            if not valid:
                result = False
                break
        return result

    def _site_ratio_normalization(self, phase):
        """
        Calculates the normalization factor based on the number of sites
        in each sublattice.
        """
        site_ratio_normalization = S.Zero
        # Normalize by the sum of site ratios times a factor
        # related to the site fraction of vacancies
        for idx, sublattice in enumerate(phase.constituents):
            if ('VA' in set(sublattice)) and ('VA' in self.components):
                site_ratio_normalization += phase.sublattices[idx] * \
                    (1 - v.SiteFraction(phase.name, idx, 'VA'))
            else:
                site_ratio_normalization += phase.sublattices[idx]
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
                if comps[0] == '*':
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
                    mixing_term *= Mul(*comp_symbols)
                # is this a higher-order interaction parameter?
                if len(comps) == 2 and param['parameter_order'] > 0:
                    # interacting sublattice, add the interaction polynomial
                    mixing_term *= Pow(comp_symbols[0] - \
                        comp_symbols[1], param['parameter_order'])
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
        return pure_energy_term / self._site_ratio_normalization(phase)

    def ideal_mixing_energy(self, dbe):
        #pylint: disable=W0613
        """
        Returns the ideal mixing energy in symbolic form.
        """
        phase = dbe.phases[self.phase_name]
        # Normalize site ratios
        site_ratio_normalization = self._site_ratio_normalization(phase)
        site_ratios = phase.sublattices
        site_ratios = [c/site_ratio_normalization for c in site_ratios]
        ideal_mixing_term = S.Zero
        for subl_index, sublattice in enumerate(phase.constituents):
            active_comps = set(sublattice).intersection(self.components)
            ratio = site_ratios[subl_index]
            for comp in active_comps:
                sitefrac = \
                    v.SiteFraction(phase.name, subl_index, comp)
                # We lose some precision here, but this makes the limit behave nicely
                # We're okay until fractions of about 1e-12 (platform-dependent)
                mixing_term = Piecewise((sitefrac*log(sitefrac), sitefrac > MIN_SITE_FRACTION/10.), (0, True))
                ideal_mixing_term += (mixing_term*ratio)
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
        return excess_term / self._site_ratio_normalization(phase)

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
        if 'ihj_magnetic_structure_factor' not in phase.model_hints:
            return S.Zero
        if 'ihj_magnetic_afm_factor' not in phase.model_hints:
            return S.Zero

        site_ratio_normalization = self._site_ratio_normalization(phase)
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
            (1., True)
            )

        curie_temp = \
            self.redlich_kister_sum(phase, param_search, tc_param_query)
        tc = curie_temp / Piecewise(
            (afm_factor, curie_temp <= 0),
            (1., True)
            )
        self.TC = self.curie_temperature = tc
        #print(tc)
        # 0.01 used to prevent singularity
        tau = v.T / (tc + 1e-9)

        # define model parameters
        p = phase.model_hints['ihj_magnetic_structure_factor']
        A = 518/1125 + (11692/15975)*(1/p - 1)
        # factor when tau < 1
        sub_tau = 1 - (1/A) * ((79/(140*p))*(tau**(-1)) + (474/497)*(1/p - 1) \
            * ((tau**3)/6 + (tau**9)/135 + (tau**15)/600)
                              )
        # factor when tau >= 1
        super_tau = -(1/A) * ((tau**-5)/10 + (tau**-15)/315 + (tau**-25)/1500)

        expr_cond_pairs = [(sub_tau, tau < 1),
                           (super_tau, True)
                           ]
        g_term = Piecewise(*expr_cond_pairs)

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

        site_ratio_normalization = self._site_ratio_normalization(phase)
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
        g_term = Piecewise(*expr_cond_pairs_curie) + Piecewise(*expr_cond_pairs_neel)

        return v.R * v.T * log(beta+1) * \
            g_term / site_ratio_normalization

    def twostate_energy(self, dbe):
        """
        Return the energy from liquid-amorphous two-state model.
        """
        phase = dbe.phases[self.phase_name]
        param_search = dbe.search
        site_ratio_normalization = self._site_ratio_normalization(phase)
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
        return result / self._site_ratio_normalization(phase)

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
            if len(sublattice) == 1 and sublattice[0] == 'VA':
                continue
            if species_name in list(sublattice):
                site_ratio_normalization += site_ratios[idx]
                numerator += site_ratios[idx] * \
                    v.SiteFraction(phase_name, idx, species_name)

        if site_ratio_normalization == 0 and species_name == 'VA':
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
        for atom in disordered_model.energy.atoms(v.SiteFraction):
            # Replace disordered phase site fractions with mole fractions of
            # ordered phase site fractions.
            # Special case: Pure vacancy sublattices
            all_species_in_sublattice = \
                dbe.phases[disordered_phase_name].constituents[
                    atom.sublattice_index]
            if atom.species == 'VA' and len(all_species_in_sublattice) == 1:
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
        for sitefrac in ordered_energy.atoms(v.SiteFraction):
            all_species_in_sublattice = \
                dbe.phases[ordered_phase_name].constituents[
                    sitefrac.sublattice_index]
            if sitefrac.species == 'VA' and len(all_species_in_sublattice) == 1:
                # pure-vacancy sublattices should not be replaced
                # this handles cases like AL,NI,VA:AL,NI,VA:VA and
                # ensures the VA's don't get mixed up
                continue
            molefraction_dict[sitefrac] = \
                self.mole_fraction(sitefrac.species,
                                   ordered_phase_name, constituents,
                                   dbe.phases[ordered_phase_name].sublattices)

        return ordered_energy - ordered_energy.subs(molefraction_dict,
                                                    simultaneous=True)

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
        self.components = {x.upper() for x in comps}
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

