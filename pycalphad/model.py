"""
The Model module provides support for using a Database to perform
calculations under specified conditions.
"""
from __future__ import division
from sympy import log, Add, And, Mul, Piecewise, Pow, S
from tinydb import where
import pycalphad.variables as v

# What about just running all self._model_*?
class Model(object):
    """
    Models use an abstract representation of the energy function
    to calculate values under specified conditions.

    Attributes
    ----------
    None yet.

    Methods
    -------
    None yet.

    Examples
    --------
    None yet.
    """
    def __init__(self, db, comps, phases):
        print("Initializing model for "+ str(phases))
        self._components = set(comps)
        self._phases = {}
        self.variables = set()
        for phase_name, phase_obj in db.phases.items():
            print('Checking '+phase_name)
            if phase_name in phases:
                cur_phase = self._build_phase(phase_obj, db.symbols, db.search)
                if cur_phase is not None:
                    # do something
                    pass
        print("Initialization complete")
    def __call__(self, substitutions=None, phases=None):
        result = {}
        if phases is None:
            for name, ast in self._phases.items():
                if substitutions is None:
                    result[name] = ast
                else:
                    result[name] = ast.subs(substitutions)
        else:
            for phase_name in phases:
                if substitutions is None:
                    result[phase_name] = self._phases[phase_name]
                else:
                    result[phase_name] = \
                        self._phases[phase_name].subs(substitutions)
        if len(result) == 1:
            return list(result.values())[0]
        else:
            return result
    def _purity_test(self, constituent_array):
        """
        Check if constituent array only has one species in its array
        This species must also be an active species
        """
        for sublattice in constituent_array:
            if len(sublattice) != 1:
                return False
            if (sublattice[0] not in self._components) and \
                (sublattice[0] != '*'):
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
            valid = set(sublattice).issubset(self._components)
            if len(sublattice) > 1 and valid:
                result = True
            if not valid:
                result = False
                break
        return result
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
        published in the pycalphad journal.
        """
        arity = len(comps)
        return_dict = {}
        correction_term = (S.One - Add(*comps)) / arity #pylint: disable=W0142
        for comp in comps:
            return_dict[comp] = comp + correction_term
        return return_dict
    def _build_phase(self, phase, symbols, param_search):
        """
        Apply phase's model hints to build a master SymPy object.
        """
        for sublattice in phase.constituents:
            if len(self._components - set(sublattice)) == \
                len(self._components):
                # None of the components in a sublattice are active
                # We cannot build a model of this phase
                return None
        total_energy = S.Zero
        # First, build the reference energy term
        total_energy += self.reference_energy(phase, symbols, param_search)

        # Next, add the ideal mixing term
        total_energy += self.ideal_mixing_energy(phase, symbols, param_search)

        # Next, add the binary, ternary and higher order mixing term
        total_energy += self.excess_mixing_energy(phase, symbols, param_search)

        # Next, we need to handle contributions from magnetic ordering
        total_energy += self.magnetic_energy(phase, symbols, param_search)

        # Next, we handle atomic ordering
        total_energy += \
            self.atomic_ordering_energy(phase, symbols, param_search)

        # Save all variables
        self.variables.update(total_energy.atoms(v.StateVariable))

        self._phases[phase.name] = total_energy
    def _redlich_kister_sum(self, phase, symbols, param_type, param_search):
        """
        Construct parameter in Redlich-Kister polynomial basis, using
        the Muggianu ternary parameter extension.
        """
        rk_term = S.Zero
        param_query = (
            (where('phase_name') == phase.name) & \
            (where('parameter_type') == param_type)
        )
        # search for desired parameters
        params = param_search(param_query)
        for param in params:
            # iterate over every sublattice
            for subl_index, comps in enumerate(param['constituent_array']):
                # consider only active components in sublattice
                # convert strings to symbols
                comp_symbols = \
                    [
                        v.SiteFraction(phase.name, subl_index, comp)
                        for comp in set(comps).intersection(self._components)
                    ]
                ratio = phase.sublattices[subl_index] / sum(phase.sublattices)
                mixing_term = Mul(*comp_symbols) #pylint: disable=W0142
                # is this a higher-order interaction parameter?
                if len(comp_symbols) > 1 and param['parameter_order'] > 0:
                    # interacting sublattice, add the interaction polynomial
                    redlich_kister_poly = Pow(comp_symbols[0] - \
                        Add(*comp_symbols[1:]), param['parameter_order'])
                    # Perform Muggianu adjustment to site fractions
                    mixing_term *= redlich_kister_poly.subs(
                        self._Muggianu_correction_dict(comp_symbols)
                    )
                rk_term += ratio * mixing_term * \
                    param['parameter'].subs(symbols)
        return rk_term
    def reference_energy(self, phase, symbols, param_search):
        """
        Returns the weighted average of the endmember energies
        in symbolic form.
        """
        total_mixing_sites = sum(phase.sublattices)
        # TODO: Handle wildcards in constituent array
        pure_energy_term = S.Zero
        pure_param_query = (
            (where('phase_name') == phase.name) & \
            (where('parameter_order') == 0) & \
            (where('parameter_type') == "G") & \
            (where('constituent_array').test(self._purity_test))
        )

        pure_params = param_search(pure_param_query)

        for param in pure_params:
            site_fraction_product = S.One
            for subl_index, comp in enumerate(param['constituent_array']):
                # We know that comp has one entry, by filtering
                sitefrac = \
                    v.SiteFraction(phase.name, subl_index, comp[0])
                site_fraction_product *= sitefrac
            pure_energy_term += (
                site_fraction_product * param['parameter'].subs(symbols)
            ) / total_mixing_sites
        return pure_energy_term
    def ideal_mixing_energy(self, phase, symbols, param_search):
        #pylint: disable=W0613
        """
        Returns the ideal mixing energy in symbolic form.
        """
        total_mixing_sites = sum(phase.sublattices)
        ideal_mixing_term = S.Zero
        for subl_index, sublattice in enumerate(phase.constituents):
            active_comps = set(sublattice).intersection(self._components)
            ratio = phase.sublattices[subl_index] / total_mixing_sites
            for comp in active_comps:
                sitefrac = \
                    v.SiteFraction(phase.name, subl_index, comp)
                mixing_term = \
                    Piecewise((sitefrac * log(sitefrac), sitefrac > 1e-16),
                              (0, True)
                             )
                ideal_mixing_term += (mixing_term*ratio)
        ideal_mixing_term *= (v.R * v.T)
        return ideal_mixing_term
    def excess_mixing_energy(self, phase, symbols, param_search):
        """
        Build the binary, ternary and higher order interaction term
        Here we use Redlich-Kister polynomial basis by default
        Here we use the Muggianu ternary extension by default
        Replace y_i -> y_i + (1 - sum(y involved in parameter)) / m,
        where m is the arity of the interaction parameter
        """
        excess_mixing_term = S.Zero
        interaction_param_query = (
            (where('phase_name') == phase.name) & \
            (
                (where('parameter_type') == "G") | \
                (where('parameter_type') == "L")
            ) & \
            (where('constituent_array').test(self._interaction_test))
        )
        # search for desired parameters
        interaction_params = param_search(interaction_param_query)
        for param in interaction_params:
            # iterate over every sublattice
            for subl_index, comps in enumerate(param['constituent_array']):
                # consider only active components in sublattice
                active_comps = set(comps).intersection(self._components)
                # convert strings to symbols
                comp_symbols = \
                    [
                        v.SiteFraction(phase.name, subl_index, comp)
                        for comp in active_comps
                    ]
                ratio = phase.sublattices[subl_index] / sum(phase.sublattices)
                mixing_term = Mul(*comp_symbols) #pylint: disable=W0142
                # is this a higher-order interaction parameter?
                # TODO: fix parameter_order handling for ternary params
                if len(active_comps) > 1 and param['parameter_order'] > 0:
                    # interacting sublattice, add the interaction polynomial
                    redlich_kister_poly = Pow(comp_symbols[0] - \
                        Add(*comp_symbols[1:]), param['parameter_order'])
                    # Perform Muggianu adjustment to site fractions
                    mixing_term *= redlich_kister_poly.subs(
                        self._Muggianu_correction_dict(comp_symbols)
                    )
                excess_mixing_term += ratio * mixing_term * \
                    param['parameter'].subs(symbols)
        return excess_mixing_term
    def magnetic_energy(self, phase, symbols, param_search):
        #pylint: disable=C0103
        """
        Return the energy from magnetic ordering in symbolic form.
        The implemented model is the Inden-Hillert-Jarl formulation.
        The approach follows from the background section of W. Xiong, 2011.
        """
        if 'ihj_magnetic_structure_factor' not in phase.model_hints:
            return S.Zero
        if 'ihj_magnetic_afm_factor' not in phase.model_hints:
            return S.Zero
        # define basic variables
        mean_magnetic_moment = \
            self._redlich_kister_sum(phase, symbols, 'BMAGN', param_search)
        afm_factor = phase.model_hints['ihj_magnetic_afm_factor']
        curie_temp = \
            self._redlich_kister_sum(phase, symbols, 'TC', param_search)
        tc = Piecewise(
            (curie_temp, curie_temp > 0),
            (curie_temp/afm_factor, curie_temp <= 0)
            )
        tau = v.T / tc

        # define model parameters
        p = phase.model_hints['ihj_magnetic_structure_factor']
        A = 518/1125 + (11592/15975)*(1/p - 1)
        # factor when tau < 1
        sub_tau = 1 - (1/A) * (79/(140*p)*(tau**(-1)) + (474/497)*(1/p - 1) * \
                     ((tau**3)/6 + (tau**9)/135 + (tau**15)/600)
                              )
        # factor when tau >= 1
        super_tau = -(1/A) * ((tau**-5)/10 + (tau**-15)/315 + (tau**-25)/1500)

        expr_cond_pairs = [(0, And(tc < 1e-4, tc > 1e6)),
                           (sub_tau, v.T < tc),
                           (super_tau, v.T >= tc)
                          ]

        g_term = Piecewise(*expr_cond_pairs) #pylint: disable=W0142
        return v.R * v.T * log(mean_magnetic_moment+1) * g_term
    def atomic_ordering_energy(self, phase, symbols, param_search):
        """
        Return the atomic ordering energy in symbolic form.
        """
        ordering_term = S.Zero
        return ordering_term
