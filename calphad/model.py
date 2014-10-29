"""
The Model module provides support for using a Database to perform
calculations under specified conditions.
"""

from sympy import log, Add, Mul, Pow, S, Symbol
from tinydb import where

SI_GAS_CONSTANT = 8.3145
T = Symbol('T')

def Y(phase_name, subl_index, comp): #pylint: disable=C0103
    """
    Convenience function for the name of site fraction variables.
    """
    return 'Y^'+phase_name+'_'+str(subl_index)+',_'+comp

class Model(object):
    """
    Models use Phases to calculate energies under specified conditions.

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
        published in the Calphad journal.
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
                # None of the components in this sublattice are active
                # We cannot build a model of this phase
                return None
        total_mixing_sites = sum(phase.sublattices)
        # First, build the reference energy term
        # TODO: Handle wildcards in constituent array
        pure_energy_term = S.Zero
        pure_param_query = (
            (where('phase_name') == phase.name) & \
            (where('parameter_order') == 0) & \
            (where('parameter_type') == "G") & \
            (where('constituent_array').test(self._purity_test))
        )
        interaction_param_query = (
            (where('phase_name') == phase.name) & \
            (
                (where('parameter_type') == "G") | \
                (where('parameter_type') == "L")
            ) & \
            (where('constituent_array').test(self._interaction_test))
        )
        pure_params = param_search(pure_param_query)
        interaction_params = param_search(interaction_param_query)

        for param in pure_params:
            site_fraction_product = S.One
            for subl_index, comp in enumerate(param['constituent_array']):
                # We know that comp has one entry, by filtering
                sitefrac = \
                    Symbol(Y(phase.name, subl_index, comp[0]))
                site_fraction_product *= sitefrac
            pure_energy_term += (
                site_fraction_product * param['parameter'].subs(symbols)
            ) / total_mixing_sites
        # Next, build the ideal mixing term
        ideal_mixing_term = S.Zero
        for subl_index, sublattice in enumerate(phase.constituents):
            active_comps = set(sublattice).intersection(self._components)
            ratio = phase.sublattices[subl_index] / total_mixing_sites
            for comp in active_comps:
                sitefrac = \
                    Symbol(Y(phase.name, subl_index, comp))
                mixing_term = SI_GAS_CONSTANT * Symbol('T') \
                    * sitefrac * log(sitefrac)
                ideal_mixing_term += (mixing_term*ratio)

        # Next, build the binary, ternary and higher order interaction term
        # Here we use Redlich-Kister polynomial basis by default
        # Here we use the Muggianu ternary extension by default
        # Replace y_i -> y_i + (1 - sum(y involved in parameter)) / m,
        #   where m is the arity of the interaction parameter
        excess_mixing_term = S.Zero
        for param in interaction_params:
            for subl_index, comps in enumerate(param['constituent_array']):
                active_comps = set(comps).intersection(self._components)
                comp_symbols = \
                    [Symbol(
                        Y(phase.name, subl_index, comp)
                        ) for comp in active_comps]
                ratio = phase.sublattices[subl_index] / total_mixing_sites
                mixing_term = Mul(*comp_symbols) #pylint: disable=W0142
                if len(active_comps) > 1 and param['parameter_order'] > 0:
                    # interacting sublattice, add the interaction polynomial
                    redlich_kister_poly = Pow(comp_symbols[0] - \
                        Add(*comp_symbols[1:]), param['parameter_order'])
                    # Perform Muggianu adjustment to site fractions
                    mixing_term *= redlich_kister_poly.subs(
                        self._Muggianu_correction_dict(comp_symbols)
                    )
                # Perform Muggianu adjustment to all site fractions
                excess_mixing_term += ratio * mixing_term * \
                    param['parameter'].subs(symbols)
        # Next, we need to handle contributions from magnetic ordering

        self._phases[phase.name] = pure_energy_term + ideal_mixing_term + \
            excess_mixing_term
