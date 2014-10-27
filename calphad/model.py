"""
The Model module provides support for using a Database to perform
calculations under specified conditions.
"""

from sympy import S, Symbol
from tinydb import where

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
            if sublattice[0] not in self._components:
                return False
        return True
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
        # First, build the reference energy term
        pure_energy_term = S.Zero
        pure_param_query = (
            (where('phase_name') == phase.name) & \
            (where('parameter_order') == 0) & \
            (where('parameter_type') == "G") & \
            (where('constituent_array').test(self._purity_test))
        )
        pure_params = param_search(pure_param_query)
        print('Found parameters: '+str(pure_params))

        for param in pure_params:
            site_fraction_product = S.One
            for subl_index, comp in enumerate(param['constituent_array']):
                # We know that comp has one entry, by filtering
                sitefrac = \
                    Symbol(Y(phase.name, subl_index, comp[0]))
                site_fraction_product *= sitefrac
            pure_energy_term += (
                site_fraction_product * param['parameter'].subs(symbols)
            )
        # Next, build the ideal mixing term
        ideal_mixing_term = S.Zero
        self._phases[phase.name] = pure_energy_term + ideal_mixing_term
