#pylint: disable=C0103,R0903,W0232
"""
Classes and constants for representing thermodynamic variables.
"""

from sympy import Float, Symbol

class StateVariable(Symbol):
    """
    State variables are symbols with built-in assumptions of being real
    and nonnegative.
    """
    def __new__(cls, name):
        return Symbol.__new__(cls, name, nonnegative=True, real=True)

class SiteFraction(StateVariable):
    """
    Site fractions are symbols with built-in assumptions of being real
    and nonnegative. The constructor handles formatting of the name.
    """
    def __new__(cls, phase_name, subl_index, species): #pylint: disable=W0221
        varname = phase_name + str(subl_index) + species
        #pylint: disable=E1121
        new_self = StateVariable.__new__(cls, varname)
        new_self.phase_name = phase_name
        new_self.sublattice_index = subl_index
        new_self.species = species
        return new_self
    def _latex(self):
        #pylint: disable=E1101
        return 'y^{'+self.phase_name.replace('_', '-') + \
            '}_{'+str(self.subl_index)+'},_{'+self.species+'}'

temperature = T = StateVariable('T')
entropy = S = StateVariable('S')
pressure = P = StateVariable('P')
volume = V = StateVariable('V')
moles = N = StateVariable('N')
site_fraction = Y = SiteFraction
si_gas_constant = R = Float(8.3145) # ideal gas constant
