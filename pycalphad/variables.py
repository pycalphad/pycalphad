#pylint: disable=C0103,R0903,W0232
"""
Classes and constants for representing thermodynamic variables.
"""

from sympy import Float, Symbol

class StateVariable(Symbol):
    """
    State variables are symbols with built-in assumptions of being real.
    """
    def __new__(cls, name, **assumptions):
        return Symbol.__new__(cls, name.upper(), real=True, **assumptions)

class SiteFraction(StateVariable):
    """
    Site fractions are symbols with built-in assumptions of being real
    and nonnegative. The constructor handles formatting of the name.
    """
    def __new__(cls, phase_name, subl_index, species): #pylint: disable=W0221
        varname = phase_name + str(subl_index) + species
        #pylint: disable=E1121
        new_self = StateVariable.__new__(cls, varname, nonnegative=True)
        new_self.phase_name = phase_name.upper()
        new_self.sublattice_index = subl_index
        new_self.species = species.upper()
        return new_self
    def _latex(self):
        "LaTeX representation."
        #pylint: disable=E1101
        return 'y^{'+self.phase_name.replace('_', '-') + \
            '}_{'+str(self.subl_index)+'},_{'+self.species+'}'
    def __str__(self):
        "String representation."
        #pylint: disable=E1101
        return 'Y(%s,%d,%s)' % \
            (self.phase_name, self.sublattice_index, self.species)

class PhaseFraction(StateVariable):
    """
    Phase fractions are symbols with built-in assumptions of being real
    and nonnegative. The constructor handles formatting of the name.
    """
    def __new__(cls, phase_name, multiplicity): #pylint: disable=W0221
        varname = phase_name + str(multiplicity)
        #pylint: disable=E1121
        new_self = StateVariable.__new__(cls, varname, nonnegative=True)
        new_self.phase_name = phase_name.upper()
        new_self.multiplicity = multiplicity
        return new_self
    def _latex(self):
        "LaTeX representation."
        #pylint: disable=E1101
        return 'f^{'+self.phase_name.replace('_', '-') + \
            '}_{'+str(self.multiplicity)+'}'

class Composition(StateVariable):
    """
    Compositions are symbols with built-in assumptions of being real
    and nonnegative.
    """
    def __new__(cls, *args): #pylint: disable=W0221
        new_self = None
        varname = None
        phase_name = None
        species = None
        if len(args) == 1:
            # this is an overall composition variable
            species = args[0].upper()
            varname = 'X_' + species
        elif len(args) == 2:
            # this is a phase-specific composition variable
            phase_name = args[0].upper()
            species = args[1].upper()
            varname = 'X_' + phase_name + '_' + species
        else:
            # not defined
            raise ValueError('Composition not defined for args: '+args)

        #pylint: disable=E1121
        new_self = StateVariable.__new__(cls, varname, nonnegative=True)
        new_self.phase_name = phase_name
        new_self.species = species
        return new_self
    def _latex(self):
        "LaTeX representation."
        #pylint: disable=E1101
        if self.phase_name:
            return 'x^{'+self.phase_name.replace('_', '-') + \
                '}_{'+self.species+'}'
        else:
            return 'x_{'+self.species+'}'

class ChemicalPotential(StateVariable):
    """
    Chemical potentials are symbols with built-in assumptions of being real.
    """
    def __new__(cls, species, **assumptions):
        varname = 'MU_' + species.upper()
        new_self = StateVariable.__new__(cls, varname, **assumptions)
        new_self.species = species
        return new_self
    def _latex(self):
        "LaTeX representation."
        return '\mu_{'+self.species+'}'
    def __str__(self):
        "String representation."
        return 'MU(%s)' % self.species

temperature = T = StateVariable('T')
entropy = S = StateVariable('S')
pressure = P = StateVariable('P')
volume = V = StateVariable('V')
moles = N = StateVariable('N')
site_fraction = Y = SiteFraction
X = Composition
MU = ChemicalPotential
si_gas_constant = R = Float(8.3145) # ideal gas constant
