#pylint: disable=C0103,R0903,W0232
"""
Classes and constants for representing thermodynamic variables.
"""

import itertools
from sympy import Float, Symbol
from pycalphad.io.grammar import chemical_formula


class Species(object):
    """
    A chemical species.

    Attributes
    ----------
    name : string
        Name of the specie
    constituents : dict
        Dictionary of {element: quantity} where the element is a string and the quantity a float.
    charge : int
        Integer charge. Can be positive or negative.
    """
    def __new__(cls, name, constituents=None, charge=0):
        if constituents is not None:
            new_self = object.__new__(cls)
            new_self.name = name
            new_self.constituents = constituents
            new_self.charge = charge
            return new_self
        else:
            arg = name
        # if a Species is passed in, return it
        if arg.__class__ == cls:
            return arg

        new_self = object.__new__(cls)

        if isinstance(arg, str):
            parse_list = chemical_formula.parseString(arg)
        else:
            parse_list = arg
        new_self.name = name
        new_self.charge = parse_list[1]
        parse_list = parse_list[0]
        new_self.constituents = {parse_list[i]: parse_list[i+1] for i in range(0, len(parse_list), 2)}
        return new_self

    def __getnewargs__(self):
        return self.name, self.constituents, self.charge

    def __eq__(self, other):
        """Two species are the same if their names and constituents are the same."""
        if isinstance(other, self.__class__):
            return (self.name == other.name) and (self.constituents == other.constituents)
        else:
            return False

    def __str__(self):
        return self.name

    def __hash__(self):
        return hash(self.name)


class StateVariable(Symbol):
    """
    State variables are symbols with built-in assumptions of being real.
    """
    def __new__(cls, name, *args, **assumptions):
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

    def __getnewargs__(self):
        return self.phase_name, self.sublattice_index, self.species

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

    def __getnewargs__(self):
        return self.phase_name, self.multiplicity

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

    def __getnewargs__(self):
        if self.phase_name is not None:
            return self.phase_name, self.species
        else:
            return self.species,

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

    def __getnewargs__(self):
        return self.species,

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
