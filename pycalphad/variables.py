#pylint: disable=C0103,R0903,W0232
"""
Classes and constants for representing thermodynamic variables.
"""

import sys
from sympy import Float, Symbol
from pycalphad.io.grammar import parse_chemical_formula


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
        if arg == '*':
            new_self = object.__new__(cls)
            new_self.name = '*'
            new_self.constituents = dict()
            new_self.charge = 0
            return new_self
        if arg is None:
            new_self = object.__new__(cls)
            new_self.name = ''
            new_self.constituents = dict()
            new_self.charge = 0
            return new_self

        if isinstance(arg, str):
            parse_list = parse_chemical_formula(arg.upper())
        else:
            parse_list = arg
        new_self.name = name
        new_self.charge = parse_list[1]
        parse_list = parse_list[0]
        new_self.constituents = dict(parse_list)
        return new_self

    def __getnewargs__(self):
        return self.name, self.constituents, self.charge

    def __eq__(self, other):
        """Two species are the same if their names and constituents are the same."""
        if isinstance(other, self.__class__):
            return (self.name == other.name) and (self.constituents == other.constituents)
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        return self.name < other.name

    def __str__(self):
        return self.name

    @property
    def escaped_name(self):
        "Name safe to embed in the variable name of complex arithmetic expressions."
        return str(self).replace('-', '_NEG').replace('+', '_POS').replace('/', 'Z')

    @property
    def number_of_atoms(self):
        "Number of atoms per formula unit. Vacancies do not count as atoms."
        return sum(value for key, value in self.constituents.items() if key != 'VA')

    @property
    def weight(self):
        "Number of grams per formula unit."
        return NotImplementedError

    def __repr__(self):
        if self.name == '*':
            return '*'
        if self.name == '':
            return 'None'
        species_constituents = ''.join(
            ['{}{}'.format(el, val) for el, val in sorted(self.constituents.items(), key=lambda t: t[0])])
        if self.charge == 0:
            repr_str = "(\'{0}\', \'{1}\')"
        else:
            repr_str = "(\'{0}\', \'{1}\', charge={2})"
        return str(self.__class__.__name__)+repr_str.format(self.name.upper(), species_constituents, self.charge)

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
        varname = phase_name + str(subl_index) + Species(species).escaped_name
        #pylint: disable=E1121
        new_self = StateVariable.__new__(cls, varname, nonnegative=True)
        new_self.phase_name = phase_name.upper()
        new_self.sublattice_index = subl_index
        new_self.species = Species(species)
        return new_self

    def __getnewargs__(self):
        return self.phase_name, self.sublattice_index, self.species

    def _latex(self, printer=None):
        "LaTeX representation."
        #pylint: disable=E1101
        return 'y^{\mathrm{'+self.phase_name.replace('_', '-') + \
            '}}_{'+str(self.sublattice_index)+',\mathrm{'+self.species.escaped_name+'}}'

    def __str__(self):
        "String representation."
        #pylint: disable=E1101
        return 'Y(%s,%d,%s)' % \
            (self.phase_name, self.sublattice_index, self.species.escaped_name)

class PhaseFraction(StateVariable):
    """
    Phase fractions are symbols with built-in assumptions of being real
    and nonnegative. The constructor handles formatting of the name.
    """
    def __new__(cls, phase_name): #pylint: disable=W0221
        varname = 'NP_' + str(phase_name)
        #pylint: disable=E1121
        new_self = StateVariable.__new__(cls, varname, nonnegative=True)
        new_self.phase_name = phase_name.upper()
        return new_self

    def __getnewargs__(self):
        return self.phase_name,

    def _latex(self, printer=None):
        "LaTeX representation."
        #pylint: disable=E1101
        return 'f^{'+self.phase_name.replace('_', '-')

class MoleFraction(StateVariable):
    """
    MoleFractions are symbols with built-in assumptions of being real
    and nonnegative.
    """
    def __new__(cls, *args): #pylint: disable=W0221
        new_self = None
        varname = None
        phase_name = None
        species = None
        if len(args) == 1:
            # this is an overall composition variable
            species = Species(args[0])
            varname = 'X_' + species.escaped_name.upper()
        elif len(args) == 2:
            # this is a phase-specific composition variable
            phase_name = args[0].upper()
            species = Species(args[1])
            varname = 'X_' + phase_name + '_' + species.escaped_name.upper()
        else:
            # not defined
            raise ValueError('MoleFraction not defined for args: '+args)

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

    def _latex(self, printer=None):
        "LaTeX representation."
        #pylint: disable=E1101
        if self.phase_name:
            return 'x^{'+self.phase_name.replace('_', '-') + \
                '}_{'+self.species.escaped_name+'}'
        else:
            return 'x_{'+self.species.escaped_name+'}'


class MassFraction(StateVariable):
    """
    Weight fractions are symbols with built-in assumptions of being real and nonnegative.
    """
    def __new__(cls, *args):  # pylint: disable=W0221
        new_self = None
        varname = None
        phase_name = None
        species = None
        if len(args) == 1:
            # this is an overall composition variable
            species = Species(args[0])
            varname = 'W_' + species.escaped_name.upper()
        elif len(args) == 2:
            # this is a phase-specific composition variable
            phase_name = args[0].upper()
            species = Species(args[1])
            varname = 'W_' + phase_name + '_' + species.escaped_name.upper()
        else:
            # not defined
            raise ValueError('Weight fraction not defined for args: '+args)

        # pylint: disable=E1121
        new_self = StateVariable.__new__(cls, varname, nonnegative=True)
        new_self.phase_name = phase_name
        new_self.species = species
        return new_self

    def __getnewargs__(self):
        if self.phase_name is not None:
            return self.phase_name, self.species
        else:
            return self.species,

    def _latex(self, printer=None):
        "LaTeX representation."
        # pylint: disable=E1101
        if self.phase_name:
            return 'w^{'+self.phase_name.replace('_', '-') + \
                '}_{'+self.species.escaped_name+'}'
        else:
            return 'w_{'+self.species.escaped_name+'}'


def get_mole_fractions(mass_fractions, dependent_species, pure_element_mass_dict):
    """
    Return a mapping of MoleFractions for a point composition.

    Parameters
    ----------
    mass_fractions : Mapping[MassFraction, float]

    dependent_species : Union[Species, str]
        Dependent species not appearing in the independent mass fractions.

    pure_element_mass_dict : Union[Mapping[str, float], pycalphad.Database]
        Either a mapping from pure elements to their mass, or a Database from
        which they can be retrieved.

    Returns
    -------
    Dict[MoleFraction, float]

    """
    if not all(isinstance(mf, MassFraction) for mf in mass_fractions):
        from pycalphad.core.errors import ConditionError
        raise ConditionError("All mass_fractions must be instances of MassFraction (v.W). Got ", mass_fractions)
    dependent_species = Species(dependent_species)
    species_mass_fracs = {mf.species: frac for mf, frac in mass_fractions.items()}
    all_species = set(species_mass_fracs.keys()) | {dependent_species}
    # Check if the mass dict is a Database, which is the source of the mass_dict
    from pycalphad import Database  # Imported here to avoid circular import
    if isinstance(pure_element_mass_dict, Database):
        pure_element_mass_dict = {el: refdict['mass'] for el, refdict in pure_element_mass_dict.refstates.items()}

    species_mass_dict = {}
    for species in all_species:
        species_mass_dict[species] = sum([pure_element_mass_dict[pe]*natoms for pe, natoms in species.constituents.items()])

    # add dependent species
    species_mass_fracs[dependent_species] = 1 - sum(species_mass_fracs.values())
    # compute moles
    species_moles = {species: mass_frac/species_mass_dict[species] for species, mass_frac in species_mass_fracs.items()}
    # normalize
    total_moles = sum(species_moles.values())
    species_mole_fractions = {species: moles/total_moles for species, moles in species_moles.items()}
    # remove dependent species
    species_mole_fractions.pop(dependent_species)
    return {MoleFraction(species): fraction for species, fraction in species_mole_fractions.items()}


def get_mass_fractions(mole_fractions, dependent_species, pure_element_mass_dict):
    """
    Return a mapping of MassFractions for a point composition.

    Parameters
    ----------
    mass_fractions : Mapping[MoleFraction, float]

    dependent_species : Union[Species, str]
        Dependent species not appearing in the independent mass fractions.

    pure_element_mass_dict : Union[Mapping[str, float], pycalphad.Database]
        Either a mapping from pure elements to their mass, or a Database from
        which they can be retrieved.

    Returns
    -------
    Dict[MassFraction, float]

    """
    if not all(isinstance(mf, MoleFraction) for mf in mole_fractions):
        from pycalphad.core.errors import ConditionError
        raise ConditionError("All mole_fractions must be instances of MoleFraction (v.X). Got ", mole_fractions)
    dependent_species = Species(dependent_species)
    species_mole_fracs = {mf.species: frac for mf, frac in mole_fractions.items()}
    all_species = set(species_mole_fracs.keys()) | {dependent_species}
    # Check if the mass dict is a Database, which is the source of the mass_dict
    from pycalphad import Database  # Imported here to avoid circular import
    if isinstance(pure_element_mass_dict, Database):
        pure_element_mass_dict = {el: refdict['mass'] for el, refdict in pure_element_mass_dict.refstates.items()}

    species_mass_dict = {}
    for species in all_species:
        species_mass_dict[species] = sum([pure_element_mass_dict[pe]*natoms for pe, natoms in species.constituents.items()])

    # add dependent species
    species_mole_fracs[dependent_species] = 1 - sum(species_mole_fracs.values())
    # compute mass
    species_mass = {species: mole_frac*species_mass_dict[species] for species, mole_frac in species_mole_fracs.items()}
    # normalize
    total_mass = sum(species_mass.values())
    species_mass_fractions = {species: mass/total_mass for species, mass in species_mass.items()}
    # remove dependent species
    species_mass_fractions.pop(dependent_species)
    return {MassFraction(species): fraction for species, fraction in species_mass_fractions.items()}


class ChemicalPotential(StateVariable):
    """
    Chemical potentials are symbols with built-in assumptions of being real.
    """
    def __new__(cls, species, **assumptions):
        species = Species(species)
        varname = 'MU_' + species.escaped_name.upper()
        new_self = StateVariable.__new__(cls, varname, **assumptions)
        new_self.species = species
        return new_self

    def __getnewargs__(self):
        return self.species,

    def _latex(self, printer=None):
        "LaTeX representation."
        return '\mu_{'+self.species.escaped_name+'}'

    def __str__(self):
        "String representation."
        return 'MU_%s' % self.species.name

temperature = T = StateVariable('T')
entropy = S = StateVariable('S')
pressure = P = StateVariable('P')
volume = V = StateVariable('V')
moles = N = StateVariable('N')
site_fraction = Y = SiteFraction
X = MoleFraction
W = MassFraction
MU = ChemicalPotential
NP = PhaseFraction
si_gas_constant = R = Float(8.3145) # ideal gas constant

CONDITIONS_REQUIRING_HESSIANS = {ChemicalPotential, PhaseFraction}
