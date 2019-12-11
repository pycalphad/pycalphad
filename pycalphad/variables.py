#pylint: disable=C0103,R0903,W0232
"""
Classes and constants for representing thermodynamic variables.
"""

import sys
import copy
# Python 2 vs 3 string types in isinstance
if sys.version_info[0] >= 3:
    string_type = str
else:
    string_type = basestring
import numpy as np
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

        if isinstance(arg, string_type):
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
    def __new__(cls, phase_name, multiplicity): #pylint: disable=W0221
        varname = phase_name + str(multiplicity)
        #pylint: disable=E1121
        new_self = StateVariable.__new__(cls, varname, nonnegative=True)
        new_self.phase_name = phase_name.upper()
        new_self.multiplicity = multiplicity
        return new_self

    def __getnewargs__(self):
        return self.phase_name, self.multiplicity

    def _latex(self, printer=None):
        "LaTeX representation."
        #pylint: disable=E1101
        return 'f^{'+self.phase_name.replace('_', '-') + \
            '}_{'+str(self.multiplicity)+'}'

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


class Composition():
    """Convenience object to facilitiate operating in multicomponent composition spaces

    Parameters
    ----------
    masses : Union[Dict[str, float], Database]
        Element to mass dictionary or database to retrive the masses from.
    composition : Dict[Union[v.MoleFraction, v.MassFraction], float]
        Mapping of MoleFraction/MassFraction objects to values.
    dependent_component: str
        Pure element that is not specified in the composition, but known
        because the sum of fractions must be one.

    Attributes
    ----------
    masses : Dict[str, float]
    """

    def __init__(self, masses, composition, dependent_component):
        # TODO: check degree of freedom (ncomps, n-1 independent compositions)
        # TODO: convert any complex  species into pure element composition
        self.components = {dependent_component}
        for xw_fraction in composition.keys():
            self.components |= xw_fraction.species.constituents.keys()
        if all([isinstance(k, MoleFraction) for k in composition.keys()]):
            self._mode = MoleFraction
        elif all([isinstance(k, MassFraction) for k in composition.keys()]):
            self._mode = MassFraction
        else:
            raise ValueError(f'Mixed MoleFraction and MassFraction compositions not supported (got {composition}).')
        from pycalphad import Database  # Imported here to avoid circular import
        if isinstance(masses, Database):
            self.masses = {c: masses.refstates[c]['mass'] for c in self.components}
        else:  # Assume masses is a dict mapping components to mass
            self.masses = masses
        self._composition = copy.deepcopy(composition)
        self.dependent_component = dependent_component

    @property
    def composition(self):
        return self._composition

    @property
    def mass_fractions(self):
        if issubclass(self._mode, MassFraction):
            return self._composition
        else:
            comp_names = {w.species.name for w in self._composition}
            mass = {MassFraction(comp): self._composition[MoleFraction(comp)]*self.masses[comp] for comp in comp_names}
            dep_comp_mass = (1-sum(self._composition.values()))*self.masses[self.dependent_component]
            total_mass = sum(mass.values()) + dep_comp_mass
            mass_fracs = {component: mass_amnt/total_mass for component, mass_amnt in mass.items()}
            return mass_fracs

    @property
    def mole_fractions(self):
        if issubclass(self._mode, MoleFraction):
            return self._composition
        else:
            comp_names = {w.species.name for w in self._composition}
            moles = {MoleFraction(comp): self._composition[MassFraction(comp)]/self.masses[comp] for comp in comp_names}
            dep_comp_moles = (1-sum(self._composition.values()))/self.masses[self.dependent_component]
            total_moles = sum(moles.values()) + dep_comp_moles
            mole_fracs = {component: mole_amnt/total_moles for component, mole_amnt in moles.items()}
            return mole_fracs

    def to_mole_fractions(self):
        """Return a new Composition object converted to mole fractions"""
        return Composition(self.masses, self.mole_fractions, self.dependent_component)

    def to_mass_fractions(self):
        """Return a new Composition object converted to mass fractions"""
        return Composition(self.masses, self.mass_fractions, self.dependent_component)

    def set_dependent_component(self, component):
        """Change the dependent component

        component : str

        Examples
        --------
        >>> raise NotImplementedError()
        """
        if self.dependent_component != component:
            self._composition[self._mode(self.dependent_component)] = (1-sum(self._composition.values()))
            del self._composition[self._mode(component)]
            self.dependent_component = component
        return self

    def mix(self, composition, amount):
        """Linearly mix two Compositions to create a new composition.

        Parameters
        ----------
        composition : Composition
            Composition to endpoint to mix with.
        amount : float
            Fraction to mix between two compositions. Between 0 (don't add any
            new), 1 (don't keep any old). ``amount=0.5`` would be halfway between.

        Examples
        --------
        >>> from pycalphad import variables as v
        >>> c1 = v.Composition({}, {v.X('A'): 0.25}, 'B')
        >>> c2 = v.Composition({}, {v.X('A'): 0.75}, 'B')
        >>> c1.mix(c2, 0.5)[v.X('A')]
        0.5
        >>> c3 = v.Composition({}, {v.X('B'): 0.25}, 'A')
        >>> c1.mix(c2, 0.5)[v.X('A')]
        0.5

        """
        if self.components != composition.components:
            raise ValueError("Compositions to mix must have the same components "
                             f"(got {sorted(self.components)} and {sorted(composition.components)})")
        # Make the end_comp here with a new instance so we don't modify the original
        if issubclass(self._mode, MoleFraction):
            end = composition.to_mole_fractions().set_dependent_component(self.dependent_component)
        else:  # Assume mass fractions
            end = composition.to_mass_fractions().set_dependent_component(self.dependent_component)
        interpolated = {}
        for c in self.composition.keys():
            interpolated[c] = self[c] + (end[c] - self[c])*amount
        return Composition(self.masses, interpolated, self.dependent_component)

    def interpolate_composition(self, composition, num):
        """Return the composition of a path between two compositions with ``num`` compositions.

        Parameters
        ----------
        composition : Composition
            Composition to endpoint to interpolate between.
        num : int
            Number of samples to generate.

        Returns
        -------
        Dict[Union[MoleFraction, MassFraction], List[float]]

        Examples
        --------
        >>>
        """
        c1 = self
        c2 = composition
        pth_l = []
        for mix in np.linspace(0, 1, num):
            pth_l.append(c1.mix(c2, mix))
        # Unzip list of dict to dict of list
        comp_keys = pth_l[0].composition.keys()
        pth = {comp: [] for comp in comp_keys}
        for comp_dict in pth_l:
            for comp in comp_keys:
                pth[comp].append(comp_dict[comp])
        return pth

    def __getitem__(self, item):
        """Return a composition for an element

        Parameters
        ----------
        item : Union[MoleFraction, MassFraction]

        Returns
        -------
        float

        """
        return self.composition[item]


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
si_gas_constant = R = Float(8.3145) # ideal gas constant

CONDITIONS_REQUIRING_HESSIANS = {ChemicalPotential, PhaseFraction}
