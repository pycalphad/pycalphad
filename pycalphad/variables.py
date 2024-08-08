#pylint: disable=C0103,R0903,W0232
"""
Classes and constants for representing thermodynamic variables.
"""

from typing import Optional
from symengine import Float, Symbol
from pycalphad.io.grammar import parse_chemical_formula
from pycalphad.property_framework.types import JanssonDerivativeDeltas
from pycalphad.core.minimizer import site_fraction_differential, state_variable_differential, \
    fixed_component_differential, chemical_potential_differential
import numpy as np
from copy import copy


class Component(object):
    """
    A chemical component, either a pure element or a linear combination of pure elements.

    Attributes
    ----------
    name : string
        Name of the component
    constituents : dict
        Dictionary of {element: quantity} where the element is a string and the quantity a float.
    """
    def __new__(cls, name, constituents=None):
        if constituents is not None:
            new_self = object.__new__(cls)
            new_self.name = name
            new_self.constituents = constituents
            return new_self
        else:
            arg = name
        # if a Component is passed in, return it
        if arg.__class__ == cls:
            return arg
        new_self = object.__new__(cls)
        if isinstance(arg, Species):
            # TODO: maybe should have some base class so Component-Species aren't coupled, but it's probably okay?
            new_self.name = arg.name
            new_self.constituents = arg.constituents
            return new_self
        if arg == '*':
            new_self = object.__new__(cls)
            new_self.name = '*'
            new_self.constituents = dict()
            return new_self
        if arg is None:
            new_self = object.__new__(cls)
            new_self.name = ''
            new_self.constituents = dict()
            return new_self

        if isinstance(arg, str):
            parse_list = parse_chemical_formula(arg.upper())
        else:
            parse_list = arg
        new_self.name = name
        parse_list = parse_list[0]
        new_self.constituents = dict(parse_list)
        return new_self

    def __getnewargs__(self):
        return self.name, self.constituents

    def __eq__(self, other):
        """Two components are the same if their names and constituents are the same."""
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

    @classmethod
    def cast_from(cls, s: str) -> "Component":
        return cls(s)

    @property
    def number_of_atoms(self):
        "Number of atoms per formula unit. Vacancies do not count as atoms."
        return sum(value for key, value in self.constituents.items() if key != 'VA')

    def __repr__(self):
        if self.name == '*':
            return str(self.__class__.__name__)+'(\'*\')'
        if self.name == '':
            return str(self.__class__.__name__)+"(None)"
        constituents = ''.join(['{}{}'.format(el, val) for el, val in sorted(self.constituents.items(), key=lambda t: t[0])])
        return str(self.__class__.__name__) + f"(\'{self.name.upper()}\', \'{constituents}\')"

    def __hash__(self):
        return hash(self.name)

    @property
    def escaped_name(self):
        "Name safe to embed in the variable name of complex arithmetic expressions."
        return str(self)

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
        if isinstance(arg, Component):
            # TODO: maybe should have some base class so Component-Species aren't coupled, but it's probably okay?
            new_self.name = arg.name
            new_self.constituents = arg.constituents
            new_self.charge = 0
            return new_self
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

    @classmethod
    def cast_from(cls, s: str) -> "Species":
        return cls(s)

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
            return str(self.__class__.__name__)+'(\'*\')'
        if self.name == '':
            return str(self.__class__.__name__)+"(None)"
        species_constituents = ''.join(
            ['{}{}'.format(el, val) for el, val in sorted(self.constituents.items(), key=lambda t: t[0])])
        if self.charge == 0:
            repr_str = "(\'{0}\', \'{1}\')"
        else:
            repr_str = "(\'{0}\', \'{1}\', charge={2})"
        return str(self.__class__.__name__)+repr_str.format(self.name.upper(), species_constituents, self.charge)

    def __hash__(self):
        return hash(self.name)


def unpack_components(components: "Sequence[Component | Species | str]", dbf: Optional["Database"] = None):
    """
    Build a set of Components from ones provided by the caller.

    The components can be given as:
    1. Component objects, which are directly used
    2. Species objects, which will be converted to Component objects
    3. str objects, which will be resolved to a Component by:

        a) matching with the name of a Species object present in the Database, or
        b) by parsing the string via the Component constructor

    Parameters
    ----------
    comps : Sequence[Component | Species | str]
        Names of components to consider in the calculation.
    dbf : Optional[Database]
        Thermodynamic database containing elements and species. If no database is
        passed, constructing Component objects via pattern matching to Species defined
        in the Database (3a) will not be possible.

    Returns
    -------
    set
        Set of Component objects

    Examples
    --------
    >>> from pycalphad import Database, variables as v
    >>> dbf = Database()
    >>> dbf.species.add(v.Species("WCL4", {"W": 1, "CL": 4}))
    >>> dbf.species.add(v.Species("H+", {"H": 1.0}, charge=1.0))
    >>> unpack_components([v.Component("NACL"), v.Species("H2O"), "SIO2", "K1CL1", "WCL4", "H+"], dbf)
    """
    desired_components = set()
    if dbf is not None:
        known_species = {sp.name: sp for sp in dbf.species}
    else:
        known_species = {}
    for c in components:
        if isinstance(c, Component):
            desired_components.add(c)
        elif isinstance(c, Species):
            desired_components.add(Component(c.name, c.constituents))
        else:
            if c in known_species:
                c = known_species[c]
                desired_components.add(Component(c.name, c.constituents))
            else:
                desired_components.add(Component(c))
    return desired_components


class StateVariable(Symbol):
    """
    State variables are symbols with built-in assumptions of being real.
    """
    implementation_units = ''
    display_units = ''

    @property
    def display_name(self):
        return self.name

    def __init__(self, name):
        super().__init__(name.upper())

    @property
    def shape(self):
        return tuple()

    @property
    def is_global_property(self):
        return (not hasattr(self, 'phase_name')) or (self.phase_name is None)

    @property
    def multiplicity(self):
        if self.phase_name is not None:
            tokens = self.phase_name.split('#')
            if len(tokens) > 1:
                return int(tokens[1])
            else:
                return 1
        else:
            return None

    @property
    def phase_name_without_suffix(self):
        if self.phase_name is not None:
            tokens = self.phase_name.split('#')
            return tokens[0]
        else:
            return None

    def filtered(self, input_compsets):
        "Return a generator of CompositionSets applicable to the current property"
        multiplicity_seen = 0

        for cs_idx, compset in enumerate(input_compsets):
            if (self.phase_name is not None) and compset.phase_record.phase_name != self.phase_name_without_suffix:
                continue
            if (compset.NP == 0) and (not compset.fixed):
                continue
            if self.phase_name is not None:
                multiplicity_seen += 1
                if self.multiplicity != multiplicity_seen:
                    continue
            yield cs_idx, compset

    def compute_property(self, compsets, cur_conds, chemical_potentials):
        if len(compsets) == 0:
            return np.nan
        state_variables = compsets[0].phase_record.state_variables
        statevar_idx = state_variables.index(self)
        return compsets[0].dof[statevar_idx]

    def jansson_derivative(self, compsets, cur_conds, chemical_potentials, deltas: JanssonDerivativeDeltas):
        "Compute Jansson derivative with self as numerator, with the given deltas"
        state_variables = compsets[0].phase_record.state_variables
        statevar_idx = state_variables.index(self)
        return deltas.delta_statevars[statevar_idx]

    def jansson_deltas(self, spec, state) -> JanssonDerivativeDeltas:
        state_variables = state.compsets[0].phase_record.state_variables
        statevar_idx = sorted(state_variables, key=str).index(self)
        delta_chemical_potentials, delta_statevars, delta_phase_amounts = \
        state_variable_differential(spec, state, statevar_idx)

        # Sundman et al, 2015, Eq. 73
        compsets_delta_sitefracs = []
        for idx, compset in enumerate(state.compsets):
            delta_sitefracs = site_fraction_differential(state.cs_states[idx], delta_chemical_potentials,
                                                         delta_statevars)
            compsets_delta_sitefracs.append(delta_sitefracs)
        return JanssonDerivativeDeltas(delta_chemical_potentials=delta_chemical_potentials, delta_statevars=delta_statevars,
                                   delta_phase_amounts=delta_phase_amounts, delta_sitefracs=compsets_delta_sitefracs,
                                   delta_parameters=None)

    def __reduce__(self):
        return self.__class__, (self.name,)

    def __eq__(self, other):
        """Two species are the same if their names are the same."""
        if isinstance(other, self.__class__):
            return self.name == other.name
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.__class__, self.name))

    def __getitem__(self, new_units: str) -> "StateVariable":
        "Get StateVariable with different display units"
        newobj = copy(self)
        newobj.display_units = new_units
        return newobj

class SiteFraction(StateVariable):
    """
    Site fractions are symbols with built-in assumptions of being real
    and nonnegative. The constructor handles formatting of the name.
    """
    implementation_units = 'fraction'
    display_units = 'fraction'
    def __init__(self, phase_name, subl_index, species): #pylint: disable=W0221
        varname = phase_name + str(subl_index) + Species(species).escaped_name
        #pylint: disable=E1121
        super().__init__(varname)
        self.phase_name = phase_name.upper()
        if subl_index == '*':
            self.sublattice_index = '*'
        else:
            self.sublattice_index = int(subl_index)
        self.species = Species(species)
        if '#' in phase_name:
            self._self_without_suffix = self.__class__(self.phase_name_without_suffix, subl_index, species)
        else:
            self._self_without_suffix = self

    def compute_property(self, compsets, cur_conds, chemical_potentials):
        result = np.atleast_1d(np.full(self.shape, fill_value=np.nan))
        state_variables = None
        for _, compset in self.filtered(compsets):
            # we'll only hit this code path if this phase is present in the list of compsets
            if state_variables is None:
                state_variables = compsets[0].phase_record.state_variables
            if np.all(np.isnan(result)):
                result[0] = 0.0
            site_fractions = compset.phase_record.variables
            sitefrac_idx = site_fractions.index(self._self_without_suffix)
            result[0] += compset.dof[len(state_variables)+sitefrac_idx]
        return result

    def __reduce__(self):
        return self.__class__, (self.phase_name, self.sublattice_index, self.species)

    def __eq__(self, other):
        """Two species are the same if their names and constituents are the same."""
        if isinstance(other, self.__class__):
            return self.name == other.name
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.phase_name, self.sublattice_index, self.species))

    def expand_wildcard(self, phase_names=None, components=None, sublattice_indices=None):
        if phase_names is not None:
            return [self.__class__(phase_name, self.sublattice_index, self.species) for phase_name in phase_names]
        elif components is not None:
            return [self.__class__(self.phase_name, self.sublattice_index, comp) for comp in components]
        elif sublattice_indices is not None:
            return [self.__class__(self.phase_name, idx, self.species) for idx in sublattice_indices]
        else:
            raise ValueError('All arguments are None')

    def _latex(self, printer=None):
        "LaTeX representation."
        return r'y^{\mathrm{'+self.phase_name.replace('_', '-') + \
            '}}_{'+str(self.sublattice_index)+r',\mathrm{'+self.species.escaped_name+'}}'

    def __str__(self):
        "String representation."
        if self.sublattice_index == '*':
            return 'Y(%s,%s,%s)' % \
                (self.phase_name, self.sublattice_index, self.species.escaped_name)
        else:
            return 'Y(%s,%d,%s)' % \
                (self.phase_name, self.sublattice_index, self.species.escaped_name)

class PhaseFraction(StateVariable):
    """
    Phase fractions are symbols with built-in assumptions of being real
    and nonnegative. The constructor handles formatting of the name.
    """
    implementation_units = 'fraction'
    display_units = 'fraction'
    def __init__(self, phase_name): #pylint: disable=W0221
        varname = 'NP_' + str(phase_name)
        super().__init__(varname)
        self.phase_name = phase_name.upper()

    def compute_property(self, compsets, cur_conds, chemical_potentials):
        result = np.atleast_1d(np.full(self.shape, fill_value=np.nan))
        for _, compset in self.filtered(compsets):
            if np.all(np.isnan(result[0])):
                result[0] = 0.0
            result[0] += compset.NP
        return result

    def jansson_derivative(self, compsets, cur_conds, chemical_potentials, deltas: JanssonDerivativeDeltas):
        "Compute Jansson derivative with self as numerator, with the given deltas"
        jansson_derivative = np.nan
        for idx, _ in self.filtered(compsets):
            if np.isnan(jansson_derivative):
                jansson_derivative = 0.0
            jansson_derivative += deltas.delta_phase_amounts[idx]
        return jansson_derivative

    def expand_wildcard(self, phase_names):
        return [self.__class__(phase_name) for phase_name in phase_names]

    def _latex(self, printer=None):
        "LaTeX representation."
        #pylint: disable=E1101
        return 'f^{'+self.phase_name.replace('_', '-')

class MoleFraction(StateVariable):
    """
    MoleFractions are symbols with built-in assumptions of being real
    and nonnegative.
    """
    implementation_units = 'fraction'
    display_units = 'fraction'
    def __init__(self, *args): #pylint: disable=W0221
        varname = None
        phase_name = None
        species = None
        if len(args) == 1:
            # this is an overall composition variable
            species = Component(args[0])
            varname = 'X_' + species.escaped_name.upper()
        elif len(args) == 2:
            # this is a phase-specific composition variable
            phase_name = args[0].upper()
            species = Component(args[1])
            varname = 'X_' + phase_name + '_' + species.escaped_name.upper()
        else:
            # not defined
            raise ValueError('MoleFraction not defined for args: '+args)

        #pylint: disable=E1121
        super().__init__(varname)
        self.phase_name = phase_name
        self.species = species

    def expand_wildcard(self, phase_names=None, components=None):
        if phase_names is not None:
            return [self.__class__(phase_name, self.species) for phase_name in phase_names]
        elif components is not None:
            if self.phase_name is None:
                return [self.__class__(comp) for comp in components]
            else:
                return [self.__class__(self.phase_name, comp) for comp in components]
        else:
            raise ValueError('Both phase_names and components are None')

    def compute_property(self, compsets, cur_conds, chemical_potentials):
        result = np.atleast_1d(np.zeros(self.shape))
        result[:] = np.nan
        for _, compset in self.filtered(compsets):
            el_idx = compset.phase_record.nonvacant_elements.index(str(self.species))
            if np.isnan(result[0]):
                result[0] = 0
            if self.phase_name is None:
                result[0] += compset.NP * compset.X[el_idx]
            else:
                result[0] += compset.X[el_idx]
        return result

    def compute_per_phase_property(self, compset, cur_conds):
        if self.phase_name is not None:
            tokens = self.phase_name.split('#')
            phase_name = tokens[0]
            if (compset.phase_record.phase_name != phase_name):
                return np.nan
        el_idx = compset.phase_record.nonvacant_elements.index(str(self.species))
        return compset.X[el_idx]

    def compute_property_gradient(self, compsets, cur_conds, chemical_potentials):
        "Compute partial derivatives of property with respect to degrees of freedom of given CompositionSets"
        result = [np.zeros(compset.dof.shape[0]) for compset in compsets]
        num_components = len(compsets[0].phase_record.nonvacant_elements)
        for cs_idx, compset in self.filtered(compsets):
            masses = np.zeros((num_components, 1))
            mass_jac = np.zeros((num_components, compset.dof.shape[0]))
            for comp_idx in range(num_components):
                compset.phase_record.formulamole_obj(masses[comp_idx, :], compset.dof, comp_idx)
                compset.phase_record.formulamole_grad(mass_jac[comp_idx, :], compset.dof, comp_idx)
            el_idx = compset.phase_record.nonvacant_elements.index(str(self.species))
            result[cs_idx][:] = (mass_jac[el_idx] * masses.sum() - masses[el_idx,0] * mass_jac.sum(axis=0)) \
                / (masses.sum(axis=0)**2)
        return result

    def jansson_derivative(self, compsets, cur_conds, chemical_potentials, deltas: JanssonDerivativeDeltas):
        "Compute Jansson derivative with self as numerator, with the given deltas"
        state_variables = compsets[0].phase_record.state_variables
        grad_values = self.compute_property_gradient(compsets, cur_conds, chemical_potentials)

        # Sundman et al, 2015, Eq. 73
        jansson_derivative = np.nan
        for idx, compset in enumerate(compsets):
            if compset.NP == 0 and not (compset.fixed):
                continue
            func_value = self.compute_per_phase_property(compset, cur_conds)
            if np.isnan(func_value):
                continue
            if np.isnan(jansson_derivative):
                jansson_derivative = 0.0
            grad_value = grad_values[idx]
            delta_sitefracs = deltas.delta_sitefracs[idx]

            if self.phase_name is None:
                jansson_derivative += deltas.delta_phase_amounts[idx] * func_value
                jansson_derivative += compset.NP * np.dot(deltas.delta_statevars, grad_value[:len(state_variables)])
                jansson_derivative += compset.NP * np.dot(delta_sitefracs, grad_value[len(state_variables):])
            else:
                jansson_derivative += np.dot(deltas.delta_statevars, grad_value[:len(state_variables)])
                jansson_derivative += np.dot(delta_sitefracs, grad_value[len(state_variables):])
        return jansson_derivative

    def jansson_deltas(self, spec, state) -> JanssonDerivativeDeltas:
        component_idx = state.compsets[0].phase_record.nonvacant_elements.index(str(self.species))
        delta_chemical_potentials, delta_statevars, delta_phase_amounts = \
        fixed_component_differential(spec, state, component_idx)

        # Sundman et al, 2015, Eq. 73
        compsets_delta_sitefracs = []
        for idx, compset in enumerate(state.compsets):
            delta_sitefracs = site_fraction_differential(state.cs_states[idx], delta_chemical_potentials,
                                                         delta_statevars)
            compsets_delta_sitefracs.append(delta_sitefracs)
        return JanssonDerivativeDeltas(delta_chemical_potentials=delta_chemical_potentials, delta_statevars=delta_statevars,
                                   delta_phase_amounts=delta_phase_amounts, delta_sitefracs=compsets_delta_sitefracs,
                                   delta_parameters=None)

    def __reduce__(self):
        if self.phase_name is None:
            return self.__class__, (self.species,)
        else:
            return self.__class__, (self.phase_name, self.species,)

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
    implementation_units = 'fraction'
    display_units = 'fraction'
    def __init__(self, *args):  # pylint: disable=W0221
        varname = None
        phase_name = None
        species = None
        if len(args) == 1:
            # this is an overall composition variable
            species = Component(args[0])
            varname = 'W_' + species.escaped_name.upper()
        elif len(args) == 2:
            # this is a phase-specific composition variable
            phase_name = args[0].upper()
            species = Component(args[1])
            varname = 'W_' + phase_name + '_' + species.escaped_name.upper()
        else:
            # not defined
            raise ValueError('Weight fraction not defined for args: '+args)

        # pylint: disable=E1121
        super().__init__(varname)
        self.phase_name = phase_name
        self.species = species

    def compute_property(self, compsets, cur_conds, chemical_potentials):
        result = np.atleast_1d(np.zeros(self.shape))
        result[:] = np.nan
        normalizer = 0.
        for _, compset in self.filtered(compsets):
            el_idx = compset.phase_record.nonvacant_elements.index(str(self.species))
            if np.isnan(result[0]):
                result[0] = 0
            if self.phase_name is None:
                result[0] += compset.NP * compset.phase_record.molar_masses[el_idx] * compset.X[el_idx]
                for n_idx in range(len(compset.phase_record.nonvacant_elements)):
                    normalizer += compset.NP * compset.phase_record.molar_masses[n_idx] * compset.X[n_idx]
            else:
                result[0] += compset.phase_record.molar_masses[el_idx] * compset.X[el_idx]
                for n_idx in range(len(compset.phase_record.nonvacant_elements)):
                    normalizer += compset.phase_record.molar_masses[n_idx] * compset.X[n_idx]
        return result / normalizer

    def __reduce__(self):
        if self.phase_name is None:
            return self.__class__, (self.species,)
        else:
            return self.__class__, (self.phase_name, self.species,)

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

    dependent_species : Union[Component, str]
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
    dependent_species = Component(dependent_species)
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

    dependent_species : Union[Component, str]
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
    dependent_species = Component(dependent_species)
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
    implementation_units = 'J / mol'
    display_units = 'J / mol'
    display_name = property(lambda self: f'Chemical Potential {self.species}')

    def __init__(self, species):
        species = Component(species)
        varname = 'MU_' + species.escaped_name.upper()
        super().__init__(varname)
        self.species = species

    def expand_wildcard(self, components):
        return [self.__class__(comp) for comp in components]

    def compute_property(self, compsets, cur_conds, chemical_potentials):
        if len(compsets) == 0:
            return np.atleast_1d(np.full(self.shape, fill_value=np.nan))
        phase_record = compsets[0].phase_record
        el_indices = [(phase_record.nonvacant_elements.index(k), v)
                       for k, v in self.species.constituents.items()]
        result = np.atleast_1d(np.zeros(self.shape))
        for el_idx, multiplicity in el_indices:
            result[0] += multiplicity * chemical_potentials[el_idx]
        return result

    def jansson_derivative(self, compsets, cur_conds, chemical_potentials, deltas: JanssonDerivativeDeltas):
        "Compute Jansson derivative with self as numerator, with the given deltas"
        if len(compsets) == 0:
            return np.full(self.shape, np.nan)
        phase_record = compsets[0].phase_record
        el_indices = [(phase_record.nonvacant_elements.index(k), v)
                       for k, v in self.species.constituents.items()]
        result = np.zeros(self.shape)
        for el_idx, multiplicity in el_indices:
            result += multiplicity * deltas.delta_chemical_potentials[el_idx]
        return result

    def jansson_deltas(self, spec, state) -> JanssonDerivativeDeltas:
        component_idx = state.compsets[0].phase_record.nonvacant_elements.index(str(self.species))
        delta_chemical_potentials, delta_statevars, delta_phase_amounts = \
        chemical_potential_differential(spec, state, component_idx)

        # Sundman et al, 2015, Eq. 73
        compsets_delta_sitefracs = []
        for idx, compset in enumerate(state.compsets):
            delta_sitefracs = site_fraction_differential(state.cs_states[idx], delta_chemical_potentials,
                                                         delta_statevars)
            compsets_delta_sitefracs.append(delta_sitefracs)
        return JanssonDerivativeDeltas(delta_chemical_potentials=delta_chemical_potentials, delta_statevars=delta_statevars,
                                   delta_phase_amounts=delta_phase_amounts, delta_sitefracs=compsets_delta_sitefracs,
                                   delta_parameters=None)

    def _latex(self, printer=None):
        "LaTeX representation."
        return r'\mu_{'+self.species.escaped_name+'}'

    def __str__(self):
        "String representation."
        return 'MU_%s' % self.species.name


class IndependentPotential(StateVariable):
    pass


class TemperatureType(IndependentPotential):
    implementation_units = 'kelvin'
    display_units = 'kelvin'
    display_name = 'Temperature'

    def __init__(self):
        super().__init__('T')
    def __reduce__(self):
        return self.__class__, ()


class PressureType(IndependentPotential):
    implementation_units = 'pascal'
    display_units = 'pascal'
    display_name = 'Pressure'

    def __init__(self):
        super().__init__('P')
    def __reduce__(self):
        return self.__class__, ()

class SystemMolesType(StateVariable):
    implementation_units = 'mol'
    display_units = 'mol'
    display_name = 'No. Moles'
    def __init__(self):
        super().__init__('N')
    def __reduce__(self):
        return self.__class__, ()

temperature = T = TemperatureType()
pressure = P = PressureType()
moles = N = SystemMolesType()
site_fraction = Y = SiteFraction
X = MoleFraction
W = MassFraction
MU = ChemicalPotential
NP = PhaseFraction
si_gas_constant = R = Float(8.3145) # ideal gas constant

CONDITIONS_REQUIRING_HESSIANS = {ChemicalPotential, PhaseFraction}
