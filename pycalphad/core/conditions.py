import numpy as np
from pycalphad.core.errors import ConditionError
from pycalphad.property_framework import as_property, as_quantity
from pycalphad.property_framework.units import Q_
from pycalphad.core.minimizer import SystemSpecification
import pycalphad.variables as v
from collections.abc import Iterable
from typing import List, NamedTuple, TYPE_CHECKING
import warnings

if TYPE_CHECKING:
    from pycalphad.core.workspace import Workspace
    from pycalphad.property_framework import ComputableProperty

class ConditionsEntry(NamedTuple):
    prop: "ComputableProperty"
    value: "Q_"

def unpack_condition(tup):
    """
    Convert a condition to a list of values.

    Notes
    -----
    Rules for keys of conditions dicts:
    (1) If it's numeric, treat as a point value
    (2) If it's a tuple with one element, treat as a point value
    (3) If it's a tuple with two elements, treat as lower/upper limits and guess a step size.
    (4) If it's a tuple with three elements, treat as lower/upper/step
    (5) If it's a list, ndarray or other non-tuple ordered iterable, use those values directly.

    """
    if isinstance(tup, tuple):
        if len(tup) == 1:
            return [float(tup[0])]
        elif len(tup) == 2:
            return np.arange(tup[0], tup[1], dtype=np.float_)
        elif len(tup) == 3:
            return np.arange(tup[0], tup[1], tup[2], dtype=np.float_)
        else:
            raise ValueError('Condition tuple is length {}'.format(len(tup)))
    elif isinstance(tup, Q_):
        return tup
    elif isinstance(tup, Iterable) and np.ndim(tup) != 0:
        return [float(x) for x in tup]
    else:
        return [float(tup)]

class Conditions:
    _wks: "Workspace"
    _conds: List[ConditionsEntry]

    minimum_composition: float = 1e-10

    def __init__(self, wks: "Workspace"):
        self._wks = wks
        self._conds = []
        # Default to N=1
        self.__setitem__(v.N, Q_(np.atleast_1d(1.0), 'mol'))
    
    def _find_matching_index(self, prop: "ComputableProperty"):
        for idx, (key, _) in enumerate(self._conds):
            # TODO: Use more sophisticated matching
            if str(prop) == str(key):
                return idx
        return None

    @classmethod
    def cast_from(cls, key) -> "Conditions":
        return key
    
    def __getitem__(self, item):
        key = as_property(item)
        idx = self._find_matching_index(key)
        if idx is None:
            raise IndexError(f"{item} is not a condition")
        entry = self._conds[idx]
        # Important to use the _key_ display_units, and not the entry.prop
        # This is because v.T['K'] == v.T['degC'], so conditions can be
        # stored and queried with distinct units
        return entry.value.to(key.display_units)

    get = __getitem__

    def __delitem__(self, item):
        idx = self._find_matching_index(as_property(item))
        if idx is None:
            raise IndexError(f"{item} is not a condition")
        del self._conds[idx]
    
    def __setitem__(self, item, value):
        prop = as_property(item)
        if isinstance(prop, v.MoleFraction):
            vals = unpack_condition(value)
            # "Zero" composition is a common pattern. Do not warn for that case.
            if np.any(np.logical_and(np.asarray(vals) < self.minimum_composition, np.asarray(vals) > 0)):
                warnings.warn(
                    f"Some specified compositions are below the minimum allowed composition of {self.minimum_composition}.")
            value = [min(max(val, self.minimum_composition), 1-self.minimum_composition) for val in vals]
        else:
            value = unpack_condition(value)
        
        value = as_quantity(prop, value).to(prop.implementation_units)

        if isinstance(prop, (v.MoleFraction, v.ChemicalPotential)) and prop.species not in self._wks.components:
            raise ConditionError('{} refers to non-existent component'.format(prop))
        
        if (prop == v.N) and np.any(value != Q_(1.0, 'mol')):
            raise ConditionError('N!=1 is not yet supported, got N={}'.format(value))
        
        entry = ConditionsEntry(prop=prop, value=value)

        idx = self._find_matching_index(prop)

        if idx is None:
            # Condition is not yet specified
            # TODO: Check number of degrees of freedom
            self._conds.append(entry)
        else:
            self._conds[idx] = entry
        
        self._conds = sorted(self._conds, key=lambda k: str(k[0]))

    def keys(self):
        for key, _ in self._conds:
            yield key

    def values(self):
        for _, value in self._conds:
            yield value

    def items(self):
        for key, value in self._conds:
            yield key, value

    def str_items(self):
        """
        Return key-value pairs suitable for the minimizer.
        This returns values as implementation units, magnitude only.
        """
        for key, value in self._conds:
            yield (str(key), value.to(key.implementation_units).magnitude)

    def __len__(self):
        return len(self._conds)

    def __iter__(self):
        yield from self.keys()

    def get_system_spec(self, composition_sets):
        """
        Create a SystemSpecification object for the specified compositions.

        Parameters
        ----------
        composition_sets : List[pycalphad.core.composition_set.CompositionSet]
            List of CompositionSet objects in the starting point. Modified in place.

        Returns
        -------
        SystemSpecification

        """
        compsets = composition_sets
        state_variables = compsets[0].phase_record.state_variables
        nonvacant_elements = compsets[0].phase_record.nonvacant_elements
        num_statevars = len(state_variables)
        num_components = len(nonvacant_elements)
        chemical_potentials = np.zeros(num_components)
        prescribed_mole_fraction_coefficients = []
        prescribed_mole_fraction_rhs = []
        for cond, value in conditions.items():
            if str(cond).startswith('X_'):
                el = str(cond)[2:]
                el_idx = list(nonvacant_elements).index(el)
                prescribed_mole_fraction_rhs.append(float(value))
                coefs = np.zeros(num_components)
                coefs[el_idx] = 1.0
                prescribed_mole_fraction_coefficients.append(coefs)
        prescribed_mole_fraction_coefficients = np.atleast_2d(prescribed_mole_fraction_coefficients)
        prescribed_mole_fraction_rhs = np.array(prescribed_mole_fraction_rhs)
        prescribed_system_amount = conditions.get('N', 1.0)
        fixed_chemical_potential_indices = np.array([nonvacant_elements.index(key[3:]) for key in conditions.keys() if key.startswith('MU_')], dtype=np.int32)
        free_chemical_potential_indices = np.array(sorted(set(range(num_components)) - set(fixed_chemical_potential_indices)), dtype=np.int32)
        for fixed_chempot_index in fixed_chemical_potential_indices:
            el = nonvacant_elements[fixed_chempot_index]
            chemical_potentials[fixed_chempot_index] = conditions.get('MU_' + str(el))
        fixed_statevar_indices = []
        for statevar_idx, statevar in enumerate(state_variables):
            if str(statevar) in [str(k) for k in conditions.keys()]:
                fixed_statevar_indices.append(statevar_idx)
        free_statevar_indices = np.array(sorted(set(range(num_statevars)) - set(fixed_statevar_indices)), dtype=np.int32)
        fixed_statevar_indices = np.array(fixed_statevar_indices, dtype=np.int32)
        fixed_stable_compset_indices = np.array([i for i, compset in enumerate(compsets) if compset.fixed], dtype=np.int32)
        spec = SystemSpecification(num_statevars, num_components, prescribed_system_amount,
                                   chemical_potentials, prescribed_mole_fraction_coefficients,
                                   prescribed_mole_fraction_rhs,
                                   free_chemical_potential_indices, free_statevar_indices,
                                   fixed_chemical_potential_indices, fixed_statevar_indices,
                                   fixed_stable_compset_indices)
        return spec