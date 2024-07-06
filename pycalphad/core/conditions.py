import numpy as np
from pycalphad.core.errors import ConditionError
from pycalphad.property_framework import as_property, as_quantity
from pycalphad.property_framework.units import Q_
import pycalphad.variables as v
from collections.abc import Iterable
from typing import List, NamedTuple, Optional, TYPE_CHECKING
import warnings
import os

if TYPE_CHECKING:
    from pycalphad.core.workspace import Workspace
    from pycalphad.property_framework import ComputableProperty

class ConditionsEntry(NamedTuple):
    prop: "ComputableProperty"
    value: "Q_"

_default = object()

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
            return np.arange(tup[0], tup[1], dtype=np.float64)
        elif len(tup) == 3:
            return np.arange(tup[0], tup[1], tup[2], dtype=np.float64)
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

    def __init__(self, wks: Optional["Workspace"]):
        self._wks = wks
        self._conds = []
        # Default to N=1
        self.__setitem__(v.N, Q_(np.atleast_1d(1.0), 'mol'))

    @classmethod
    def from_dict(cls, d):
        if isinstance(d, Conditions):
            return d
        obj = cls(wks=None)
        obj.update(d)
        return obj
    
    def _find_matching_index(self, prop: "ComputableProperty"):
        for idx, (key, _) in enumerate(self._conds):
            # TODO: Use more sophisticated matching
            if str(prop) == str(key):
                return idx
        return None

    @classmethod
    def cast_from(cls, key) -> "Conditions":
        return cls.from_dict(key)
    
    def __getitem__(self, item):
        key = as_property(item)
        idx = self._find_matching_index(key)
        if idx is None:
            raise IndexError(f"{item} is not a condition")
        entry = self._conds[idx]
        # Important to use the _key_ display_units, and not the entry.prop
        # This is because v.T['K'] == v.T['degC'], so conditions can be
        # stored and queried with distinct units
        return entry.value.to(key.display_units).magnitude

    def get(self, item, default=_default):
        try:
            return self.__getitem__(item)
        except IndexError:
            if default is not _default:
                return default
            else:
                raise

    def __delitem__(self, item):
        idx = self._find_matching_index(as_property(item))
        if idx is None:
            raise IndexError(f"{item} is not a condition")
        del self._conds[idx]
    
    def __setitem__(self, item, value):
        prop = as_property(item)
        if isinstance(prop, (v.MoleFraction, v.MassFraction, v.SiteFraction)):
            vals = unpack_condition(value)
            if isinstance(vals, Q_):
                vals = vals.to(prop.implementation_units).magnitude
            # "Zero" composition is a common pattern. Do not warn for that case.
            if np.any(np.logical_and(np.asarray(vals) < self.minimum_composition, np.asarray(vals) > 0)):
                warnings.warn(
                    f"Some specified compositions are below the minimum allowed composition of {self.minimum_composition}.")
            value = [min(max(val, self.minimum_composition), 1-self.minimum_composition) for val in vals]
        else:
            value = unpack_condition(value)
        
        value = as_quantity(prop, value).to(prop.implementation_units)

        if isinstance(prop, (v.MoleFraction, v.MassFraction, v.ChemicalPotential)) and prop.species not in self._wks.components:
            raise ConditionError('{} refers to non-existent component'.format(prop))

        if isinstance(prop, v.SiteFraction) and prop not in self._wks.phase_record_factory[prop.phase_name].variables:
            raise ConditionError('{} refers to non-existent constituent'.format(prop))

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

    def str_keys(self):
        for key, _ in self._conds:
            yield str(key)

    def values(self, units='display_units'):
        for key, value in self._conds:
            yield value.to(getattr(key, units, '')).magnitude

    def update(self, d):
        for key, value in d.items():
            self.__setitem__(key, value)

    def items(self, units='display_units'):
        for key, value in self._conds:
            yield key, value.to(getattr(key, units, '')).magnitude

    def __len__(self):
        return len(self._conds)

    def __iter__(self):
        yield from self.keys()

    def __str__(self):
        result = ""
        with np.printoptions(threshold=10):
            for key, value in self._conds:
                result += str(key) + "=" + str(value) + os.linesep
        return result

    __repr__ = __str__