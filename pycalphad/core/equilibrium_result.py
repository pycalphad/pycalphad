"""Defines a class for internally representing arrays used in equilibrium calculations"""

import numpy as np
from xarray import Dataset


class EquilibriumResult:
    """

    Attributes
    ----------
    data_vars : dict
    coords : dict
    attrs : dict

    """
    def __init__(self, data_vars=None, coords=None, attrs=None):
        """

        Parameters
        ----------
        data_vars :
            Dictionary of {Variable: (Dimensions, Values)}
        coords :
            Mapping of {Dimension: Values}
        attrs :

        Returns
        -------
        EquilibriumResult

        Notes
        -----
        Takes on same format as xarray.Dataset initializer

        """
        self.data_vars = data_vars or dict()
        self.coords = coords or dict()
        self.attrs = attrs or dict()
        for var, (coord, values) in data_vars.items():
            setattr(self, var, values)
        for coord, values in coords.items():
            setattr(self, coord, values)

    def get_dataset(self):
        """Build an xarray Dataset"""
        return Dataset(self.data_vars, self.coords, self.attrs)

    def __getitem__(self, item):
        try:
            return getattr(self, item)
        except:
            raise KeyError("`{}` is not a variable or coordinate".format(item))

    def remove(self, item):
        self.data_vars.pop(item)
        delattr(self, item)

    def merge(self, other, inplace=False, compat='no_conflicts'):
        if compat != 'equals':
            raise ValueError("Only `compat='equals'` is supported. Passed `compat={}`".format(compat))
        if inplace:
            for var, coords_vals in other.data_vars.items():
                if var not in self.data_vars.keys():
                    self.data_vars[var] = coords_vals
                    setattr(self, var, coords_vals[1])
                else:
                    if np.all(self.data_vars[var] != coords_vals):
                        raise ValueError("Cannot merge EquilibriumResults because data variable `{}` is not equal to preexisting variable".format(var))
            for coord, values in other.coords.items():
                if coord not in self.coords.keys():
                    self.coords[var] = values
                    setattr(self, coord, values)
                else:
                    if np.all(self.coords[coord] != values):
                        raise ValueError("Cannot merge EquilibriumResults because coordinate `{}` is not equal to preexisting coordinate".format(coord))
        else:
            raise NotImplementedError("Copy merges not implemented")
        return self

    def add_variable(self, var, coord, value):
        self.data_vars[var] = (coord, value)
        setattr(self, var, value)
