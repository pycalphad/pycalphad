from pycalphad.property_framework import ComputableProperty, as_property
from pycalphad.property_framework.units import ureg
from typing import Tuple
from abc import ABCMeta
import weakref
from collections import defaultdict
import numpy as np

class Renderer(metaclass=ABCMeta):
    def __init__(self, wks: "pycalphad.core.workspace.Workspace"):
        self.workspace = wks
    def __enter__(self):
        # Object will be deleted when __exit__ is called
        return weakref.proxy(self)
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.workspace = None

def _property_axis_label(prop: ComputableProperty) -> str:
    propname = getattr(prop, 'display_name', None)
    if propname is not None:
        result = str(propname)
        display_units = ureg.Unit(getattr(prop, 'display_units', ''))
        if len(f'{display_units:~P}') > 0:
            result += f' [{display_units:~P}]'
        return result
    else:
        return str(prop)

class MatplotlibRenderer(Renderer):
    def __call__(self, x: ComputableProperty, *ys: Tuple[ComputableProperty], ax=None):
        if self.workspace.ndim > 1:
            raise ValueError('Dimension of calculation is greater than one')
        import matplotlib.pyplot as plt
        ax = ax if ax is not None else plt.gca()
        x = as_property(x)
        data = self.workspace.get(x, *ys, values_only=False, return_units=False)

        num_y = 0
        for y in data.keys():
            if y == x:
                continue
            num_y += 1

        if num_y > 1:
            display_groupings = defaultdict(lambda: [])
            for y in data.keys():
                if y == x:
                    continue
                display_units = ureg.Unit(getattr(y, 'display_units', None))
                display_groupings[display_units].append(y)
            if len(display_groupings) > 1:
                # TODO: Add more axes for each distinct grouping
                raise ValueError('Cannot plot distinct quantities on same plot')
            else:
                unit_to_display = list(display_groupings.keys())[0]
                ylabel = f'{unit_to_display:~P}'
        else:
            ylabel = None
            for y in data.keys():
                if y == x:
                    continue
                ylabel = _property_axis_label(y)
        
        for y in data.keys():
            if y == x:
                continue
            if np.all(np.isnan(data[y])):
                continue
            ax.plot(data[x], data[y], label=getattr(y, 'display_name', str(y)))
        ax.set_ylabel(ylabel)
        ax.set_xlabel(_property_axis_label(x))
        # Suppress legend if there is only one line
        if num_y > 1:
            ax.legend(fontsize='small')

class PandasRenderer(Renderer):
    def __call__(self, *ys: Tuple[ComputableProperty]):
        import pandas as pd
        data = self.workspace.get(*ys, values_only=False, return_units=False)
        stripped_data = {}
        for key, value in data.items():
            stripped_data[_property_axis_label(key)] = value
        return pd.DataFrame.from_dict(stripped_data)

def set_plot_renderer(klass):
    global DEFAULT_PLOT_RENDERER
    if not isinstance(klass, type):
        klass = getattr(globals(), klass, None)
    if klass is None:
        raise AttributeError(f"Unknown plot renderer: {klass}")
    DEFAULT_PLOT_RENDERER = klass


DEFAULT_PLOT_RENDERER = MatplotlibRenderer
