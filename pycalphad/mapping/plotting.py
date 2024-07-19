import itertools
from typing import Union

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from pycalphad import variables as v
from pycalphad.plot.utils import phase_legend
from pycalphad.plot import triangular  # register triangular projection

from pycalphad.mapping.primitives import _get_phase_specific_variable
from pycalphad.mapping.strategy.step_strategy import StepStrategy
from pycalphad.mapping.strategy.binary_strategy import BinaryStrategy
from pycalphad.mapping.strategy.ternary_strategy import TernaryStrategy
from pycalphad.mapping.strategy.isopleth_strategy import IsoplethStrategy
import pycalphad.mapping.utils as map_utils

def get_label(var: v.StateVariable):
    # If user just passes v.NP rather than an instance of v.NP, then label is just NP
    if var == v.NP:
        return 'Phase Fraction'
    elif isinstance(var, v.X):
        if var.phase_name is None:
            return 'X({})'.format(var.species.name.capitalize())
        else:
            return 'X({}, {})'.format(var.phase_name, var.species.name.capitalize())
    elif isinstance(var, v.W):
        if var.phase_name is None:
            return 'W({})'.format(var.species.name.capitalize())
        else:
            return 'W({}, {})'.format(var.phase_name, var.species.name.capitalize())
    elif isinstance(var, v.MU):
        return 'MU({})'.format(var.species.name.capitalize())
    # Otherwise, we can just use the display name
    else:
        return var.display_name

def plot_step(strategy: StepStrategy, x: v.StateVariable = None, y: v.StateVariable = None, ax = None, legend_generator = phase_legend, set_nan_to_zero = True, *args, **kwargs):
    """
    Plots step map using matplotlib

    Parameters
    ----------
    strategy : StepStrategy
    x : v.StateVariable
    y : v.StateVariable
    ax : matplotlib axes (optional)
        A new axis object will be made if not supplied
    legend_generator : function that creates legend handles and colors for list of phases

    Returns
    -------
    Matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots()

    # If x is None, then use axis variable and state that x is global
    # (this is useful for v.X where we can distinguish v.X(sp,ph) vs. v.X(sp)
    x_is_global = False
    if x is None:
        x = strategy.axis_vars[0]
    if x == strategy.axis_vars[0]:
        x_is_global = True

    # If y is None, then use phase fractions
    if y is None:
        y = v.NP

    step_data = strategy.get_data(x, y, x_is_global, set_nan_to_zero=set_nan_to_zero)
    data = step_data['data']
    xlim = step_data['xlim']
    ylim = step_data['ylim']

    handles, colors = legend_generator(sorted(data.keys()))

    for p in data:
        x_data = data[p]['x']
        y_data = data[p]['y']
        ax.plot(x_data, y_data, color=colors[p], lw=1, solid_capstyle="butt")

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # Add legend
    ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1, 0.5))
    plot_title = '-'.join([component.title() for component in sorted(strategy.components) if component != 'VA'])
    ax.set_title(plot_title)
    ax.set_xlabel(get_label(x))
    ax.set_ylabel(get_label(y))

    return ax

def plot_invariants(ax, strategy: Union[BinaryStrategy, TernaryStrategy], x: v.StateVariable, y: v.StateVariable, phase_colors, label_end_points: bool = False, tie_triangle_color = (1, 0, 0, 1)):
    """
    Plots node data from BinaryStrategy or TernaryStrategy onto matplotlib axis

    Parameters
    ----------
    ax : matplotlib axis
    strategy : BinaryStrategy or TernaryStrategy
    x : v.StateVariable
    y : v.StateVariable
    phase_colors : dict[str, color]
        Color to plot end points if set
    label_end_points : bool
    tie_triangle_color : color
        Color to plot node
    """
    invariant_data = strategy.get_invariant_data(x, y)
    for data in invariant_data:
        x_data, y_data, phases = data['x'], data['y'], data['phases']
        ax.plot(x_data + [x_data[0]], y_data + [y_data[0]], color=tie_triangle_color, zorder=2.5, lw=1, solid_capstyle="butt")

        # If labeling end points, add a scatter point on each phase coordinate in the node
        if label_end_points:
            for xp, yp, p in zip(x_data, y_data, phases):
                ax.scatter([xp], [yp], color=phase_colors[p], s=8, zorder=3)

def plot_tielines(ax, strategy: Union[BinaryStrategy, TernaryStrategy], x: v.StateVariable, y: v.StateVariable, phase_colors, tielines = 1, tieline_color=(0, 1, 0, 1)):
    """
    Plots tieline data from BinaryStrategy or TernaryStrategy onto matplotlib axis

    Parameters
    ----------
    ax : matplotlib axis
    strategy : BinaryStrategy or TernaryStrategy
    x : v.StateVariable
    y : v.StateVariable
    phase_colors : dict[str, color]
        Color to plot end points if set
    tielines : int or False
        int - plots every n tielines
        False - only plots phase boundaries
    tieline_color : color
    """
    zpf_data = strategy.get_tieline_data(x, y)
    for data in zpf_data:
        for p in data:
            x_data, y_data = data[p]['x'], data[p]['y']
            if not all((y_data == 0) | (y_data == np.nan)):
                ax.plot(x_data, y_data, color=phase_colors[p], lw=1, solid_capstyle="butt")

        if tielines:
            x_list = [data[p]['x'] for p in data]
            y_list = [data[p]['y'] for p in data]
            tieline_collection = LineCollection(np.asarray([x_list, y_list]).T[::tielines, ...], zorder=1, linewidths=0.5, capstyle="butt", colors=[tieline_color for _ in range(len(x_list[0]))])
            ax.add_collection(tieline_collection)

def plot_binary(strategy: BinaryStrategy, x: v.StateVariable = None, y: v.StateVariable = None, ax = None, tielines = 1, label_nodes = False, legend_generator = phase_legend, tieline_color=(0, 1, 0, 1), tie_triangle_color=(1, 0, 0, 1), *args, **kwargs):
    """
    Plots binary map using matplotlib

    Parameters
    ----------
    strategy : BinaryStrategy or TernaryStrategy
    x : v.StateVariable
    y : v.StateVariable
    ax : matplotlib axes (optional)
        A new axis object will be made if not supplied
    tielines : int or False (optional)
        Default = 1
        int - plots every n tieline
        False - only plots phase boundaries
    label_node : bool (optional)
        Default = False
        Plots points on nodes for each phase if true
    legend_generator : function that creates legend handles and colors for list of phases
    tieline_color : color
    tie_triangle_color : color

    Returns
    -------
    Matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots()

    # If variables are not given, then plot composition vs. temperature
    sorted_axis_var = map_utils._sort_axis_by_state_vars(strategy.axis_vars)
    if x is None:
        x = sorted_axis_var[1]
    if y is None:
        y = sorted_axis_var[0]

    phases = sorted(strategy.get_all_phases())
    handles, colors = legend_generator(phases)

    plot_tielines(ax, strategy, x, y, phase_colors=colors, tielines=tielines, tieline_color=tieline_color)
    plot_invariants(ax, strategy, x, y, phase_colors=colors, label_end_points=label_nodes, tie_triangle_color=tie_triangle_color)

    # Adjusts axis limits
    # 1. Autoscale axis
    # 2. If x or y is a strategy axis variable
    #    Set lower limit to max of strategy limits to rescaled limits
    #    Set upper limit to min of strategy limits or rescaled limits
    # 3. If x or y is not a strategy axis variable, then use the rescaled limits
    ax.autoscale()
    if x in strategy.axis_vars:
        xlim = list(ax.get_xlim())
        xlim[0] = np.amax((np.amin(strategy.axis_lims[x]), xlim[0]))
        xlim[1] = np.amin((np.amax(strategy.axis_lims[x]), xlim[1]))
        ax.set_xlim(xlim)
    if y in strategy.axis_vars:
        ylim = list(ax.get_ylim())
        ylim[0] = np.amax((np.amin(strategy.axis_lims[y]), ylim[0]))
        ylim[1] = np.amin((np.amax(strategy.axis_lims[y]), ylim[1]))
        ax.set_ylim(ylim)

    ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1, 0.5))
    plot_title = '-'.join([component.title() for component in sorted(strategy.components) if component != 'VA'])
    ax.set_title(plot_title)
    ax.set_xlabel(get_label(x))
    ax.set_ylabel(get_label(y))

    return ax

def plot_ternary(strategy: TernaryStrategy, x: v.StateVariable = None, y: v.StateVariable = None, ax = None, tielines = 1, label_nodes = False, legend_generator = phase_legend, tieline_color=(0, 1, 0, 1), tie_triangle_color=(1, 0, 0, 1), *args, **kwargs):
    """
    Plots ternary map using matplotlib

    Pretty much the same as binary mapping but some extra stuff
    to create defualt triangular axis, limit axis limits to (0,1) and 
    set y label position if axis is triangular

    Parameters
    ----------
    strategy : Ternary strategy
    x : v.StateVariable
    y : v.StateVariable
    ax : matplotlib axes (optional)
        A new axis object will be made if not supplied
    tielines : int or False (optional)
        Default = 1
        int - plots every n tieline
        False - only plots phase boundaries
    label_node : bool (optional)
        Default = False
        Plots points on nodes for each phase if true
    legend_generator : function that creates legend handles and colors for list of phases
    tieline_color : color
    tie_triangle_color : color

    Returns
    -------
    Matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots(subplot_kw={'projection': "triangular"})

    plot_binary(strategy, x, y, ax, tielines=tielines, label_nodes=label_nodes, legend_generator=legend_generator, tieline_color=tieline_color, tie_triangle_color=tie_triangle_color, *args, **kwargs)
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])

    # Projection is stored in the default axis name, so only adjust y label if the axis is triangular
    # Note: this is assuming that triangular is the only option for making a ternary plot and that the user doesn't change the default name
    if 'triangular' in ax.name:
        ax.yaxis.label.set_rotation(60)  # rotate ylabel
        ax.yaxis.set_label_coords(x=0.12, y=0.5)  # move the label to a pleasing position

    return ax

def plot_isopleth(strategy: IsoplethStrategy, x: v.StateVariable = None, y: v.StateVariable = None, ax = None, legend_generator = phase_legend, tie_triangle_color=(1, 0, 0, 1), *args, **kwargs):
    """
    Plots isopleth map using matplotlib

    Parameters
    ----------
    strategy : IsoplethStrategy
    x : v.StateVariable
    y : v.StateVariable
    ax : matplotlib axes (optional)
        A new axis object will be made if not supplied
    legend_generator : function that creates legend handles and colors for list of phases
    tie_triangle_color : color

    Returns
    -------
    Matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots()

    sorted_axis_var = map_utils._sort_axis_by_state_vars(strategy.axis_vars)
    if x is None:
        x = sorted_axis_var[1]
    if y is None:
        y = sorted_axis_var[0]

    phases = sorted(strategy.get_all_phases())
    handles, colors = legend_generator(phases)

    # Plot zpf lines
    zpf_data = strategy.get_zpf_data(x, y)
    xlim, ylim = zpf_data['xlim'], zpf_data['ylim']
    for data in zpf_data['data']:
        zero_phase = data['phase']
        x_data, y_data = data['x'], data['y']
        if not all((y_data == 0) | (y_data == np.nan)):
            ax.plot(x_data, y_data, color=colors[zero_phase], lw=1, solid_capstyle="butt")

    # Plot nodes
    node_data = strategy.get_invariant_data(x, y)
    for data in node_data:
        x_data, y_data = data['x'], data['y']
        ax.plot(x_data, y_data, color=tie_triangle_color, zorder=2.5, lw=1, solid_capstyle="butt")

    # Set axis limits
    # If variable is a strategy axis variables, then set limits to axis variable limits
    # If variable is not a strategy axis variable, then set the lower (left, bottom) limits
    #    to the min of the zpf data
    if x in strategy.axis_vars:
        ax.set_xlim([np.amin(strategy.axis_lims[x]), np.amax(strategy.axis_lims[x])])
    else:
        ax.set_xlim(left=xlim[0])

    if y in strategy.axis_vars:
        ax.set_ylim([np.amin(strategy.axis_lims[y]), np.amax(strategy.axis_lims[y])])
    else:
        ax.set_ylim(bottom=ylim[0])

    ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1, 0.5))
    plot_title = '-'.join([component.title() for component in sorted(strategy.components) if component != 'VA'])
    ax.set_title(plot_title)
    ax.set_xlabel(get_label(x))
    ax.set_ylabel(get_label(y))

    return ax