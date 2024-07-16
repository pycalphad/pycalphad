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

"""
Two types of functions in this module

Functions that return plot data
    _get_step_data
    _get_node_data
    _get_tieline_data
    _get_isopleth_zpf_data
    _get_isopleth_node_data

    These just returns dictionaries of data from a strategy

    Ideally, users who want to interface with other plotting libraries
    can still use these functions to get an easy to read data rather than
    having to interface with the strategies themselves

Functions that interface with matplotlib for plotting
    plot_step
    plot_binary
    plot_ternary
    plot_isopleth

    These functions call the previous _get_..._data functions and plots
    them using matplotlib
"""

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

def get_step_data(strategy: StepStrategy, x: v.StateVariable, y: v.StateVariable, x_is_global: bool = False, set_nan_to_zero = False):
    """
    Utility function to get data from StepStrategy for plotting

    Parameters
    ----------
    strategy : StepStrategy
    x : v.StateVariable
    y : v.StateVariable
    x_is_global : bool
        Whether x is a phase local or global variable

    Return
    ------
    step_data : {
        "data" : {
            <phase_name> : {
                "x" : [float]
                "y" : [float]
            }
        }
        "xlim" : [float] - min and max for all x values
        "ylim" : [float] - min and max for all y values
    }
    """
    # Get all phases in strategy (including multiplicity)
    phases = sorted(strategy.get_all_phases())

    # Axis limits for x and y
    xlim = [np.inf, -np.inf]
    ylim = [np.inf, -np.inf]

    # For each phase, grab x and y values and plot, setting all nan values to 0 (if phase is unstable in zpf line, it will return nan for any variable)
    # Then get the max and min of x and y values to update xlim and ylim
    phase_data = {}
    for p in phases:
        x_array = []
        y_array = []
        for zpf_lines in strategy.zpf_lines:
            x_data = zpf_lines.get_var_list(_get_phase_specific_variable(p, x, x_is_global))
            y_data = zpf_lines.get_var_list(_get_phase_specific_variable(p, y))
            if set_nan_to_zero:
                x_data[np.isnan(x_data)] = 0
                y_data[np.isnan(y_data)] = 0
            x_array.append(x_data)
            y_array.append(y_data)

        # We return a single x, y array for all zpf_lines per phase
        x_array = np.concatenate(x_array, axis=0)
        y_array = np.concatenate(y_array, axis=0)

        # Sort arrays by x
        argsort = np.argsort(x_array)
        x_array = x_array[argsort]
        y_array = y_array[argsort]

        phase_data[p] = {'x': x_array, 'y': y_array}

        # Can this be done outside of this loop, or is this the easiest way?
        xlim[0] = np.amin([xlim[0], np.amin(x_array[~np.isnan(x_array)])])
        xlim[1] = np.amax([xlim[1], np.amax(x_array[~np.isnan(x_array)])])
        ylim[0] = np.amin([ylim[0], np.amin(y_array[~np.isnan(y_array)])])
        ylim[1] = np.amax([ylim[1], np.amax(y_array[~np.isnan(y_array)])])

    step_data = {
        'data': phase_data,
        'xlim': xlim,
        'ylim': ylim,
    }

    return step_data

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
        fig, ax = plt.subplots(1,1)

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

    step_data = get_step_data(strategy, x, y, x_is_global, set_nan_to_zero=set_nan_to_zero)
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

def get_node_data(strategy: Union[BinaryStrategy, TernaryStrategy], x: v.StateVariable, y: v.StateVariable):
    """
    Creates dictionary of data for node plotting in binary and ternary plots

    Parameters
    ----------
    strategy : BinaryStrategy or TernaryStrategy
    x : v.StateVariable
    y : v.StateVariable

    Returns
    -------
    node_data : list[dict]
        Each dict will be
        {
            "phases" : [str]
            "x" : [float]
            "y" : [float]
        }
        Indices in x and y will match the indices in phases
    """
    node_data = []
    for node in strategy.node_queue.nodes:
        # Nodes in binary and ternary mappings are always 3 composition sets
        if len(node.stable_composition_sets) == 3:
            node_phases = node.stable_phases_with_multiplicity
            x_data = [node.get_property(_get_phase_specific_variable(p, x)) for p in node_phases]
            y_data = [node.get_property(_get_phase_specific_variable(p, y)) for p in node_phases]
            data = {
                'phases': node_phases,
                'x': x_data,
                'y': y_data,
            }
            node_data.append(data)

    return node_data

def plot_nodes(ax, strategy: Union[BinaryStrategy, TernaryStrategy], x: v.StateVariable, y: v.StateVariable, phase_colors, label_end_points: bool = False, tie_triangle_color = (1, 0, 0, 1)):
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
    node_data = get_node_data(strategy, x, y)
    for data in node_data:
        x_data, y_data, phases = data['x'], data['y'], data['phases']
        ax.plot(x_data + [x_data[0]], y_data + [y_data[0]], color=tie_triangle_color, zorder=2.5, lw=1, solid_capstyle="butt")

        # If labeling end points, add a scatter point on each phase coordinate in the node
        if label_end_points:
            for xp, yp, p in zip(x_data, y_data, phases):
                ax.scatter([xp], [yp], color=phase_colors[p], s=8, zorder=3)

def get_tieline_data(strategy: Union[BinaryStrategy, TernaryStrategy], x: v.StateVariable, y: v.StateVariable):
    """
    Creates dictionary of data for plotting zpf lines

    Parameters
    ----------
    strategy : BinaryStrategy or TernaryStrategy
    x : v.StateVariable
    y : v.StateVariable

    Returns
    -------
    zpf_data : list[dict]
        Each dict will be
        {
            <phase_name> : {
                "x": [float]
                "y": [float]
            }
        }
        Length of x and y for each phase in a ZPFLine should be equal
    """
    zpf_data = []
    for zpf_line in strategy.zpf_lines:
        phases = zpf_line.stable_phases_with_multiplicity
        phase_data = {}
        for p in phases:
            x_data = zpf_line.get_var_list(_get_phase_specific_variable(p, x))
            y_data = zpf_line.get_var_list(_get_phase_specific_variable(p, y))
            phase_data[p] = {
                'x': x_data,
                'y': y_data,
                }
        zpf_data.append(phase_data)
    return zpf_data

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
    zpf_data = get_tieline_data(strategy, x, y)
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
        fig, ax = plt.subplots(1,1)

    # If variables are not given, then plot composition vs. temperature
    sorted_axis_var = map_utils._sort_axis_by_state_vars(strategy.axis_vars)
    if x is None:
        x = sorted_axis_var[1]
    if y is None:
        y = sorted_axis_var[0]

    phases = sorted(strategy.get_all_phases())
    handles, colors = legend_generator(phases)

    plot_tielines(ax, strategy, x, y, phase_colors=colors, tielines=tielines, tieline_color=tieline_color)
    plot_nodes(ax, strategy, x, y, phase_colors=colors, label_end_points=label_nodes, tie_triangle_color=tie_triangle_color)

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
        fig, ax = plt.subplots(1, 1, subplot_kw={'projection': "triangular"})

    plot_binary(strategy, x, y, ax, tielines=tielines, label_nodes=label_nodes, legend_generator=legend_generator, tieline_color=tieline_color, tie_triangle_color=tie_triangle_color, *args, **kwargs)
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])

    # Projection is stored in the default axis name, so only adjust y label if the axis is triangular
    # Note: this is assuming that triangular is the only option for making a ternary plot and that the user doesn't change the default name
    if 'triangular' in ax.name:
        ax.yaxis.label.set_rotation(60)  # rotate ylabel
        ax.yaxis.set_label_coords(x=0.12, y=0.5)  # move the label to a pleasing position

    return ax

def get_isopleth_zpf_data(strategy: IsoplethStrategy, x: v.StateVariable, y: v.StateVariable):
    """
    Creates dictionary of data for plotting zpf lines for isopleths

    Parameters
    ----------
    strategy : BinaryStrategy or TernaryStrategy
    x : v.StateVariable
    y : v.StateVariable

    Returns
    -------
    zpf_data : {
        "data" : [
            {
                "phase" : str
                "x" : [float]
                "y" : [float]
            }
        ]
        "xlim" : [float]
        "ylim" : [float]
    }
    Phase in "data" is the fixed phase at zero-phase fraction
    """
    xlim = [np.inf, -np.inf]
    ylim = [np.inf, -np.inf]
    data = []
    for zpf_line in strategy.zpf_lines:
        zero_phase = zpf_line.fixed_phases[0]
        x_data = zpf_line.get_var_list(x)
        y_data = zpf_line.get_var_list(y)

        zpf_data = {
            'phase': zero_phase,
            'x': x_data,
            'y': y_data,
        }
        data.append(zpf_data)

        xlim[0] = np.amin([xlim[0], np.amin(x_data[~np.isnan(x_data)])])
        xlim[1] = np.amin([xlim[1], np.amin(x_data[~np.isnan(x_data)])])
        ylim[0] = np.amin([ylim[0], np.amin(y_data[~np.isnan(y_data)])])
        ylim[1] = np.amin([ylim[1], np.amin(y_data[~np.isnan(y_data)])])

    zpf_data = {
        'data': data,
        'xlim': xlim,
        'ylim': ylim,
    }

    return zpf_data

def get_isopleth_node_data(strategy: IsoplethStrategy, x: v.StateVariable, y: v.StateVariable):
    """
    Creates dictionary of data for plotting nodes for isopleths

    End points of the node is adjusted to be the intersection of the isopleth polytope on the node in n-composition space

    Parameters
    ----------
    strategy : BinaryStrategy or TernaryStrategy
    x : v.StateVariable
    y : v.StateVariable

    Returns
    -------
    node_data : [
        {
            "x" : [float]
            "y" : [float]
        }
    ]
    """
    node_data = []
    for node in strategy.node_queue.nodes:
        is_invariant = map_utils.degrees_of_freedom(node, strategy.components, strategy.num_potential_condition) == 0
        if is_invariant:
            x_vals = []
            y_vals = []
            for trial_stable_compsets in itertools.permutations(node.stable_composition_sets, len(node.stable_composition_sets)-2):
                # Get phase fraction for combination of phases and node conditions
                phase_NP = strategy._invariant_phase_fractions(node, trial_stable_compsets)
                if phase_NP is None:
                    continue

                # If phase combination is value, then extract x and y values
                if all(phase_NP > 0):
                    if map_utils.is_state_variable(x):
                        x_vals.append(node.get_property(x))
                    else:
                        x_vals.append(sum(node.get_local_property(cs, x)*cs_NP for cs, cs_NP in zip(trial_stable_compsets, phase_NP)))
                    if map_utils.is_state_variable(y):
                        y_vals.append(node.get_property(y))
                    else:
                        y_vals.append(sum(node.get_local_property(cs, y)*cs_NP for cs, cs_NP in zip(trial_stable_compsets, phase_NP)))

            for p1, p2 in itertools.combinations(range(len(x_vals)), 2):
                x_data = [x_vals[p1], x_vals[p2]]
                y_data = [y_vals[p1], y_vals[p2]]
                data = {
                    'x': x_data,
                    'y': y_data,
                }
                node_data.append(data)
    return node_data


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
        fig, ax = plt.subplots(1,1)

    sorted_axis_var = map_utils._sort_axis_by_state_vars(strategy.axis_vars)
    if x is None:
        x = sorted_axis_var[1]
    if y is None:
        y = sorted_axis_var[0]

    phases = sorted(strategy.get_all_phases())
    handles, colors = legend_generator(phases)

    # Plot zpf lines
    zpf_data = get_isopleth_zpf_data(strategy, x, y)
    xlim, ylim = zpf_data['xlim'], zpf_data['ylim']
    for data in zpf_data['data']:
        zero_phase = data['phase']
        x_data, y_data = data['x'], data['y']
        if not all((y_data == 0) | (y_data == np.nan)):
            ax.plot(x_data, y_data, color=colors[zero_phase], lw=1, solid_capstyle="butt")

    # Plot nodes
    node_data = get_isopleth_node_data(strategy, x, y)
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