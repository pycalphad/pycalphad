import itertools

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from pycalphad import variables as v
from pycalphad.plot.utils import phase_legend
from pycalphad.plot import triangular  # register triangular projection

from pycalphad.mapping.primitives import STATEVARS, Node
from pycalphad.mapping.strategy.step_strategy import StepStrategy
from pycalphad.mapping.strategy.binary_strategy import BinaryStrategy
from pycalphad.mapping.strategy.ternary_strategy import TernaryStrategy
from pycalphad.mapping.strategy.isopleth_strategy import IsoplethStrategy
import pycalphad.mapping.utils as map_utils

def _get_phase_specific_variable(phase: str, var: v.StateVariable, is_global = False):
    if is_global:
        return var
    if isinstance(var, v.X):
        return v.X(phase, var.species)
    elif isinstance(var, v.NP) or var == v.NP:
        return v.NP(phase)
    else:
        return var

def plot_step(strategy: StepStrategy, x: v.StateVariable = None, y: v.StateVariable = None, ax = None, legend_generator = phase_legend, *args, **kwargs):
    """
    API for plotting step maps
    """
    if ax is None:
        fig, ax = plt.subplots(1,1)

    #If x is None, then use axis variable and state that x is global (this is useful for v.X where we can distinguish v.X(sp,ph) vs. v.X(sp)
    x_is_global = False
    if x is None:
        x = strategy.axis_vars[0]
    if x == strategy.axis_vars[0]:
        x_is_global = True
    #If y is None, then use v.NP
    if y is None:
        y = v.NP

    #Get all phases in strategy (including multiplicity)
    phases = strategy.get_all_phases()

    handles, colors = legend_generator(phases)

    #Axis limits for x and y
    xlim = [np.inf, -np.inf]
    ylim = [np.inf, -np.inf]

    #For each phase, grab x and y values and plot, setting all nan values to 0 (if phase is unstable in zpf line, it will return nan for any variable)
    #Then get the max and min of x and y values to update xlim and ylim
    #TODO: I don't like the "setting nan values to 0" since it can lead to some awkward plotting for variables such as MU, for NP it seems to be okay though
    for p in phases:
        x_array = []
        y_array = []
        for zpf_lines in strategy.zpf_lines:
            x_data = zpf_lines.get_var_list(_get_phase_specific_variable(p, x, x_is_global))
            y_data = zpf_lines.get_var_list(_get_phase_specific_variable(p, y))
            x_data[np.isnan(x_data)] = 0
            y_data[np.isnan(y_data)] = 0
            x_array.append(x_data)
            y_array.append(y_data)
        x_array = np.concatenate(x_array, axis=0)
        y_array = np.concatenate(y_array, axis=0)
        argsort = np.argsort(x_array)
        ax.plot(x_array[argsort], y_array[argsort], color=colors[p], lw=1, solid_capstyle="butt")

        xlim[0] = np.amin([xlim[0], np.amin(x_array[~np.isnan(x_array)])])
        xlim[1] = np.amax([xlim[1], np.amax(x_array[~np.isnan(x_array)])])
        ylim[0] = np.amin([ylim[0], np.amin(y_array[~np.isnan(y_array)])])
        ylim[1] = np.amax([ylim[1], np.amax(y_array[~np.isnan(y_array)])])

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    #Add legend
    ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1, 0.5))
    plot_title = '-'.join([component.title() for component in sorted(strategy.components) if component != 'VA'])
    ax.set_title(plot_title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    
    return ax

def _plot_nodes(ax, nodes: list[Node], x: v.StateVariable, y: v.StateVariable, phases: list[str], phase_colors, label_end_points: bool = False, tie_triangle_color=(1, 0, 0, 1)):
    """
    Plotting nodes in binary and ternary plots
    """
    for node in nodes:
        #For binary and ternary, a node will always have 3 phases
        #We could also check the degrees of freedon which should return 0 for 3 phases
        if len(node.stable_composition_sets) == 3:
            x_data = [node.get_property(_get_phase_specific_variable(p, x)) for p in node.stable_phases_with_multiplicity]
            y_data = [node.get_property(_get_phase_specific_variable(p, y)) for p in node.stable_phases_with_multiplicity]
            ax.plot(x_data + [x_data[0]], y_data + [y_data[0]], color=tie_triangle_color, zorder=2.5, lw=1, solid_capstyle="butt")

            if label_end_points:
                for xp, yp, p in zip(x_data, y_data, node.stable_phases_with_multiplicity):
                    ax.scatter([xp], [yp], color=phase_colors[p], s=8, zorder=3)

def plot_binary(strategy: BinaryStrategy, x: v.StateVariable = None, y: v.StateVariable = None, ax = None, tielines = 1, label_node = False, legend_generator = phase_legend, tieline_color=(0, 1, 0, 1), tie_triangle_color=(1, 0, 0, 1), *args, **kwargs):
    """
    Binary plotting
    """
    if ax is None:
        fig, ax = plt.subplots(1,1)

    #If variables are not given, then plot composition vs. temperature
    sorted_axis_var = map_utils._sort_axis_by_state_vars(strategy.axis_vars)
    if x is None:
        x = sorted_axis_var[1]
    if y is None:
        y = sorted_axis_var[0]

    phases = strategy.get_all_phases()
    handles, colors = legend_generator(phases)

    for zpf_lines in strategy.zpf_lines:
        phases = zpf_lines.stable_phases_with_multiplicity
        x_arrays = []
        y_arrays = []
        for p in phases:
            x_data = zpf_lines.get_var_list(_get_phase_specific_variable(p, x))
            y_data = zpf_lines.get_var_list(_get_phase_specific_variable(p, y))
            x_arrays.append(x_data)
            y_arrays.append(y_data)
            if not all((y_data == 0) | (y_data == np.nan)):
                ax.plot(x_data, y_data, color=colors[p], lw=1, solid_capstyle="butt")

        if tielines:
            tieline_collection = LineCollection(np.asarray([[x_arrays[0], x_arrays[1]], [y_arrays[0], y_arrays[1]]]).T[::tielines, ...], zorder=1, linewidths=0.5, capstyle="butt", colors=[tieline_color for _ in range(len(x_arrays[0]))])
            ax.add_collection(tieline_collection)

    _plot_nodes(ax, strategy.node_queue.nodes, x, y, phases=phases, phase_colors=colors, label_end_points=label_node, tie_triangle_color=tie_triangle_color)

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
    ax.set_xlabel(x)
    ax.set_ylabel(y)

    return ax

def plot_ternary(strategy: TernaryStrategy, x: v.StateVariable = None, y: v.StateVariable = None, ax = None, tielines = 1, label_nodes = False, legend_generator = phase_legend, tieline_color=(0, 1, 0, 1), tie_triangle_color=(1, 0, 0, 1), *args, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(1, 1, subplot_kw={'projection': "triangular"})

    plot_binary(strategy, x, y, ax, tielines=tielines, label_node=label_nodes, legend_generator=legend_generator, tieline_color=tieline_color, tie_triangle_color=tie_triangle_color, *args, **kwargs)
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.yaxis.label.set_rotation(60)  # rotate ylabel
    ax.yaxis.set_label_coords(x=0.12, y=0.5)  # move the label to a pleasing position

    return ax


def plot_isopleth(strategy: IsoplethStrategy, x: v.StateVariable = None, y: v.StateVariable = None, ax = None, legend_generator = phase_legend, *args, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(1,1)

    sorted_axis_var = map_utils._sort_axis_by_state_vars(strategy.axis_vars)
    if x is None:
        x = sorted_axis_var[1]
    if y is None:
        y = sorted_axis_var[0]

    phases = strategy.get_all_phases()
    handles, colors = legend_generator(phases)

    xmin, ymin = np.inf, np.inf
    for zpf_lines in strategy.zpf_lines:
        phases = zpf_lines.stable_phases_with_multiplicity
        x_data = zpf_lines.get_var_list(x)
        y_data = zpf_lines.get_var_list(y)
        if not all((y_data == 0) | (y_data == np.nan)):
            ax.plot(x_data, y_data, color=colors[zpf_lines.fixed_phases[0]], lw=1, solid_capstyle="butt")

        xmin = np.amin([xmin, np.amin(x_data[~np.isnan(x_data)])])
        ymin = np.amin([ymin, np.amin(y_data[~np.isnan(y_data)])])

    for node in strategy.node_queue.nodes:
        #NOTE: This is pretty much copied from the isopleth strategy, so there's probably a way to avoid redundancy here
        is_invariant = map_utils.degrees_of_freedom(node, strategy.components, strategy.num_potential_condition) == 0
        if is_invariant:
            x_vals = []
            y_vals = []
            for trial_stable_compsets in itertools.permutations(node.stable_composition_sets, len(node.stable_composition_sets)-2):
                phase_NP = strategy._invariant_phase_fractions(node, trial_stable_compsets)
                if phase_NP is None:
                    continue

                if all(phase_NP > 0):
                    if x in STATEVARS:
                        x_vals.append(node.get_property(x))
                    else:
                        x_vals.append(sum(node.get_local_property(cs, x)*cs_NP for cs, cs_NP in zip(trial_stable_compsets, phase_NP)))
                    if y in STATEVARS:
                        y_vals.append(node.get_property(y))
                    else:
                        y_vals.append(sum(node.get_local_property(cs, y)*cs_NP for cs, cs_NP in zip(trial_stable_compsets, phase_NP)))

            for p1, p2 in itertools.combinations(range(len(x_vals)), 2):
                xx = [x_vals[p1], x_vals[p2]]
                yy = [y_vals[p1], y_vals[p2]]
                ax.plot(xx, yy, color=(1,0,0,1), zorder=2.5, lw=1, solid_capstyle="butt")

    if x in strategy.axis_vars:
        ax.set_xlim([np.amin(strategy.axis_lims[x]), np.amax(strategy.axis_lims[x])])
    else:
        ax.set_xlim(left=xmin)
    
    if y in strategy.axis_vars:
        ax.set_ylim([np.amin(strategy.axis_lims[y]), np.amax(strategy.axis_lims[y])])
    else:
        ax.set_ylim(bottom=ymin)

    ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1, 0.5))
    plot_title = '-'.join([component.title() for component in sorted(strategy.components) if component != 'VA'])
    ax.set_title(plot_title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)

    return ax