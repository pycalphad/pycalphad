from collections import defaultdict
import warnings
import numpy as np
import matplotlib.pyplot as plt
from pycalphad.plot.utils import phase_legend
from pycalphad import variables as v
from pycalphad.mapping.primitives import STATEVARS
from .primitives import Point, ZPFLine, _get_value_for_var, _get_global_value_for_var
from .mapper import Mapper

_plot_labels = {
    v.T: 'Temperature (K)',
    v.P: 'Pressure (Pa)',
}

#Generic utility function that tests whether the mapper is stepping or tielines or whatever
#TODO: User should still be allowed to define plotting variables
def plot_map(mapper, tielines=False, ax=None, legend_generator=phase_legend):
    fig = None
    ms = mapper.strategy
    if ms.stepping:
        if ax is None:
            fig, ax = plt.subplots(1,1)
        plot_step(ms, ms.axis_vars[0], v.NP(''), ax, legend_generator=legend_generator)
        ax.set_xlim(ms.axis_lims[ms.axis_vars[0]][0], ms.axis_lims[ms.axis_vars[0]][1])
    else:
        v1 = v.T if v.T in ms.axis_vars else v.P if v.P in ms.axis_vars else ms.axis_vars[0]
        v2 = ms.axis_vars[1] if v1 == ms.axis_vars[0] else ms.axis_vars[0]
        if v.T in ms.axis_vars or v.P in ms.axis_vars:
            if ax is None:
                fig, ax = plt.subplots(1,1)
        else:
            if ax is None:
                fig = plt.figure()
                ax = fig.add_subplot(projection='triangular')
        if ms.tielines:
            plot_2d(ms, v2, v1, ax, tielines=tielines, legend_generator=legend_generator)
        else:
            plot_isopleth(ms, v2, v1, ax)
        ax.set_xlim(ms.axis_lims[v2][0], ms.axis_lims[v2][1])
        ax.set_ylim(ms.axis_lims[v1][0], ms.axis_lims[v1][1])
    return fig, ax

def _axis_label(ax_var):
    if isinstance(ax_var, v.MoleFraction):
        return 'X({})'.format(ax_var.species.name)
    elif isinstance(ax_var, v.PhaseFraction):
        return "Phase Fraction"  # TODO: handle particular phase vs. "*" case
    elif isinstance(ax_var, v.StateVariable):
        return _plot_labels[ax_var]
    else:
        return ax_var

# NOTE: default zorders: https://matplotlib.org/3.1.1/gallery/misc/zorder_demo.html
# - Patch / PatchCollection : 1
# - Line2D / LineCollection : 2
# - Text                    : 3

# NOTE: On options;
# solid_capstyle choose between "butt", "round", "projecting"(default). "butt" should help the line from appearing to go past the point, see

def is_true_node(node: Point, mapper):
    #Quick check that node is actually a node
    #In Sundman's paper, the starting point for the step/map is considered a node to start algorithm C1;
    #   It is a node in the sense that it defines the start of a zpf line
    #   However, it is also not a node in this sense where it doesn't represent an invariant
    #Since we store all the conditions in the node, we can check this by gibbs phase rule f = n+2-p-c = 0
    f = len(mapper.elements)-1 + 2 - len(node.stable_composition_sets) - (2-mapper.num_potential_conditions)
    return f == 0

# Case for tie-lines in planes - plot lines between the composition at each stable composition sets
# Case for isopleths - we want to set up linear combinations of pairs of phase to get values for xvar and yvar, then plot between the calualated xvar and yvar of each solution
#       This will be kinda copied over from the exit strategy in general_strategy
def _plot_node(node: Point, xvar, yvar, ax, isopleth = False):
    tie_triangle_color = (1, 0, 0, 1)  # RGBA
    import itertools
    # Plot all pairwise lines
    if not isopleth:
        for cs1, cs2 in itertools.combinations(node.stable_composition_sets, 2):
            xx = [_get_value_for_var(cs1, xvar), _get_value_for_var(cs2, xvar)]
            yy = [_get_value_for_var(cs1, yvar), _get_value_for_var(cs2, yvar)]
            # Plot on top of other lines
            ax.plot(xx, yy, color=tie_triangle_color, zorder=2.5, lw=1, solid_capstyle="butt")  # z-order just under line 2D
    else:
        xvals = []
        yvals = []
        for trial_stable_compsets in itertools.combinations(node.stable_composition_sets, len(node.stable_composition_sets)-2):
            phase_X_matrix = np.array([np.array(cs.X) for cs in trial_stable_compsets])
            fixed_var = [av for av in node.global_conditions if (av != v.T and av != v.P and av not in [xvar, yvar])]
            phase_X_matrix = np.zeros((len(fixed_var), len(trial_stable_compsets)))
            b = np.zeros((len(fixed_var),1))
            for i in range(len(fixed_var)):
                for j in range(len(trial_stable_compsets)):
                    phase_X_matrix[i,j] = _get_value_for_var(trial_stable_compsets[j], fixed_var[i])
                b[i,0] = node.global_conditions[fixed_var[i]]
            if np.linalg.matrix_rank(phase_X_matrix) != phase_X_matrix.shape[0]:
                continue
            phase_NP = np.matmul(np.linalg.inv(phase_X_matrix), b).flatten()
            if all(phase_NP > 0):
                if xvar in STATEVARS:
                    xvals.append(_get_global_value_for_var(node, xvar))
                else:
                    xvals.append(sum(_get_value_for_var(cs, xvar)*cs_NP for cs,cs_NP in zip(trial_stable_compsets, phase_NP)))
                if yvar in STATEVARS:
                    yvals.append(_get_global_value_for_var(node, yvar))
                else:
                    yvals.append(sum(_get_value_for_var(cs, yvar)*cs_NP for cs,cs_NP in zip(trial_stable_compsets, phase_NP)))
            for p1, p2 in itertools.combinations(range(len(xvals)), 2):
                xx = [xvals[p1], xvals[p2]]
                yy = [yvals[p2], yvals[p2]]
                ax.plot(xx, yy, color=tie_triangle_color, zorder=2.5, lw=1, solid_capstyle="butt")

# TODO: binary assumption with one fixed and one free phase
#   to drop this assumption, we could do loops over the pt.fixed/free/stable compsets
#   easily, but currently the compset lists are unordered and there is not a way to
#   uniquely identify composition sets so we can plot _lines_ instead of points
#   phase_name could be one way, but it would break if there was a miscibility gap
def _plot_zpf_line(zpf_line: ZPFLine, xvar, yvar, ax, colors, with_free_phases=False, tielines=True):
    # tielines : int - slice step for tielines. False or zero means no tie-lines. True or 1 means every tie-line. Other integers are every n-th tie-line.
    if zpf_line.num_fixed_phases() >= 1:
        xplot, yplot = [], []
        for pt in zpf_line.points:
                xplot.append(_get_value_for_var(pt.fixed_composition_sets[0], xvar))
                yplot.append(_get_value_for_var(pt.fixed_composition_sets[0], yvar))
        ax.plot(xplot, yplot, color=colors[zpf_line.fixed_phases[0]], lw=1, solid_capstyle="butt")

    if with_free_phases:
        xplot_free, yplot_free = [], []
        for pt in zpf_line.points:
            xplot_free.append(_get_value_for_var(pt.free_composition_sets[0], xvar))
            yplot_free.append(_get_value_for_var(pt.free_composition_sets[0], yvar))
        ax.plot(xplot_free, yplot_free, color=colors[zpf_line.free_phases[0]], lw=1, solid_capstyle="butt")

        if zpf_line.num_fixed_phases() == 1 and zpf_line.num_free_phases() == 1 and tielines:
            from matplotlib.collections import LineCollection
            tieline_color=(0, 1, 0, 1)
            tieline_collection = LineCollection(np.asarray([[xplot, xplot_free], [yplot, yplot_free]]).T[::tielines, ...], zorder=1, linewidths=0.5, capstyle="butt", colors=[tieline_color for _ in range(len(yplot))])
            ax.add_collection(tieline_collection)

def plot_2d(mapper: Mapper, xvar, yvar, ax, tielines=1, legend_generator=phase_legend):
    # TODO: Hardcoded for tielines in place case
    #Get phases that are included only in zpf lines
    phases = []
    for zpfline in mapper.zpf_lines:
        zpfline_phases = zpfline.fixed_phases + zpfline.free_phases
        phases = np.unique(np.concatenate([phases, zpfline_phases]))
    handles, colors = legend_generator(phases)

    for zpf_line in mapper.zpf_lines:
        _plot_zpf_line(zpf_line, xvar, yvar, ax, colors, with_free_phases=True, tielines=tielines)

    for node in mapper.node_queue.nodes:
        if is_true_node(node, mapper):
            _plot_node(node, xvar, yvar, ax)

    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1.01, 0.5))
    ax.tick_params(axis='both', which='major')
    plot_title = '-'.join([component for component in sorted(mapper.components) if component != 'VA'])
    ax.set_title(plot_title)
    ax.set_xlabel(_axis_label(xvar))
    ax.set_ylabel(_axis_label(yvar))
    # autoscale needs to be used in case boundaries are plotted as lines because
    # only plotting line collections will not rescale the axes
    ax.autoscale(axis='y')
    ax.grid(False)

def plot_isopleth(mapper: Mapper, xvar, yvar, ax):
    handles, colors = phase_legend(mapper.phases)

    for zpf_line in mapper.zpf_lines:
        xplot_fixed, yplot_fixed = [], []
        for pt in zpf_line.points:
            xplot_fixed.append(pt.global_conditions[xvar])
            yplot_fixed.append(pt.global_conditions[yvar])
        ax.plot(xplot_fixed, yplot_fixed, color=colors[zpf_line.fixed_phases[0]], lw=1, solid_capstyle="butt")

    for node in mapper.node_queue.nodes:
        if is_true_node(node, mapper):
            _plot_node(node, xvar, yvar, ax, True)

    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1.01, 0.5))
    ax.tick_params(axis='both', which='major')
    plot_title = '-'.join([component for component in sorted(mapper.components) if component != 'VA'])
    ax.set_title(plot_title)
    ax.set_xlabel(_axis_label(xvar))
    ax.set_ylabel(_axis_label(yvar))
    # autoscale needs to be used in case boundaries are plotted as lines because
    # only plotting line collections will not rescale the axes
    ax.autoscale(axis='y')
    ax.grid(False)


def plot_step(mapper: Mapper, xvar, yvar, ax, legend_generator=phase_legend):
    #Get list of unique phases
    unique_phases = []
    for zpf_line in mapper.zpf_lines:
        unique_phases += zpf_line.unique_phases()
    unique_phases = np.unique(unique_phases)
    # Ploting for stepping
    handles, colors = legend_generator(unique_phases)

    for zpf_line in mapper.zpf_lines:
        # TODO: assumption here that global var a user plots is always X and local is y
        cs_x = zpf_line.get_global_condition_var_list(xvar)
        cs_y = zpf_line.get_local_var_list(yvar)
        for p in cs_y:
            ax.plot(cs_x, cs_y[p], color=colors[p])

    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1, 0.5))
    ax.tick_params(axis='both', which='major')
    plot_title = '-'.join([component for component in sorted(mapper.components) if component != 'VA'])
    ax.set_title(plot_title)
    ax.set_xlabel(_axis_label(xvar))
    ax.set_ylabel(_axis_label(yvar))
    # autoscale needs to be used in case boundaries are plotted as lines because
    # only plotting line collections will not rescale the axes
    ax.autoscale(axis='y')
    ax.grid(False)
