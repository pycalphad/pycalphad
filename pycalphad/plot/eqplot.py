"""
The eqplot module contains functions for general plotting of
the results of equilibrium calculations.
"""
from pycalphad.core.utils import unpack_condition
import pycalphad.variables as v
import matplotlib.patches as mpatches
from matplotlib import collections as mc
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict


_plot_labels = {v.T: 'Temperature (K)', v.P: 'Pressure (Pa)'}


def _map_coord_to_variable(coord):
    """
    Map a coordinate to a StateVariable object.

    Parameters
    ----------
    coord : str
        Name of coordinate in equilibrium object.

    Returns
    -------
    pycalphad StateVariable
    """
    vals = {'T': v.T, 'P': v.P}
    if coord.startswith('X_'):
        return v.X(coord[2:])
    elif coord in vals:
        return vals[coord]
    else:
        return coord


def _binplot_setup(ax, phases, tie_lines, tie_line_colors, tie_line_widths):
    "Setup the plot for a binary phase diagram."
    colorlist = {}

    # colors from Junwei Huang, March 21 2013
    # exclude green and red because of their special meaning on the diagram
    colorvalues = ["0000FF", "FFFF00", "FF00FF", "00FFFF", "000000",
                   "800000", "008000", "000080", "808000", "800080", "008080",
                   "808080", "C00000", "00C000", "0000C0", "C0C000", "C000C0",
                   "00C0C0", "C0C0C0", "400000", "004000", "000040", "404000",
                   "400040", "004040", "404040", "200000", "002000", "000020",
                   "202000", "200020", "002020", "202020", "600000", "006000",
                   "000060", "606000", "600060", "006060", "606060", "A00000",
                   "00A000", "0000A0", "A0A000", "A000A0", "00A0A0", "A0A0A0",
                   "E00000", "00E000", "0000E0", "E0E000", "E000E0", "00E0E0",
                   "E0E0E0"]

    phasecount = 0
    legend_handles = []
    for phase in phases:
        phase = phase.upper()
        colorlist[phase] = "#"+colorvalues[np.mod(phasecount, len(colorvalues))]
        legend_handles.append(mpatches.Patch(color=colorlist[phase], label=phase))
        phasecount += 1
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.grid(True)

    # Get the configured plot colors
    #plotcolors = list(map(lambda x: [colorlist[x[0]], colorlist[x[1]]],
    #                      tie_lines[:, :, 2]))
    if len(tie_lines) > 0:
        lc = mc.LineCollection(
            tie_lines, zorder=1, colors=tie_line_colors, linewidths=tie_line_widths
        )
        ax.add_collection(lc)
        ax.scatter(tie_lines[:, :, 0].ravel(), tie_lines[:, :, 1].ravel(), s=3, zorder=2)


    # position the phase legend
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1, 0.5))

    return ax


def eqplot(eq, ax=None, x=None, y=None, z=None, **kwargs):
    """
    Plot the result of an equilibrium calculation.

    Parameters
    ----------
    eq : xarray.Dataset
        Result of equilibrium calculation.
    ax : matplotlib.Axes
        Default axes used if not specified.
    x : StateVariable, optional
    y : StateVariable, optional
    z : StateVariable, optional
    kwargs : kwargs
        Passed to matplotlib.pyplot.plot()

    Returns
    -------
    matplotlib Figure
    """
    ax = plt.gca() if ax is None else ax
    tie_lines = []
    tie_line_colors = []
    tie_line_widths = []
    phase_boundaries = OrderedDict()
    conds = OrderedDict([(_map_coord_to_variable(key), unpack_condition(np.asarray(value)))
                         for key, value in sorted(eq.coords.items(), key=str)
                         if (key == 'T') or (key == 'P') or (key.startswith('X_'))])
    indep_comp = [key for key, value in conds.items() if isinstance(key, v.Composition) and len(value) > 1]
    indep_pot = [key for key, value in conds.items() if ((key == v.T) or (key == v.P)) and len(value) > 1]
    if (len(indep_comp) != 1) or (len(indep_pot) != 1):
        raise ValueError('Plot currently requires exactly one composition and one potential coordinate')
    indep_comp = indep_comp[0]
    indep_pot = indep_pot[0]
    x = indep_comp if x is None else x
    y = indep_pot if y is None else y
    if z is not None:
        raise NotImplementedError('3D plotting is not yet implemented')
    # TODO: Temporary workaround to fix string encoding issue when loading netcdf files from disk
    phases = map(str, sorted(set(np.array(eq.Phase.values.ravel(), dtype='U')) - {''}, key=str))
    comps = map(str, sorted(np.array(eq.coords['component'].values, dtype='U'), key=str))
    fixed_conds = {str(key): value[0] for key, value in conds.items() if len(value) == 1}
    eq['component'] = np.array(eq['component'], dtype='U')
    eq['Phase'].values = np.array(eq['Phase'].values, dtype='U')

    full_eq_by_temp = eq.sel(**fixed_conds).groupby(str(indep_pot))
    for current_temp, temp_slice in full_eq_by_temp:
        full_eq_by_comp = temp_slice.groupby(str(indep_comp))
        for current_comp, slice_eq in full_eq_by_comp:
            # Select all two-phase regions in this composition
            num_phases = np.sum((slice_eq.Phase != ''), dtype=np.int)
            if num_phases != 2:
                continue
            first_vertex = slice_eq.sel(vertex=0, component=indep_comp.species)
            first_tuple = (float(first_vertex[str(indep_pot)].values), float(first_vertex['X'].values), first_vertex['Phase'].values)
            second_vertex = slice_eq.sel(vertex=1, component=indep_comp.species)
            second_tuple = (float(second_vertex[str(indep_pot)].values), float(second_vertex['X'].values), second_vertex['Phase'].values)
            tie_lines.append([[first_tuple[1], first_tuple[0]], [second_tuple[1], second_tuple[0]]])
            # Green for a tie line (as is tradition)
            tie_line_colors.append([0, 1, 0, 1])
            tie_line_widths.append(0.5)
    tie_lines = np.atleast_3d(tie_lines)
    tie_line_colors = np.atleast_2d(tie_line_colors)
    tie_line_widths = np.atleast_1d(tie_line_widths)

    if ax is None:
        ax = plt.gca()
    ax = _binplot_setup(ax, phases, tie_lines, tie_line_colors, tie_line_widths)
    plot_title = '-'.join([x.title() for x in sorted(comps) if x != 'VA'])
    ax.set_title(plot_title, fontsize=20)
    ax.set_xlim([np.min(conds[indep_comp])-1e-4, np.max(conds[indep_comp])+1e-4])
    ax.set_ylim([np.min(conds[indep_pot]), np.max(conds[indep_pot])])
    ax.set_xlabel(indep_comp, labelpad=15, fontsize=20)
    ax.set_ylabel(_plot_labels[indep_pot], fontsize=20)
    return ax