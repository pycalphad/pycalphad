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
        Passed to `matplotlib.pyplot.scatter`.

    Returns
    -------
    matplotlib AxesSubplot
    """
    ax = plt.gca() if ax is None else ax
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
    eq['component'] = np.array(eq['component'], dtype='U')
    eq['Phase'].values = np.array(eq['Phase'].values, dtype='U')

    # Select all two-phase regions
    two_phase_indices = np.nonzero(np.sum(eq.Phase.values != '', axis=-1, dtype=np.int) == 2)
    found_phases = eq.Phase.values[two_phase_indices][..., :2]

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
    # position the phase legend
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1, 0.5))
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.grid(True)
    plotcolors = np.array(list(map(lambda x: [colorlist[x[0]], colorlist[x[1]]], found_phases)), dtype='U')

    if isinstance(x, v.Composition):
        compositions = eq.X.sel(component=x.species).values[two_phase_indices][..., :2]
    else:
        raise NotImplementedError('Plotting {} is not yet implemented'.format(x))
    # Have to do some extra work to get all potential values for the given tie lines
    temps = np.take(eq[str(y)].values, two_phase_indices[list(str(i) for i in conds.keys()).index(str(y))])
    if ax is None:
        ax = plt.gca()
    # Draw zero phase-fraction lines
    ax.scatter(compositions[..., 0], temps, s=3, c=plotcolors[..., 0], edgecolors='None', zorder=2, **kwargs)
    ax.scatter(compositions[..., 1], temps, s=3, c=plotcolors[..., 1], edgecolors='None', zorder=2, **kwargs)
    # Draw tie-lines
    tielines = np.array([np.concatenate((compositions[..., 0][..., np.newaxis], temps[..., np.newaxis]), axis=-1),
                         np.concatenate((compositions[..., 1][..., np.newaxis], temps[..., np.newaxis]), axis=-1)])
    tielines = np.rollaxis(tielines, 1)
    lc = mc.LineCollection(tielines, zorder=1, colors=[0, 1, 0, 1], linewidths=[0.5, 0.5])
    ax.add_collection(lc)
    plot_title = '-'.join([x.title() for x in sorted(comps) if x != 'VA'])
    ax.set_title(plot_title, fontsize=20)
    ax.set_xlim([np.min(conds[indep_comp])-1e-2, np.max(conds[indep_comp])+1e-2])
    ax.set_ylim([np.min(conds[indep_pot]), np.max(conds[indep_pot])])
    if isinstance(x, v.Composition):
        ax.set_xlabel('X({})'.format(indep_comp.species), labelpad=15, fontsize=20)
    else:
        ax.set_xlabel(indep_comp, labelpad=15, fontsize=20)
    ax.set_ylabel(_plot_labels[indep_pot], fontsize=20)
    return ax