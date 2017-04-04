"""
The eqplot module contains functions for general plotting of
the results of equilibrium calculations.
"""
from pycalphad.core.utils import unpack_condition
from pycalphad.plot.utils import phase_legend
import pycalphad.variables as v
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

def eqplot(eq, ax=None, x=None, y=None, z=None, phases=None, **kwargs):
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
    phases = phases if phases is not None else \
        map(str, sorted(set(np.array(eq.Phase.values.ravel(), dtype='U')) - {''}, key=str))
    comps = map(str, sorted(np.array(eq.coords['component'].values, dtype='U'), key=str))
    eq['component'] = np.array(eq['component'], dtype='U')
    eq['Phase'].values = np.array(eq['Phase'].values, dtype='U')

    # Select all two-phase regions
    two_phase_indices = np.nonzero(np.sum(eq.Phase.values != '', axis=-1, dtype=np.int) == 2)
    found_phases = eq.Phase.values[two_phase_indices][..., :2]

    legend_handles, colorlist = phase_legend(phases)
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
