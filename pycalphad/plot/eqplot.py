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

# TODO: support other state variables here or make isinstance elif == v.T or v.P
_plot_labels = {v.T: 'Temperature (K)', v.P: 'Pressure (Pa)'}


def _axis_label(ax_var):
    if isinstance(ax_var, v.MoleFraction):
        return 'X({})'.format(ax_var.species.name)
    elif isinstance(ax_var, v.StateVariable):
        return _plot_labels[ax_var]
    else:
        return ax_var

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


def eqplot(eq, ax=None, x=None, y=None, z=None, tielines=True, tieline_color=(0, 1, 0, 1), tie_triangle_color=(1, 0, 0, 1), legend_generator=phase_legend, **kwargs):
    """
    Plot the result of an equilibrium calculation.

    The type of plot is controlled by the degrees of freedom in the equilibrium calculation.

    Parameters
    ----------
    eq : xarray.Dataset
        Result of equilibrium calculation.
    ax : matplotlib.Axes
        Default axes used if not specified.
    x : StateVariable, optional
    y : StateVariable, optional
    z : StateVariable, optional
    tielines : bool
        If True, will plot tielines
    tieline_color: color
        A valid matplotlib color, such as a named color string, hex RGB
        string, or a tuple of RGBA components to set the color of the two
        phase region tielines. The default is an RGBA tuple for green:
        (0, 1, 0, 1).
    tie_triangle_color: color
        A valid matplotlib color, such as a named color string, hex RGB
        string, or a tuple of RGBA components to set the color of the two
        phase region tielines. The default is an RGBA tuple for red:
        (1, 0, 0, 1).
    legend_generator : Callable
        A function that will be called with the list of phases and will
        return legend labels and colors for each phase. By default
        pycalphad.plot.utils.phase_legend is used
    kwargs : kwargs
        Passed to `matplotlib.pyplot.scatter`.

    Returns
    -------
    matplotlib AxesSubplot
    """
    conds = OrderedDict([(_map_coord_to_variable(key), unpack_condition(np.asarray(value)))
                         for key, value in sorted(eq.coords.items(), key=str)
                         if (key in ('T', 'P', 'N')) or (key.startswith('X_'))])
    indep_comps = sorted([key for key, value in conds.items() if isinstance(key, v.MoleFraction) and len(value) > 1], key=str)
    indep_pots = [key for key, value in conds.items() if (type(key) is v.StateVariable) and len(value) > 1]

    # determine what the type of plot will be
    if len(indep_comps) == 1 and len(indep_pots) == 1:
        projection = None
    elif len(indep_comps) == 2 and len(indep_pots) == 0:
        projection = 'triangular'
    else:
        raise ValueError('The eqplot projection is not defined and cannot be autodetected. There are {} independent compositions and {} indepedent potentials.'.format(len(indep_comps), len(indep_pots)))
    if z is not None:
        raise NotImplementedError('3D plotting is not yet implemented')
    if ax is None:
        fig, (ax) = plt.subplots(subplot_kw={'projection': projection})

    # Handle cases for different plot types
    if projection is None:
        x = indep_comps[0] if x is None else x
        y = indep_pots[0] if y is None else y
        # plot settings
        ax.set_xlim([np.min(conds[x]) - 1e-2, np.max(conds[x]) + 1e-2])
        ax.set_ylim([np.min(conds[y]), np.max(conds[y])])
    elif projection == 'triangular':
        x = indep_comps[0] if x is None else x
        y = indep_comps[1] if y is None else y
        # Here we adjust the x coordinate of the ylabel.
        # We make it reasonably comparable to the position of the xlabel from the xaxis
        # As the figure size gets very large, the label approaches ~0.55 on the yaxis
        # 0.55*cos(60 deg)=0.275, so that is the xcoord we are approaching.
        ax.yaxis.label.set_va('baseline')
        fig_x_size = ax.figure.get_size_inches()[0]
        y_label_offset = 1 / fig_x_size
        ax.yaxis.set_label_coords(x=(0.275 - y_label_offset), y=0.5)

    # get the active phases and support loading netcdf files from disk
    phases = map(str, sorted(set(np.array(eq.Phase.values.ravel(), dtype='U')) - {''}, key=str))
    comps = map(str, sorted(np.array(eq.coords['component'].values, dtype='U'), key=str))
    eq['component'] = np.array(eq['component'], dtype='U')
    eq['Phase'].values = np.array(eq['Phase'].values, dtype='U')

    # Select all two- and three-phase regions
    three_phase_idx = np.nonzero(np.sum(eq.Phase.values != '', axis=-1, dtype=np.int_) == 3)
    two_phase_idx = np.nonzero(np.sum(eq.Phase.values != '', axis=-1, dtype=np.int_) == 2)

    legend_handles, colorlist = legend_generator(phases)

    # For both two and three phase, cast the tuple of indices to an array and flatten
    # If we found two phase regions:
    if two_phase_idx[0].size > 0:
        found_two_phase = eq.Phase.values[two_phase_idx][..., :2]
        # get tieline endpoint compositions
        two_phase_x = eq.X.sel(component=x.species.name).values[two_phase_idx][..., :2]
        # handle special case for potential
        if isinstance(y, v.MoleFraction):
            two_phase_y = eq.X.sel(component=y.species.name).values[two_phase_idx][..., :2]
        else:
            # it's a StateVariable. This must be True
            two_phase_y = np.take(eq[str(y)].values, two_phase_idx[list(str(i) for i in conds.keys()).index(str(y))])
            # because the above gave us a shape of (n,) instead of (n,2) we are going to create it ourselves
            two_phase_y = np.array([two_phase_y, two_phase_y]).swapaxes(0, 1)

        # plot two phase points
        two_phase_plotcolors = np.array(list(map(lambda x: [colorlist[x[0]], colorlist[x[1]]], found_two_phase)), dtype='U')
        ax.scatter(two_phase_x[..., 0], two_phase_y[..., 0], s=3, c=two_phase_plotcolors[:, 0], edgecolors='None', zorder=2, **kwargs)
        ax.scatter(two_phase_x[..., 1], two_phase_y[..., 1], s=3, c=two_phase_plotcolors[:, 1], edgecolors='None', zorder=2, **kwargs)

        if tielines:
            # construct and plot tielines
            two_phase_tielines = np.array([np.concatenate((two_phase_x[..., 0][..., np.newaxis], two_phase_y[..., 0][..., np.newaxis]), axis=-1),
                                           np.concatenate((two_phase_x[..., 1][..., np.newaxis], two_phase_y[..., 1][..., np.newaxis]), axis=-1)])
            two_phase_tielines = np.rollaxis(two_phase_tielines, 1)
            lc = mc.LineCollection(two_phase_tielines, zorder=1, colors=tieline_color, linewidths=[0.5, 0.5])
            ax.add_collection(lc)

    # If we found three phase regions:
    if (three_phase_idx[0].size > 0) and (len(indep_comps) == 2):
        found_three_phase = eq.Phase.values[three_phase_idx][..., :3]
        # get tieline endpoints
        three_phase_x = eq.X.sel(component=x.species.name).values[three_phase_idx][..., :3]
        three_phase_y = eq.X.sel(component=y.species.name).values[three_phase_idx][..., :3]
        # three phase tielines, these are tie triangles and we always plot them
        three_phase_tielines = np.array([np.concatenate((three_phase_x[..., 0][..., np.newaxis], three_phase_y[..., 0][..., np.newaxis]), axis=-1),
                                         np.concatenate((three_phase_x[..., 1][..., np.newaxis], three_phase_y[..., 1][..., np.newaxis]), axis=-1),
                                         np.concatenate((three_phase_x[..., 2][..., np.newaxis], three_phase_y[..., 2][..., np.newaxis]), axis=-1)])
        three_phase_tielines = np.rollaxis(three_phase_tielines, 1)
        three_lc = mc.LineCollection(three_phase_tielines, zorder=1, colors=tie_triangle_color, linewidths=[0.5, 0.5])
        # plot three phase points and tielines
        three_phase_plotcolors = np.array(list(map(lambda x: [colorlist[x[0]], colorlist[x[1]], colorlist[x[2]]], found_three_phase)), dtype='U')
        ax.scatter(three_phase_x[..., 0], three_phase_y[..., 0], s=3, c=three_phase_plotcolors[:, 0], edgecolors='None', zorder=2, **kwargs)
        ax.scatter(three_phase_x[..., 1], three_phase_y[..., 1], s=3, c=three_phase_plotcolors[:, 1], edgecolors='None', zorder=2, **kwargs)
        ax.scatter(three_phase_x[..., 2], three_phase_y[..., 2], s=3, c=three_phase_plotcolors[:, 2], edgecolors='None', zorder=2, **kwargs)
        ax.add_collection(three_lc)

    # position the phase legend and configure plot
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1, 0.5))
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.grid(True)
    plot_title = '-'.join([component.title() for component in sorted(comps) if component != 'VA'])
    ax.set_title(plot_title, fontsize=20)
    ax.set_xlabel(_axis_label(x), labelpad=15, fontsize=20)
    ax.set_ylabel(_axis_label(y), fontsize=20)

    return ax
