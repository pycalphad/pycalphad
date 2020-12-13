"""
The plot module handles plotting ZPF boundaries and provides a user-facing
function `binplot` for users to plot binary phase diagrams with a similar API
as `equilibrium`.
"""

import pycalphad.variables as v
import matplotlib.pyplot as plt
from pycalphad.plot.eqplot import _axis_label
from pycalphad.plot.utils import phase_legend
from .map import map_binary


def plot_boundaries(zpf_boundary_sets, tielines=True, tieline_color=(0, 1, 0, 1), scatter=True, legend_generator=phase_legend, ax=None, gridlines=False):
    """
    Plot a set of ZPFBoundarySets

    Parameters
    ----------
    zpf_boundary_sets : pycalphad.mapping.zpf_boundary_sets.ZPFBoundarySets
    tielines : optional, bool
        Whether the plot the tielines (defaults to True)
    tieline_color: color
        A valid matplotlib color, such as a named color string, hex RGB
        string, or a tuple of RGBA components to set the color of the two
        phase region tielines. The default is an RGBA tuple for green:
        (0, 1, 0, 1).
    scatter : optional, bool
        Whether to use scatter plot the phase boundaries (True, the default) or
        to connect lines in the same two phase region by lines. Note that lines
        may appear broken when the set of phases change, even if the boundary
        does not change.
    legend_generator : Callable
        A function that will be called with the list of phases and will
        return legend labels and colors for each phase. By default
        pycalphad.plot.utils.phase_legend is used
    ax : plt.Axes
        Matplotlib axes to plot to. If none are pasesed, a new figure will be
        created.
    gridlines : False
        Whether to plot the grid lines in the plot. Defaults to False.

    Returns
    -------
    plt.Axes

    """
    if ax is None:
        ax = plt.figure().gca()
    if scatter:
        scatter_dict, tieline_coll, legend_handles = zpf_boundary_sets.get_scatter_plot_boundaries(tieline_color=tieline_color, legend_generator=legend_generator)
        ax.scatter(scatter_dict['x'], scatter_dict['y'], c=scatter_dict['c'], edgecolor='None', s=3, zorder=2)
    else:
        boundary_collection, tieline_coll, legend_handles = zpf_boundary_sets.get_line_plot_boundaries(tieline_color=tieline_color, legend_generator=legend_generator)
        ax.add_collection(boundary_collection)
    if tielines:
        ax.add_collection(tieline_coll)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1, 0.5))
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.grid(True)
    plot_title = '-'.join([component for component in sorted(zpf_boundary_sets.components) if component != 'VA'])
    ax.set_title(plot_title, fontsize=20)
    ax.set_xlabel(_axis_label(zpf_boundary_sets.indep_comp_cond), labelpad=15, fontsize=20)
    ax.set_ylabel(_axis_label(v.T), fontsize=20)
    ax.set_xlim(0, 1)
    # autoscale needs to be used in case boundaries are plotted as lines because
    # only plotting line collections will not rescale the axes
    ax.autoscale(axis='y')
    ax.grid(gridlines)
    return ax


def binplot(database, components, phases, conditions, plot_kwargs=None, **map_kwargs):
    """
    Calculate the binary isobaric phase diagram.

    This function is a convenience wrapper around map_binary() and plot_boundaries()

    Parameters
    ----------
    database : Database
        Thermodynamic database containing the relevant parameters.
    components : list
        Names of components to consider in the calculation.
    phases : list
        Names of phases to consider in the calculation.
    conditions : dict
        Maps StateVariables to values and/or iterables of values.
        For binplot only one changing composition and one potential coordinate each is supported.
    eq_kwargs : dict, optional
        Keyword arguments to use in equilibrium() within map_binary(). If
        eq_kwargs is defined in map_kwargs, this argument takes precedence.
    map_kwargs : dict, optional
        Additional keyword arguments to map_binary().
    plot_kwargs : dict, optional
        Keyword arguments to plot_boundaries().

    Returns
    -------
    Axes
        Matplotlib Axes of the phase diagram

    Examples
    --------
    None yet.
    """
    plot_kwargs = plot_kwargs if plot_kwargs is not None else dict()
    zpf_boundaries = map_binary(database, components, phases, conditions, **map_kwargs)
    ax = plot_boundaries(zpf_boundaries, **plot_kwargs)
    return ax
