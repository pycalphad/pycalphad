from pycalphad import Database, variables as v
from pycalphad_mapping.mapper import Mapper
from pycalphad_mapping.starting_points import automatic_starting_points_from_axis_limits
import matplotlib.pyplot as plt
from pycalphad_mapping.plotting import plot_map
import numpy as np
from pycalphad.core.utils import unpack_components, filter_phases
from pycalphad.plot.utils import phase_legend
from pycalphad.plot import triangular  # register triangular projection


def binplot(database, components, phases, conditions, plot_kwargs=None, **map_kwargs):
    """
    Calculate the binary isobaric phase diagram.

    This function is a convenience wrapper around map_binary() and plot_boundaries()

    Parameters
    ----------
    database : Database
        Thermodynamic database containing the relevant parameters.
    components : Sequence[str]
        Names of components to consider in the calculation.
    phases : Sequence[str]
        Names of phases to consider in the calculation.
    conditions : Mapping[v.StateVariable, Union[float, Tuple[float, float, float]]]
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
    indep_comps = [key for key, value in conditions.items() if isinstance(key, v.MoleFraction) and len(np.atleast_1d(value)) > 1]
    indep_pots = [key for key, value in conditions.items() if (type(key) is v.StateVariable) and len(np.atleast_1d(value)) > 1]
    if (len(indep_comps) != 1) or (len(indep_pots) != 1):
        raise ValueError('binplot() requires exactly one composition coordinate and one potential coordinate')
    # TODO: try to give full backwards compatible support for plot_kwargs and map_kwargs
    # remaining plot_kwargs from pycalphad.plot.binary.plot.plot_boundaries:
    # tieline_color=(0, 1, 0, 1)
    # remaining map_kwargs from pycalphad.plot.binary.map.map_binary:
    # calc_kwargs=None
    plot_kwargs = plot_kwargs if plot_kwargs is not None else dict()

    # TODO: filtering phases should be done by the mapper
    phases = filter_phases(database, unpack_components(database, components), phases)

    eq_kwargs = map_kwargs.get("eq_kwargs", {})  # only used for start points, not in the mapping currently
    mapper = Mapper(database, components, phases, conditions)
    start_points, start_dir = automatic_starting_points_from_axis_limits(database, components, phases, conditions, **eq_kwargs)
    mapper.strategy.add_starting_points_with_axes(start_points, start_dir)
    mapper.do_map()

    ax = plot_kwargs.get("ax")
    if ax is None:
        ax = plt.figure().gca()
    legend_generator = plot_kwargs.get('legend_generator', phase_legend)
    plot_map(mapper, tielines=plot_kwargs.get("tielines", True), ax=ax, legend_generator=legend_generator)
    ax.grid(plot_kwargs.get("gridlines", False))

    return ax


def ternplot(dbf, comps, phases, conds, x=None, y=None, eq_kwargs=None, **plot_kwargs):
    """
    Calculate the ternary isothermal, isobaric phase diagram.
    This function is a convenience wrapper around equilibrium() and eqplot().

    Parameters
    ----------
    dbf : Database
        Thermodynamic database containing the relevant parameters.
    comps : Sequence[str]
        Names of components to consider in the calculation.
    phases : Sequence[str]
        Names of phases to consider in the calculation.
    conds : Mapping[v.StateVariable, Union[float, Tuple[float, float, float]]]
        Maps StateVariables to values and/or iterables of values.
        For ternplot only one changing composition and one potential coordinate each is supported.
    x : v.MoleFraction
        instance of a pycalphad.variables.composition to plot on the x-axis.
        Must correspond to an independent condition.
    y : v.MoleFraction
        instance of a pycalphad.variables.composition to plot on the y-axis.
        Must correspond to an independent condition.
    eq_kwargs : optional
        Keyword arguments to equilibrium().
    plot_kwargs : optional
        Keyword arguments to eqplot().

    Returns
    -------
    A phase diagram as a figure.

    Examples
    --------
    None yet.
    """
    # remaining plot_kwargs from pycalphad.plot.eqplot
    # x=None, y=None, z=None, tieline_color=(0, 1, 0, 1), tie_triangle_color=(1, 0, 0, 1), **kwargs
    # kwargs passed ot ax.scatter
    indep_comps = [key for key, value in conds.items() if isinstance(key, v.MoleFraction) and len(np.atleast_1d(value)) > 1]
    indep_pots = [key for key, value in conds.items() if (type(key) is v.StateVariable) and len(np.atleast_1d(value)) > 1]
    if (len(indep_comps) != 2) or (len(indep_pots) != 0):
        raise ValueError('ternplot() requires exactly two composition coordinates')

    phases = filter_phases(dbf, unpack_components(dbf, comps), phases)
    mapper = Mapper(dbf, comps, phases, conds)
    start_points, start_dir = automatic_starting_points_from_axis_limits(dbf, comps, phases, conds)
    mapper.strategy.add_starting_points_with_axes(start_points, start_dir)
    mapper.do_map()

    ax = plot_kwargs.get("ax")
    if ax is None:
        fig, ax = plt.subplots(subplot_kw={'projection': "triangular"})

    legend_generator = plot_kwargs.get('legend_generator', phase_legend)
    plot_map(mapper, tielines=plot_kwargs.get("tielines", True), ax=ax, legend_generator=legend_generator)

    return ax
