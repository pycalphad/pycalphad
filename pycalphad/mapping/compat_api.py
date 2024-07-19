import numpy as np

from pycalphad.mapping import BinaryStrategy, TernaryStrategy, plot_binary, plot_ternary
import pycalphad.mapping.utils as map_utils

def binplot(database, components, phases, conditions, return_strategy=False, plot_kwargs=None, **map_kwargs):
    """
    Calculate the binary isobaric phase diagram.

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
    return_strategy : bool, optional
        Return the BinaryStrategy object in addition to the Axes. Defaults to False.
    map_kwargs : dict, optional
        Additional keyword arguments to BinaryStrategy().
    plot_kwargs : dict, optional
        Keyword arguments to plot_binary()
        Possible key,val pairs in plot_kwargs
            label_nodes : bool
                Whether to plot points for phases on three-phase regions
                Default = False
            tieline_color : tuple
                Color for tielines
                Default = (0,1,0,1)
            tie_triangle_color : tuple
                Color for tie triangles
                Default = (1,0,0,1)

    Returns
    -------
    Axes
        Matplotlib Axes of the phase diagram
    (Axes, BinaryStrategy)
        If return_strategy is True.

    """
    indep_comps = [key for key, value in conditions.items() if not map_utils.is_state_variable(key) and len(np.atleast_1d(value)) > 1]
    indep_pots = [key for key, value in conditions.items() if map_utils.is_state_variable(key) and len(np.atleast_1d(value)) > 1]
    if (len(indep_comps) != 1) or (len(indep_pots) != 1):
        raise ValueError('binplot() requires exactly one composition coordinate and one potential coordinate')

    strategy = BinaryStrategy(database, components, phases, conditions, **map_kwargs)
    strategy.initialize()
    strategy.do_map()

    plot_kwargs = plot_kwargs if plot_kwargs is not None else dict()
    ax = plot_binary(strategy, **plot_kwargs)
    ax.grid(plot_kwargs.get("gridlines", False))

    if return_strategy:
        return ax, strategy
    else:
        return ax


def ternplot(dbf, comps, phases, conds, x=None, y=None, return_strategy=False, map_kwargs=None, **plot_kwargs):
    """
    Calculate the ternary isothermal, isobaric phase diagram.

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
        For ternplot only two changing composition coordinates is supported.
    x : v.MoleFraction
        instance of a pycalphad.variables.composition to plot on the x-axis.
        Must correspond to an independent condition.
    y : v.MoleFraction
        instance of a pycalphad.variables.composition to plot on the y-axis.
        Must correspond to an independent condition.
    return_strategy : bool, optional
        Return the TernaryStrategy object in addition to the Axes. Defaults to False.
    label_nodes : bool (optional)
        Whether to plot points for phases on three-phase regions
        Default = False
    tieline_color : tuple (optional)
        Color for tielines
        Default = (0,1,0,1)
    tie_triangle_color : tuple (optional)
        Color for tie triangles
        Default = (1,0,0,1)
    map_kwargs : dict, optional
        Additional keyword arguments to TernaryStrategy().
    plot_kwargs : dict, optional
        Keyword arguments to plot_ternary().

    Returns
    -------
    Axes
        Matplotlib Axes of the phase diagram
    (Axes, TernaryStrategy)
        If return_strategy is True.

    """
    indep_comps = [key for key, value in conds.items() if not map_utils.is_state_variable(key) and len(np.atleast_1d(value)) > 1]
    indep_pots = [key for key, value in conds.items() if map_utils.is_state_variable(key) and len(np.atleast_1d(value)) > 1]
    if (len(indep_comps) != 2) or (len(indep_pots) != 0):
        raise ValueError('ternplot() requires exactly two composition coordinates')

    map_kwargs = map_kwargs if map_kwargs is not None else dict()
    strategy = TernaryStrategy(dbf, comps, phases, conds, **map_kwargs)
    strategy.initialize()
    strategy.do_map()

    ax = plot_ternary(strategy, x, y, **plot_kwargs)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    if return_strategy:
        return ax, strategy
    else:
        return ax