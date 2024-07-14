import numpy as np

from pycalphad.mapping.primitives import STATEVARS
from pycalphad.mapping.strategy import StepStrategy, BinaryStrategy, TernaryStrategy, IsoplethStrategy
from pycalphad.mapping.plotting import plot_step, plot_binary, plot_ternary, plot_isopleth

def binplot(database, components, phases, conditions, return_strategy = False, plot_kwargs=None, **map_kwargs):
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
    indep_comps = [key for key, value in conditions.items() if key not in STATEVARS and len(np.atleast_1d(value)) > 1]
    indep_pots = [key for key, value in conditions.items() if key in STATEVARS and len(np.atleast_1d(value)) > 1]
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


def ternplot(dbf, comps, phases, conds, x=None, y=None, return_strategy = False, map_kwargs=None, **plot_kwargs):
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
    indep_comps = [key for key, value in conds.items() if key not in STATEVARS and len(np.atleast_1d(value)) > 1]
    indep_pots = [key for key, value in conds.items() if key in STATEVARS and len(np.atleast_1d(value)) > 1]
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

def isoplethplot(database, components, phases, conditions, return_strategy = False, plot_kwargs=None, **map_kwargs):
    """
    Calculates an isopleth phase diagram.
    For now, we'll define isopleths as having 1 potential condition and 1 non-potential condition

    TODO: if we can confirm that isopleths work with 2 non-potential conditions, then remove the check at the beginning

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
    indep_comps = [key for key, value in conditions.items() if key not in STATEVARS and len(np.atleast_1d(value)) > 1]
    indep_pots = [key for key, value in conditions.items() if key in STATEVARS and len(np.atleast_1d(value)) > 1]
    if (len(indep_comps) != 1) or (len(indep_pots) != 1):
        raise ValueError('isoplethplot() requires exactly one composition coordinate and one potential coordinate')
    
    strategy = IsoplethStrategy(database, components, phases, conditions, **map_kwargs)
    strategy.initialize()
    strategy.do_map()

    plot_kwargs = plot_kwargs if plot_kwargs is not None else dict()
    ax = plot_isopleth(strategy, **plot_kwargs)
    ax.grid(plot_kwargs.get("gridlines", False))

    if return_strategy:
        return ax, strategy
    else:
        return ax

def stepplot(database, components, phases, conditions, return_strategy = False, plot_kwargs=None, **map_kwargs):
    """
    Calculates and plot a step diagram
        Default axes will be axis variable vs. phase fraction

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
    indep_vars = [key for key, value in conditions.items() if len(np.atleast_1d(value)) > 1]
    if len(indep_vars) != 1:
        raise ValueError('stepplot() requires exactly one coordinate')

    strategy = StepStrategy(database, components, phases, conditions, **map_kwargs)
    strategy.initialize()
    strategy.do_map()

    plot_kwargs = plot_kwargs if plot_kwargs is not None else dict()
    ax = plot_step(strategy, **plot_kwargs)
    ax.grid(plot_kwargs.get("gridlines", False))

    if return_strategy:
        return ax, strategy
    else:
        return ax