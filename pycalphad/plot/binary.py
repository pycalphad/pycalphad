"""
The binary module provides an interface to map_binary and plot_binary for
plotting binary isobaric phase diagrams.
"""

from .mapping import map_binary, plot_binary

def binplot(dbf, comps, phases, conds, plot_kwargs=None, **map_kwargs):
    """
    Calculate the binary isobaric phase diagram.

    This function is a convenience wrapper around map_binary() and plot_binary()

    Parameters
    ----------
    dbf : Database
        Thermodynamic database containing the relevant parameters.
    comps : list
        Names of components to consider in the calculation.
    phases : list
        Names of phases to consider in the calculation.
    conds : dict
        Maps StateVariables to values and/or iterables of values.
        For binplot only one changing composition and one potential coordinate each is supported.
    eq_kwargs : dict, optional
        Keyword arguments to use in equilibrium() within map_binary(). If
        eq_kwargs is defined in map_kwargs, this argument takes precedence.
    map_kwargs : dict, optional
        Keyword arguments to map_binary().
    plot_kwargs : dict, optional
        Keyword arguments to plot_binary().

    Returns
    -------
    Axes
        Matplotlib Axes of the phase diagram

    Examples
    --------
    None yet.
    """
    plot_kwargs = plot_kwargs if plot_kwargs is not None else dict()
    zpf_boundaries = map_binary(dbf, comps, phases, conds, **map_kwargs)
    ax = plot_binary(zpf_boundaries, **plot_kwargs)
    return ax
