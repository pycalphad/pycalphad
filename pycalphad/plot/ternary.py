"""
The ternary module enables plotting of ternary isobaric phase diagrams.
"""
import numpy as np

from pycalphad import equilibrium
import pycalphad.variables as v
from pycalphad.plot.eqplot import eqplot


def ternplot(dbf, comps, phases, conds, x=None, y=None, eq_kwargs=None, **plot_kwargs):
    """
    Calculate the ternary isothermal, isobaric phase diagram.
    This function is a convenience wrapper around equilibrium() and eqplot().

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
    eq_kwargs = eq_kwargs if eq_kwargs is not None else dict()
    indep_comps = [key for key, value in conds.items() if isinstance(key, v.MoleFraction) and len(np.atleast_1d(value)) > 1]
    indep_pots = [key for key, value in conds.items() if (type(key) is v.StateVariable) and len(np.atleast_1d(value)) > 1]
    if (len(indep_comps) != 2) or (len(indep_pots) != 0):
        raise ValueError('ternplot() requires exactly two composition coordinates')
    full_eq = equilibrium(dbf, comps, phases, conds, **eq_kwargs)
    # TODO: handle x and y as strings with #87
    x = x if x in indep_comps else indep_comps[0]
    y = y if y in indep_comps else indep_comps[1]
    return eqplot(full_eq, x=x, y=y, **plot_kwargs)
