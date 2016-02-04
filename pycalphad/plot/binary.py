"""
The binary module enables plotting of binary
isobaric phase diagrams.
"""
import numpy as np

from pycalphad import equilibrium
import pycalphad.variables as v
from pycalphad.plot.eqplot import eqplot


def binplot(dbf, comps, phases, conds, eq_kwargs=None, **plot_kwargs):
    """
    Calculate the binary isobaric phase diagram.
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
        For binplot only one changing composition and one potential coordinate each is supported.
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
    indep_comp = [key for key, value in conds.items() if isinstance(key, v.Composition) and len(np.atleast_1d(value)) > 1]
    indep_pot = [key for key, value in conds.items() if ((key == v.T) or (key == v.P)) and len(np.atleast_1d(value)) > 1]
    if (len(indep_comp) != 1) or (len(indep_pot) != 1):
        raise ValueError('binplot() requires exactly one composition and one potential coordinate')
    indep_comp = indep_comp[0]
    indep_pot = indep_pot[0]

    full_eq = equilibrium(dbf, comps, phases, conds, **eq_kwargs)
    return eqplot(full_eq, x=indep_comp, y=indep_pot, **plot_kwargs)
