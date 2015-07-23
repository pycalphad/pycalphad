"""
The constraints module contains definitions for equilibrium constraints and
their Jacobian.
"""

import pycalphad.variables as v
from sympy import S, Add

def mole_fraction(phase, active_comps, species):
    """
    Return a SymPy object representing the mole fraction as a function of
    site fractions.

    Parameters
    ----------
    phase : Phase
        Phase object corresponding to the phase of interest.
    active_comps : list of str
        Names of components to consider.
    species : str
        Names of species to consider.

    Returns
    -------
    SymPy object representing the mole fraction.

    Examples
    --------
    >>> dbf = Database('alfe_sei.TDB')
    >>> mole_fraction(dbf.phases['FCC_A1'], ['AL', 'FE', 'VA'], 'AL')
    """
    result = S.Zero
    site_ratio_normalization = S.Zero
    # Calculate normalization factor
    for idx, sublattice in enumerate(phase.constituents):
        active = set(sublattice).intersection(set(active_comps))
        if 'VA' in active:
            site_ratio_normalization += phase.sublattices[idx] * \
                (1.0 - v.SiteFraction(phase.name, idx, 'VA'))
        else:
            site_ratio_normalization += phase.sublattices[idx]
    site_ratios = [c/site_ratio_normalization for c in phase.sublattices]
    # Sum up site fraction contributions from each sublattice
    for idx, sublattice in enumerate(phase.constituents):
        active = set(sublattice).intersection(set(active_comps))
        if species in active:
            result += site_ratios[idx] * \
                v.SiteFraction(phase.name, idx, species)
    return result

