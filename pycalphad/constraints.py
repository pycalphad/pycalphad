"""
The constraints module contains definitions for equilibrium constraints and
their Jacobian.
"""

import pycalphad.variables as v
import numpy as np
from sympy import S

# An index range is a list of (ordered pairs of indices).
def sitefrac_cons(input_x, idx_range):
    """
    Accepts input vector and index range and returns
    site fraction constraint.
    """
    return 1.0 - sum(input_x[idx_range[0]:idx_range[1]])**2

def sitefrac_jac(input_x, idx_range):
    """
    Accepts input vector and index range and returns
    Jacobian of site fraction constraint.
    """
    output_x = np.zeros(len(input_x))
    output_x[idx_range[0]:idx_range[1]] = \
        -2.0*sum(input_x[idx_range[0]:idx_range[1]])
    return output_x

def molefrac_ast(phase, species):
    """
    Return a SymPy object representing the mole fraction as a function of
    site fractions.
    TODO: Assumes all phase constituents are active
    """
    result = S.Zero
    site_ratio_normalization = S.Zero
    # Calculate normalization factor
    for idx, sublattice in enumerate(phase.constituents):
        if 'VA' in set(sublattice):
            site_ratio_normalization += phase.sublattices[idx] * \
                (1.0 - v.SiteFraction(phase.name, idx, 'VA'))
        else:
            site_ratio_normalization += phase.sublattices[idx]
    site_ratios = [c/site_ratio_normalization for c in phase.sublattices]
    # Sum up site fraction contributions from each sublattice
    for idx, sublattice in enumerate(phase.constituents):
        if species in set(sublattice):
            result += site_ratios[idx] * \
                v.SiteFraction(phase.name, idx, species)
    return result
