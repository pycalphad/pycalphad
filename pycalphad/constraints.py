"""
The constraints module contains definitions for equilibrium constraints and
their Jacobian.
"""

import pycalphad.variables as v
import numpy as np
from sympy import S

def msum(iterable):
    "Full precision summation using multiple floats for intermediate values"
    # Rounded x+y stored in hi with the round-off stored in lo.  Together
    # hi+lo are exactly equal to x+y.  The inner loop applies hi/lo summation
    # to each partial so that the list of partial sums remains exact.
    # Depends on IEEE-754 arithmetic guarantees.  See proof of correctness at:
    # www-2.cs.cmu.edu/afs/cs/project/quake/public/papers/robust-arithmetic.ps

    partials = []               # sorted, non-overlapping partial sums
    for x in iterable:
        i = 0
        for y in partials:
            if abs(x) < abs(y):
                x, y = y, x
            hi = x + y
            lo = y - (hi - x)
            if lo:
                partials[i] = lo
                i += 1
            x = hi
        partials[i:] = [x]
    return sum(partials, 0.0)

# An index range is a list of (ordered pairs of indices).
def sitefrac_cons(input_x, idx_range):
    """
    Accepts input vector and index range and returns
    site fraction constraint.
    """
    print('sitefrac cons '+str(idx_range)+': '+str(sum(input_x[idx_range[0]:idx_range[1]]) - 1))
    return sum(input_x[idx_range[0]:idx_range[1]]) - 1.0

def sitefrac_jac(input_x, idx_range):
    """
    Accepts input vector and index range and returns
    Jacobian of site fraction constraint.
    """
    output_x = np.zeros(len(input_x))
    output_x[idx_range[0]:idx_range[1]] = 1.0
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
