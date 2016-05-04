"""
The constants module contains some numerical constants for use
in the module.
Note that modifying these may yield unpredictable results.
"""
# Force zero values to this amount, for numerical stability
MIN_SITE_FRACTION = 1e-12
# For each phase pair with composition difference below tolerance, eliminate phase with largest index
COMP_DIFFERENCE_TOL = 0.01