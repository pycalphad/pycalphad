"""
The constants module contains some numerical constants for use
in the module.
Note that modifying these may yield unpredictable results.
"""
# Force zero values to this amount, for numerical stability
MIN_SITE_FRACTION = 1e-14
MIN_PHASE_FRACTION = 1e-6
# Phases with mole fractions less than COMP_DIFFERENCE_TOL apart (by Chebyshev distance) are considered "the same" for
# the purposes of CompositionSet addition and removal during energy minimization.
COMP_DIFFERENCE_TOL = 1e-4

# Constraint scaling factors, for numerical stability
INTERNAL_CONSTRAINT_SCALING = 1.0
