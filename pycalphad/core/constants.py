"""
The constants module contains some numerical constants for use
in the module.
Note that modifying these may yield unpredictable results.
"""
# Force zero values to this amount, for numerical stability
MIN_SITE_FRACTION = 1e-12
MIN_PHASE_FRACTION = 1e-6
# Phases with mole fractions less than COMP_DIFFERENCE_TOL apart (by Chebyshev distance) are considered "the same" for
# the purposes of CompositionSet addition and removal during energy minimization.
COMP_DIFFERENCE_TOL = 1e-4

# 'infinity' for numerical purposes
BIGNUM = 1e60

# Maximum residual driving force (J/mol-atom) allowed for convergence
MAX_SOLVE_DRIVING_FORCE = 1e-4
# Maximum number of multi-phase solver iterations
MAX_SOLVE_ITERATIONS = 300
# Minimum energy (J/mol-atom) difference between iterations before stopping solver
MIN_SOLVE_ENERGY_PROGRESS = 1e-3
# Maximum absolute value of a Lagrange multiplier before it's recomputed with an alternative method
MAX_ABS_LAGRANGE_MULTIPLIER = 1e16
