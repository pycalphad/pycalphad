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

# Singular values at or below this fraction of the largest are treated as zero
# when diagnosing rank deficiency under Solver(debugging_output=True).
# Deliberately looser than the rcond=1e-16 that lstsq passes to dgelsd, which
# only catches exact deficiency.
RANK_DEFICIENCY_RTOL = 1e-12
# Full-rank matrices with s_min/s_max below this get a softer "ill-conditioned" note
ILL_CONDITIONED_RATIO = 1e-8
# Null space components smaller than this (after normalization) are omitted from diagnostics
NULLSPACE_SIGNIFICANCE = 0.1

# Prevent excessive sampling for very complex phase models
# This avoids running out of RAM
MAX_ENDMEMBER_PAIRS = 5000 # ~100 endmembers
MAX_EXTRA_POINTS = 90000
