"""
Rank and conditioning diagnostics for the energy minimizer.

Everything in this module is called only from inside the ``if spec.debugging_output:``
blocks of :func:`pycalphad.core.minimizer.solve_state`, so it has no effect on the
normal solve path. It is kept in pure Python (rather than in ``minimizer.pyx``) so
the report wording and thresholds can be tuned without a Cython rebuild, and so the
analysis can be inspected on its own.

The entry point is :func:`analyze_rank`, which runs one SVD on a matrix and returns
either ``None`` (the matrix is healthy and nothing should be printed), an
:class:`IllConditioned` note, or a :class:`RankDeficiency` report naming the redundant
equations (left null space) and the undetermined unknowns (right null space).
"""

import numpy as np

from pycalphad.core.constants import (ILL_CONDITIONED_RATIO, NULLSPACE_SIGNIFICANCE,
                                      RANK_DEFICIENCY_RTOL)

__all__ = ["RankDeficiency", "IllConditioned", "analyze_rank"]


def _significant_terms(v, labels, threshold=NULLSPACE_SIGNIFICANCE):
    """
    Render a null space vector as a short, readable sum of labeled terms.

    The vector is normalized, given a deterministic sign (the largest-magnitude
    component is made positive), stripped of components at or below ``threshold``,
    and sorted by descending magnitude, e.g.::

        +0.707*stable_phase[TIO_ALPHA] -0.707*stable_phase[TI3O2]

    Structural degeneracies produce sparse, near-canonical null vectors, so this is
    typically two or three terms.
    """
    v = np.asarray(v, dtype=np.float64).ravel()
    norm = np.linalg.norm(v)
    if norm > 0:
        v = v / norm
    magnitudes = np.abs(v)
    # Compare magnitudes at the precision we print them, so that components which are
    # equal in exact arithmetic (the +/-1/sqrt(2) pair a structural degeneracy
    # produces) do not order or sign themselves off a last-bit difference
    ranking = np.round(magnitudes, 3)
    largest = int(np.argmax(ranking))
    # Deterministic sign convention: the largest component is positive. Otherwise the
    # printed signs would flip with the LAPACK driver, since -w is as valid as w.
    if v[largest] < 0:
        v = -v
    significant = np.flatnonzero(magnitudes > threshold)
    if significant.shape[0] == 0:
        # Every component is diffuse; report the largest one rather than nothing
        significant = np.array([largest])
    # Stable sort, so ties fall back to the natural (matrix index) order
    order = significant[np.argsort(-ranking[significant], kind="stable")]
    return " ".join(f"{v[i]:+.3f}*{labels[i]}" for i in order)


class RankDeficiency:
    """A matrix with at least one singular value at or below ``RANK_DEFICIENCY_RTOL * s_max``."""

    def __init__(self, rank, shape, s_max, s_min, ratio, dependent_rows, undetermined):
        self.rank = rank
        self.shape = shape
        self.s_max = s_max
        self.s_min = s_min
        self.ratio = ratio
        #: Left null space, rendered: which equations are redundant
        self.dependent_rows = dependent_rows
        #: Right null space, rendered: which unknowns are undetermined
        self.undetermined = undetermined

    def format(self, title):
        lines = [f"!! RANK DEFICIENT {title}  rank {self.rank}/{min(self.shape)}  "
                 f"(s_min/s_max = {self.ratio:.1e})"]
        for terms in self.dependent_rows:
            lines.append(f"   dependent rows:        {terms}")
        for terms in self.undetermined:
            lines.append(f"   undetermined unknowns: {terms}")
        return "\n".join(lines)

    def __repr__(self):
        return (f"RankDeficiency(rank={self.rank}, shape={self.shape}, "
                f"ratio={self.ratio:.3e})")


class IllConditioned:
    """A full-rank matrix whose smallest singular value is below ``ILL_CONDITIONED_RATIO * s_max``."""

    def __init__(self, ratio, s_max, s_min):
        self.ratio = ratio
        self.s_max = s_max
        self.s_min = s_min

    def format(self, title):
        return (f"!  ill-conditioned {title}  (s_min/s_max = {self.ratio:.1e}, "
                f"s_max = {self.s_max:.3e}, s_min = {self.s_min:.3e})")

    def __repr__(self):
        return f"IllConditioned(ratio={self.ratio:.3e})"


def analyze_rank(A, row_labels, col_labels, rtol=RANK_DEFICIENCY_RTOL,
                 cond_tol=ILL_CONDITIONED_RATIO,
                 want_row_nullspace=True, want_col_nullspace=True):
    """
    Diagnose the rank and conditioning of ``A``.

    Parameters
    ----------
    A : array-like
        The matrix to analyze. It is not modified.
    row_labels : Sequence[str]
        One label per row of ``A``, e.g. ``stable_phase[FCC_A1:cs0]``.
    col_labels : Sequence[str]
        One label per column of ``A``, e.g. ``MU(AL)``.
    rtol : float
        Singular values at or below ``rtol * s_max`` are treated as zero.
    cond_tol : float
        Full-rank matrices with ``s_min/s_max`` below this get an ill-conditioned note.
    want_row_nullspace : bool
        Report the left null space (which equations are redundant).
    want_col_nullspace : bool
        Report the right null space (which unknowns are undetermined). Set this to
        False for a symmetric matrix, where the two null spaces coincide, or where
        the interesting answer is only about the rows.

    Returns
    -------
    RankDeficiency, IllConditioned, or None
        ``None`` means the matrix is healthy and nothing should be printed.
    """
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2 or A.size == 0:
        return None
    if not np.all(np.isfinite(A)):
        # NaN or +/-1e19 sentinels from a failed lstsq/invert_matrix. SVD would raise
        # on these, and the raw values are already visible in the surrounding dump.
        return None
    try:
        U, s, Vt = np.linalg.svd(A)
    except np.linalg.LinAlgError:
        # A diagnostic must never break the run it is diagnosing
        return None
    s_max = float(s[0])
    s_min = float(s[-1])
    if s_max > 0:
        ratio = s_min / s_max
        rank = int(np.count_nonzero(s > rtol * s_max))
    else:
        ratio = 0.0
        rank = 0
    if rank == min(A.shape):
        if ratio < cond_tol:
            return IllConditioned(ratio=ratio, s_max=s_max, s_min=s_min)
        # Healthy: emit nothing, so the diagnostic is signal rather than more firehose
        return None
    dependent_rows = []
    if want_row_nullspace:
        # Left null space: w.T @ A == 0, i.e. sum_i w_i * row_i == 0
        for k in range(rank, A.shape[0]):
            dependent_rows.append(_significant_terms(U[:, k], row_labels))
    undetermined = []
    if want_col_nullspace:
        # Right null space: A @ v == 0, the direction the solution is arbitrary along
        for k in range(rank, A.shape[1]):
            undetermined.append(_significant_terms(Vt[k, :], col_labels))
    return RankDeficiency(rank=rank, shape=A.shape, s_max=s_max, s_min=s_min, ratio=ratio,
                          dependent_rows=dependent_rows, undetermined=undetermined)
