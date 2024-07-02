"""
This module provides functions to uniformly sample points subject to a system of linear
inequality constraints, :math:`Ax <= b` (convex polytope), and linear equality
constraints, :math:`Ax = b` (affine projection).

A comparison of MCMC algorithms to generate uniform samples over a convex polytope is
given in [Chen2018]_. Here, we use the Hit & Run algorithm described in [Smith1984]_.
The R-package `hitandrun`_ provides similar functionality to this module.

Based on https://github.com/DavidWalz/polytope-sampling
Used under the terms of the MIT license. License information can be found in the pycalphad LICENSE.txt.

References
----------
.. [Chen2018] Chen Y., Dwivedi, R., Wainwright, M., Yu B. (2018) Fast MCMC Sampling
    Algorithms on Polytopes. JMLR, 19(55):1−86
    https://arxiv.org/abs/1710.08165
.. [Smith1984] Smith, R. (1984). Efficient Monte Carlo Procedures for Generating
    Points Uniformly Distributed Over Bounded Regions. Operations Research,
    32(6), 1296-1308.
    www.jstor.org/stable/170949
.. _`hitandrun`: https://cran.r-project.org/web/packages/hitandrun/index.html
"""
import numpy as np
import scipy.linalg
import scipy.optimize


def check_Ab(A, b):
    """Check if matrix equation Ax=b is well defined.

    Parameters
    ----------
    A : 2d-array of shape (n_constraints, dimension)
        Left-hand-side of Ax <= b.
    b : 1d-array of shape (n_constraints)
        Right-hand-side of Ax <= b.

    """
    assert A.ndim == 2
    assert b.ndim == 1
    assert A.shape[0] == b.shape[0]


def chebyshev_center(A, b):
    """Find the center of the polytope Ax <= b.

    Parameters
    ----------
    A : 2d-array of shape (n_constraints, dimension)
        Left-hand-side of Ax <= b.
    b : 1d-array of shape (n_constraints)
        Right-hand-side of Ax <= b.

    Returns
    -------
    1d-array of shape (dimension)
        Chebyshev center of the polytope
    """
    res = scipy.optimize.linprog(
        np.r_[np.zeros(A.shape[1]), -1],
        A_ub=np.hstack([A, np.linalg.norm(A, axis=1, keepdims=True)]),
        b_ub=b,
        bounds=(None, None),
    )
    if not res.success:
        raise Exception("Unable to find Chebyshev center")
    return res.x[:-1]


def constraints_from_bounds(lower, upper):
    """Construct the inequality constraints Ax <= b that correspond to the given
    lower and upper bounds.

    Parameters
    ----------
    lower : array-like
        lower bound in each dimension
    upper : array-like
        upper bound in each dimension

    Returns
    -------
    A: 2d-array of shape (2 * dimension, dimension)
        Left-hand-side of Ax <= b.
    b: 1d-array of shape (2 * dimension)
        Right-hand-side of Ax <= b.
    """
    n = len(lower)
    A = np.row_stack([-np.eye(n), np.eye(n)])
    b = np.r_[-np.array(lower), np.array(upper)]
    return A, b


def affine_subspace(A, b):
    """Compute a basis of the nullspace of A, and a particular solution to Ax = b.
    This allows to to construct arbitrary solutions as the sum of any vector in the
    nullspace, plus the particular solution.

    Parameters
    ----------
    A : 2d-array of shape (n_constraints, dimension)
        Left-hand-side of Ax <= b.
    b : 1d-array of shape (n_constraints)
        Right-hand-side of Ax <= b.

    Returns
    -------
    N: 2d-array of shape (dimension, dimension)
        Orthonormal basis of the nullspace of A.
    xp: 1d-array of shape (dimension)
        Particular solution to Ax = b.
    """
    N = scipy.linalg.null_space(A)
    xp = np.linalg.pinv(A) @ b
    return N, xp

def sample(n_points, lower, upper, A1=None, b1=None, A2=None, b2=None):
    """Sample a number of points from a convex polytope A1 x <= b1 using the Hit & Run
    algorithm.

    Lower and upper bounds need to be provided to ensure that the polytope is bounded.
    Equality constraints A2 x = b2 may be optionally provided.

    Parameters
    ----------
    n_points : int
        Number of samples to generate.
    lower: 1d-array of shape (dimension)
        Lower bound in each dimension. If not wanted set to -np.inf.
    upper: 1d-array of shape (dimension)
        Upper bound in each dimension. If not wanted set to np.inf.
    A1 : 2d-array of shape (n_constraints, dimension)
        Left-hand-side of A1 x <= b1.
    b1 : 1d-array of shape (n_constraints)
        Right-hand-side of A1 x <= b1.
    A2 : 2d-array of shape (n_constraints, dimensions), optional
        Left-hand-side of A2 x = b2.
    b2 : 1d-array of shape (n_constraints), optional
        Right-hand-side of A2 x = b2.

    Returns
    -------
    2d-array of shape (n_points)
        Points sampled from the polytope.
    """
    A, b = constraints_from_bounds(lower, upper)
    if (A1 is not None) and (b1 is not None):
        A1 = np.r_[A, A1]
        b1 = np.r_[b, b1]
    else:
        A1, b1 = A, b

    if (A2 is not None) and (b2 is not None):
        check_Ab(A2, b2)
        N, xp = affine_subspace(A2, b2)
    else:
        N = np.eye(A1.shape[1])
        xp = np.zeros(A1.shape[1])

    if N.shape[1] == 0:
        # zero-dimensional polytope, return unique solution
        # Use lstsq instead of solve, to allow for redundant constraints (non-square constraint matrix)
        solution = np.linalg.lstsq(A2, b2, rcond=None)
        X = np.atleast_2d(solution[0])
        # Check residuals to ensure system was fully determined, or constraints were redundant
        if solution[1].size > 0:
            residual = float(solution[1])
            if residual > 1e-10:
                # Starting point is not feasible
                return np.empty((0, A1.shape[1]))
        return X

    # project to the affine subspace of the equality constraints
    At = A1 @ N
    bt = b1 - A1 @ xp

    try:
        x0 = chebyshev_center(At, bt)
    except:
        # Unable to find center
        return np.empty((0, A1.shape[1]))

    test_point = x0[np.newaxis, :] @ N.T + xp
    if np.any(test_point < lower-1e-10) or np.any(test_point > upper+1e-10):
        # Starting point is not feasible
        return np.empty((0, A1.shape[1]))

    X = np.empty((n_points, At.shape[1]))
    x = x0
    rng = np.random.RandomState(1769)
    with np.errstate(divide='ignore', invalid='ignore'):
        directions = rng.randn(n_points, At.shape[1])
        directions /= np.linalg.norm(directions, axis=0)
        for i in range(n_points):
            # sample random direction from unit hypersphere
            direction = directions[i]

            # distances to each face from the current point in the sampled direction
            D = (bt - x @ At.T) / (direction @ At.T)

            # distance to the closest face in and opposite to direction
            lo = max(D[D < 1e-10])
            hi = min(D[D > -1e-10])
            print('DEBUG', lo, hi)
            if hi < lo:
                # Amount of 'wiggle room' is down in the numerical noise
                lo = 0.0
                hi = 0.0
            # make random step
            x += rng.uniform(lo, hi) * direction
            X[i] = x

    # project back
    X = X @ N.T + xp
    X = np.clip(X, lower, upper)
    return X