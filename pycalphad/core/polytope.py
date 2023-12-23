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


def hitandrun(A, b, x0):
    """Generator for uniform sampling from the convex polytope Ax <= b using the
    Hit & Run algorithm described in [Smith1984].

    Parameters
    ----------
    A : 2d-array of shape (n_constraints, dimension)
        Left-hand-side of Ax <= b.
    b : 1d-array of shape (n_constraints)
        Right-hand-side of Ax <= b.
    x0 : 1d-array of shape (dimension)
        Initial point that satisfies A x0 <= b.

    Yields
    -------
    1d-array of shape (dimension)
        Point sampled from the polytope.
    """
    check_Ab(A, b)
    assert A.shape[1] == len(x0)

    x = x0
    rng = np.random.RandomState(1769)

    with np.errstate(divide='ignore', invalid='ignore'):
        while True:
            # sample random direction from unit hypersphere
            direction = rng.randn(A.shape[1])
            direction /= np.linalg.norm(direction)

            # distances to each face from the current point in the sampled direction
            D = (b - x @ A.T) / (direction @ A.T)

            # distance to the closest face in and opposite to direction
            lo = np.max(D[D < 1e-13])
            hi = np.min(D[D > -1e-13])
            # make random step
            x += rng.uniform(lo, hi) * direction
            yield x


def sample(n_points, lower, upper, A1=None, b1=None, A2=None, b2=None, thin=1):
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
    thin : int, optional
        The thinning factor of the generated samples. A thinning of 10 means a sample
        is taken every 10 steps.

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
        X = np.atleast_2d(np.linalg.solve(A2, b2))
        return X

    # project to the affine subspace of the equality constraints
    At = A1 @ N
    bt = b1 - A1 @ xp

    x0 = chebyshev_center(At, bt)
    sampler = hitandrun(At, bt, x0)

    X = np.empty((n_points, At.shape[1]))
    for i in range(n_points):
        for _ in range(thin - 1):
            next(sampler)
        X[i] = next(sampler)

    # project back
    X = X @ N.T + xp
    X = np.clip(X, lower, upper)
    return X