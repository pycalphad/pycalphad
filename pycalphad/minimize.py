"""
The minimize module handles helper routines for equilibrium calculation.
"""
from __future__ import division
# monkey patch for theano/sympy integration
import theano.tensor as tt
import sympy
import sympy.printing.theanocode
from sympy.printing.theanocode import theano_function
sympy.printing.theanocode.mapping[sympy.And] = tt.and_

from sympy.utilities.lambdify import lambdify
import numpy as np
import itertools
try:
    set
except NameError:
    from sets import Set as set #pylint: disable=W0622

def point_sample(comp_count, size=10):
    """
    Sample 'size' ** len('comp_count') points in composition space
    for the sublattice configuration specified by 'comp_count'.
    Points are sampled psuedo-randomly from a symmetric Dirichlet distribution.
    Note: For sublattices with only one component, only one point will be
        returned, regardless of 'size'.

    Parameters
    ----------
    comp_count : list
        Number of components in each sublattice.
    size : int
        Number of points to sample _per sublattice_.

    Returns
    -------
    ndarray of generated points satisfying the mass balance.

    Examples
    --------
    >>> comps = [8,1] # 8 components in sublattice 1; only 1 in sublattice 2
    >>> pts = point_sample(comps, size=20)
    """
    subl_points = []
    for ctx in comp_count:
        if ctx > 1:
            pts = np.random.dirichlet(tuple(np.ones(ctx)), size)
            subl_points.append(pts)
        elif ctx == 1:
            # only 1 component; no degrees of freedom
            subl_points.append(np.atleast_2d(1))
        else:
            raise ValueError('Number of components must be >= 1')

    # Calculate Cartesian product over all sublattices
    #pylint: disable=W0142
    prod = itertools.product(*subl_points)
    # join together the coordinates in each sublattice
    result = [np.concatenate(x) for x in prod]
    return np.asarray(result)

def make_callable(model, variables, mode='numpy'):
    """
    Take a SymPy object and create a callable function.

    Parameters
    ----------
    model, SymPy object or iterable of SymPy objects
        Abstract representation of function
    variables, list
        Input variables, ordered in the way the return function will expect
    mode, ['numpy', 'theano', 'theano-debug', 'sympy'], optional
        Method to use when 'compiling' the function. SymPy mode is
        slow and should only be used for debugging. If Theano is installed,
        then it can offer speed-ups when calling the energy function many
        times, at a one-time cost of compiling in advance.

    Returns
    -------
    Function that takes arguments in the same order as 'variables'
    and returns the energy according to 'model.'

    Examples
    --------
    None yet.
    """
    energy = None
    if mode == 'theano':
        energy = \
            theano_function(variables, [model], on_unused_input='ignore')
    elif mode == 'theano-debug':
        energy = \
            theano_function(variables, [model], on_unused_input='warn',
                            mode='DebugMode')
    elif mode == 'numpy':
        if hasattr(model, "__iter__"):
            # is iterable, apply lambdify to each element
            energy = [lambdify(tuple(variables), elem, dummify=True,
                               modules='numpy') for elem in model]
        else:
            energy = lambdify(tuple(variables), model, dummify=True,
                              modules='numpy')
    elif mode == 'sympy':
        energy = lambda *vs: model.subs(zip(variables, vs)).evalf()
    else:
        raise ValueError('Unsupported function mode: '+mode)

    return energy
