"""
The minimize module handles helper routines for equilibrium calculation.
"""
from __future__ import division
# monkey patch for theano/sympy integration
#import theano.tensor as tt
import sympy
import scipy.spatial.distance
#import sympy.printing.theanocode
#from sympy.printing.theanocode import theano_function
#sympy.printing.theanocode.mapping[sympy.And] = tt.and_

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
    model, SymPy object
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
    #if mode == 'theano':
    #    #energy = \
    #    #    theano_function(variables, [model], on_unused_input='ignore')
    #elif mode == 'theano-debug':
    #    #energy = \
    #    #    theano_function(variables, [model], on_unused_input='warn',
    #    #                    mode='DebugMode')
    if mode == 'numpy':
        energy = lambdify(tuple(variables), model, dummify=True,
                          modules='numpy')
    elif mode == 'numexpr':
        energy = lambdify(tuple(variables), model, dummify=True,
                  modules='numexpr')
    elif mode == 'sympy':
        energy = lambda *vs: model.subs(zip(variables, vs)).evalf()
    else:
        raise ValueError('Unsupported function mode: '+mode)

    return energy

def check_degenerate_phases(phase_compositions, mindist=0.1):
    """
    Because the global minimization procedure returns a simplex as an
    output, our starting point will always assume the maximum number of
    phases. In many cases, one or more of these phases will be redundant,
    i.e., the simplex is narrow. These redundant or degenerate phases can
    be eliminated from the computation.

    Here we perform edge-wise comparisons of all the simplex vertices.
    Vertices which are from the same phase and "sufficiently" close to
    each other in composition space are redundant, and one of them is
    eliminated from the computation.

    This function accepts a DataFrame of the estimated phase compositions
    and returns the indices of the "independent" phases in the DataFrame.
    """
    output_vertices = set(range(len(phase_compositions)))
    edges = list(itertools.combinations(output_vertices, 2))
    sitefrac_columns = \
        [c for c in phase_compositions.columns.values \
            if str(c).startswith('Y')]
    for edge in edges:
        # check if both end-points are still in output_vertices
        # if not, we should skip this edge
        if not set(edge).issubset(output_vertices):
            continue
        first_vertex = phase_compositions.iloc[edge[0]]
        second_vertex = phase_compositions.iloc[edge[1]]
        if first_vertex.loc['Phase'] != second_vertex.loc['Phase']:
                # phases along this edge do not match; leave them alone
                continue
        # phases match; check the distance between their respective
        # site fractions; if below the threshold, eliminate one of them
        first_coords = first_vertex.loc[sitefrac_columns].fillna(0)
        second_coords = second_vertex.loc[sitefrac_columns].fillna(0)
        edge_length = \
            scipy.spatial.distance.euclidean(first_coords, second_coords)
        if edge_length < mindist and len(output_vertices) > 1:
            output_vertices.discard(edge[1])
    return list(output_vertices)
