"""
The minimize module handles helper routines for equilibrium calculation.
"""
from __future__ import division
import scipy.spatial.distance
from sympy.utilities import default_sort_key
from sympy.utilities.lambdify import lambdify
from sympy.printing.lambdarepr import LambdaPrinter, NumExprPrinter
from sympy import Piecewise
import numpy as np
import itertools
try:
    set
except NameError:
    from sets import Set as set #pylint: disable=W0622

class NumPyPrinter(LambdaPrinter):
    """
    Special numpy lambdify printer which handles vectorized
    piecewise functions.
    """
    def _print_seq(self, seq, delimiter=', '):
        # simplified _print_seq taken from pretty.py
        s = [self._print(item) for item in seq]
        if s:
            return delimiter.join(s)
        else:
            return ""

    def _print_Piecewise(self, expr):
        expr_list = []
        cond_list = []
        for arg in expr.args:
            expr_list.append(self._print(arg.expr))
            cond_list.append(self._print(arg.cond))
        exprs = '['+','.join(expr_list)+']'
        conds = '['+','.join(cond_list)+']'
        return 'select('+conds+', '+exprs+')'

    def _print_And(self, expr):
        return self._print_Function(expr)

    def _print_Or(self, expr):
        return self._print_Function(expr)

    def _print_Function(self, e):
        return "%s(%s)" % (e.func.__name__, self._print_seq(e.args))

class SpecialNumExprPrinter(NumExprPrinter):
    "numexpr printing for vectorized piecewise functions"
    def _print_And(self, expr):
        result = ['(']
        for arg in sorted(expr.args, key=default_sort_key):
            result.extend(['(', self._print(arg), ')'])
            result.append(' & ')
        result = result[:-1]
        result.append(')')
        return ''.join(result)

    def _print_Or(self, expr):
        result = ['(']
        for arg in sorted(expr.args, key=default_sort_key):
            result.extend(['(', self._print(arg), ')'])
            result.append(' | ')
        result = result[:-1]
        result.append(')')
        return ''.join(result)

    def _print_Piecewise(self, expr, **kwargs):
        e, cond = expr.args[0].args
        if len(expr.args) == 1:
            return 'where(%s, %s, %f)' % (self._print(cond, **kwargs),
                             self._print(e, **kwargs),
                             0)
        return 'where(%s, %s, %s)' % (self._print(cond, **kwargs),
                         self._print(e, **kwargs),
                         self._print(Piecewise(*expr.args[1:]), **kwargs))

def walk(num_dims, samples_per_dim):
    """
    A generator that returns lattice points on an n-simplex.
    """
    max_ = samples_per_dim + num_dims - 1
    for c in itertools.combinations(range(max_), num_dims):
        c = list(c)
        yield [(y - x - 1) / (samples_per_dim - 1)
               for x, y in zip([-1] + c, c + [max_])]

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
    if mode == 'sympy':
        energy = lambda *vs: model.subs(zip(variables, vs)).evalf()
    elif mode == 'numpy':
        logical_np = [{'And': np.logical_and, 'Or': np.logical_or}, 'numpy']
        energy = lambdify(tuple(variables), model, dummify=True,
                          modules=logical_np, printer=NumPyPrinter)
    elif mode == 'numexpr':
        energy = lambdify(tuple(variables), model, dummify=True,
                          modules='numexpr', printer=SpecialNumExprPrinter)
    else:
        energy = lambdify(tuple(variables), model, dummify=True,
                          modules=mode)

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
