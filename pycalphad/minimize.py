"""
The minimize module handles calculation of equilibrium.
"""
from __future__ import division
from pycalphad import Model
import pycalphad.variables as v
from sympy.printing.theanocode import theano_function
import theano
import pandas as pd
import numpy as np
from scipy.spatial import ConvexHull
import itertools
try:
    set
except NameError:
    from sets import Set as set #pylint: disable=W0622

# monkey patch for theano_function handling sympy
import sympy.printing.theanocode
sympy.printing.theanocode.mapping[sympy.And] = theano.tensor.and_
def _special_print_Piecewise(self, expr, **kwargs):
    import numpy.nan
    from theano.ifelse import ifelse
    e, cond = expr.args[0].args
    if len(expr.args) == 1:
        return ifelse(self._print(cond, **kwargs),
                      self._print(e, **kwargs),
                      numpy.nan)
    return ifelse(self._print(cond, **kwargs),
                  self._print(e, **kwargs),
                  self._print(sympy.Piecewise(*expr.args[1:]), **kwargs))
sympy.printing.theanocode._print_Piecewise = _special_print_Piecewise

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
    result = list(map(np.concatenate, prod))
    return np.asarray(result)

class CompositionSet(object):
    """
    CompositionSets are used by equilibrium calculations to represent
    phases during the calculation.

    Attributes
    ----------
    ast, SymPy object
        Abstract representation of energy function
    energy, Theano function
        Compiled energy function optimized for call speed
    variables, list
        Input variables, ordered in the way 'energy' expects

    Methods
    -------
    None yet.

    Examples
    --------
    None yet.
    """
    def __init__(self, mod, statevars):
        self.ast = mod.ast.subs(statevars)
        self.variables = self.ast.atoms(v.StateVariable)
        print(self.variables)
        self.energy = theano_function(self.variables, [self.ast])

def eq(db, comps, phases, **kwargs):
    """
    Calculate the equilibrium state of a system containing the specified
    components and phases. Model parameters are taken from 'db' and any
    state variables (T, P, etc.) can be specified as keyword arguments.

    Parameters
    ----------
    db : Database
        Thermodynamic database containing the relevant parameters.
    comps : list
        Names (case-sensitive) of components to consider in the calculation.
    phases : list
        Names (case-sensitive) of phases to consider in the calculation.

    Returns
    -------
    Structured equilibrium calculation.

    Examples
    --------
    None yet.
    """
    # Here we would check for any keyword arguments that are special, i.e.,
    # there may be keyword arguments that aren't state variables

    # Convert keyword strings to proper state variable objects
    # If we don't do this, sympy will get confused during substitution
    statevars = {v.StateVariable(k): val for k, val in kwargs.items()}
    comp_sets = {}
    phases_df = pd.DataFrame()
    for phase_name in phases:
        mod = Model(db, comps, phase_name)
        comp_sets[phase_name] = CompositionSet(mod, statevars)
        sublattice_dof = list(map(len, db.phases[phase_name].constituents))
        num_points = int(10000**(1/len(sublattice_dof)))
        points = point_sample(sublattice_dof, size=num_points)
        energies = np.zeros(len(points))
        # TODO: not very efficient point sampling strategy
        # in principle, this could be parallelized
        for idx, p in enumerate(points):
            energies[idx] = comp_sets[phase_name].energy(*p)

        # Construct a matrix to calculate the convex hull of this phase
        gibbs_matrix = np.ndarray((len(points),sum(sublattice_dof)+1))
        gibbs_matrix[:,0:-1] = points
        gibbs_matrix[:,-1] = energies

        # Strip out the dependent degrees of freedom before finding the hull
        all_dof = set(np.arange(sum(sublattice_dof)))
        cur_idx = 0
        dependent_dof = []
        for dof in sublattice_dof:
            cur_idx += dof
            dependent_dof.append(cur_idx-1)
        dependent_dof = set(dependent_dof)
        independent_dof = list(all_dof - dependent_dof)
        hull = ConvexHull(gibbs_matrix[:,independent_dof])
        
        # Now map the internal degrees of freedom to global coordinates
    print(comp_sets)
