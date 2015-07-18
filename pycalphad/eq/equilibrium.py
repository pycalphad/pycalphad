"""
The equilibrium module defines routines for interacting with
calculated phase equilibria.
"""
from __future__ import print_function
import pycalphad.variables as v
from pycalphad.log import logger
from pycalphad.eq.utils import make_callable
from pycalphad.eq.utils import check_degenerate_phases
from pycalphad.eq.utils import unpack_kwarg
from pycalphad.eq.utils import sizeof_fmt
from pycalphad import calculate, Model
from pycalphad.eq.geometry import lower_convex_hull
from pycalphad.eq.eqresult import EquilibriumResult
from sympy import Matrix, hessian
import xray
import numpy as np
from collections import namedtuple, Iterable, OrderedDict
import itertools

try:
    set
except NameError:
    from sets import Set as set #pylint: disable=W0622

# Refine points within REFINEMENT_DISTANCE J/mol-atom of convex hull
REFINEMENT_DISTANCE = 10

class EquilibriumError(Exception):
    "Exception related to calculation of equilibrium"
    pass

def _unpack_condition(tup):
    """
    Convert a condition to a list of values.
    Rules for keys of conditions dicts:
    (1) If it's numeric, treat as a point value
    (2) If it's a tuple with one element, treat as a point value
    (3) If it's a tuple with two elements, treat as lower/upper limits and
        guess a step size
    (4) If it's a tuple with three elements, treat as lower/upper/num
    (5) If it's a list, ndarray or other non-tuple ordered iterable,
        use those values directly
    """
    if isinstance(tup, tuple):
        if len(tup) == 1:
            return [float(tup[0])]
        elif len(tup) == 2:
            return np.linspace(float(tup[0]), float(tup[1]), num=20)
        elif len(tup) == 3:
            return np.linspace(float(tup[0]), float(tup[1]), num=tup[2])
        else:
            raise ValueError('Condition tuple is length {}'.format(len(tup)))
    elif isinstance(tup, Iterable):
        return tup
    else:
        return [float(tup)]

def _unpack_phases(phases):
    "Convert a phases list/dict into a sorted list."
    if isinstance(phases, list):
        active_phases = sorted(phases)
    elif isinstance(phases, dict):
        active_phases = sorted([phn for phn, status in phases.items() \
            if status == 'entered'])
    elif type(phases) is str:
        active_phases = [phases]
    return active_phases

def equilibrium(dbf, comps, phases, conditions, **kwargs):
    """
    Calculate the equilibrium state of a system containing the specified
    components and phases, under the specified conditions.
    Model parameters are taken from 'dbf'.

    Parameters
    ----------
    dbf : Database
        Thermodynamic database containing the relevant parameters.
    comps : list
        Names of components to consider in the calculation.
    phases : list or dict
        Names of phases to consider in the calculation.
    conditions : dict or (list of dict)
        StateVariables and their corresponding value.
    verbose : bool, optional
        Show progress of calculations. (Default: True)
    grid_opts : dict, optional
        Keyword arguments to pass to the initial grid routine.

    Returns
    -------
    Structured equilibrium calculation.

    Examples
    --------
    None yet.
    """
    active_phases = _unpack_phases(phases)
    comps = sorted(set(comps))
    indep_vars = ['T', 'P']
    grid_opts = kwargs.pop('grid_opts', dict())
    verbose = kwargs.pop('verbose', True)
    PhaseRecord = namedtuple('PhaseRecord', ['variables', 'obj', 'grad', 'hess'])
    phase_records = dict()
    # Construct models for each phase; prioritize user models
    models = unpack_kwarg(kwargs.pop('model', Model), default_arg=Model)
    if verbose:
        print('Components:', ' '.join(comps))
        print('Models:', end=' ')
    for name in active_phases:
        mod = models[name]
        if isinstance(mod, type):
            models[name] = mod = mod(dbf, comps, name)
        variables = sorted(mod.energy.atoms(v.StateVariable), key=str)
        obj_func = lambda x, f=mod.energy, vxx=variables: make_callable(f, vxx)(*x)
        grad_func = make_callable(Matrix([mod.energy]).jacobian(variables), variables)
        #grad_func = ndt.Gradient(obj_func)
        hess_func = make_callable(hessian(mod.energy, variables), variables)
        #hess_func = ndt.Hessian(obj_func)
        phase_records[name.upper()] = PhaseRecord(variables=variables, obj=obj_func,
                                                  grad=grad_func, hess=hess_func)
        if verbose:
            print(name, end=' ')
    if verbose:
        print('[done]', end='\n')

    conds = {key: _unpack_condition(value) for key, value in conditions.items()}
    str_conds = OrderedDict({str(key): value for key, value in conds.items()})
    indep_vals = list(val for key, val in str_conds.items() if key in indep_vars)
    components = [x for x in sorted(comps) if not x.startswith('VA')]
    # 'calculate' accepts conditions through its keyword arguments
    grid_opts.update({key: value for key, value in str_conds.items() if key in indep_vars})
    if 'pdens' not in grid_opts:
        grid_opts['pdens'] = 100

    coord_dict = str_conds.copy()
    coord_dict['vertex'] = np.arange(len(components))
    grid_shape = np.meshgrid(*coord_dict.values(),
                             indexing='ij', sparse=False)[0].shape
    coord_dict['component'] = components
    if verbose:
        print('Computing initial grid', end=' ')

    grid = calculate(dbf, comps, active_phases, output='GM',
                     model=models, fake_points=True, **grid_opts)
    if verbose:
        print('[{0} points, {1}]'.format(len(grid.points), sizeof_fmt(grid.nbytes)))

    properties = xray.Dataset({'NP': (list(str_conds.keys()) + ['vertex'],
                                      np.empty(grid_shape)),
                               'GM': (list(str_conds.keys()),
                                      np.empty(grid_shape[:-1])),
                               'MU': (list(str_conds.keys()) + ['component'],
                                      np.empty(grid_shape)),
                               'points': (list(str_conds.keys()) + ['vertex'],
                                          np.empty(grid_shape, dtype=np.int))
                              },
                              coords=coord_dict,
                              attrs={'iterations': 0},
                             )
    if verbose:
        print('Computing convex hull [iteration {}]'.format(properties.attrs['iterations']))
    # lower_convex_hull will modify properties
    lower_convex_hull(grid, properties)

    if verbose:
        print('Refining points within {}J/mol of hull'.format(REFINEMENT_DISTANCE))
    # Insert extra dimensions for non-T,P conditions so GM broadcasts correctly
    energy_broadcast_shape = grid.GM.values.shape[:len(indep_vals)] + \
        (1,) * (len(str_conds) - len(indep_vals)) + (grid.GM.values.shape[-1],)
    driving_forces = np.inner(properties.MU.values, grid.X.values) - \
        grid.GM.values.reshape(energy_broadcast_shape)
    near_stability = driving_forces > -REFINEMENT_DISTANCE
    # If one point is "close to stability" for even one set of conditions
    #    we refine it for _all_ conditions. This is very conservative.
    near_stability = np.any(near_stability, axis=tuple(np.arange(len(near_stability.shape) - 1)))

    for name in active_phases:
        print(name)
        # The Y arrays have been padded, so we should slice off the padding
        dof = len(models[name].energy.atoms(v.SiteFraction))
        points = grid.Y.values[np.logical_and(near_stability, grid.Phase.values == name),
                               :dof]
        if (len(points) == 0) or (np.sum(points[0]) == dof):
            # No nearly stable points: skip this phase
            # or
            # All site fractions are 1, aka zero internal degrees of freedom
            # Impossible to refine these points, so skip this phase
            continue
        num_vars = len(models[name].energy.atoms(v.StateVariable))
        statevar_grid = np.meshgrid(*itertools.chain(indep_vals, [np.empty(points.shape[0])]),
                                    sparse=True, indexing='ij')[:-1]
        grad = phase_records[name].grad(*itertools.chain(statevar_grid, points.T))
        grad.shape = grad.shape[1:]
        grad = grad.T
        print('gradient shape', grad.shape)
        hess = phase_records[name].hess(*itertools.chain(statevar_grid, points.T))
        if hess.dtype == 'object':
            # Workaround a bug in zero Hessian entries
            hess_zeros = np.zeros(tuple(len(vals) for vals in indep_vals) + (len(points),), dtype=np.float)
            for i in np.arange(hess.shape[0]):
                for j in np.arange(hess.shape[1]):
                    if isinstance(hess[i,j], int):
                        hess[i, j] = hess_zeros
            hess = np.array(hess.tolist(), dtype=np.float)
        hess = hess.T
        print('hessian shape', hess.shape)
        e_matrix = np.linalg.inv(hess)
        print('e_matrix shape', e_matrix.shape)
        dy_unconstrained = -np.einsum('...ij,...j->...i', e_matrix, grad)
        print(dy_unconstrained.shape)
        print(dy_unconstrained)
        constraint_jac = None
    return properties
    # 1. Call energy_surf to sample the entire energy surface at some density (whatever T, P required)
    # 2. Call lower_convex_hull to find equilibrium simplices/potentials along the calculation path
    # 3. Perform a driving force calculation.
    # 4. While any driving force is positive:
    #       a. Using the determined potentials for each point on the calculation path, find the constrained minimum
    #          for _all_ phases in the system. This will be done with N-R by an iterative calculation.
    #       b. If the solver failed to converge, call energy_surf to add more random points for that phase, T, P, etc.
    #       c. Add all computed points to the global energy surface calculated by energy_surf.
    #       d. Call lower_convex_hull again, including the added points. These should be the equilibrium values.
    #       e. Perform a driving force calculation with the new potentials.
    # 5. Recompute simplices for points where driving force is zero. This is to catch invariant regions.
    # 6. Return the potentials, simplices and fractions found for the calculation path.
    points = grid.sel(points=phase_compositions)
    for iteration in range(10):
        #print('x: ', points)
        hess = np.rollaxis(hess_func(*points.T), -1)
        print('eigenvalues: ', np.linalg.eigvals(hess))
        gradient = np.transpose(gradient_func(*points.T), (2, 0, 1))
        energies = energy_func(*points.T)
        #print('energy: ', energies[..., 0])
        e_matrix = np.linalg.inv(hess)
        e_matrix = e_matrix[0, ...]
        constraint_jac = np.atleast_2d(np.ones(len(comps)))
        # unconstrained Newton-Raphson solution
        dy_unconstrained = -np.dot(e_matrix, gradient)
        # extra terms to handle constraints
        proj_matrix = np.dot(e_matrix, constraint_jac.T)
        inv_term = np.linalg.inv(np.dot(constraint_jac, proj_matrix))
        cons_summation = (points.sum(axis=-1)-1.0) + np.dot(constraint_jac, dy_unconstrained)
        cons_correction = np.dot(proj_matrix, inv_term).dot(cons_summation)
        dy = dy_unconstrained - cons_correction
        driving_force = energies - np.tensordot(points, potentials, axes=(-1, -1))
        print('DF: ', driving_force)
        #diff /= np.linalg.norm(diff, ord=1)
        alpha = 1
        new_points = points+alpha*dy
        new_energies = energy_func(*new_points.T)
        while (new_energies > energies - 0.1*np.dot(dy.T, gradient)) or np.any(new_points < 0):
            alpha *= 0.5
            new_points = points+alpha*dy
            new_energies = energy_func(*new_points.T)
            if alpha < 1e-6:
                break
        points = new_points
        if np.linalg.norm(alpha*dy) < 1e-12:
            break
        #if np.any(points < 0):
        #    points -= alpha*dy
        #    break
        #print('diff ', diff)
        #print('sum ', diff.sum(axis=-1))
        #print('new x = ', points)
    print('iterations: ', iteration)
    print('x: ', points)
    print('alpha: ', alpha)
    print('gradient: ', gradient)
    print('potentials: ', potentials)
    print('x_sum_residual: ', 1 - points.sum(axis=-1))
    print('energy: ', energies[..., 0])
    print('dy ', dy)

def newton_raphson_solve(grid, guess_simplices, **kwargs):
    pass
