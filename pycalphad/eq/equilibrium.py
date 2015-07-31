"""
The equilibrium module defines routines for interacting with
calculated phase equilibria.
"""
from __future__ import print_function
import pycalphad.variables as v
from pycalphad.eq.utils import make_callable
from pycalphad.eq.utils import check_degenerate_phases
from pycalphad.eq.utils import unpack_kwarg
from pycalphad.eq.utils import sizeof_fmt
from pycalphad import calculate, Model
from pycalphad.eq.calculate import _compute_phase_values
from pycalphad.constraints import mole_fraction
from pycalphad.eq.geometry import lower_convex_hull
from pycalphad.eq.eqresult import EquilibriumResult
from sympy import Add, Matrix, Mul, hessian
import xray
import numpy as np
from collections import namedtuple, Iterable, OrderedDict
import itertools

try:
    set
except NameError:
    from sets import Set as set  #pylint: disable=W0622

# Refine points within REFINEMENT_DISTANCE J/mol-atom of convex hull
REFINEMENT_DISTANCE = 10
# initial value of 'alpha' in Newton-Raphson procedure
INITIAL_STEP_SIZE = 1

PhaseRecord = namedtuple('PhaseRecord', ['variables', 'grad', 'hess'])

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
    (4) If it's a tuple with three elements, treat as lower/upper/step
    (5) If it's a list, ndarray or other non-tuple ordered iterable,
        use those values directly
    """
    if isinstance(tup, tuple):
        if len(tup) == 1:
            return [float(tup[0])]
        elif len(tup) == 2:
            return np.arange(tup[0], tup[1], dtype=np.float)
        elif len(tup) == 3:
            return np.arange(tup[0], tup[1], tup[2], dtype=np.float)
        else:
            raise ValueError('Condition tuple is length {}'.format(len(tup)))
    elif isinstance(tup, Iterable):
        return tup
    else:
        return [float(tup)]

def _unpack_phases(phases):
    "Convert a phases list/dict into a sorted list."
    active_phases = None
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
    verbose : bool, optional (Default: True)
        Show progress of calculations.
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
    phase_records = dict()
    points_dict = dict()
    # Construct models for each phase; prioritize user models
    models = unpack_kwarg(kwargs.pop('model', Model), default_arg=Model)
    if verbose:
        print('Components:', ' '.join(comps))
        print('Phases:', end=' ')
    for name in active_phases:
        mod = models[name]
        if isinstance(mod, type):
            models[name] = mod = mod(dbf, comps, name)
        variables = sorted(mod.energy.atoms(v.StateVariable), key=str)
        grad_func = make_callable(Matrix([mod.energy]).jacobian(variables), variables)
        hess_func = make_callable(hessian(mod.energy, variables), variables)
        phase_records[name.upper()] = PhaseRecord(variables=variables,
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
        print('[{0} points, {1}]'.format(len(grid.points), sizeof_fmt(grid.nbytes)), end='\n')

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
                              attrs={'iterations': 1},
                              )

    for iteration in range(10):
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
        #print('driving_forces', driving_forces)
        #print('grid.GM', grid.GM.values)
        near_stability = driving_forces > -REFINEMENT_DISTANCE
        # If one point is "close to stability" for even one set of conditions
        #    we refine it for _all_ conditions. This is very conservative.
        near_stability = np.any(near_stability, axis=tuple(np.arange(len(near_stability.shape) - 1)))
        #print('near_stability', near_stability)

        for name in active_phases:
            # The Y arrays have been padded, so we should slice off the padding
            dof = len(models[name].energy.atoms(v.SiteFraction))
            points = grid.Y.values[np.logical_and(near_stability, grid.Phase.values == name),
                                   :dof]
            if len(points) == 0:
                if name in points_dict:
                    del points_dict[name]
                # No nearly stable points: skip this phase
                continue
            if np.sum(points[0]) == dof:
                # All site fractions are 1, aka zero internal degrees of freedom
                # Impossible to refine these points, so skip this phase
                points_dict[name] = np.atleast_2d(points)
                continue

            num_vars = len(models[name].energy.atoms(v.StateVariable))
            statevar_grid = np.meshgrid(*itertools.chain(indep_vals, [np.empty(points.shape[0])]),
                                        sparse=True, indexing='ij')[:-1]
            # Adjust gradient and Hessian by the approximate chemical potentials
            plane_vars = sorted(models[name].energy.atoms(v.SiteFraction), key=str)
            hyperplane = Add(*[v.MU(i)*mole_fraction(dbf.phases[name], comps, i) \
                                for i in comps if i != 'VA'])
            # Workaround an annoying bug with zero gradient/Hessian entries
            # This forces numerically zero entries to broadcast correctly
            hyperplane += 1e-100 * Mul(*([v.MU(i) for i in comps if i != 'VA'] + plane_vars + [v.T, v.P])) ** 3

            plane_grad = make_callable(Matrix([hyperplane]).jacobian(phase_records[name].variables),
                                       [v.MU(i) for i in comps if i != 'VA'] + plane_vars + [v.T, v.P])
            plane_hess = make_callable(hessian(hyperplane, phase_records[name].variables),
                                       [v.MU(i) for i in comps if i != 'VA'] + plane_vars + [v.T, v.P])
            grad = phase_records[name].grad(*itertools.chain(statevar_grid, points.T))
            if grad.dtype == 'object':
                # Workaround a bug in zero gradient entries
                grad_zeros = np.zeros(tuple(len(vals) for vals in indep_vals) + (len(points),), dtype=np.float)
                for i in np.arange(grad.shape[0]):
                    if isinstance(grad[i], int):
                        grad[i] = grad_zeros
                grad = np.array(grad.tolist(), dtype=np.float)
            # Add additional dimensions to grad so it'll broadcast against cast_grad
            grad.shape = grad.shape[:2] + (1,) * (len(str_conds) - len(indep_vals)) + grad.shape[2:]
            #print('grad.shape', grad.shape)
            bcasts = np.broadcast_arrays(*itertools.chain(properties.MU.values.T[..., np.newaxis],
                                                          points.T.reshape((points.T.shape[0],) + (1,) * len(str_conds) + (-1,))))
            #print(bcasts[0].shape)
            cast_grad = -plane_grad(*itertools.chain(bcasts, [0], [0]))
            cast_grad = cast_grad.T + grad.T
            grad = cast_grad
            grad.shape = grad.shape[:-1] # Remove extraneous dimension
            #print('gradient shape', grad.shape)
            #print('gradient dtype', grad.dtype)
            hess = phase_records[name].hess(*itertools.chain(statevar_grid, points.T))
            if hess.dtype == 'object':
                # Workaround a bug in zero Hessian entries
                hess_zeros = np.zeros(tuple(len(vals) for vals in indep_vals) + (len(points),), dtype=np.float)
                for i in np.arange(hess.shape[0]):
                    for j in np.arange(hess.shape[1]):
                        if isinstance(hess[i, j], int):
                            hess[i, j] = hess_zeros
                hess = np.array(hess.tolist(), dtype=np.float)
            # Add additional dimensions to hess so it'll broadcast against cast_hess
            hess.shape = hess.shape[:2] + (1,) * (len(str_conds) - len(indep_vals)) + hess.shape[2:]
            #print('hess shape', hess.shape)
            #print('hess dtype', hess.dtype)
            cast_hess = -plane_hess(*itertools.chain(bcasts, [0], [0])).T + hess.T
            hess = cast_hess
            #print('hessian shape', hess.shape)
            #print('hessian dtype', hess.dtype)
            e_matrix = np.linalg.inv(hess)
            #print('e_matrix shape', e_matrix.shape)
            dy_unconstrained = -np.einsum('...ij,...j->...i', e_matrix, grad)
            #print('dy_unconstrained.shape', dy_unconstrained.shape)
            # TODO: A more sophisticated treatment of constraints
            num_constraints = len(indep_vals) + len(dbf.phases[name].sublattices)
            constraint_jac = np.zeros((num_constraints, num_vars))
            # Independent variables are always fixed (in this limited implementation)
            for idx in range(len(indep_vals)):
                constraint_jac[idx, idx] = 1
            # This is for site fraction balance constraints
            var_idx = len(indep_vals)
            for idx in range(len(dbf.phases[name].sublattices)):
                constraint_jac[len(indep_vals) + idx,
                               var_idx:var_idx + len(dbf.phases[name].constituents[idx])] = 1
                var_idx += len(dbf.phases[name].constituents[idx])
            proj_matrix = np.dot(e_matrix, constraint_jac.T)
            #print('proj_matrix.shape', proj_matrix.shape)
            inv_matrix = np.rollaxis(np.dot(constraint_jac, proj_matrix), 0, -1)
            inv_term = np.linalg.inv(inv_matrix)
            first_term = np.einsum('...ij,...jk->...ik', proj_matrix, inv_term)
            # Normally a term for the residual here
            # We only choose starting points which obey the constraints, so r = 0
            cons_summation = np.einsum('...i,...ji->...j', dy_unconstrained, constraint_jac)
            cons_correction = np.einsum('...ij,...j->...i', first_term, cons_summation)
            dy_constrained = dy_unconstrained - cons_correction
            dy_constrained = np.rollaxis(dy_constrained, 0, -1)
            print('points', points)
            print('dy_constrained', dy_constrained)
            #print('dy_constrained.shape', dy_constrained.shape)
            # TODO: Support for adaptive changing independent variable steps
            alpha = 1
            new_points = points + alpha*dy_constrained[..., len(indep_vals):]
            while np.any(new_points < 0):
                alpha *= 0.1
                new_points = points + alpha*dy_constrained[..., len(indep_vals):]
                if alpha < 1e-6:
                    break
            print('alpha =', alpha)
            points_dict[name] = np.concatenate((points, new_points.reshape((-1, new_points.shape[-1]))), axis=0)

        if verbose:
            print('Rebuilding grid', end=' ')
        grid = calculate(dbf, comps, active_phases, output='GM',
                         model=models, fake_points=True, points=points_dict, **grid_opts)
        if verbose:
            print('[{0} points, {1}]'.format(len(grid.points), sizeof_fmt(grid.nbytes)), end='\n')
        properties.attrs['iterations'] += 1

    return properties
