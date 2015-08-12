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
from pycalphad.eq.utils import unpack_condition, unpack_phases
from pycalphad import calculate, Model
from pycalphad.eq.calculate import _compute_phase_values
from pycalphad.constraints import mole_fraction
from pycalphad.eq.geometry import lower_convex_hull
from pycalphad.eq.eqresult import EquilibriumResult
from sympy import Add, Matrix, Mul, hessian
import xray
import numpy as np
from collections import namedtuple, OrderedDict
import itertools

try:
    set
except NameError:
    from sets import Set as set  #pylint: disable=W0622

# Maximum number of refinements
MAX_ITERATIONS = 50
# If the norm of the energy difference between iterations is less than
# MIN_PROGRESS J/mol-atom, stop the refinement
MIN_PROGRESS = 1e-4
# initial value of 'alpha' in Newton-Raphson procedure
INITIAL_STEP_SIZE = 1.

PhaseRecord = namedtuple('PhaseRecord', ['variables', 'grad', 'hess'])

class EquilibriumError(Exception):
    "Exception related to calculation of equilibrium"
    pass

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
    active_phases = unpack_phases(phases)
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
        variables = sorted(mod.energy.atoms(v.StateVariable).union({key for key in conditions.keys() if key in [v.T, v.P]}), key=str)
        # Extra factor '1e-100...' is to work around an annoying broadcasting bug for zero gradient entries
        models[name].models['_broadcaster'] = 1e-100 * Mul(*variables) ** 3
        grad_func = make_callable(Matrix([mod.energy]).jacobian(variables), variables)
        hess_func = make_callable(hessian(mod.energy, variables), variables)
        phase_records[name.upper()] = PhaseRecord(variables=variables,
                                                  grad=grad_func, hess=hess_func)
        if verbose:
            print(name, end=' ')
    if verbose:
        print('[done]', end='\n')

    conds = OrderedDict((key, unpack_condition(value)) for key, value in sorted(conditions.items(), key=str))
    str_conds = OrderedDict((str(key), value) for key, value in conds.items())
    indep_vals = list([float(x) for x in np.atleast_1d(val)]
                      for key, val in str_conds.items() if key in indep_vars)
    components = [x for x in sorted(comps) if not x.startswith('VA')]
    # 'calculate' accepts conditions through its keyword arguments
    grid_opts.update({key: value for key, value in str_conds.items() if key in indep_vars})
    if 'pdens' not in grid_opts:
        grid_opts['pdens'] = 10

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
    # Store the energies from the previous iteration
    current_energies = np.zeros(grid_shape[:-1], dtype=np.float)

    for iteration in range(MAX_ITERATIONS):
        if verbose:
            print('Computing convex hull [iteration {}]'.format(properties.attrs['iterations']))
        # lower_convex_hull will modify properties
        lower_convex_hull(grid, properties)
        progress = np.abs((current_energies - properties.GM.values)).max()
        if verbose:
            print('progress', progress)
        if progress < MIN_PROGRESS:
            if verbose:
                print('Convergence achieved')
            break
        current_energies[...] = properties.GM.values
        if verbose:
            print('Refining convex hull')
        # Insert extra dimensions for non-T,P conditions so GM broadcasts correctly
        energy_broadcast_shape = grid.GM.values.shape[:len(indep_vals)] + \
            (1,) * (len(str_conds) - len(indep_vals)) + (grid.GM.values.shape[-1],)
        driving_forces = np.multiply(properties.MU.values[..., np.newaxis, :],
                                     grid.X.values[np.index_exp[...] +
                                                   (np.newaxis,) * (len(str_conds) - len(indep_vals)) +
                                                   np.index_exp[:, :]]).sum(axis=-1) - \
            grid.GM.values.reshape(energy_broadcast_shape)
        #print('grid.Y.values', grid.Y.values)
        #print('driving_forces.shape', driving_forces)

        for name in active_phases:
            dof = len(models[name].energy.atoms(v.SiteFraction))
            current_phase_indices = (grid.Phase.values == name).reshape(energy_broadcast_shape[:-1] + (-1,))
            # Broadcast to capture all conditions
            current_phase_indices = np.broadcast_arrays(current_phase_indices,
                                                        np.empty(driving_forces.shape))[0]
            #print('current_phase_indices.shape', current_phase_indices.shape)
            # This reshape is safe as long as phases have the same number of points at all indep. conditions
            current_phase_driving_forces = driving_forces[current_phase_indices].reshape(
                current_phase_indices.shape[:-1] + (-1,))
            #print('current_phase_driving_forces.shape', current_phase_driving_forces.shape)
            # Find the N points with largest driving force for a given set of conditions
            # Remember that driving force has a sign, so we want the "most positive" values
            # N is the number of components, in this context
            # N points define a 'best simplex' for every set of conditions
            # We also need to restrict ourselves to one phase at a time
            trial_indices = np.argpartition(current_phase_driving_forces,
                                            -len(components), axis=-1)[..., -len(components):]
            #print('trial_indices', trial_indices)
            trial_indices = trial_indices.ravel()
            # Note: This works as long as all points are in the same phase order for all T, P
            current_site_fractions = grid.Y.values[..., current_phase_indices[(0,) * len(str_conds)], :]
            statevar_indices = np.unravel_index(np.arange(np.multiply.reduce(properties.MU.values.shape)),
                                                properties.MU.values.shape)[:len(indep_vals)]
            points = current_site_fractions[statevar_indices, trial_indices, :]
            points.shape = properties.points.shape[:-1] + (-1, properties.points.shape[-1])
            # The Y arrays have been padded, so we should slice off the padding
            points = points[..., :dof]
            #print('points.shape', points.shape)
            #print('points', points)
            if len(points) == 0:
                if name in points_dict:
                    del points_dict[name]
                # No nearly stable points: skip this phase
                continue
            if np.any(np.sum(points, axis=-1) == dof):
                # All site fractions are 1, aka zero internal degrees of freedom
                # Impossible to refine these points, so skip this phase
                points_dict[name] = np.atleast_2d(points)
                continue

            num_vars = len(phase_records[name].variables)
            # Adjust gradient and Hessian by the approximate chemical potentials
            plane_vars = sorted(models[name].energy.atoms(v.SiteFraction), key=str)
            hyperplane = Add(*[v.MU(i)*mole_fraction(dbf.phases[name], comps, i)
                               for i in comps if i != 'VA'])
            # Workaround an annoying bug with zero gradient/Hessian entries
            # This forces numerically zero entries to broadcast correctly
            hyperplane += 1e-100 * Mul(*([v.MU(i) for i in comps if i != 'VA'] + plane_vars + [v.T, v.P])) ** 3

            plane_grad = make_callable(Matrix([hyperplane]).jacobian(phase_records[name].variables),
                                       [v.MU(i) for i in comps if i != 'VA'] + plane_vars + [v.T, v.P])
            plane_hess = make_callable(hessian(hyperplane, phase_records[name].variables),
                                       [v.MU(i) for i in comps if i != 'VA'] + plane_vars + [v.T, v.P])
            statevar_grid = np.meshgrid(*itertools.chain(indep_vals, [np.empty(points.shape[-2])]),
                                        sparse=True, indexing='xy')[:-1]
            #print('statevar_grid[0].shape', statevar_grid[0].shape)
            #print('points.T.shape', points.T.shape)
            grad = phase_records[name].grad(*itertools.chain(statevar_grid, points.T))
            if grad.dtype == 'object':
                # Workaround a bug in zero gradient entries
                grad_zeros = np.zeros(points.T.shape[1:], dtype=np.float)
                for i in np.arange(grad.shape[0]):
                    if isinstance(grad[i], int):
                        grad[i] = grad_zeros
                grad = np.array(grad.tolist(), dtype=np.float)
            bcasts = np.broadcast_arrays(*itertools.chain(properties.MU.values.T, points.T))
            cast_grad = -plane_grad(*itertools.chain(bcasts, [0], [0]))
            cast_grad = cast_grad.T + grad.T
            grad = cast_grad
            grad.shape = grad.shape[:-1]  # Remove extraneous dimension
            #print('grad.shape', grad.shape)
            hess = phase_records[name].hess(*itertools.chain(statevar_grid, points.T))
            if hess.dtype == 'object':
                # Workaround a bug in zero Hessian entries
                hess_zeros = np.zeros(points.T.shape[1:], dtype=np.float)
                for i in np.arange(hess.shape[0]):
                    for j in np.arange(hess.shape[1]):
                        if isinstance(hess[i, j], int):
                            hess[i, j] = hess_zeros
                hess = np.array(hess.tolist(), dtype=np.float)
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
            #print('cons_correction.shape', cons_correction.shape)
            #print('cons_correction[..., len(indep_vals):]', cons_correction[..., len(indep_vals)])
            #print('dy_unconstrained.shape', dy_unconstrained.shape)
            #print('dy_unconstrained[..., len(indep_vals):]', dy_unconstrained[..., len(indep_vals):])
            dy_constrained = dy_unconstrained - cons_correction
            #print('dy_constrained[..., len(indep_vals):]', dy_constrained[..., len(indep_vals):])
            # TODO: Support for adaptive changing independent variable steps
            # Backtracking line search
            new_points = points + INITIAL_STEP_SIZE*dy_constrained[..., len(indep_vals):]
            alpha = np.full(new_points.shape[:-1], INITIAL_STEP_SIZE, dtype=np.float)
            negative_points = np.any(new_points < 0., axis=-1)
            while np.any(negative_points):
                alpha[negative_points] *= 0.1
                new_points = points + alpha[..., np.newaxis]*dy_constrained[..., len(indep_vals):]
                negative_points = np.any(new_points < 0., axis=-1)
            #print('points', points)
            #print('alpha', alpha)
            #print('new_points', new_points)
            #new_points = np.concatenate((points, new_points), axis=-2)
            new_points = new_points.reshape(new_points.shape[:len(indep_vals)] + (-1, new_points.shape[-1]))
            new_points = np.concatenate((current_site_fractions, new_points), axis=-2)
            points_dict[name] = new_points

        if verbose:
            print('Rebuilding grid', end=' ')
        grid = calculate(dbf, comps, active_phases, output='GM',
                         model=models, fake_points=True, points=points_dict, **grid_opts)
        if verbose:
            print('[{0} points, {1}]'.format(len(grid.points), sizeof_fmt(grid.nbytes)), end='\n')
        properties.attrs['iterations'] += 1

    # One last call to ensure 'properties' and 'grid' are consistent with one another
    lower_convex_hull(grid, properties)
    ravelled_X_view = grid['X'].values.view().reshape(-1, grid['X'].values.shape[-1])
    ravelled_Y_view = grid['Y'].values.view().reshape(-1, grid['Y'].values.shape[-1])
    ravelled_Phase_view = grid['Phase'].values.view().reshape(-1)
    # Copy final point values from the grid and drop the index array
    # For some reason direct construction doesn't work. We have to create empty and then assign.
    properties['X'] = xray.DataArray(np.empty_like(ravelled_X_view[properties['points'].values]),
                                     dims=properties['points'].dims + ('component',))
    properties['X'].values[...] = ravelled_X_view[properties['points'].values]
    properties['Y'] = xray.DataArray(np.empty_like(ravelled_Y_view[properties['points'].values]),
                                     dims=properties['points'].dims + ('internal_dof',))
    properties['Y'].values[...] = ravelled_Y_view[properties['points'].values]
    # TODO: What about invariant reactions? We should perform a final driving force calculation here.
    # We can handle that in the same post-processing step where we identify single-phase regions.
    properties['Phase'] = xray.DataArray(np.empty_like(ravelled_Phase_view[properties['points'].values]),
                                     dims=properties['points'].dims)
    properties['Phase'].values[...] = ravelled_Phase_view[properties['points'].values]
    del properties['points']
    return properties
