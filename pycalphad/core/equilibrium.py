"""
The equilibrium module defines routines for interacting with
calculated phase equilibria.
"""
from __future__ import print_function
import pycalphad.variables as v
from pycalphad.core.utils import broadcast_to
from pycalphad.core.utils import make_callable
from pycalphad.core.utils import unpack_kwarg
from pycalphad.core.utils import sizeof_fmt
from pycalphad.core.utils import unpack_condition, unpack_phases
from pycalphad import calculate, Model
from pycalphad.constraints import mole_fraction
from pycalphad.core.lower_convex_hull import lower_convex_hull
from sympy import Add, Matrix, Mul, Symbol, hessian
import xray
import numpy as np
from collections import namedtuple, OrderedDict
import itertools
try:
    set
except NameError:
    from sets import Set as set #pylint: disable=W0622

# Maximum number of refinements
MAX_ITERATIONS = 100
# Maximum number of Newton steps to take
MAX_NEWTON_ITERATIONS = 50
# If the max of the potential difference between iterations is less than
# MIN_PROGRESS J/mol-atom, stop the refinement
MIN_PROGRESS = 1e-4
# Minimum length of a Newton step before Hessian update is skipped
MIN_STEP_LENGTH = 1e-09
# Force zero values to this amount, for numerical stability
MIN_SITE_FRACTION = 1e-16
# initial value of 'alpha' in Newton-Raphson procedure
INITIAL_STEP_SIZE = 1.

PhaseRecord = namedtuple('PhaseRecord', ['variables', 'grad', 'plane_grad', 'plane_hess'])

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
    active_phases = unpack_phases(phases) or sorted(dbf.phases.keys())
    comps = sorted(comps)
    indep_vars = ['T', 'P']
    grid_opts = kwargs.pop('grid_opts', dict())
    verbose = kwargs.pop('verbose', True)
    phase_records = dict()
    callable_dict = kwargs.pop('callables', dict())
    grad_callable_dict = kwargs.pop('grad_callables', dict())
    points_dict = dict()
    maximum_internal_dof = 0
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
        site_fracs = sorted(mod.energy.atoms(v.SiteFraction), key=str)
        maximum_internal_dof = max(maximum_internal_dof, len(site_fracs))
        # Extra factor '1e-100...' is to work around an annoying broadcasting bug for zero gradient entries
        models[name].models['_broadcaster'] = 1e-100 * Mul(*variables) ** 3
        out = models[name].energy
        if name not in callable_dict:
            undefs = list(out.atoms(Symbol) - out.atoms(v.StateVariable))
            for undef in undefs:
                out = out.xreplace({undef: float(0)})
            # callable_dict takes variables in a different order due to calculate() pecularities
            callable_dict[name] = make_callable(out,
                                                sorted((key for key in conditions.keys() if key in [v.T, v.P]),
                                                       key=str) + site_fracs)
        if name not in grad_callable_dict:
            grad_func = make_callable(Matrix([out]).jacobian(variables), variables)
        else:
            grad_func = grad_callable_dict[name]
        # Adjust gradient by the approximate chemical potentials
        plane_vars = sorted(models[name].energy.atoms(v.SiteFraction), key=str)
        hyperplane = Add(*[v.MU(i)*mole_fraction(dbf.phases[name], comps, i)
                           for i in comps if i != 'VA'])
        # Workaround an annoying bug with zero gradient entries
        # This forces numerically zero entries to broadcast correctly
        hyperplane += 1e-100 * Mul(*([v.MU(i) for i in comps if i != 'VA'] + plane_vars + [v.T, v.P])) ** 3

        plane_grad = make_callable(Matrix([hyperplane]).jacobian(variables),
                                   [v.MU(i) for i in comps if i != 'VA'] + plane_vars + [v.T, v.P])
        plane_hess = make_callable(hessian(hyperplane, variables),
                                   [v.MU(i) for i in comps if i != 'VA'] + plane_vars + [v.T, v.P])
        phase_records[name.upper()] = PhaseRecord(variables=variables,
                                                  grad=grad_func,
                                                  plane_grad=plane_grad,
                                                  plane_hess=plane_hess)
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
                     model=models, callables=callable_dict, fake_points=True, **grid_opts)

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
    # Store the potentials from the previous iteration
    current_potentials = properties.MU.copy()

    for iteration in range(MAX_ITERATIONS):
        if verbose:
            print('Computing convex hull [iteration {}]'.format(properties.attrs['iterations']))
        # lower_convex_hull will modify properties
        lower_convex_hull(grid, properties)
        progress = np.abs(current_potentials - properties.MU).max().values
        if verbose:
            print('progress', progress)
        if progress < MIN_PROGRESS:
            if verbose:
                print('Convergence achieved')
            break
        current_potentials[...] = properties.MU.values
        if verbose:
            print('Refining convex hull')
        # Insert extra dimensions for non-T,P conditions so GM broadcasts correctly
        energy_broadcast_shape = grid.GM.values.shape[:len(indep_vals)] + \
            (1,) * (len(str_conds) - len(indep_vals)) + (grid.GM.values.shape[-1],)
        driving_forces = np.einsum('...i,...i',
                                   properties.MU.values[..., np.newaxis, :].astype(np.float),
                                   grid.X.values[np.index_exp[...] +
                                                 (np.newaxis,) * (len(str_conds) - len(indep_vals)) +
                                                 np.index_exp[:, :]].astype(np.float)) - \
            grid.GM.values.view().reshape(energy_broadcast_shape)

        for name in active_phases:
            dof = len(models[name].energy.atoms(v.SiteFraction))
            current_phase_indices = (grid.Phase.values == name).reshape(energy_broadcast_shape[:-1] + (-1,))
            # Broadcast to capture all conditions
            current_phase_indices = np.broadcast_arrays(current_phase_indices,
                                                        np.empty(driving_forces.shape))[0]
            # This reshape is safe as long as phases have the same number of points at all indep. conditions
            current_phase_driving_forces = driving_forces[current_phase_indices].reshape(
                current_phase_indices.shape[:-1] + (-1,))
            # Note: This works as long as all points are in the same phase order for all T, P
            current_site_fractions = grid.Y.values[..., current_phase_indices[(0,) * len(str_conds)], :]
            if np.sum(current_site_fractions[(0,) * len(indep_vals)][..., :dof]) == dof:
                # All site fractions are 1, aka zero internal degrees of freedom
                # Impossible to refine these points, so skip this phase
                points_dict[name] = current_site_fractions[(0,) * len(indep_vals)][..., :dof]
                continue
            # Find the N points with largest driving force for a given set of conditions
            # Remember that driving force has a sign, so we want the "most positive" values
            # N is the number of components, in this context
            # N points define a 'best simplex' for every set of conditions
            # We also need to restrict ourselves to one phase at a time
            trial_indices = np.argpartition(current_phase_driving_forces,
                                            -len(components), axis=-1)[..., -len(components):]
            trial_indices = trial_indices.ravel()
            statevar_indices = np.unravel_index(np.arange(np.multiply.reduce(properties.GM.values.shape + (len(components),))),
                                                properties.GM.values.shape + (len(components),))[:len(indep_vals)]
            points = current_site_fractions[np.index_exp[statevar_indices + (trial_indices,)]]
            points.shape = properties.points.shape[:-1] + (-1, maximum_internal_dof)
            # The Y arrays have been padded, so we should slice off the padding
            points = points[..., :dof]
            # Workaround for derivative issues at endmembers
            points[points == 0.] = MIN_SITE_FRACTION
            if len(points) == 0:
                if name in points_dict:
                    del points_dict[name]
                # No nearly stable points: skip this phase
                continue

            num_vars = len(phase_records[name].variables)
            plane_grad = phase_records[name].plane_grad
            plane_hess = phase_records[name].plane_hess
            statevar_grid = np.meshgrid(*itertools.chain(indep_vals), sparse=True, indexing='xy')
            # TODO: A more sophisticated treatment of constraints
            num_constraints = len(indep_vals) + len(dbf.phases[name].sublattices)
            constraint_jac = np.zeros((num_constraints, num_vars))
            # Independent variables are always fixed (in this limited implementation)
            for idx in range(len(indep_vals)):
                constraint_jac[idx, idx] = 1
            # This is for site fraction balance constraints
            var_idx = len(indep_vals)
            for idx in range(len(dbf.phases[name].sublattices)):
                active_in_subl = set(dbf.phases[name].constituents[idx]).intersection(comps)
                constraint_jac[len(indep_vals) + idx,
                               var_idx:var_idx + len(active_in_subl)] = 1
                var_idx += len(active_in_subl)

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
            grad = grad.astype(np.float, copy=False)
            # This Hessian is an approximation updated using the BFGS method
            # See Nocedal and Wright, ch.3, p. 198
            # Initialize as identity matrix
            hess = broadcast_to(np.eye(num_vars), grad.shape + (grad.shape[-1],)).copy()
            hess = hess.astype(np.float, copy=False)
            newton_iteration = 0
            while newton_iteration < MAX_NEWTON_ITERATIONS:
                e_matrix = np.linalg.inv(hess)
                dy_unconstrained = -np.einsum('...ij,...j->...i', e_matrix, grad)
                proj_matrix = np.dot(e_matrix, constraint_jac.T)
                inv_matrix = np.rollaxis(np.dot(constraint_jac, proj_matrix), 0, -1)
                inv_term = np.linalg.inv(inv_matrix)
                first_term = np.einsum('...ij,...jk->...ik', proj_matrix, inv_term)
                # Normally a term for the residual here
                # We only choose starting points which obey the constraints, so r = 0
                cons_summation = np.einsum('...i,...ji->...j', dy_unconstrained, constraint_jac)
                cons_correction = np.einsum('...ij,...j->...i', first_term, cons_summation)
                dy_constrained = dy_unconstrained - cons_correction
                # TODO: Support for adaptive changing independent variable steps
                new_direction = dy_constrained[..., len(indep_vals):]
                # Backtracking line search
                new_points = points + INITIAL_STEP_SIZE * new_direction
                alpha = np.full(new_points.shape[:-1], INITIAL_STEP_SIZE, dtype=np.float)
                negative_points = np.any(new_points < 0., axis=-1)
                while np.any(negative_points):
                    alpha[negative_points] *= 0.1
                    new_points = points + alpha[..., np.newaxis] * new_direction
                    negative_points = np.any(new_points < 0., axis=-1)

                # Workaround for derivative issues at endmembers
                new_points[new_points == 0.] = 1e-16
                new_grad = phase_records[name].grad(*itertools.chain(statevar_grid, new_points.T))
                if new_grad.dtype == 'object':
                    # Workaround a bug in zero gradient entries
                    grad_zeros = np.zeros(new_points.T.shape[1:], dtype=np.float)
                    for i in np.arange(new_grad.shape[0]):
                        if isinstance(new_grad[i], int):
                            new_grad[i] = grad_zeros
                    new_grad = np.array(new_grad.tolist(), dtype=np.float)
                bcasts = np.broadcast_arrays(*itertools.chain(properties.MU.values.T, new_points.T))
                cast_grad = -plane_grad(*itertools.chain(bcasts, [0], [0]))
                cast_grad = cast_grad.T + new_grad.T
                new_grad = cast_grad
                new_grad.shape = new_grad.shape[:-1]  # Remove extraneous dimension
                new_grad = new_grad.astype(np.float)
                # BFGS update to Hessian
                # Notation used here consistent with Nocedal and Wright
                s_k = np.empty(points.shape[:-1] + (points.shape[-1] + len(indep_vals),))
                # Zero out independent variable changes for now
                s_k[..., :len(indep_vals)] = 0
                s_k[..., len(indep_vals):] = new_points - points
                tiny_step_filter = np.abs(s_k) < MIN_STEP_LENGTH
                # if all steps were too small, skip all Hessian updating
                # Nocedal and Wright recommend against skipping Hessian updates
                # They recommend using a damped update approach, pp. 538-539 of their book
                if ~np.all(tiny_step_filter):
                    y_k = new_grad - grad
                    s_s_term = np.einsum('...j,...k->...jk', s_k, s_k)
                    s_b_s_term = np.einsum('...i,...ij,...j', s_k, hess, s_k)
                    y_y_y_s_term = np.einsum('...j,...k->...jk', y_k, y_k) / \
                        np.einsum('...i,...i', y_k, s_k)[..., np.newaxis, np.newaxis]
                    update = np.einsum('...ij,...jk,...kl->...il', hess, s_s_term, hess) / \
                        s_b_s_term[..., np.newaxis, np.newaxis] + y_y_y_s_term
                    # Skip Hessian updates where step was too small to compute update
                    update[tiny_step_filter] = 0
                    hess = hess - update
                    cast_hess = -plane_hess(*itertools.chain(bcasts, [0], [0])).T + hess
                    hess = -cast_hess.astype(np.float, copy=False) #TODO: Why does this fix things?

                points = new_points
                grad = new_grad
                newton_iteration += 1
            new_points = new_points.reshape(new_points.shape[:len(indep_vals)] + (-1, new_points.shape[-1]))
            new_points = np.concatenate((current_site_fractions[..., :dof], new_points), axis=-2)
            points_dict[name] = new_points

        if verbose:
            print('Rebuilding grid', end=' ')
        grid = calculate(dbf, comps, active_phases, output='GM',
                         model=models, callables=callable_dict,
                         fake_points=True, points=points_dict, **grid_opts)
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
