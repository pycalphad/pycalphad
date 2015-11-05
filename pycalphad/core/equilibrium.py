"""
The equilibrium module defines routines for interacting with
calculated phase equilibria.
"""
from __future__ import print_function
import pycalphad.variables as v
from pycalphad.core.utils import broadcast_to
from pycalphad.core.utils import unpack_kwarg
from pycalphad.core.utils import sizeof_fmt
from pycalphad.core.utils import unpack_condition, unpack_phases
from pycalphad import calculate, Model
from pycalphad.constraints import mole_fraction
from pycalphad.core.lower_convex_hull import lower_convex_hull
from pycalphad.core.autograd_utils import build_functions
from sympy import Add, Mul, Symbol

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

PhaseRecord = namedtuple('PhaseRecord', ['variables', 'grad', 'hess', 'plane_grad', 'plane_hess'])

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
    hess_callable_dict = kwargs.pop('hess_callables', dict())
    points_dict = dict()
    maximum_internal_dof = 0
    conds = OrderedDict((key, unpack_condition(value)) for key, value in sorted(conditions.items(), key=str))
    str_conds = OrderedDict((str(key), value) for key, value in conds.items())
    indep_vals = list([float(x) for x in np.atleast_1d(val)]
                      for key, val in str_conds.items() if key in indep_vars)
    components = [x for x in sorted(comps) if not x.startswith('VA')]
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
        undefs = list(out.atoms(Symbol) - out.atoms(v.StateVariable))
        for undef in undefs:
            out = out.xreplace({undef: float(0)})
        callable_dict[name], grad_callable_dict[name], hess_callable_dict[name] = \
            build_functions(out, [v.P, v.T] + site_fracs)

        # Adjust gradient by the approximate chemical potentials
        hyperplane = Add(*[v.MU(i)*mole_fraction(dbf.phases[name], comps, i)
                           for i in comps if i != 'VA'])
        plane_obj, plane_grad, plane_hess = build_functions(hyperplane, [v.MU(i) for i in comps if i != 'VA']+site_fracs)
        phase_records[name.upper()] = PhaseRecord(variables=variables,
                                                  grad=grad_callable_dict[name],
                                                  hess=hess_callable_dict[name],
                                                  plane_grad=plane_grad,
                                                  plane_hess=plane_hess)
        if verbose:
            print(name, end=' ')
    if verbose:
        print('[done]', end='\n')

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
        if np.isnan(grid.GM.values).any():
            print(grid.Y.values[np.isnan(grid.GM.values)])
            print(grid.Phase.values[np.isnan(grid.GM.values)])

        lower_convex_hull(grid, properties)
        progress = np.abs(current_potentials - properties.MU).values
        converged = (progress < MIN_PROGRESS).all(axis=-1)
        if verbose:
            print('progress', progress.max(), '[{} conditions updated]'.format(np.sum(~converged)))
        if progress.max() < MIN_PROGRESS:
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
            #print('Starting points shape: ', points.shape)
            #print(points)
            if len(points) == 0:
                if name in points_dict:
                    del points_dict[name]
                # No nearly stable points: skip this phase
                continue

            num_vars = len(phase_records[name].variables)
            plane_grad = phase_records[name].plane_grad
            plane_hess = phase_records[name].plane_hess
            statevar_grid = np.meshgrid(*itertools.chain(indep_vals), sparse=True, indexing='ij')
            # TODO: A more sophisticated treatment of constraints
            num_constraints = len(dbf.phases[name].sublattices)
            constraint_jac = np.zeros((num_constraints, num_vars-len(indep_vars)))
            # Independent variables are always fixed (in this limited implementation)
            #for idx in range(len(indep_vals)):
            #    constraint_jac[idx, idx] = 1
            # This is for site fraction balance constraints
            var_idx = 0#len(indep_vals)
            for idx in range(len(dbf.phases[name].sublattices)):
                active_in_subl = set(dbf.phases[name].constituents[idx]).intersection(comps)
                constraint_jac[idx,
                               var_idx:var_idx + len(active_in_subl)] = 1
                var_idx += len(active_in_subl)

            # Theano functions require the same number of dimensions for variables as initially defined
            # It's easier to flatten and reshape after the fact
            #print('points.shape', points.shape)
            flattened_points = points.reshape(points.shape[:len(indep_vals)] + (-1, points.shape[-1]))
            #print('Starting flattened points shape:', flattened_points.shape)
            #print('flattened_points.shape', flattened_points.shape)
            grad_args = itertools.chain([i[..., None] for i in statevar_grid],
                                        [flattened_points[..., i] for i in range(flattened_points.shape[-1])])
            grad = np.array(phase_records[name].grad(*grad_args), dtype=np.float)
            # Remove derivatives wrt T,P
            grad = grad[..., len(indep_vars):]
            grad.shape = points.shape
            hess_args = itertools.chain([i[..., None] for i in statevar_grid],
                                        [flattened_points[..., i] for i in range(flattened_points.shape[-1])])
            hess = np.array(phase_records[name].hess(*hess_args), dtype=np.float)
            # Remove derivatives wrt T,P
            hess = hess[..., len(indep_vars):, len(indep_vars):]
            hess.shape = points.shape + (hess.shape[-1],)
            #print(grad)
            #print('Grad check: ', np.isnan(grad).any())
            if np.isnan(grad).any():
                print(points)
            if np.isnan(hess).any():
                print(points)
            #print('after reshape', grad.shape)
            #print([properties.MU.values[..., i][..., None].shape for i in range(properties.MU.shape[-1])])
            #print([points[..., i].shape for i in range(points.shape[-1])])
            plane_args = itertools.chain([properties.MU.values[..., i][..., None] for i in range(properties.MU.shape[-1])],
                                         [points[..., i] for i in range(points.shape[-1])])
            cast_grad = np.array(plane_grad(*plane_args), dtype=np.float)
            # Remove derivatives wrt chemical potentials
            #print('cast_grad_before.shape', cast_grad.shape)
            #print('cast_grad_before', cast_grad)
            cast_grad = cast_grad[..., properties.MU.shape[-1]:]
            #print('properties.MU.values', properties.MU.values)
            #print('new_points', points)
            #print('cast_grad', cast_grad)
            grad = grad - cast_grad
            grad[np.isnan(grad).any(axis=-1)] = 0  # This is necessary for gradients on the edge of composition space
            plane_args = itertools.chain([properties.MU.values[..., i][..., None] for i in range(properties.MU.shape[-1])],
                                         [points[..., i] for i in range(points.shape[-1])])
            cast_hess = np.array(plane_hess(*plane_args), dtype=np.float)
            # Remove derivatives wrt chemical potentials
            cast_hess = cast_hess[..., properties.MU.shape[-1]:, properties.MU.shape[-1]:]
            hess = hess - cast_hess
            hess[np.isnan(hess).any(axis=(-2, -1))] = np.eye(hess.shape[-1])
            #print(grad)
            #print(grad.shape)
            newton_iteration = 0
            while newton_iteration < MAX_NEWTON_ITERATIONS:
                try:
                    e_matrix = np.linalg.inv(hess)
                except np.linalg.LinAlgError:
                    print(hess)
                    raise
                current = calculate(dbf, comps, name, output='GM',
                                    model=models, callables=callable_dict,
                                    fake_points=False,
                                    points=points.reshape(points.shape[:len(indep_vals)] + (-1, points.shape[-1])),
                                    **grid_opts)
                current_plane = np.multiply(current.X.values.reshape(points.shape[:-1] + (len(components),)),
                                            properties.MU.values[..., np.newaxis, :]).sum(axis=-1)
                current_df = current.GM.values.reshape(points.shape[:-1]) - current_plane
                #print('Inv hess check: ', np.isnan(e_matrix).any())
                #print('grad check: ', np.isnan(grad).any())
                dy_unconstrained = -np.einsum('...ij,...j->...i', e_matrix, grad)
                #print('dy_unconstrained check: ', np.isnan(dy_unconstrained).any())
                proj_matrix = np.dot(e_matrix, constraint_jac.T)
                inv_matrix = np.rollaxis(np.dot(constraint_jac, proj_matrix), 0, -1)
                inv_term = np.linalg.inv(inv_matrix)
                #print('inv_term check: ', np.isnan(inv_term).any())
                first_term = np.einsum('...ij,...jk->...ik', proj_matrix, inv_term)
                #print('first_term check: ', np.isnan(first_term).any())
                # Normally a term for the residual here
                # We only choose starting points which obey the constraints, so r = 0
                cons_summation = np.einsum('...i,...ji->...j', dy_unconstrained, constraint_jac)
                #print('cons_summation check: ', np.isnan(cons_summation).any())
                cons_correction = np.einsum('...ij,...j->...i', first_term, cons_summation)
                #print('cons_correction check: ', np.isnan(cons_correction).any())
                dy_constrained = dy_unconstrained - cons_correction
                #print('dy_constrained check: ', np.isnan(dy_constrained).any())
                # TODO: Support for adaptive changing independent variable steps
                new_direction = dy_constrained
                #print('new_direction', new_direction)
                #print('points', points)
                # Backtracking line search
                if np.isnan(new_direction).any():
                    print('new_direction', new_direction)
                new_points = points + INITIAL_STEP_SIZE * new_direction
                # TODO: Breakpoint for if new_direction gets tiny
                #print('new_points check: ', np.isnan(new_points).any())
                alpha = np.full(new_points.shape[:-1], INITIAL_STEP_SIZE, dtype=np.float)
                alpha[np.all(np.abs(new_direction) < 1e-12, axis=-1)] = 0
                negative_points = np.any(new_points < 0., axis=-1)
                while np.any(negative_points):
                    alpha[negative_points] *= 0.5
                    new_points = points + alpha[..., np.newaxis] * new_direction
                    negative_points = np.any(new_points < 0., axis=-1)
                # Backtracking line search
                # alpha now contains maximum possible values that keep us inside the space
                # but we don't just want to take the biggest step; we want the biggest step which reduces energy
                new_points = new_points.reshape(new_points.shape[:len(indep_vals)] + (-1, new_points.shape[-1]))
                candidates = calculate(dbf, comps, name, output='GM',
                                       model=models, callables=callable_dict,
                                       fake_points=False, points=new_points, **grid_opts)
                candidate_plane = np.multiply(candidates.X.values.reshape(points.shape[:-1] + (len(components),)),
                                              properties.MU.values[..., np.newaxis, :]).sum(axis=-1)
                energy_diff = (candidates.GM.values.reshape(new_direction.shape[:-1]) - candidate_plane) - current_df
                new_points.shape = new_direction.shape
                bad_steps = energy_diff > alpha * 0.5 * (new_direction * grad).sum(axis=-1)
                safe_break = 0
                #print('points', points)
                while np.any(bad_steps):
                    safe_break += 1
                    if safe_break > 500:
                        print('SAFE BREAK')
                        break
                    alpha[bad_steps] *= 0.5
                    new_points = points + alpha[..., np.newaxis] * new_direction
                    #print('new_points', new_points)
                    #print('bad_steps', bad_steps)
                    new_points = new_points.reshape(new_points.shape[:len(indep_vals)] + (-1, new_points.shape[-1]))
                    candidates = calculate(dbf, comps, name, output='GM',
                                           model=models, callables=callable_dict,
                                           fake_points=False, points=new_points, **grid_opts)
                    candidate_plane = np.multiply(candidates.X.values.reshape(points.shape[:-1] + (len(components),)),
                                                  properties.MU.values[..., np.newaxis, :]).sum(axis=-1)
                    energy_diff = (candidates.GM.values.reshape(new_direction.shape[:-1]) - candidate_plane) - current_df
                    #print('energy_diff', energy_diff)
                    new_points.shape = new_direction.shape
                    bad_steps = energy_diff > alpha * 0.5 * (new_direction * grad).sum(axis=-1)
                biggest_step = np.max(np.linalg.norm(new_points - points, axis=-1))
                if biggest_step < 1e-12:
                    if verbose:
                        print('N-R convergence on mini-iteration', newton_iteration)
                    points = new_points
                    break
                if verbose:
                    #print('Biggest step:', biggest_step)
                    #print('points', points)
                    #print('grad of points', grad)
                    #print('cast grad', cast_grad)
                    #print('alpha', alpha)
                    #print('new_points', new_points)
                    pass

                flattened_points = new_points.reshape(new_points.shape[:len(indep_vals)] + (-1, new_points.shape[-1]))
                grad_args = itertools.chain([i[..., None] for i in statevar_grid],
                                            [flattened_points[..., i] for i in range(flattened_points.shape[-1])])
                grad = np.array(phase_records[name].grad(*grad_args), dtype=np.float)
                # Remove derivatives wrt T,P
                grad = grad[..., len(indep_vars):]
                grad.shape = new_points.shape
                grad[np.isnan(grad).any(axis=-1)] = 0  # This is necessary for gradients on the edge of composition space
                hess_args = itertools.chain([i[..., None] for i in statevar_grid],
                                            [flattened_points[..., i] for i in range(flattened_points.shape[-1])])
                hess = np.array(phase_records[name].hess(*hess_args), dtype=np.float)
                # Remove derivatives wrt T,P
                hess = hess[..., len(indep_vars):, len(indep_vars):]
                hess.shape = new_points.shape + (hess.shape[-1],)
                hess[np.isnan(hess).any(axis=(-2, -1))] = np.eye(hess.shape[-1])
                plane_args = itertools.chain([properties.MU.values[..., i][..., None] for i in range(properties.MU.shape[-1])],
                                             [new_points[..., i] for i in range(new_points.shape[-1])])
                cast_grad = np.array(plane_grad(*plane_args), dtype=np.float)
                # Remove derivatives wrt chemical potentials
                cast_grad = cast_grad[..., properties.MU.shape[-1]:]
                grad = grad - cast_grad
                plane_args = itertools.chain([properties.MU.values[..., i][..., None] for i in range(properties.MU.shape[-1])],
                                             [new_points[..., i] for i in range(new_points.shape[-1])])
                cast_hess = np.array(plane_hess(*plane_args), dtype=np.float)
                # Remove derivatives wrt chemical potentials
                cast_hess = cast_hess[..., properties.MU.shape[-1]:, properties.MU.shape[-1]:]
                cast_hess = -cast_hess + hess
                hess = cast_hess.astype(np.float, copy=False)
                points = new_points
                newton_iteration += 1
            new_points = points.reshape(points.shape[:len(indep_vals)] + (-1, points.shape[-1]))
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
