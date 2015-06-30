"""
The geometry module handles geometric calculations associated with
equilibrium calculation.
"""

from pycalphad.log import logger
from pycalphad.eq.cartesian import cartesian
import numpy as np
import xray
import itertools

def _initialize_array(global_grid, result_array):
    "Fill in starting values for the energy array."
    max_energies = global_grid['GM'].max(dim='points', skipna=False)
    len_comps = result_array.dims['component']
    if max_energies.isnull().any():
        raise ValueError('Input energy surface contains one or more NaNs.')
    result_array['GM'] = xray.broadcast_arrays(max_energies, result_array['GM'])[0].copy()
    result_array['MU'] = xray.broadcast_arrays(max_energies, result_array['MU'])[0].copy()
    result_array['NP'] = xray.broadcast_arrays(xray.DataArray(0), result_array['NP'])[0].copy()
    # Initial simplex for each target point in will be
    #     the fictitious hyperplane
    # This hyperplane sits above the system's energy surface
    # The reason for this is to guarantee our initial simplex contains
    #     the target point
    # Note: We're assuming that the max energy is in the first few, presumably
    # fictitious points instead of more rigorously checking with argmax.
    result_array['points'] = xray.broadcast_arrays(xray.DataArray(np.arange(len_comps),
                                                                  dims='vertex'),
                                                   result_array['points'])[0].copy()

def lower_convex_hull(global_grid, result_array):
    """
    Find the simplices on the lower convex hull satisfying the specified
    conditions in the result array.

    Parameters
    ----------
    global_grid : Dataset
        A sample of the energy surface of the system.
    result_array : Dataset
        This object will be modified!
        Coordinates correspond to conditions axes.

    Returns
    -------
    None. Results are written to result_array.

    Notes
    -----
    This routine will not check if any simplex is degenerate.
    Degenerate simplices will manifest with duplicate or NaN indices.

    Examples
    --------
    None yet.
    """
    conditions = [x for x in result_array.coords.keys() if x not in ['vertex',
                                                                     'trial',
                                                                     'component']]
    indep_conds = sorted([x for x in result_array.coords.keys() if x in ['T', 'P', 'V']])
    comp_conds = sorted([x for x in result_array.coords.keys() if x.startswith('X_')])
    pot_conds = sorted([x for x in result_array.coords.keys() if x.startswith('MU_')])
    # force conditions to have particular ordering
    conditions = indep_conds + pot_conds + comp_conds
    comps = result_array.coords['component']
    if result_array.attrs['iterations'] == 0:
        _initialize_array(global_grid, result_array)

    # This is a view
    trial_simplices = result_array['points']
    # Enforce ordering of shape
    trial_simplices = trial_simplices.transpose(*(conditions + ['trial', 'vertex']))

    # Determine starting combinations of chemical potentials and compositions
    # Check Gibbs phase rule compliance

    if len(pot_conds) > 0:
        raise NotImplementedError('Chemical potential conditions are not yet supported')

    # FIRST CASE: Only composition conditions specified
    # We only need to compute the dependent composition value directly
    # Initialize trial points as lowest energy point in the system
    if (len(comp_conds) > 0) and (len(pot_conds) == 0):
        trial_points = global_grid['GM'].argmin(dim='points')
        comp_values = cartesian([result_array.coords[cond] for cond in comp_conds])
        # Insert dependent composition value
        comp_values = np.append(comp_values, 1 - np.sum(comp_values, keepdims=True,
                                                        axis=-1), axis=-1)
        # Force additional axes for broadcasting against other conditions
        additional_axes = (1,) * (len(conditions) - len(comp_conds))
        comp_values = np.reshape(comp_values, additional_axes + comp_values.shape)
        print(comp_values)

    # SECOND CASE: Only chemical potential conditions specified
    # TODO: Implementation of chemical potential

    # THIRD CASE: Mixture of composition and chemical potential conditions
    # TODO: Implementation of mixed conditions



    max_iterations = 50
    iterations = 0
    while iterations < max_iterations:
        iterations += 1
        # Trial simplices will be the current simplex with each vertex
        #     replaced by the trial point
        # Exactly one of those simplices will contain a given test point,
        #     excepting edge cases
        for diag in np.arange(len(comps)):
            trial_simplices[dict(vertex=diag, trial=diag)] = trial_points.values[..., np.newaxis]
        trial_matrix = global_grid.X.values[trial_simplices.values]
        # TODO: Fix this to use unravel_index() instead to save memory
        #[..., np.newaxis, np.newaxis, :] = [..., trial, vertex, component]
        dof_values = np.broadcast_arrays(comp_values[..., np.newaxis, np.newaxis, :],
                                         trial_matrix)[0]
        dof_values = dof_values.reshape(-1, dof_values.shape[-1])
        # Partially ravel the array to make indexing operations easier
        old_trial_matrix_shape = trial_matrix.shape
        trial_matrix.shape = (-1,) + trial_matrix.shape[-2:]
        # We have to filter out degenerate simplices before
        #     phase fraction computation
        # This is because even one degenerate simplex causes the entire tensor
        #     to be singular
        nondegenerate_indices = np.all(np.linalg.svd(trial_matrix,
                                                     compute_uv=False) > 1e-12,
                                       axis=-1)
        # Determine how many trial simplices remain for each target point.
        # In principle this would always be one simplex per point, but once
        # some target values reach equilibrium, trial_points starts
        # to contain points already on our best guess simplex.
        # This causes trial_simplices to create degenerate simplices.
        # We can safely filter them out since those target values are
        # already at equilibrium.
        fractions = np.linalg.solve(np.swapaxes(trial_matrix[nondegenerate_indices], -2, -1),
                                    dof_values[nondegenerate_indices])
        # Partially ravel the array to make indexing operations easier
        unraveled_simplices = trial_simplices.values.reshape(-1, trial_simplices.values.shape[-1])
        # A simplex only contains a point if its barycentric coordinates
        # (phase fractions) are positive.
        bounding_indices = np.all(fractions >= 0, axis=-1)
        candidate_indices = nondegenerate_indices & bounding_indices
        candidate_simplices = unraveled_simplices[candidate_indices]
        print(candidate_simplices)
        print(candidate_indices)
        print(old_trial_matrix_shape)
        aligned_sums = candidate_indices.reshape(old_trial_matrix_shape[0:-2]
                                                ).sum(axis=-1, keepdims=True)
        print(aligned_sums)
        aligned_indices = np.repeat(np.arange(len(aligned_sums.flat)), aligned_sums.flat)
        print(aligned_indices)
        aligned_energies = global_grid.GM.values.flat[candidate_simplices]
        print(candidate_simplices.shape)
        print(global_grid.X.values[candidate_simplices].shape)
        print(aligned_energies.shape)
        candidate_potentials = np.linalg.solve(global_grid.X.values[candidate_simplices],
                                               aligned_energies)
        logger.debug('candidate_simplices: %s', candidate_simplices)
        print(candidate_potentials)
        candidate_energies = np.multiply(candidate_potentials,
                                         dof_values[candidate_indices]).sum(axis=-1)
        print(candidate_energies)
        print(candidate_energies.shape)
        print(result_array.GM.shape)
        broadcast_energies = np.empty(result_array.GM.shape)
        broadcast_energies[...] = np.inf
        broadcast_energies.flat[candidate_indices] = candidate_energies
        broadcast_energies.sort(axis=-1)
        is_lower_energy = broadcast_energies < result_array.GM.values
        is_lower_energy.shape = (-1,)
        result_array.GM.values.flat[is_lower_energy] = broadcast_energies.ravel()
        result_array.MU.values.flat[is_lower_energy] = candidate_potentials.ravel()
        print(result_array.NP.values.flat[is_lower_energy])
        print(fractions.ravel())
        result_array.NP.values.flat[is_lower_energy] = fractions.ravel()

        return
        #logger.debug('target_values: %s', target_values.shape)
        #logger.debug('candidate_energies: %s', candidate_energies)
        # Generate a matrix of energies comparing our calculations for this iteration
        # to each other; one axis is for each trial, the other is for target values
        # This matrix may not have full rank because some target values had multiple trials
        # and some target values may not have been computed at all for this iteration
        # Empty values are filled in with infinity
        comparison_matrix = np.empty(dof_values.shape[0:-1] + \
                                        (candidate_simplices.shape[-2],),
                                     dtype=np.float)
        #logger.debug('comparison_matrix shape: %s', comparison_matrix.shape)
        comparison_matrix[...] = np.inf
        comparison_matrix[..., np.arange(dof_values.shape[-2]),
                          dof_index_array] = np.swapaxes(candidate_energies, -1, -2)
        # Extract indices for trials with the lowest energy for each target point
        lowest_candidate_indices = np.argmin(comparison_matrix, axis=-1)
        # Update simplices and energies when a trial simplex is lower energy for a point
        # than the current best guess
        is_lower_energy = (candidate_energies[..., np.arange(dof_values.shape[-2]),
                                              lowest_candidate_indices] <= dof_energies)
        #logger.debug('is_lower_energy: %s', is_lower_energy)
        dof_simplices = np.where(is_lower_energy[..., np.newaxis],
                                 candidate_simplices[lowest_candidate_indices], dof_simplices)
        dof_energies = np.minimum(dof_energies,
                                  candidate_energies[..., np.arange(dof_values.shape[-2]),
                                                     lowest_candidate_indices])
        #logger.debug('dof_energies: %s', dof_energies)
        #logger.debug('dof_simplices: %s', dof_simplices)
        #logger.debug('dat_coords[dof_simplices[is_lower_energy]].shape: %s',
        #             dat_coords[dof_simplices[is_lower_energy]].shape)
        dof_potentials[is_lower_energy] = \
            np.linalg.solve(dat_coords[dof_simplices[is_lower_energy]],
                            dat_energies[dof_simplices[is_lower_energy]])
        driving_forces[is_lower_energy, ...] = \
            np.tensordot(dof_potentials[is_lower_energy],
                         dat_coords, axes=(-1, -1)) - dat_energies
        # Update trial points to choose points with largest remaining driving force
        trial_points = np.argmax(driving_forces, axis=-1)
        logger.debug('trial_points: %s', trial_points)
        # If all driving force (within some tolerance) is consumed, we found equilibrium
        if np.all(driving_forces[..., trial_points] < 1e-4 * 8.3145 * temperature):
            final_matrix = np.swapaxes(dat_coords[dof_simplices], -1, -2)
            fractions = np.linalg.solve(final_matrix, dof_values)
            # Fix candidate simplex indices to remove fictitious points
            dof_simplices = dof_simplices - (len(dof)-1)
            logger.debug('Adjusted dof_simplices: %s', dof_simplices)
            # TODO: Remove fictitious points from the candidate simplex
            # These can inadvertently show up if we only calculate a phase with
            # limited solubility
            # Also remove points with very small estimated phase fractions
            return dof_simplices, fractions, dof_potentials

    logger.error('Iterations exceeded')
    logger.debug(driving_forces[..., trial_points])
    return None, None, None
