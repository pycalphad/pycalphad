"""
The geometry module handles geometric calculations associated with
equilibrium calculation.
"""

from pycalphad.log import logger
from pycalphad.eq.cartesian import cartesian
import numpy as np
import xray
from itertools import chain

def _initialize_array(global_grid, result_array):
    "Fill in starting values for the energy array."
    max_energies = global_grid['GM'].max(dim='points', skipna=False)
    len_comps = result_array.dims['component']
    if max_energies.isnull().any():
        raise ValueError('Input energy surface contains one or more NaNs.')
    result_array['GM'] = xray.broadcast_arrays(max_energies, result_array['GM'])[0].copy()
    result_array['MU'] = xray.broadcast_arrays(max_energies, result_array['MU'])[0].copy()
    result_array['MU'].values[:] = np.nan
    result_array['NP'] = xray.broadcast_arrays(xray.DataArray(np.nan), result_array['NP'])[0].copy()
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
                                                                     'component']]
    indep_conds = sorted([x for x in sorted(result_array.coords.keys()) if x in ['T', 'P']])
    indep_shape = tuple(len(result_array.coords[x]) for x in indep_conds)
    comp_conds = sorted([x for x in sorted(result_array.coords.keys()) if x.startswith('X_')])
    comp_shape = tuple(len(result_array.coords[x]) for x in comp_conds)
    pot_conds = sorted([x for x in sorted(result_array.coords.keys()) if x.startswith('MU_')])
    # force conditions to have particular ordering
    conditions = indep_conds + pot_conds + comp_conds
    trial_shape = (len(result_array.coords['component']),)
    _initialize_array(global_grid, result_array)
    # Simplify indexing by partially ravelling the array
    grid_view = global_grid.X.values.view().reshape(-1, global_grid.X.values.shape[-1])

    # Enforce ordering of shape
    result_array['points'] = result_array['points'].transpose(*(conditions + ['vertex']))
    result_array['GM'] = result_array['GM'].transpose(*(conditions))
    result_array['NP'] = result_array['NP'].transpose(*(conditions + ['vertex']))

    # Determine starting combinations of chemical potentials and compositions
    # Check Gibbs phase rule compliance

    if len(pot_conds) > 0:
        raise NotImplementedError('Chemical potential conditions are not yet supported')

    # FIRST CASE: Only composition conditions specified
    # We only need to compute the dependent composition value directly
    # Initialize trial points as lowest energy point in the system
    if (len(comp_conds) > 0) and (len(pot_conds) == 0):
        trial_points = np.empty(result_array['GM'].T.shape)
        trial_points.fill(np.inf)
        trial_points[...] = global_grid['GM'].argmin(dim='points').values.T
        trial_points = trial_points.T
        comp_values = cartesian([result_array.coords[cond] for cond in comp_conds])
        # Insert dependent composition value
        comp_values = np.append(comp_values, 1 - np.sum(comp_values, keepdims=True,
                                                        axis=-1), axis=-1)

    # SECOND CASE: Only chemical potential conditions specified
    # TODO: Implementation of chemical potential

    # THIRD CASE: Mixture of composition and chemical potential conditions
    # TODO: Implementation of mixed conditions

    max_iterations = 50
    iterations = 0
    while iterations < max_iterations:
        iterations += 1

        trial_simplices = np.empty(result_array['points'].values.shape + \
                                   (result_array['points'].values.shape[-1],), dtype=np.int)
        # Initialize trial simplices with values from best guess simplices
        trial_simplices[..., :, :] = result_array['points'].values[..., np.newaxis, :]
        # Trial simplices will be the current simplex with each vertex
        #     replaced by the trial point
        # Exactly one of those simplices will contain a given test point,
        #     excepting edge cases
        trial_simplices.T[np.diag_indices(trial_shape[0])] = trial_points.T
        #print('trial_simplices.shape', trial_simplices.shape)
        #print('global_grid.X.values.shape', global_grid.X.values.shape)
        trial_matrix = grid_view[trial_simplices]
        #print('trial_matrix.shape', trial_matrix.shape)
        # Partially ravel the array to make indexing operations easier
        trial_matrix.shape = (-1,) + trial_matrix.shape[-2:]

        # We have to filter out degenerate simplices before
        #     phase fraction computation
        # This is because even one degenerate simplex causes the entire tensor
        #     to be singular
        nondegenerate_indices = np.all(np.linalg.svd(trial_matrix,
                                                     compute_uv=False) > 1e-12,
                                       axis=-1, keepdims=True)
        # Determine how many trial simplices remain for each target point.
        # In principle this would always be one simplex per point, but once
        # some target values reach equilibrium, trial_points starts
        # to contain points already on our best guess simplex.
        # This causes trial_simplices to create degenerate simplices.
        # We can safely filter them out since those target values are
        # already at equilibrium.
        sum_array = np.sum(nondegenerate_indices, axis=-1, dtype=np.int)
        index_array = np.repeat(np.arange(trial_matrix.shape[0], dtype=np.int),
                                sum_array)
        comp_shape = trial_simplices.shape[:len(indep_conds)+len(pot_conds)] + \
                     (comp_values.shape[0], trial_simplices.shape[-2])

        comp_indices = np.unravel_index(index_array, comp_shape)[len(indep_conds)+len(pot_conds)]

        fractions = np.linalg.solve(np.swapaxes(trial_matrix[index_array], -2, -1),
                                    comp_values[comp_indices])

        # A simplex only contains a point if its barycentric coordinates
        # (phase fractions) are positive.
        bounding_indices = np.all(fractions >= 0, axis=-1)
        index_array = np.atleast_1d(index_array[bounding_indices])

        raveled_simplices = trial_simplices.reshape((-1,) + trial_simplices.shape[-1:])
        candidate_simplices = raveled_simplices[..., index_array, :]

        # We need to convert the flat index arrays into multi-index tuples.
        # These tuples will tell us which state variable combinations are relevant
        # for the calculation. We can drop the last dimension, 'trial'.

        statevar_indices = np.unravel_index(index_array, trial_simplices.shape[:-1]
                                           )[:len(indep_conds)+len(pot_conds)]
        aligned_indices = np.broadcast_arrays(*chain(statevar_indices, [candidate_simplices.T]))
        aligned_energies = global_grid.GM.values[aligned_indices].T
        candidate_potentials = np.linalg.solve(grid_view[candidate_simplices],
                                               aligned_energies)
        logger.debug('candidate_simplices: %s', candidate_simplices)
        comp_indices = np.unravel_index(index_array, comp_shape)[len(indep_conds)+len(pot_conds)]
        candidate_energies = np.multiply(candidate_potentials,
                                         comp_values[comp_indices]).sum(axis=-1)

        # Generate a matrix of energies comparing our calculations for this iteration
        # to each other.
        # 'conditions' axis followed by a 'trial' axis
        # Empty values are filled in with infinity
        comparison_matrix = np.empty([trial_matrix.shape[0] / trial_shape[0],
                                      trial_shape[0]])
        comparison_matrix.fill(np.inf)
        comparison_matrix[np.divide(index_array, trial_shape[0]).astype(np.int),
                          np.mod(index_array, trial_shape[0])] = candidate_energies

        # If a condition point is all infinities, it means we did not calculate it
        # We should filter those out from any comparisons
        calculated_indices = ~np.all(comparison_matrix == np.inf, axis=-1)
        # Extract indices for trials with the lowest energy for each target point
        lowest_energy_indices = np.argmin(comparison_matrix[calculated_indices],
                                          axis=-1)
        # Filter conditions down to only those calculated this iteration
        calculated_conditions_indices = np.arange(comparison_matrix.shape[0])[calculated_indices]

        is_lower_energy = comparison_matrix[calculated_conditions_indices,
                                            lowest_energy_indices] < \
            result_array['GM'].values.flat[calculated_conditions_indices]

        # These are the conditions we will update this iteration
        final_indices = calculated_conditions_indices[is_lower_energy]
        # Convert to multi-index form so we can index the result array
        final_multi_indices = np.unravel_index(final_indices,
                                               result_array['GM'].values.shape)

        result_array['points'].values[final_multi_indices] = candidate_simplices[is_lower_energy]
        result_array['GM'].values[final_multi_indices] = candidate_energies[is_lower_energy]
        result_array['MU'].values[final_multi_indices] = candidate_potentials[is_lower_energy]
        result_array['NP'].values[final_multi_indices] = \
            fractions[bounding_indices][is_lower_energy]

        global_energies = global_grid.GM.values[final_multi_indices[:len(indep_conds)]]
        #print('global_energies.shape', global_energies.shape)
        #print('candidate_potentials[is_lower_energy][..., np.newaxis, :].shape', candidate_potentials[is_lower_energy][..., np.newaxis, :].shape)
        #print('global_grid.X.values[final_multi_indices[:len(indep_conds)]].shape', global_grid.X.values[final_multi_indices[:len(indep_conds)]].shape)
        raw_driving_forces = np.multiply(candidate_potentials[is_lower_energy][..., np.newaxis, :],
                                         global_grid.X.values[final_multi_indices[:len(indep_conds)]]).sum(axis=-1) - \
            global_energies
        #print('raw_driving_forces.shape', raw_driving_forces.shape)
        # In principle, any driving forces not in raw_driving_forces should be zero
        # This is why we only evaluate those points.
        # Update trial points to choose points with largest remaining driving force
        trial_points[final_multi_indices] = np.argmax(raw_driving_forces, axis=-1)
        logger.debug('trial_points: %s', trial_points)

        # If all driving force (within some tolerance) is consumed, we found equilibrium
        if np.all(raw_driving_forces < 1e-4):
            return
    logger.error('Iterations exceeded')
