"""
The geometry module handles geometric calculations associated with
equilibrium calculation.
"""

import pycalphad.variables as v
import numpy as np
from pycalphad.log import logger

def lower_convex_hull(data, comps, conditions):
    """
    Find the simplex on the lower convex hull satisfying the specified
    conditions.

    Parameters
    ----------
    data : DataFrame
        A sample of the energy surface of the system.
    comps : list
        All the components in the system.
    conditions : dict
        StateVariables and their corresponding value.

    Returns
    -------
    A tuple containing:
    (1) A numpy array of indices corresponding to vertices of the simplex.
    (2) A numpy array corresponding to the phase fractions.
    (3) A numpy array of chemical potentials in sorted(comps) order (no 'VA')
    Note: This routine will not check if the simplex is degenerate.

    Examples
    --------
    None yet.
    """
    # determine column indices for degrees of freedom
    comps = sorted(list(comps))
    dof = ['X({0})'.format(c) for c in comps if c != 'VA']
    dof_values = np.zeros(len(dof))
    marked_dof_values = list(range(len(dof)))
    for cond, value in conditions.items():
        if not isinstance(cond, v.Composition):
            continue
        # ignore phase-specific composition conditions
        if cond.phase_name is not None:
            continue
        if cond.species == 'VA':
            continue
        dof_values[dof.index('X({0})'.format(cond.species))] = value
        marked_dof_values.remove(dof.index('X({0})'.format(cond.species)))

    dof.append('GM')

    if len(marked_dof_values) == 1:
        dof_values[marked_dof_values[0]] = 1-sum(dof_values)
    else:
        logger.error('Not enough composition conditions specified')
        raise ValueError('Not enough composition conditions specified.')

    # convert DataFrame of independent columns to ndarray
    dat_coords = np.concatenate((np.eye(len(dof)-1),
                                 data[dof].values[:, :-1]), axis=-2)
    max_energy = np.amax(data[dof].values[:, -1])
    if np.isnan(max_energy):
        raise ValueError('Input energy surface contains one or more NaNs.')
    if max_energy < 0:
        max_energy *= 0.5
    else:
        max_energy *= 2.0
    dat_energies = np.concatenate((np.repeat(max_energy, len(dof)-1),
                                   data[dof].values[:, -1]), axis=-2)
    temperature = data.iloc[0].loc['T']
    dof_values = dof_values[np.newaxis, :]

    dof_energies = np.empty(dof_values.shape[0:-1], dtype=np.float)
    dof_energies[...] = np.inf
    dof_simplices = np.empty(dof_values.shape, dtype=np.int)
    # Initial simplex for each target point in dof_values will be
    #     the fictitious hyperplane
    # This hyperplane sits above the system's energy surface
    # The reason for this is to guarantee our initial simplex contains
    #     the target point
    dof_simplices[..., :] = np.arange(dat_coords.shape[-1])
    dof_potentials = np.empty(dof_values.shape, dtype=np.float)
    dof_potentials[...] = np.inf
    driving_forces = np.empty(dof_values.shape[0:-1] + (dat_coords.shape[-2],),
                              dtype=np.float)
    trial_points = np.empty(dof_values.shape[0:-1], dtype=np.int)
    # Initialize trial points as lowest energy point in the system
    trial_points[...] = np.argmin(dat_energies, axis=-1)
    max_iterations = 50
    iterations = 0
    while iterations < max_iterations:
        iterations += 1
        trial_simplices = np.empty(dof_values.shape + (dat_coords.shape[-1],),
                                   dtype=np.int)
        # Trial simplices based on current best guess simplex for each
        #     target point in dof_values
        trial_simplices[..., :, :] = dof_simplices[..., np.newaxis, :]
        # Trial simplices will be the current simplex with each vertex
        #     replaced by the trial point
        # Exactly one of those simplices will contain a given test point,
        #     excepting edge cases
        trial_simplices[..., np.arange(dat_coords.shape[-1]),
                        np.arange(dat_coords.shape[-1])] = \
            trial_points[..., np.newaxis]
        trial_matrix = np.swapaxes(dat_coords[trial_simplices], -1, -2)
        # We have to filter out degenerate simplices before
        #     phase fraction computation
        # This is because even one degenerate simplex causes the entire tensor
        #     to be singular
        nondegenerate_indices = np.all(np.linalg.svd(trial_matrix,
                                                     compute_uv=False) > 1e-12,
                                       axis=-1)
        nondegenerate_simplices = trial_simplices[nondegenerate_indices]
        # Determine how many trial simplices remain for each target point
        # in dof_values. In principle this would always be one simplex per
        # point, but once some target values reach equilibrium, trial_points starts
        # to contain points already on our best guess simplex.
        # This causes trial_simplices to create degenerate simplices.
        # We can safely filter them out since those target values are
        # already at equilibrium.
        dof_sum_array = np.sum(nondegenerate_indices, axis=-1, dtype=np.int)
        dof_index_array = np.repeat(np.arange(dof_values.shape[-2],
                                              dtype=np.int),
                                    dof_sum_array.astype(np.int))
        fractions = np.linalg.solve(trial_matrix[nondegenerate_indices],
                                    dof_values[dof_index_array])
        # A simplex only contains a point if its barycentric coordinates
        # (phase fractions) are positive.
        bounding_simplices = np.all(fractions >= 0, axis=-1)
        candidate_simplices = nondegenerate_simplices[bounding_simplices]
        candidate_potentials = np.linalg.solve(dat_coords[candidate_simplices],
                                               dat_energies[candidate_simplices])
        logger.debug('candidate_simplices: %s', candidate_simplices)
        dof_index_array = dof_index_array[bounding_simplices]

        target_values = dof_values[dof_index_array]
        candidate_energies = np.empty(candidate_potentials.shape[0:-1] + (1,))
        candidate_energies[...] = np.multiply(candidate_potentials,
                                              target_values).sum(axis=-1,
                                                                 keepdims=True)
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
