"""
The lower_convex_hull module handles geometric calculations associated with
equilibrium calculation.
"""
from __future__ import print_function
from pycalphad.log import logger
from pycalphad.core.cartesian import cartesian
from pycalphad.core.constants import MIN_SITE_FRACTION
import numpy as np

# The energetic difference, in J/mol-atom, below which is considered 'zero'
DRIVING_FORCE_TOLERANCE = 1e-8

def _initialize_array(global_grid, result_array):
    "Fill in starting values for the energy array."
    # Profiling says it's way faster to compute one global max_energy value
    # than to compute it per condition and do a broadcast assignment
    # This will cause some minor differences in the driving force for the first few iterations
    # but it shouldn't be a big deal
    max_energy = global_grid['GM'].values.max()
    len_comps = result_array.dims['component']
    if max_energy == np.nan:
        raise ValueError('Input energy surface contains one or more NaNs.')
    result_array['GM'].values[...] = max_energy
    result_array['MU'].values[...] = np.nan
    result_array['NP'].values[...] = np.nan
    # Initial simplex for each target point in will be
    #     the fictitious hyperplane
    # This hyperplane sits above the system's energy surface
    # The reason for this is to guarantee our initial simplex contains
    #     the target point
    # Note: We're assuming that the max energy is in the first few, presumably
    # fictitious points instead of more rigorously checking with argmax.
    result_array['points'].values[...] = np.arange(len_comps)

def stacked_lstsq(L, b, rcond=1e-10):
    """
    Solve L x = b, via SVD least squares cutting of small singular values
    L is an array of shape (..., M, N) and b of shape (..., M).
    Returns x of shape (..., N)
    This code is necessary until stacked arrays are supported in numpy.linalg.lstsq.
    Source: http://stackoverflow.com/questions/30442377/how-to-solve-many-overdetermined-systems-of-linear-equations-using-vectorized-co
    """
    u, s, v = np.linalg.svd(L, full_matrices=False)
    s_max = s.max(axis=-1, keepdims=True)
    s_min = rcond*s_max
    inv_s = np.zeros_like(s)
    inv_s[s >= s_min] = 1/s[s>=s_min]
    x = np.einsum('...ji,...j->...i', v,
                  inv_s * np.einsum('...ji,...j->...i', u, b.conj()))
    return np.conj(x, x)

def lower_convex_hull(global_grid, result_array, verbose=False):
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
    verbose : bool
        Display details to stdout. Useful for debugging.

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
    trial_points = None
    _initialize_array(global_grid, result_array)

    # Enforce ordering of shape if this is the first iteration
    if result_array.attrs['hull_iterations'] == 1:
        result_array['points'] = result_array['points'].transpose(*(conditions + ['vertex']))
        result_array['GM'] = result_array['GM'].transpose(*conditions)
        result_array['NP'] = result_array['NP'].transpose(*(conditions + ['vertex']))

    # Determine starting combinations of chemical potentials and compositions
    # TODO: Check Gibbs phase rule compliance

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
        # TODO: Handle W(comp) as well as X(comp) here
        specified_components = set([x[2:] for x in comp_conds])
        dependent_component = set(result_array.coords['component'].values) - specified_components
        dependent_component = list(dependent_component)
        if len(dependent_component) != 1:
            raise ValueError('Number of dependent components is different from one')
        insert_idx = sorted(result_array.coords['component'].values).index(dependent_component[0])
        comp_values = np.concatenate((comp_values[..., :insert_idx],
                                      1 - np.sum(comp_values, keepdims=True, axis=-1),
                                      comp_values[..., insert_idx:]),
                                     axis=-1)
        # Prevent compositions near an edge from going negative
        comp_values[np.nonzero(comp_values < MIN_SITE_FRACTION)] = MIN_SITE_FRACTION*10
        # TODO: Assumes N=1
        comp_values /= comp_values.sum(axis=-1, keepdims=True)
        #print(comp_values)

    # SECOND CASE: Only chemical potential conditions specified
    # TODO: Implementation of chemical potential

    # THIRD CASE: Mixture of composition and chemical potential conditions
    # TODO: Implementation of mixed conditions

    if trial_points is None:
        raise ValueError('Invalid conditions')

    driving_forces = np.zeros(result_array.GM.values.shape + (len(global_grid.points),),
                                   dtype=np.float)

    max_iterations = 200
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
        flat_statevar_indices = np.unravel_index(np.arange(np.multiply.reduce(result_array.MU.values.shape)),
                                                 result_array.MU.values.shape)[:len(indep_conds)]
        #print('flat_statevar_indices', flat_statevar_indices)
        trial_matrix = global_grid.X.values[np.index_exp[flat_statevar_indices +
                                                         (trial_simplices.reshape(-1, trial_simplices.shape[-1]).T,)]]
        trial_matrix = np.rollaxis(trial_matrix, 0, -1)
        #print('trial_matrix', trial_matrix)
        # Partially ravel the array to make indexing operations easier
        trial_matrix.shape = (-1,) + trial_matrix.shape[-2:]

        # We have to filter out degenerate simplices before
        #     phase fraction computation
        # This is because even one degenerate simplex causes the entire tensor
        #     to be singular
        nondegenerate_indices = np.all(np.linalg.svd(trial_matrix,
                                                     compute_uv=False) > 1e-09,
                                       axis=-1, keepdims=True)
        #('NONDEGENERATE INDICES', nondegenerate_indices)
        # Determine how many trial simplices remain for each target point.
        # In principle this would always be one simplex per point, but once
        # some target values reach equilibrium, trial_points starts
        # to contain points already on our best guess simplex.
        # This causes trial_simplices to create degenerate simplices.
        # We can safely filter them out since those target values are
        # already at equilibrium.
        #sum_array = np.sum(nondegenerate_indices, axis=-1, dtype=np.int)
        #index_array = np.repeat(np.arange(trial_matrix.shape[0], dtype=np.int),
        #                        sum_array)
        index_array = np.arange(trial_matrix.shape[0], dtype=np.int)
        comp_shape = trial_simplices.shape[:len(indep_conds)+len(pot_conds)] + \
                     (comp_values.shape[0], trial_simplices.shape[-2])

        comp_indices = np.unravel_index(index_array, comp_shape)[len(indep_conds)+len(pot_conds)]
        fractions = np.full(result_array['points'].values.shape + \
                            (result_array['points'].values.shape[-1],), -1.)
        fractions[np.unravel_index(index_array, fractions.shape[:-1])] = \
            stacked_lstsq(np.swapaxes(trial_matrix[index_array], -2, -1),
                          comp_values[comp_indices])
        fractions /= fractions.sum(axis=-1, keepdims=True)
        #print('fractions', fractions)
        # A simplex only contains a point if its barycentric coordinates
        # (phase fractions) are non-negative.
        bounding_indices = np.all(fractions >= -MIN_SITE_FRACTION*100, axis=-1)
        #print('BOUNDING INDICES', bounding_indices)
        if ~np.any(bounding_indices):
            raise ValueError('Desired composition is not inside any candidate simplex. This is a bug.')
        multiple_success_trials = np.sum(bounding_indices, axis=-1, dtype=np.int, keepdims=False) != 1
        #print('MULTIPLE SUCCESS TRIALS SHAPE', np.nonzero(multiple_success_trials))
        if np.any(multiple_success_trials):
            saved_trial = np.zeros_like(multiple_success_trials, dtype=np.int)
            # Case of only degenerate simplices (zero bounding)
            # Choose trial with "least negative" fraction
            zero_success_indices = np.logical_and(~nondegenerate_indices.reshape(bounding_indices.shape),
                                                  multiple_success_trials[..., np.newaxis])
            saved_trial[np.nonzero(zero_success_indices.any(axis=-1))] = \
                np.argmax(fractions[np.nonzero(zero_success_indices.any(axis=-1))].min(axis=-1), axis=-1)
            # Case of multiple bounding non-degenerate simplices
            # Choose the first one. This addresses gh-28.
            multiple_bounding_indices = \
                np.logical_and(np.logical_and(bounding_indices, nondegenerate_indices.reshape(bounding_indices.shape)),
                               multiple_success_trials[..., np.newaxis])
            #print('MULTIPLE SUCCESS TRIALS.shape', multiple_success_trials.shape)
            #print('BOUNDING INDICES.shape', bounding_indices.shape)
            #print('MULTIPLE_BOUNDING_INDICES.shape', multiple_bounding_indices.shape)
            saved_trial[np.nonzero(multiple_bounding_indices.any(axis=-1))] = \
                np.argmax(multiple_bounding_indices[np.nonzero(multiple_bounding_indices.any(axis=-1))], axis=-1)
            #print('SAVED TRIAL.shape', saved_trial.shape)
            #print('BOUNDING INDICES BEFORE', bounding_indices)
            bounding_indices[np.nonzero(multiple_success_trials)] = False
            #print('BOUNDING INDICES FALSE', bounding_indices)
            bounding_indices[np.nonzero(multiple_success_trials) + \
                             (saved_trial[np.nonzero(multiple_success_trials)],)] = True
            #print('BOUNDING INDICES AFTER', bounding_indices)
        fractions.shape = (-1, fractions.shape[-1])
        bounding_indices.shape = (-1,)
        index_array = np.arange(trial_matrix.shape[0], dtype=np.int)[bounding_indices]

        raveled_simplices = trial_simplices.reshape((-1,) + trial_simplices.shape[-1:])
        candidate_simplices = raveled_simplices[index_array, :]
        #print('candidate_simplices', candidate_simplices)

        # We need to convert the flat index arrays into multi-index tuples.
        # These tuples will tell us which state variable combinations are relevant
        # for the calculation. We can drop the last dimension, 'trial'.
        #print('trial_simplices.shape[:-1]', trial_simplices.shape[:-1])
        statevar_indices = np.unravel_index(index_array, trial_simplices.shape[:-1]
                                            )[:len(indep_conds)+len(pot_conds)]
        aligned_energies = global_grid.GM.values[statevar_indices + (candidate_simplices.T,)].T
        statevar_indices = tuple(x[..., np.newaxis] for x in statevar_indices)
        #print('statevar_indices', statevar_indices)
        aligned_compositions = global_grid.X.values[np.index_exp[statevar_indices + (candidate_simplices,)]]
        #print('aligned_compositions', aligned_compositions)
        #print('aligned_energies', aligned_energies)
        candidate_potentials = stacked_lstsq(aligned_compositions.astype(np.float, copy=False),
                                             aligned_energies.astype(np.float, copy=False))
        #print('candidate_potentials', candidate_potentials)
        logger.debug('candidate_simplices: %s', candidate_simplices)
        comp_indices = np.unravel_index(index_array, comp_shape)[len(indep_conds)+len(pot_conds)]
        #print('comp_values[comp_indices]', comp_values[comp_indices])
        candidate_energies = np.multiply(candidate_potentials,
                                         comp_values[comp_indices]).sum(axis=-1)
        #print('candidate_energies', candidate_energies)

        # Generate a matrix of energies comparing our calculations for this iteration
        # to each other.
        # 'conditions' axis followed by a 'trial' axis
        # Empty values are filled in with infinity
        comparison_matrix = np.empty([int(trial_matrix.shape[0] / trial_shape[0]),
                                      trial_shape[0]])
        if comparison_matrix.shape[0] != aligned_compositions.shape[0]:
            raise ValueError('Arrays have become misaligned. This is a bug. Try perturbing your composition conditions '
                             'by a small amount (1e-4). If you would like, you can report this issue to the development'
                             ' team and they will fix it for future versions.')
        comparison_matrix.fill(np.inf)
        comparison_matrix[np.divide(index_array, trial_shape[0]).astype(np.int),
                          np.mod(index_array, trial_shape[0])] = candidate_energies
        #print('comparison_matrix', comparison_matrix)

        # If a condition point is all infinities, it means we did not calculate it
        # We should filter those out from any comparisons
        calculated_indices = ~np.all(comparison_matrix == np.inf, axis=-1)
        # Extract indices for trials with the lowest energy for each target point
        lowest_energy_indices = np.argmin(comparison_matrix[calculated_indices],
                                          axis=-1)
        # Filter conditions down to only those calculated this iteration
        calculated_conditions_indices = np.arange(comparison_matrix.shape[0])[calculated_indices]

        #print('comparison_matrix[calculated_conditions_indices,lowest_energy_indices]',comparison_matrix[calculated_conditions_indices,
        #                                    lowest_energy_indices])
        # This has to be greater-than-or-equal because, in the case where
        # the desired condition is right on top of a simplex vertex (gh-28), there
        # will be no change in energy changing a "_FAKE_" vertex to a real one.
        is_lower_energy = comparison_matrix[calculated_conditions_indices,
                                            lowest_energy_indices] <= \
            result_array['GM'].values.flat[calculated_conditions_indices]
        #print('is_lower_energy', is_lower_energy)

        # These are the conditions we will update this iteration
        final_indices = calculated_conditions_indices[is_lower_energy]
        #print('final_indices', final_indices)
        # Convert to multi-index form so we can index the result array
        final_multi_indices = np.unravel_index(final_indices,
                                               result_array['GM'].values.shape)

        updated_potentials = candidate_potentials[is_lower_energy]
        result_array['points'].values[final_multi_indices] = candidate_simplices[is_lower_energy]
        result_array['GM'].values[final_multi_indices] = candidate_energies[is_lower_energy]
        result_array['MU'].values[final_multi_indices] = updated_potentials
        result_array['NP'].values[final_multi_indices] = \
            fractions[np.nonzero(bounding_indices)][is_lower_energy]
        #print('result_array.GM.values', result_array.GM.values)

        # By profiling, it's faster to recompute all driving forces in-place
        # versus doing fancy indexing to only update "changed" driving forces
        # This avoids the creation of expensive temporaries
        np.einsum('...i,...i',
                  result_array.MU.values[..., np.newaxis, :],
                  global_grid.X.values[np.index_exp[...] + ((np.newaxis,) * len(comp_conds)) + np.index_exp[:, :]],
                  out=driving_forces)
        np.subtract(driving_forces,
                    global_grid.GM.values[np.index_exp[...] + ((np.newaxis,) * len(comp_conds)) + np.index_exp[:]],
                    out=driving_forces)


        # Update trial points to choose points with largest remaining driving force
        trial_points = np.argmax(driving_forces, axis=-1)
        #print('trial_points', trial_points)
        logger.debug('trial_points: %s', trial_points)

        # If all driving force (within some tolerance) is consumed, we found equilibrium
        if np.all(driving_forces <= DRIVING_FORCE_TOLERANCE):
            return
    if verbose:
        print('Max hull iterations exceeded. Remaining driving force: ', driving_forces.max())
