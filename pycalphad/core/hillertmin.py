"""
This module defines an implementation of Hillert's two-step method.
"""

import numpy as np
import numba

# (1) Initial grid for all phases and all T, P
# (2) Compute chemical potentials for each (unconverged) condition set using hyperplane()
# (3) Discard all but N closest points to hyperplane for each condition set for each phase (pad with nan's for simplicity)
# (4) Find new points for each condition set for each phase, minimized subject to that potential, using hillertmin()
# (5) Stop if energy/potential progress gets too small; else go to step 2



def hillertmin(temperature, pressure, chemical_potentials, obj_func, grad_func, initial_grid_points):
    # Minimize subject to given chemical potentials
    # For each phase
    # Generate a grid (or use initial_grid_points)
    # Compute chemical potentials based on initial grid
    points = initial_grid_points
    hess = np.zeros((1 + initial_grid_points.shape[-1], 1 + initial_grid_points.shape[-1]))
    pass

@numba.guvectorize(['float64[:,:], float64[:], float64[:]'], '(k,l),(m)->(m)')
def hyperplane(points, composition, chemical_potentials):
    """
    Find chemical potentials which approximate the tangent hyperplane
    at the given composition.

    Parameters
    ----------
    points : ndarray
        A sample of the energy surface of the system.
        First N columns are composition; last column is energy.
        Shape of (M, N+1)
    composition : ndarray
        Shape of (N,)
    chemical_potentials : ndarray
        Shape of (N,)
        Will be overwritten

    Returns
    -------
    chemical_potentials : ndarray
        Shape of (N,)

    Examples
    --------
    None yet.
    """
    num_components = points.shape[-1] - 1
    best_guess_simplex = np.arange(num_components).astype(np.int32)

    max_iterations = 100
    iterations = 0
    while iterations < max_iterations:
        iterations += 1
        driving_forces = points[:, -1] - np.multiply(chemical_potentials, points[:, :-1]).sum(axis=-1)
        if np.abs(driving_forces.min()) < 1e-4:
            break
        trial_simplices = np.zeros((num_components, num_components), dtype=np.int32)
        for idx in range(trial_simplices.shape[0]):
            trial_simplices[idx, :] = best_guess_simplex
        # Trial simplices will be the current simplex with each vertex
        #     replaced by the trial point
        # Exactly one of those simplices will contain a given test point,
        #     excepting edge cases
        for idx in range(num_components):
            trial_simplices[idx, idx] = np.argmin(driving_forces)
        trial_matrix = points[trial_simplices]
        # We have to filter out degenerate simplices before
        #     phase fraction computation
        # This is because even one degenerate simplex causes the entire tensor
        #     to be singular
        nondegenerate_indices = np.all(np.linalg.svd(trial_matrix,
                                                     compute_uv=False) > 1e-12,
                                       axis=-1, keepdims=False)
        trial_matrix = trial_matrix[nondegenerate_indices]
        #print('trial_matrix', trial_matrix)
        fractions = np.dot(np.linalg.inv(np.swapaxes(trial_matrix[:, :, :-1], -2, -1)),
                           composition)
        # A simplex only contains a point if its barycentric coordinates
        # (phase fractions) are non-negative.
        bounding_indices = np.all(fractions >= 0, axis=-1)
        # If more than one trial simplex satisfies the non-negativity criteria
        # then just choose the first non-degenerate one. This addresses gh-28.
        # There is also the possibility that *none* of the trials were successful.
        # This is usually due to numerical problems at the limit of composition space.
        # We will sidestep the issue here by forcing the last first non-degenerate simplex to match in that case.
        multiple_success_trials = np.sum(bounding_indices, dtype=int, keepdims=False) != 1
        #print('MULTIPLE SUCCESS TRIALS SHAPE', np.nonzero(multiple_success_trials))
        if multiple_success_trials:
            saved_trial = np.argmax(np.logical_or(bounding_indices[np.nonzero(multiple_success_trials)],
                                                  nondegenerate_indices[np.nonzero(multiple_success_trials)]))
            #print('SAVED TRIAL', saved_trial)
            #print('BOUNDING INDICES BEFORE', bounding_indices)
            bounding_indices[np.nonzero(multiple_success_trials)] = False
            #print('BOUNDING INDICES FALSE', bounding_indices)
            bounding_indices[np.nonzero(multiple_success_trials), saved_trial] = True
            #print('BOUNDING INDICES AFTER', bounding_indices)

        # Should be exactly one candidate simplex
        candidate_simplex = trial_simplices[np.nonzero(bounding_indices), :][0][0]

        candidate_potentials = np.dot(np.linalg.inv(points[candidate_simplex, :-1]),
                                      points[candidate_simplex, -1])
        chemical_potentials[:] = candidate_potentials
        best_guess_simplex[:] = candidate_simplex
