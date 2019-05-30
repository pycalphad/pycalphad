# distutils: language = c++
cimport numpy as np
import numpy as np
cimport cython
from libc.stdlib cimport malloc, free
cimport scipy.linalg.cython_lapack as cython_lapack


@cython.boundscheck(False)
cdef void solve(double[::1, :] A, double[::1] x, int[::1] ipiv) nogil:
    cdef int N = A.shape[0]
    cdef int info = 0
    cdef int NRHS = 1
    cython_lapack.dgesv(&N, &NRHS, &A[0,0], &N, &ipiv[0], &x[0], &N, &info)
    # Special for our case: singular matrix results get set to a special value
    if info != 0:
        x[:] = -1e19


@cython.boundscheck(False)
cdef void prodsum(double[::1] chempots, double[:,::1] points, double[::1] result) nogil:
    for i in range(chempots.shape[0]):
        for j in range(result.shape[0]):
            result[j] -= chempots[i]*points[j,i]


@cython.boundscheck(False)
cdef int argmin(double[::1] a, double[::1] lowest) nogil:
    cdef int result = 0
    for i in range(a.shape[0]):
        if a[i] < lowest[0]:
            lowest[0] = a[i]
            result = i
    return result


@cython.boundscheck(False)
cdef int argmax(double[::1] a) nogil:
    cdef int result = 0
    cdef double highest = -1e30
    for i in range(a.shape[0]):
        if a[i] > highest:
            highest = a[i]
            result = i
    return result


cpdef double hyperplane(double[:,::1] compositions,
                        double[::1] energies,
                        double[::1] composition,
                        double[::1] chemical_potentials,
                        double total_moles,
                        size_t[::1] fixed_chempot_indices,
                        size_t[::1] fixed_comp_indices,
                        double[::1] result_fractions,
                        int[::1] result_simplex) except *:
    """
    Find chemical potentials which approximate the tangent hyperplane
    at the given composition.
    Parameters
    ----------
    compositions : ndarray
        A sample of the energy surface of the system.
        Aligns with 'energies'.
        Shape of (M, N)
    energies : ndarray
        A sample of the energy surface of the system.
        Aligns with 'compositions'.
        Shape of (M,)
    composition : ndarray
        Target composition for the hyperplane.
        Shape of (N,)
    chemical_potentials : ndarray
        Shape of (N,)
        Will be overwritten
    total_moles : double
        Total number of moles in the system.
    fixed_chempot_indices : ndarray
        Variable shape from (0,) to (N-1,)
    fixed_comp_indices : ndarray
        Variable shape from (0,) to (N-1,)
    result_fractions : ndarray
        Relative amounts of the points making up the hyperplane simplex. Shape of (P,).
        Will be overwritten. Output sums to 1.
    result_simplex : ndarray
        Energies of the points making up the hyperplane simplex. Shape of (P,).
        Will be overwritten. Output*result_fractions sums to out_energy (return value).

    Returns
    -------
    out_energy : double
        Energy of the output configuration.

    Examples
    --------
    None yet.

    Notes
    -----
    M: number of energy points that have been sampled
    N: number of components
    P: N+1, max phases by gibbs phase rule that we can find in a point calculations
    """
    cdef int num_components = compositions.shape[1]
    cdef int num_fixed_chempots = fixed_chempot_indices.shape[0]
    cdef int simplex_size = num_components - num_fixed_chempots
    # composition index of -1 indicates total number of moles, i.e., N=1 condition
    cdef int[::1] included_composition_indices = \
        np.array(sorted(fixed_comp_indices) + [-1], dtype=np.int32)
    cdef int[::1] best_guess_simplex = np.array(sorted(set(range(num_components)) - set(fixed_chempot_indices)),
                                                dtype=np.int32)
    cdef int[::1] free_chempot_indices = np.array(sorted(set(range(num_components)) - set(fixed_chempot_indices)),
                                                dtype=np.int32)
    cdef int[::1] candidate_simplex = best_guess_simplex
    cdef int[:,::1] trial_simplices = np.empty((simplex_size, simplex_size), dtype=np.int32)
    cdef double[:,::1] fractions = np.empty((simplex_size, simplex_size))
    cdef int[::1] int_tmp = np.empty(simplex_size, dtype=np.int32)
    cdef double[::1] driving_forces = np.empty(compositions.shape[0])
    for i in range(trial_simplices.shape[0]):
        trial_simplices[i, :] = best_guess_simplex
    cdef double[::1,:,:] trial_matrix = np.empty((simplex_size, simplex_size, simplex_size), order='F')
    cdef double[::1,:] candidate_tieline = np.empty((simplex_size, simplex_size), order='F')
    cdef double[::1] candidate_energies = np.empty(simplex_size)
    cdef double[::1] candidate_potentials = np.empty(simplex_size)
    cdef double[::1] smallest_fractions = np.empty(simplex_size)
    cdef double[::1] tmp = np.empty(simplex_size)
    cdef double[::1, :] f_contig_trial
    # Not sure how to create scalar memoryviews...
    cdef double[::1] lowest_df = np.empty(1)
    cdef bint tmp3
    cdef int saved_trial = 0
    cdef int min_df
    cdef int max_iterations = 1000
    cdef int iterations = 0
    cdef int idx, ici, comp_idx, simplex_idx, trial_idx, chempot_idx

    while iterations < max_iterations:
        iterations += 1
        for trial_idx in range(simplex_size):
            #smallest_fractions[trial_idx] = 0
            for comp_idx in range(simplex_size):
                ici = included_composition_indices[comp_idx]
                for simplex_idx in range(simplex_size):
                    if ici >= 0:
                        trial_matrix[comp_idx, simplex_idx, trial_idx] = \
                            compositions[trial_simplices[trial_idx, simplex_idx], ici]
                    else:
                        # ici = -1, refers to N=1 condition
                        trial_matrix[comp_idx, simplex_idx, trial_idx] = 1 # 1 mole-formula per formula unit of a phase

        for trial_idx in range(simplex_size):
            f_contig_trial = np.asfortranarray(trial_matrix[:, :, trial_idx].copy())
            for simplex_idx in range(fractions.shape[1]):
                ici = included_composition_indices[simplex_idx]
                if ici >= 0:
                    fractions[trial_idx, simplex_idx] = composition[ici]
                else:
                    # ici = -1, refers to N=1 condition
                    fractions[trial_idx, simplex_idx] = total_moles
            solve(f_contig_trial, fractions[trial_idx, :], int_tmp)
            smallest_fractions[trial_idx] = min(fractions[trial_idx, :])
            if iterations > 1:
                # Penalize inclusion of fake starting points in subsequent iterations
                if np.any(np.array(trial_simplices[trial_idx, :]) < result_fractions.shape[0]):
                    smallest_fractions[trial_idx] -= 1
        # Choose simplex with the largest smallest-fraction
        saved_trial = argmax(smallest_fractions)
        if smallest_fractions[saved_trial] < -simplex_size:
            break
        # Should be exactly one candidate simplex
        candidate_simplex = trial_simplices[saved_trial, :]
        for i in range(candidate_simplex.shape[0]):
            idx = candidate_simplex[i]
            ici = 0
            for ici, chempot_idx in enumerate(free_chempot_indices):
                candidate_tieline[i, ici] = compositions[idx, chempot_idx]
            candidate_potentials[i] = energies[idx]
            for j in fixed_chempot_indices:
                candidate_potentials[i] -= chemical_potentials[j] * compositions[idx, j]
        solve(candidate_tieline, candidate_potentials, int_tmp)
        if candidate_potentials[0] == -1e19:
            break
        driving_forces[:] = energies
        for ici, chempot_idx in enumerate(free_chempot_indices):
            for idx in range(driving_forces.shape[0]):
                driving_forces[idx] -= candidate_potentials[ici] * compositions[idx, chempot_idx]
        for chempot_idx in fixed_chempot_indices:
            for idx in range(driving_forces.shape[0]):
                driving_forces[idx] -= chemical_potentials[chempot_idx] * compositions[idx, chempot_idx]
        best_guess_simplex[:] = candidate_simplex
        for i in range(trial_simplices.shape[0]):
            trial_simplices[i, :] = best_guess_simplex
        # Trial simplices will be the current simplex with each vertex
        #     replaced by the trial point
        # Exactly one of those simplices will contain a given test point,
        #     excepting edge cases
        lowest_df[0] = 1e10
        min_df = argmin(driving_forces, lowest_df)
        for i in range(simplex_size):
            trial_simplices[i, i] = min_df
        if lowest_df[0] > -1e-8:
            break
    out_energy = 0
    for i in range(best_guess_simplex.shape[0]):
        idx = best_guess_simplex[i]
        out_energy += fractions[saved_trial, i] * energies[idx]
    result_fractions[:simplex_size] = fractions[saved_trial, :]
    for ici, chempot_idx in enumerate(free_chempot_indices):
        chemical_potentials[chempot_idx] = candidate_potentials[ici]
    result_simplex[:simplex_size] = best_guess_simplex
    # Hack to enforce Gibbs phase rule, shape of result is comp+1, shape of hyperplane is comp
    result_fractions[simplex_size:] = 0.0
    result_simplex[simplex_size:] = 0
    return out_energy
