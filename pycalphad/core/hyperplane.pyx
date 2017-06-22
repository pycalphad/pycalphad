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
    # Special for our case: singular matrix results get set to -1
    if info != 0:
        x[:] = -1


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


cpdef double hyperplane(double[:,::1] compositions,
                        double[::1] energies,
                        double[::1] composition,
                        double[::1] chemical_potentials,
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
    result_fractions : ndarray
        Shape of (N,)
        Will be overwritten
    best_guess_simplex : ndarray
        Shape of (N,)
        Will be overwritten

    Returns
    -------
    out_energy : double
        Energy of the output configuration.

    Examples
    --------
    None yet.
    """
    cdef int num_components = compositions.shape[1]
    cdef int[::1] best_guess_simplex = np.arange(num_components, dtype=np.int32)
    cdef int[::1] candidate_simplex = best_guess_simplex
    cdef int[:,::1] trial_simplices = np.empty((num_components, num_components), dtype=np.int32)
    cdef double[:,::1] fractions = np.empty((num_components, num_components))
    cdef int[::1] int_tmp = np.empty(num_components, dtype=np.int32)
    cdef double[::1] driving_forces = np.empty(compositions.shape[0])
    for i in range(trial_simplices.shape[0]):
        trial_simplices[i, :] = best_guess_simplex
    cdef double[::1,:,:] trial_matrix = np.empty((num_components, num_components, num_components), order='F')
    cdef double[::1,:] candidate_tieline = np.empty((num_components, num_components), order='F')
    cdef double[::1] candidate_energies = np.empty(num_components)
    cdef double[::1] candidate_potentials = np.empty(num_components)
    cdef bint[::1] bounding_indices = np.ones(num_components, dtype=np.int32)
    cdef double[::1] tmp = np.empty(num_components)
    cdef double[::1, :] f_contig_trial
    # Not sure how to create scalar memoryviews...
    cdef double[::1] lowest_df = np.empty(1)
    cdef bint tmp3
    cdef int saved_trial = 0
    cdef int min_df
    cdef int max_iterations = 1000
    cdef int iterations = 0
    cdef int idx

    while iterations < max_iterations:
        iterations += 1
        for i in range(num_components):
            for j in range(num_components):
                trial_matrix[:, j, i] = compositions[trial_simplices[i,j]]
            f_contig_trial = trial_matrix[:, :, i]
            fractions[i, :] = composition
            solve(f_contig_trial, fractions[i, :], int_tmp)
            for j in range(num_components):
                bounding_indices[i] &= fractions[i,j]>=-1e-12

        # If more than one trial simplex satisfies the non-negativity criteria
        # then just choose the first non-degenerate one. This addresses gh-28.
        # There is also the possibility that *none* of the trials were successful.
        # This is usually due to numerical problems at the limit of composition space.
        # We will sidestep the issue here by forcing the last first non-degenerate simplex to match in that case.
        for i in range(bounding_indices.shape[0]):
            tmp3 = bounding_indices[i]
            if tmp3:
                saved_trial = i
                break
        bounding_indices[...] = True
        # Should be exactly one candidate simplex
        candidate_simplex = trial_simplices[saved_trial, :]
        for i in range(candidate_simplex.shape[0]):
            idx = candidate_simplex[i]
            candidate_tieline[i, :] = compositions[idx]
            candidate_potentials[i] = energies[idx]
        solve(candidate_tieline, candidate_potentials, int_tmp)
        driving_forces[:] = energies
        prodsum(candidate_potentials, compositions, driving_forces)
        best_guess_simplex[:] = candidate_simplex
        for i in range(trial_simplices.shape[0]):
            trial_simplices[i, :] = best_guess_simplex
        # Trial simplices will be the current simplex with each vertex
        #     replaced by the trial point
        # Exactly one of those simplices will contain a given test point,
        #     excepting edge cases
        lowest_df[0] = 1e30
        min_df = argmin(driving_forces, lowest_df)
        for i in range(num_components):
            trial_simplices[i, i] = min_df
        if lowest_df[0] > -1e-8:
            break
    if lowest_df[0] < -1e-8:
        raise ValueError('Max hull iterations exceeded. Remaining driving force: ', lowest_df[0])
    out_energy = 0
    for i in range(best_guess_simplex.shape[0]):
        idx = best_guess_simplex[i]
        out_energy += fractions[saved_trial, i] * energies[idx]
    result_fractions[:] = fractions[saved_trial, :]
    chemical_potentials[:] = candidate_potentials
    result_simplex[:] = best_guess_simplex
    return out_energy