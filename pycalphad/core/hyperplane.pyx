# distutils: language = c++
cimport numpy as np
import numpy as np
cimport cython
from libc.stdlib cimport malloc, free
cimport scipy.linalg.cython_lapack as cython_lapack


@cython.boundscheck(False)
cdef void solve(double* A, int N, double* x, int* ipiv) nogil:
    cdef int i
    cdef int info = 0
    cdef int NRHS = 1
    cython_lapack.dgesv(&N, &NRHS, A, &N, ipiv, x, &N, &info)
    # Special for our case: singular matrix results get set to a special value
    if info != 0:
        for i in range(N):
            x[i] = -1e19


@cython.boundscheck(False)
cdef void prodsum(double[::1] chempots, double[:,::1] points, double[::1] result) nogil:
    for i in range(chempots.shape[0]):
        for j in range(result.shape[0]):
            result[j] -= chempots[i]*points[j,i]


@cython.boundscheck(False)
cdef double _min(double* a, int a_shape) nogil:
    cdef int i
    cdef double result = 1e300
    for i in range(a_shape):
        if a[i] < result:
            result = a[i]
    return result

@cython.boundscheck(False)
cdef int argmin(double* a, int a_shape, double* lowest) nogil:
    cdef int i
    cdef int result = 0
    for i in range(a_shape):
        if a[i] < lowest[0]:
            lowest[0] = a[i]
            result = i
    return result


@cython.boundscheck(False)
cdef int argmax(double* a, int a_shape) nogil:
    cdef int i
    cdef int result = 0
    cdef double highest = -1e30
    for i in range(a_shape):
        if a[i] > highest:
            highest = a[i]
            result = i
    return result

@cython.boundscheck(False)
cpdef double hyperplane(double[:,::1] compositions,
                        double[::1] energies,
                        double[::1] composition,
                        double[::1] chemical_potentials,
                        double total_moles,
                        size_t[::1] fixed_chempot_indices,
                        size_t[::1] fixed_comp_indices,
                        double[::1] result_fractions,
                        int[::1] result_simplex) nogil except *:
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
    # Scalars
    cdef int num_points = compositions.shape[0]
    cdef int num_components = compositions.shape[1]
    cdef int num_fixed_chempots = fixed_chempot_indices.shape[0]
    cdef int simplex_size = num_components - num_fixed_chempots
    cdef int i, j
    cdef int fixed_index = 0
    cdef int saved_trial = 0
    cdef int min_df
    cdef int max_iterations = 1000
    cdef int iterations = 0
    cdef int idx, ici, comp_idx, simplex_idx, trial_idx, chempot_idx
    cdef bint tmp3
    cdef bint skip_index = False
    cdef double lowest_df = 0
    cdef double out_energy = 0
    # 1-D
    cdef int* remaining_point_indices = <int*>malloc(num_points * sizeof(int))
    for i in range(num_points):
        remaining_point_indices[i] = i
    # composition index of -1 indicates total number of moles, i.e., N=1 condition
    cdef int* included_composition_indices = <int*>malloc((fixed_comp_indices.shape[0] + 1) * sizeof(int))
    for i in range(fixed_comp_indices.shape[0]):
        included_composition_indices[i] = fixed_comp_indices[i]
    included_composition_indices[fixed_comp_indices.shape[0]] = -1
    cdef int* best_guess_simplex = <int*>malloc(simplex_size * sizeof(int))
    for i in range(num_components):
        skip_index = False
        for j in range(num_fixed_chempots):
            if i == fixed_chempot_indices[j]:
                skip_index = True
        if skip_index:
            pass
        else:
            best_guess_simplex[fixed_index] = i
            fixed_index += 1
    cdef int* free_chempot_indices = <int*>malloc(simplex_size * sizeof(int))
    cdef int* candidate_simplex = <int*>malloc(simplex_size * sizeof(int))
    for i in range(simplex_size):
        free_chempot_indices[i] = best_guess_simplex[i]
        candidate_simplex[i] = best_guess_simplex[i]
    cdef int* int_tmp = <int*>malloc(simplex_size * sizeof(int)) # np.empty(simplex_size, dtype=np.int32)
    cdef double* candidate_potentials = <double*>malloc(simplex_size * sizeof(double)) # np.empty(simplex_size)
    cdef double* smallest_fractions = <double*>malloc(simplex_size * sizeof(double)) # np.empty(simplex_size)
    cdef double* driving_forces = <double*>malloc(num_points * sizeof(double)) # np.empty(compositions.shape[0])
    # 2-D
    cdef int* trial_simplices = <int*>malloc(simplex_size * simplex_size * sizeof(int)) # np.empty((simplex_size, simplex_size), dtype=np.int32)
    cdef double* fractions = <double*>malloc(simplex_size * simplex_size * sizeof(double)) # np.empty((simplex_size, simplex_size))
    for i in range(simplex_size):
        for j in range(simplex_size):
            trial_simplices[i*simplex_size + j] = best_guess_simplex[j]
    cdef double* f_contig_trial = <double*>malloc(simplex_size * simplex_size * sizeof(double)) # np.empty((simplex_size, simplex_size), order='F')
    cdef double* f_candidate_tieline = <double*>malloc(simplex_size * simplex_size * sizeof(double)) # np.empty((simplex_size, simplex_size), order='F')
    # 3-D
    cdef double* f_trial_matrix = <double*>malloc(simplex_size * simplex_size * simplex_size * sizeof(double)) # np.empty((simplex_size, simplex_size, simplex_size), order='F')


    while iterations < max_iterations:
        iterations += 1
        for trial_idx in range(simplex_size):
            #smallest_fractions[trial_idx] = 0
            for comp_idx in range(simplex_size):
                ici = included_composition_indices[comp_idx]
                for simplex_idx in range(simplex_size):
                    if ici >= 0:
                        f_trial_matrix[comp_idx + simplex_idx*simplex_size + trial_idx*simplex_size*simplex_size] = \
                            compositions[trial_simplices[trial_idx*simplex_size + simplex_idx], ici]
                    else:
                        # ici = -1, refers to N=1 condition
                        f_trial_matrix[comp_idx + simplex_idx*simplex_size + trial_idx*simplex_size*simplex_size] = 1 # 1 mole-formula per formula unit of a phase

        for trial_idx in range(simplex_size):
            for i in range(simplex_size):
                for j in range(simplex_size):
                    f_contig_trial[i + j*simplex_size] = f_trial_matrix[i + j*simplex_size + trial_idx*simplex_size*simplex_size]
            for simplex_idx in range(simplex_size):
                ici = included_composition_indices[simplex_idx]
                if ici >= 0:
                    fractions[trial_idx*simplex_size + simplex_idx] = composition[ici]
                else:
                    # ici = -1, refers to N=1 condition
                    fractions[trial_idx*simplex_size + simplex_idx] = total_moles
            solve(f_contig_trial, simplex_size, &fractions[trial_idx*simplex_size], int_tmp)
            smallest_fractions[trial_idx] = _min(&fractions[trial_idx*simplex_size], simplex_size)

        # Choose simplex with the largest smallest-fraction
        saved_trial = argmax(smallest_fractions, simplex_size)
        if smallest_fractions[saved_trial] < -simplex_size:
            break
        # Should be exactly one candidate simplex
        for i in range(simplex_size):
            candidate_simplex[i] = trial_simplices[saved_trial*simplex_size + i]
        for i in range(simplex_size):
            idx = candidate_simplex[i]
            for ici in range(simplex_size):
                chempot_idx = free_chempot_indices[ici]
                f_candidate_tieline[i + simplex_size*ici] = compositions[idx, chempot_idx]
            candidate_potentials[i] = energies[idx]
            for ici in range(fixed_chempot_indices.shape[0]):
                chempot_idx = fixed_chempot_indices[ici]
                candidate_potentials[i] -= chemical_potentials[chempot_idx] * compositions[idx, chempot_idx]
        solve(f_candidate_tieline, simplex_size, candidate_potentials, int_tmp)
        if candidate_potentials[0] == -1e19:
            break
        for i in range(num_points):
            driving_forces[i] = energies[remaining_point_indices[i]]
        for ici in range(simplex_size):
            chempot_idx = free_chempot_indices[ici]
            for idx in range(num_points):
                driving_forces[idx] -= candidate_potentials[ici] * compositions[remaining_point_indices[idx], chempot_idx]
        for ici in range(fixed_chempot_indices.shape[0]):
            chempot_idx = fixed_chempot_indices[ici]
            for idx in range(num_points):
                driving_forces[idx] -= chemical_potentials[chempot_idx] * compositions[remaining_point_indices[idx], chempot_idx]
        for i in range(simplex_size):
            best_guess_simplex[i] = candidate_simplex[i]
        for i in range(simplex_size):
            for j in range(simplex_size):
                trial_simplices[i*simplex_size + j] = best_guess_simplex[j]

        # Only points below, or at, the candidate hyperplane can possibly be part of the solution
        ici = 0
        lowest_df = 1e10
        min_df = -1
        for i in range(num_points):
            if driving_forces[i] < 1.0:
                remaining_point_indices[ici] = remaining_point_indices[i]
                if driving_forces[i] < lowest_df:
                    lowest_df = driving_forces[i]
                    min_df = ici
                ici += 1
        num_points = ici

        # Trial simplices will be the current simplex with each vertex
        #     replaced by the trial point
        # Exactly one of those simplices will contain a given test point,
        #     excepting edge cases
        for i in range(simplex_size):
            trial_simplices[i*simplex_size + i] = remaining_point_indices[min_df]
        if lowest_df > -1e-8:
            break
    out_energy = 0
    for i in range(simplex_size):
        idx = best_guess_simplex[i]
        out_energy += fractions[saved_trial*simplex_size + i] * energies[idx]
    for i in range(simplex_size):
        result_fractions[i] = fractions[saved_trial*simplex_size + i]
    for ici in range(simplex_size):
        chempot_idx = free_chempot_indices[ici]
        chemical_potentials[chempot_idx] = candidate_potentials[ici]
        result_simplex[ici] = best_guess_simplex[ici]

    # Hack to enforce Gibbs phase rule, shape of result is comp+1, shape of hyperplane is comp
    result_fractions[simplex_size:] = 0.0
    result_simplex[simplex_size:] = 0

    # 1-D
    free(remaining_point_indices)
    free(included_composition_indices)
    free(best_guess_simplex)
    free(free_chempot_indices)
    free(candidate_simplex)
    free(int_tmp)
    free(candidate_potentials)
    free(smallest_fractions)
    free(driving_forces)
    # 2-D
    free(trial_simplices)
    free(fractions)
    free(f_contig_trial)
    free(f_candidate_tieline)
    # 3-D
    free(f_trial_matrix)

    return out_energy
