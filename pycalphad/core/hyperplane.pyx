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
cpdef void hyperplane_coefficients(double[:,::1] compositions,
                                   size_t[::1] fixed_chempot_indices,
                                   int[::1] trial_simplex,
                                   double[::1] out_plane_coefs) except * nogil:
    cdef int i, j
    cdef int plane_rows = trial_simplex.shape[0] + fixed_chempot_indices.shape[0]
    if plane_rows != compositions.shape[1]:
        raise ValueError('Hyperplane coefficient matrix is not square')
    cdef double* f_plane_matrix = <double*>malloc(plane_rows * compositions.shape[1] * sizeof(double))
    cdef int* int_tmp = <int*>malloc(plane_rows * sizeof(int))
    for i in range(trial_simplex.shape[0]):
        for j in range(compositions.shape[1]):
            f_plane_matrix[i + j*plane_rows] = compositions[trial_simplex[i], j]
        out_plane_coefs[i] = 1
    for i in range(fixed_chempot_indices.shape[0]):
        for j in range(compositions.shape[1]):
            f_plane_matrix[i + trial_simplex.shape[0] + j*plane_rows] = 0
        f_plane_matrix[i + trial_simplex.shape[0] + fixed_chempot_indices[i]*plane_rows] = 1
        out_plane_coefs[i + trial_simplex.shape[0]] = 0
    solve(f_plane_matrix, plane_rows, &out_plane_coefs[0], int_tmp)
    free(f_plane_matrix)
    free(int_tmp)

@cython.boundscheck(False)
cpdef void intersecting_point(double[:,::1] compositions,
                              size_t[::1] fixed_chempot_indices,
                              int[::1] trial_simplex,
                              double[:,::1] fixed_lincomb_molefrac_coefs,
                              double[::1] fixed_lincomb_molefrac_rhs,
                              double[::1] out_intersecting_point) except * nogil:
    cdef int i, j
    if trial_simplex.shape[0] == 1:
        # Simplex is zero-dimensional, so there is no intersection; just return the point defining the 0-simplex
        for i in range(compositions.shape[1]):
            out_intersecting_point[i] = compositions[trial_simplex[0], i]
        return
    #with gil:
    #    print('trial_simplex ', np.asarray(trial_simplex))
    if (fixed_lincomb_molefrac_rhs.shape[0] + 1 != compositions.shape[1]) and fixed_chempot_indices.shape[0] > 0:
        raise ValueError('Constraint matrix is not square')
    cdef int* int_tmp = <int*>malloc(compositions.shape[1] * sizeof(int))
    cdef double* constraint_matrix = <double*>malloc((fixed_lincomb_molefrac_rhs.shape[0] + 1) * compositions.shape[1] * sizeof(double))
    cdef double* constraint_rhs = <double*>malloc((fixed_lincomb_molefrac_rhs.shape[0] + 1) * sizeof(double))
    out_intersecting_point[:] = 0
    cdef double[::1] plane_coefs = out_intersecting_point
    hyperplane_coefficients(compositions, fixed_chempot_indices, trial_simplex, plane_coefs)
    for j in range(compositions.shape[1]):
        for i in range(fixed_lincomb_molefrac_rhs.shape[0]):
            constraint_matrix[i + j*compositions.shape[1]] = fixed_lincomb_molefrac_coefs[i, j]
            constraint_rhs[i] = fixed_lincomb_molefrac_rhs[i]
        constraint_matrix[fixed_lincomb_molefrac_rhs.shape[0] + j*compositions.shape[1]] = plane_coefs[j]
        constraint_rhs[fixed_lincomb_molefrac_rhs.shape[0]] = 1
    solve(constraint_matrix, compositions.shape[1], constraint_rhs, int_tmp)
    for i in range(compositions.shape[1]):
        out_intersecting_point[i] = constraint_rhs[i]
    free(int_tmp)
    free(constraint_matrix)
    free(constraint_rhs)

#@cython.boundscheck(False)
cdef void simplex_fractions(double[:,::1] compositions,
                             size_t[::1] fixed_chempot_indices,
                             int[::1] trial_simplex,
                             double[:,::1] fixed_lincomb_molefrac_coefs,
                             double[::1] fixed_lincomb_molefrac_rhs,
                             double* out_fractions) except *:
    cdef int simplex_size = trial_simplex.shape[0]
    # Note that compositions.shape[1] = simplex_size + fixed_chempot_indices.shape[0], by construction
    cdef int i, j
    cdef double* f_coord_matrix = <double*>malloc(simplex_size * simplex_size * sizeof(double))
    cdef double* target_point = <double*>malloc(compositions.shape[1] * sizeof(double))
    cdef int* int_tmp = <int*>malloc(simplex_size * sizeof(int))
    cdef size_t[::1] free_chempot_indices = np.array(list(set(range(compositions.shape[1])) - set(fixed_chempot_indices)), dtype=np.uintp)
    # Get target point for calculation
    intersecting_point(compositions, fixed_chempot_indices, trial_simplex,
                       fixed_lincomb_molefrac_coefs, fixed_lincomb_molefrac_rhs,
                       <double[:compositions.shape[1]]>target_point)
    # Fill coordinate matrix
    for j in range(simplex_size):
        for i in range(simplex_size):
            f_coord_matrix[j + simplex_size*i] = compositions[trial_simplex[i], free_chempot_indices[j]]
        out_fractions[j] = target_point[free_chempot_indices[j]]
    solve(f_coord_matrix, simplex_size, &out_fractions[0], int_tmp)
    free(f_coord_matrix)
    free(target_point)
    free(int_tmp)

@cython.boundscheck(False)
cpdef double hyperplane(double[:,::1] compositions,
                        double[::1] energies,
                        double[::1] chemical_potentials,
                        size_t[::1] fixed_chempot_indices,
                        double[:, ::1] fixed_lincomb_molefrac_coefs,
                        double[::1] fixed_lincomb_molefrac_rhs,
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
    chemical_potentials : ndarray
        Shape of (N,)
        Will be overwritten
    fixed_chempot_indices : ndarray
        Variable shape from (0,) to (N-1,)
    fixed_lincomb_molefrac_coefs : ndarray
        Variable shape from (0,P) to (N-1, P)
    fixed_lincomb_molefrac_rhs : ndarray
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
            #print('trial simplex ', np.asarray(<int[:simplex_size]>&trial_simplices[trial_idx*simplex_size]))
            for simplex_idx in range(simplex_size):
                fractions[trial_idx*simplex_size + simplex_idx] = 0
            simplex_fractions(compositions, fixed_chempot_indices, <int[:simplex_size]>&trial_simplices[trial_idx*simplex_size],
                              fixed_lincomb_molefrac_coefs, fixed_lincomb_molefrac_rhs, &fractions[trial_idx*simplex_size])
            smallest_fractions[trial_idx] = _min(&fractions[trial_idx*simplex_size], simplex_size)
        #print('smallest_fractions ', np.asarray(<double[:simplex_size]>&smallest_fractions[0]))
        # Choose simplex with the largest smallest-fraction
        saved_trial = argmax(smallest_fractions, simplex_size)
        #print('saved_trial', saved_trial)
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
            driving_forces[i] = energies[i]
        for ici in range(simplex_size):
            chempot_idx = free_chempot_indices[ici]
            for idx in range(num_points):
                driving_forces[idx] -= candidate_potentials[ici] * compositions[idx, chempot_idx]
        for ici in range(fixed_chempot_indices.shape[0]):
            chempot_idx = fixed_chempot_indices[ici]
            for idx in range(num_points):
                driving_forces[idx] -= chemical_potentials[chempot_idx] * compositions[idx, chempot_idx]
        for i in range(simplex_size):
            best_guess_simplex[i] = candidate_simplex[i]
        for i in range(simplex_size):
            for j in range(simplex_size):
                trial_simplices[i*simplex_size + j] = best_guess_simplex[j]

        lowest_df = 1e10
        min_df = -1
        for i in range(num_points):
            if driving_forces[i] < lowest_df:
                lowest_df = driving_forces[i]
                min_df = i

        # Trial simplices will be the current simplex with each vertex
        #     replaced by the trial point
        # Exactly one of those simplices will contain a given test point,
        #     excepting edge cases
        for i in range(simplex_size):
            trial_simplices[i*simplex_size + i] = min_df
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
