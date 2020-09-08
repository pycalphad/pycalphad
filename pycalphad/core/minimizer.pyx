cimport cython
import numpy as np
cimport numpy as np
from pycalphad.core.composition_set cimport CompositionSet
from pycalphad.core.phase_rec cimport PhaseRecord
from pycalphad.core.constants import MIN_SITE_FRACTION
import scipy.optimize
cimport scipy.linalg.cython_lapack as cython_lapack
from libc.stdlib cimport malloc, free
import xarray as xr

@cython.boundscheck(False)
cdef void lstsq(double *A, int M, int N, double* x, double rcond) nogil:
    cdef int i
    cdef int NRHS = 1
    cdef int iwork = 0
    cdef int info = 0
    cdef int SMLSIZ = 50  # this is a guess
    cdef int NLVL = 10  # this is also a guess
    cdef int lwork = 12*N + 2*N*SMLSIZ + 8*N*NLVL + N*NRHS + (SMLSIZ+1)**2
    cdef int rank = 0
    cdef double* work = <double*>malloc(lwork * sizeof(double))
    cdef double* singular_values = <double*>malloc(N * sizeof(double))
    # with lwork=-1, we calculate the optimal workspace size,
    cython_lapack.dgelsd(&M, &N, &NRHS, A, &N, x, &M, singular_values, &rcond, &rank,
                         work, &lwork, &iwork, &info)
    free(singular_values)
    free(work)
    if info != 0:
        for i in range(N):
            x[i] = -1e19

@cython.boundscheck(False)
cdef void solve(double* A, int N, double* x, int* ipiv) nogil:
    cdef int i
    cdef int info = 0
    cdef int NRHS = 1
    cython_lapack.dgesv(&N, &NRHS, A, &N, ipiv, x, &N, &info)
    if info != 0:
        for i in range(N):
            x[i] = -1e19

@cython.boundscheck(False)
cdef void invert_matrix(double *A, int N, double *A_inv_out, int* ipiv) nogil:
    "A_inv_out should be the identity matrix; it will be overwritten."
    cdef int info = 0

    cython_lapack.dgesv(&N, &N, A, &N, ipiv, A_inv_out, &N, &info)
    if info != 0:
        for i in range(N**2):
            A_inv_out[i] = -1e19

cdef void compute_phase_matrix(double[:,::1] phase_matrix, double[:,::1] hess, CompositionSet compset,
                               int num_statevars, double[::1] chemical_potentials, double[::1] phase_dof):
    "Compute the LHS of Eq. 41, Sundman 2015."
    cdef int comp_idx, i, j
    cdef int num_components = chemical_potentials.shape[0]
    cdef double[:, ::1] cons_jac_tmp = np.zeros((compset.phase_record.num_internal_cons,
                                                num_statevars + compset.phase_record.phase_dof))
    cdef double[:,:, ::1] mass_hess_tmp = np.zeros((num_components,
                                                    num_statevars + compset.phase_record.phase_dof,
                                                    num_statevars + compset.phase_record.phase_dof))
    compset.phase_record.internal_cons_jac(cons_jac_tmp, phase_dof)
    phase_matrix[:compset.phase_record.phase_dof, :compset.phase_record.phase_dof] = hess[
                                                                                     num_statevars:,
                                                                                     num_statevars:]
    for comp_idx in range(num_components):
        compset.phase_record.mass_hess(mass_hess_tmp[comp_idx, :, :], phase_dof, comp_idx)
    for comp_idx in range(num_components):
        for i in range(compset.phase_record.phase_dof):
            for j in range(i, compset.phase_record.phase_dof):
                phase_matrix[i, j] -= chemical_potentials[comp_idx] * mass_hess_tmp[comp_idx,
                                                                                    num_statevars+i,
                                                                                    num_statevars+j]
                phase_matrix[j, i] = phase_matrix[i, j]

    phase_matrix[compset.phase_record.phase_dof:compset.phase_record.phase_dof+compset.phase_record.num_internal_cons,
                 :compset.phase_record.phase_dof] = cons_jac_tmp[:, num_statevars:]
    phase_matrix[:compset.phase_record.phase_dof,
                 compset.phase_record.phase_dof:compset.phase_record.phase_dof+compset.phase_record.num_internal_cons] \
        = cons_jac_tmp[:, num_statevars:].T


cdef double compute_phase_system(double[:,::1] phase_matrix, double[::1] phase_rhs, CompositionSet compset,
                                 double[::1] delta_statevars, double[::1] chemical_potentials, double[::1] phase_dof):
    "Compute the system of equations in Eq. 41, Sundman 2015."
    cdef int i, cons_idx, comp_idx, sv_idx
    cdef int num_statevars = delta_statevars.shape[0]
    cdef int num_components = chemical_potentials.shape[0]
    cdef int num_internal_cons = compset.phase_record.num_internal_cons
    cdef double[::1] cons_tmp = np.zeros(num_internal_cons)
    cdef double[::1] grad_tmp = np.zeros(num_statevars + compset.phase_record.phase_dof)
    cdef double[:, ::1] mass_jac_tmp = np.zeros((num_components, num_statevars + compset.phase_record.phase_dof))
    cdef double[:, ::1] hess_tmp = np.zeros((num_statevars + compset.phase_record.phase_dof,
                                             num_statevars + compset.phase_record.phase_dof))
    cdef double max_cons = 0

    compset.phase_record.internal_cons_func(cons_tmp, phase_dof)
    compset.phase_record.hess(hess_tmp, phase_dof)
    compset.phase_record.grad(grad_tmp, phase_dof)

    for comp_idx in range(num_components):
        compset.phase_record.mass_grad(mass_jac_tmp[comp_idx, :], phase_dof, comp_idx)

    compute_phase_matrix(phase_matrix, hess_tmp, compset, num_statevars, chemical_potentials, phase_dof)

    # Compute right-hand side of Eq. 41, Sundman 2015
    for i in range(compset.phase_record.phase_dof):
        phase_rhs[i] = -grad_tmp[num_statevars+i]
        for sv_idx in range(num_statevars):
            phase_rhs[i] -= hess_tmp[num_statevars + i, sv_idx] * delta_statevars[sv_idx]
        for comp_idx in range(num_components):
            phase_rhs[i] += chemical_potentials[comp_idx] * mass_jac_tmp[comp_idx, num_statevars + i]

    for cons_idx in range(num_internal_cons):
        phase_rhs[compset.phase_record.phase_dof + cons_idx] = -cons_tmp[cons_idx]
        if abs(cons_tmp[cons_idx]) > max_cons:
            max_cons = abs(cons_tmp[cons_idx])
    return max_cons


cdef void fill_equilibrium_system_for_phase(double[::1,:] equilibrium_matrix, double[::1] equilibrium_rhs,
                                            double energy, double[::1] grad, double[:, ::1] hess,
                                            double[:, ::1] masses, double[:, ::1] mass_jac, int num_phase_dof,
                                            double[:, ::1] full_e_matrix, double[::1] chemical_potentials,
                                            double[::1] phase_amt, int[::1] free_chemical_potential_indices,
                                            int[::1] free_statevar_indices, int[::1] free_stable_compset_indices,
                                            int[::1] fixed_chemical_potential_indices,
                                            int[::1] prescribed_element_indices, double[::1] prescribed_elemental_amounts,
                                            int idx, int stable_idx, int num_statevars) except +:
    cdef int comp_idx, component_idx, chempot_idx, compset_idx, statevar_idx, fixed_component_idx, component_row_offset, i, j
    cdef int num_components = chemical_potentials.shape[0]
    cdef int num_stable_phases = free_stable_compset_indices.shape[0]
    cdef int num_fixed_components = prescribed_elemental_amounts.shape[0]
    # Eq. 44
    cdef double[::1] c_G = np.zeros(num_phase_dof)
    cdef double[:, ::1] c_statevars = np.zeros((num_phase_dof, num_statevars))
    cdef double[:, ::1] c_component = np.zeros((num_components, num_phase_dof))
    for i in range(num_phase_dof):
        for j in range(num_phase_dof):
            c_G[i] -= full_e_matrix[i, j] * grad[num_statevars+j]
    for i in range(num_phase_dof):
        for j in range(num_phase_dof):
            for statevar_idx in range(num_statevars):
                c_statevars[i, statevar_idx] -= full_e_matrix[i, j] * hess[num_statevars + j, statevar_idx]
    for comp_idx in range(num_components):
        for i in range(num_phase_dof):
            for j in range(num_phase_dof):
                c_component[comp_idx, i] += mass_jac[comp_idx, num_statevars + j] * full_e_matrix[i, j]
    # KEY STEPS for filling equilibrium matrix
    # 1. Contribute to the row corresponding to this composition set
    # 1a. Loop through potential conditions to fill out each column
    # 2. Contribute to the rows of all fixed components
    # 2a. Loop through potential conditions to fill out each column
    # 3. Contribute to RHS of each component row
    # 4. Add energies to RHS of each stable composition set
    # 5. Subtract contribution from RHS due to any fixed chemical potentials
    # 6. Subtract fixed chemical potentials from each fixed component RHS

    # 1a. This phase row: free chemical potentials
    cdef int free_variable_column_offset = 0
    for i in range(free_chemical_potential_indices.shape[0]):
        chempot_idx = free_chemical_potential_indices[i]
        equilibrium_matrix[stable_idx, free_variable_column_offset + i] = masses[chempot_idx, 0]
    free_variable_column_offset += free_chemical_potential_indices.shape[0]
    # 1a. This phase row: free stable composition sets = zero contribution
    free_variable_column_offset += free_stable_compset_indices.shape[0]
    # 1a. This phase row: free state variables
    for i in range(free_statevar_indices.shape[0]):
        statevar_idx = free_statevar_indices[i]
        equilibrium_matrix[stable_idx, free_variable_column_offset + i] = -grad[statevar_idx]

    # 2. Contribute to the row of all fixed components
    component_row_offset = num_stable_phases
    for fixed_component_idx in range(num_fixed_components):
        component_idx = prescribed_element_indices[fixed_component_idx]
        free_variable_column_offset = 0
        # 2a. This component row: free chemical potentials
        for i in range(free_chemical_potential_indices.shape[0]):
            chempot_idx = free_chemical_potential_indices[i]
            for j in range(c_component.shape[1]):
                equilibrium_matrix[component_row_offset + fixed_component_idx, free_variable_column_offset + i] += \
                    phase_amt[idx] * mass_jac[component_idx, num_statevars+j] * c_component[chempot_idx, j]
        free_variable_column_offset += free_chemical_potential_indices.shape[0]
        # 2a. This component row: free stable composition sets
        for i in range(free_stable_compset_indices.shape[0]):
            compset_idx = free_stable_compset_indices[i]
            # Only fill this out if the current idx is equal to a free composition set
            if compset_idx == idx:
                equilibrium_matrix[
                    component_row_offset + fixed_component_idx, free_variable_column_offset + i] = \
                    masses[component_idx, 0]
        free_variable_column_offset += free_stable_compset_indices.shape[0]
        # 2a. This component row: free state variables
        for i in range(free_statevar_indices.shape[0]):
            statevar_idx = free_statevar_indices[i]
            for j in range(c_statevars.shape[0]):
                equilibrium_matrix[component_row_offset + fixed_component_idx, free_variable_column_offset + i] += \
                    phase_amt[idx] * mass_jac[component_idx, num_statevars+j] * c_statevars[j, statevar_idx]
        # 3.
        for j in range(c_G.shape[0]):
            equilibrium_rhs[component_row_offset + fixed_component_idx] += -phase_amt[idx] * \
                mass_jac[component_idx, num_statevars+j] * c_G[j]

    system_amount_index = component_row_offset + num_fixed_components
    # 2X. Also handle the N=1 row
    for component_idx in range(num_components):
        free_variable_column_offset = 0
        # 2a. This component row: free chemical potentials
        for i in range(free_chemical_potential_indices.shape[0]):
            chempot_idx = free_chemical_potential_indices[i]
            for j in range(c_component.shape[1]):
                equilibrium_matrix[system_amount_index, free_variable_column_offset + i] += \
                    phase_amt[idx] * mass_jac[component_idx, num_statevars+j] * c_component[chempot_idx, j]
        free_variable_column_offset += free_chemical_potential_indices.shape[0]
        # 2a. This component row: free stable composition sets
        for i in range(free_stable_compset_indices.shape[0]):
            compset_idx = free_stable_compset_indices[i]
            # Only fill this out if the current idx is equal to a free composition set
            if compset_idx == idx:
                equilibrium_matrix[system_amount_index, free_variable_column_offset + i] += masses[component_idx, 0]
        free_variable_column_offset += free_stable_compset_indices.shape[0]
        # 2a. This component row: free state variables
        for i in range(free_statevar_indices.shape[0]):
            statevar_idx = free_statevar_indices[i]
            for j in range(c_statevars.shape[0]):
                equilibrium_matrix[system_amount_index, free_variable_column_offset + i] += \
                    phase_amt[idx] * mass_jac[component_idx, num_statevars+j] * c_statevars[j, statevar_idx]
        # 3.
        for j in range(c_G.shape[0]):
            equilibrium_rhs[system_amount_index] += -phase_amt[idx] * mass_jac[component_idx, num_statevars+j] * c_G[j]
    # 4.
    equilibrium_rhs[idx] = energy
    # 5. Subtract fixed chemical potentials from each phase RHS
    for i in range(fixed_chemical_potential_indices.shape[0]):
        chempot_idx = fixed_chemical_potential_indices[i]
        equilibrium_rhs[idx] -= masses[chempot_idx, 0] * chemical_potentials[chempot_idx]
        # 6. Subtract fixed chemical potentials from each fixed component RHS
        for fixed_component_idx in range(num_fixed_components):
            component_idx = prescribed_element_indices[fixed_component_idx]
            for j in range(c_component.shape[1]):
                equilibrium_rhs[component_row_offset + fixed_component_idx] -= phase_amt[idx] * chemical_potentials[
                    chempot_idx] * mass_jac[component_idx, num_statevars+j] * c_component[chempot_idx, j]


cdef double fill_equilibrium_system(double[::1,:] equilibrium_matrix, double[::1] equilibrium_rhs,
                                    object compsets, double[::1] chemical_potentials,
                                    double[::1] current_elemental_amounts, double[::1] phase_amt,
                                    int[::1] free_chemical_potential_indices,
                                    int[::1] free_statevar_indices, int[::1] free_stable_compset_indices,
                                    int[::1] fixed_chemical_potential_indices,
                                    int[::1] prescribed_element_indices, double[::1] prescribed_elemental_amounts,
                                    int num_statevars, double prescribed_system_amount, object dof) except +:
    cdef int stable_idx, idx, component_row_offset, component_idx, fixed_component_idx, comp_idx, system_amount_index
    cdef CompositionSet compset
    cdef int num_components = chemical_potentials.shape[0]
    cdef int num_stable_phases = free_stable_compset_indices.shape[0]
    cdef int num_fixed_components = len(prescribed_elemental_amounts)
    # Placeholder (output unused)
    cdef int[::1] ipiv = np.empty(10*num_components*num_stable_phases, dtype=np.int32)
    cdef double mass_residual, current_system_amount
    cdef double[::1] x
    cdef double[::1,:] energy_tmp
    cdef double[::1] grad_tmp
    cdef double[:,::1] hess_tmp
    cdef double[:,::1] masses_tmp
    cdef double[:,::1] mass_jac_tmp
    cdef double[:,::1] phase_matrix
    cdef double[:,::1] e_matrix, full_e_matrix
    for stable_idx in range(free_stable_compset_indices.shape[0]):
        idx = free_stable_compset_indices[stable_idx]
        compset = compsets[idx]
        # TODO: Use better dof storage
        # Calculate key phase quantities starting here
        x = dof[idx]
        energy_tmp = np.zeros((1, 1))
        masses_tmp = np.zeros((num_components, 1))
        mass_jac_tmp = np.zeros((num_components, num_statevars + compset.phase_record.phase_dof))
        # Compute phase matrix (LHS of Eq. 41, Sundman 2015)
        phase_matrix = np.zeros(
            (compset.phase_record.phase_dof + compset.phase_record.num_internal_cons,
             compset.phase_record.phase_dof + compset.phase_record.num_internal_cons))
        full_e_matrix = np.eye(compset.phase_record.phase_dof + compset.phase_record.num_internal_cons)
        hess_tmp = np.zeros((num_statevars + compset.phase_record.phase_dof,
                             num_statevars + compset.phase_record.phase_dof))
        grad_tmp = np.zeros(num_statevars + compset.phase_record.phase_dof)

        compset.phase_record.obj(energy_tmp[:, 0], x)
        for comp_idx in range(num_components):
            compset.phase_record.mass_grad(mass_jac_tmp[comp_idx, :], x, comp_idx)
            compset.phase_record.mass_obj(masses_tmp[comp_idx, :], x, comp_idx)
        compset.phase_record.hess(hess_tmp, x)
        compset.phase_record.grad(grad_tmp, x)

        compute_phase_matrix(phase_matrix, hess_tmp, compset, num_statevars, chemical_potentials, x)

        invert_matrix(&phase_matrix[0,0], phase_matrix.shape[0], &full_e_matrix[0,0], &ipiv[0])

        fill_equilibrium_system_for_phase(equilibrium_matrix, equilibrium_rhs, energy_tmp[0, 0], grad_tmp, hess_tmp,
                                          masses_tmp, mass_jac_tmp, compset.phase_record.phase_dof, full_e_matrix, chemical_potentials,
                                          phase_amt, free_chemical_potential_indices, free_statevar_indices,
                                          free_stable_compset_indices, fixed_chemical_potential_indices,
                                          prescribed_element_indices, prescribed_elemental_amounts,
                                          idx, stable_idx, num_statevars)

    # Add mass residual to fixed component row RHS, plus N=1 row
    mass_residual = 0.0
    component_row_offset = num_stable_phases
    system_amount_index = component_row_offset + num_fixed_components
    current_system_amount = float(np.sum(phase_amt))
    for fixed_component_idx in range(num_fixed_components):
        component_idx = prescribed_element_indices[fixed_component_idx]
        mass_residual += abs(
            current_elemental_amounts[component_idx] - prescribed_elemental_amounts[fixed_component_idx]) \
            / abs(prescribed_elemental_amounts[fixed_component_idx])
        equilibrium_rhs[component_row_offset + fixed_component_idx] -= current_elemental_amounts[component_idx] - \
                                                                       prescribed_elemental_amounts[
                                                                           fixed_component_idx]
    mass_residual += abs(current_system_amount - prescribed_system_amount)
    equilibrium_rhs[system_amount_index] -= current_system_amount - prescribed_system_amount
    return mass_residual


cdef void extract_equilibrium_solution(double[::1] chemical_potentials, double[::1] phase_amt, double[::1] delta_statevars,
                                 int[::1] free_chemical_potential_indices, int[::1] free_statevar_indices,
                                 int[::1] free_stable_compset_indices, double[::1] equilibrium_soln,
                                 double[:] largest_statevar_change, double[:] largest_phase_amt_change, list dof):
    cdef int i, idx, chempot_idx, compset_idx, statevar_idx
    cdef int num_statevars = delta_statevars.shape[0]
    cdef int soln_index_offset = 0
    cdef double chempot_change, percent_chempot_change, phase_amt_change, psc
    for i in range(free_chemical_potential_indices.shape[0]):
        chempot_idx = free_chemical_potential_indices[i]
        chempot_change = equilibrium_soln[soln_index_offset + i] - chemical_potentials[chempot_idx]
        #percent_chempot_change = abs(chempot_change / chemical_potentials[chempot_idx])
        chemical_potentials[chempot_idx] = equilibrium_soln[soln_index_offset + i]
        #largest_statevar_change[0] = max(largest_statevar_change[0], percent_chempot_change)
    soln_index_offset += free_chemical_potential_indices.shape[0]
    largest_delta_phase_amt = np.max(np.abs(equilibrium_soln[soln_index_offset:soln_index_offset+free_stable_compset_indices.shape[0]]))
    for i in range(free_stable_compset_indices.shape[0]):
        compset_idx = free_stable_compset_indices[i]
        phase_amt_change = float(phase_amt[compset_idx])
        scale_factor = 1.0
        if phase_amt_change > 0.5:
            if np.max(np.abs(equilibrium_soln[soln_index_offset + i])) > 0.1:
                scale_factor = 1./(1+largest_delta_phase_amt)
            clipped_change = np.clip(equilibrium_soln[soln_index_offset + i], -0.1, 0.1)
        else:
            clipped_change = np.clip(equilibrium_soln[soln_index_offset + i], -1, 1)
        phase_amt[compset_idx] += scale_factor * equilibrium_soln[soln_index_offset + i]
        phase_amt[compset_idx] = np.minimum(1.0, phase_amt[compset_idx])
        phase_amt[compset_idx] = np.maximum(0.0, phase_amt[compset_idx])
        phase_amt_change = phase_amt[compset_idx] - phase_amt_change
        largest_phase_amt_change[0] = max(largest_phase_amt_change[0], phase_amt_change)

    soln_index_offset += free_stable_compset_indices.shape[0]
    delta_statevars[:] = 0
    for i in range(free_statevar_indices.shape[0]):
        statevar_idx = free_statevar_indices[i]
        delta_statevars[statevar_idx] = equilibrium_soln[soln_index_offset + i]
    for i in range(delta_statevars.shape[0]):
        psc = abs(delta_statevars[i] / dof[0][i])
        largest_statevar_change[0] = max(largest_statevar_change[0], psc)
    for idx in range(len(dof)):
        for i in range(delta_statevars.shape[0]):
            dof[idx][i] += delta_statevars[i]


def check_convergence_and_change_phases(phase_amt, current_free_stable_compset_indices, metastable_phase_iterations,
                                        times_compset_removed, driving_forces,
                                        largest_internal_dof_change, largest_phase_amt_change, largest_statevar_change, can_add_phases):
    # Only add phases with positive driving force which have been metastable for at least 5 iterations, which have been removed fewer than 4 times
    if can_add_phases:
        newly_metastable_compsets = set(np.nonzero((np.array(metastable_phase_iterations) < 5))[0]) - set(current_free_stable_compset_indices)
        add_criteria = np.logical_and(np.array(driving_forces[:, 0]) > -1e-5, np.array(times_compset_removed) < 4)
        compsets_to_add = set((np.nonzero(add_criteria)[0])) - newly_metastable_compsets
    else:
        compsets_to_add = set()
    #print('compsets_to_add', compsets_to_add)
    compsets_to_remove = set(np.nonzero(np.array(phase_amt) < 1e-9)[0])
    #print('current_free_stable', set(current_free_stable_compset_indices))
    #print('compsets_to_remove', compsets_to_remove)
    new_free_stable_compset_indices = np.array(sorted((set(current_free_stable_compset_indices) - compsets_to_remove) | compsets_to_add))
    #print('new_free_stable', set(new_free_stable_compset_indices))
    removed_compset_indices = set(current_free_stable_compset_indices) - set(new_free_stable_compset_indices)
    #print('removed_compset_indices', removed_compset_indices)
    for idx in removed_compset_indices:
        times_compset_removed[idx] += 1
    converged = False
    if len(set(current_free_stable_compset_indices) - set(new_free_stable_compset_indices)) == 0:
        # feasible system, and no phases to add or remove
        if (largest_internal_dof_change < 1e-11) and (largest_phase_amt_change[0] < 1e-10) and \
                (largest_statevar_change[0] < 1e-1) and can_add_phases:
            converged = True
    return converged, new_free_stable_compset_indices

cpdef compute_infeasibility(list compsets, object dof, double[::1] lmul, double[::1] phase_amt,
                           int[::1] free_stable_compset_indices, int num_statevars, int num_components,
                           double prescribed_system_amount, int[::1] free_chemical_potential_indices,
                           int[::1] prescribed_element_indices, double[::1] prescribed_elemental_amounts):
    cdef int phase_idx, phase_idx_2, output_phase_idx, output_phase_idx_2, constraint_idx, dof_idx, dof_idx_2, pei_idx, comp_idx
    cdef CompositionSet compset
    cdef int constraint_offset = 0
    cdef int num_constraints = lmul.shape[0]
    cdef int dof_offset = num_statevars
    cdef double comp_amt
    cdef double total_infeasibility = 0.0
    cdef int infeasibility_dof = num_statevars
    cdef double[::1] total_infeasibility_gradient
    cdef double[:,::1] total_infeasibility_hess
    cdef double[::1] tmp_internal_cons, x
    cdef double[:,::1] masses_tmp  = np.zeros((num_components, 1))
    cdef double[:, ::1] mass_jac
    cdef double[:,:,::1] mass_hess, tmp_internal_cons_hess
    cdef double[:,::1] all_phase_amounts = np.zeros((len(compsets), num_components))
    cdef double[:,::1] tmp_internal_cons_jac
    cdef double[::1] tmp_mass_jac

    for phase_idx in free_stable_compset_indices:
        compset = compsets[phase_idx]
        infeasibility_dof += compset.phase_record.phase_dof
    infeasibility_dof += len(compsets) # phase amount
    total_infeasibility_gradient = np.zeros(infeasibility_dof+num_constraints)
    total_infeasibility_hess = np.zeros((infeasibility_dof+num_constraints, infeasibility_dof+num_constraints))
    mass_jac = np.zeros((num_components, infeasibility_dof))
    mass_hess = np.zeros((num_components, infeasibility_dof, infeasibility_dof))
    tmp_mass_jac = np.zeros(infeasibility_dof)

    for phase_idx in free_stable_compset_indices:
        output_phase_idx = mass_jac.shape[1] - phase_amt.shape[0] + phase_idx
        compset = compsets[phase_idx]
        tmp_internal_cons_jac = np.zeros((compset.phase_record.num_internal_cons, num_statevars + compset.phase_record.phase_dof))
        tmp_internal_cons = np.zeros(compset.phase_record.num_internal_cons)
        tmp_internal_cons_hess = np.zeros((tmp_internal_cons_jac.shape[0],
                                           tmp_internal_cons_jac.shape[1], tmp_internal_cons_jac.shape[1]))
        tmp_mass_jac = np.zeros(num_statevars + compset.phase_record.phase_dof)
        tmp_mass_hess = np.zeros((tmp_mass_jac.shape[0], tmp_mass_jac.shape[0]))
        x = dof[phase_idx]

        # Compute mass, mass Jacobian and mass Hessian of this phase
        for comp_idx in range(num_components):
            compset.phase_record.mass_obj(masses_tmp[comp_idx, :], x, comp_idx)
            compset.phase_record.mass_grad(tmp_mass_jac, x, comp_idx)
            compset.phase_record.mass_hess(tmp_mass_hess, x, comp_idx)
            all_phase_amounts[phase_idx, comp_idx] = masses_tmp[comp_idx, 0]
            for dof_idx in range(num_statevars):
                mass_jac[comp_idx, dof_idx] += phase_amt[phase_idx] * tmp_mass_jac[dof_idx]
                for dof_idx_2 in range(dof_idx, num_statevars):
                    mass_hess[comp_idx, dof_idx, dof_idx_2] += phase_amt[phase_idx] * tmp_mass_hess[dof_idx, dof_idx_2]
                    if dof_idx != dof_idx_2:
                        mass_hess[comp_idx, dof_idx_2, dof_idx] += phase_amt[phase_idx] * tmp_mass_hess[dof_idx, dof_idx_2]
                for dof_idx_2 in range(compset.phase_record.phase_dof):
                    mass_hess[comp_idx, dof_idx, dof_offset+dof_idx_2] = phase_amt[phase_idx] * tmp_mass_hess[dof_idx, num_statevars+dof_idx_2]
                    mass_hess[comp_idx, dof_offset+dof_idx_2, dof_idx] = phase_amt[phase_idx] * tmp_mass_hess[dof_idx, num_statevars+dof_idx_2]
                mass_hess[comp_idx, output_phase_idx, dof_idx] = tmp_mass_jac[dof_idx]
                mass_hess[comp_idx, dof_idx, output_phase_idx] = tmp_mass_jac[dof_idx]
            for dof_idx in range(compset.phase_record.phase_dof):
                mass_jac[comp_idx, dof_offset + dof_idx] += phase_amt[phase_idx] * tmp_mass_jac[num_statevars + dof_idx]
                for dof_idx_2 in range(dof_idx, compset.phase_record.phase_dof):
                    mass_hess[comp_idx, dof_offset + dof_idx, dof_offset + dof_idx_2] += phase_amt[phase_idx] * tmp_mass_hess[num_statevars + dof_idx, num_statevars + dof_idx_2]
                    if dof_idx != dof_idx_2:
                        mass_hess[comp_idx, dof_offset + dof_idx_2, dof_offset + dof_idx] += phase_amt[phase_idx] * tmp_mass_hess[num_statevars + dof_idx, num_statevars + dof_idx_2]
                mass_hess[comp_idx, output_phase_idx, dof_offset + dof_idx] = tmp_mass_jac[num_statevars + dof_idx]
                mass_hess[comp_idx, dof_offset + dof_idx, output_phase_idx] = tmp_mass_jac[num_statevars + dof_idx]
            mass_jac[comp_idx, output_phase_idx] += masses_tmp[comp_idx, 0]
            tmp_mass_jac[:] = 0
            tmp_mass_hess[:,:] = 0

        # Compute internal constraints, Jacobian and Hessian
        compset.phase_record.internal_cons_func(tmp_internal_cons, x)
        compset.phase_record.internal_cons_jac(tmp_internal_cons_jac, x)
        compset.phase_record.internal_cons_hess(tmp_internal_cons_hess, x)

        for constraint_idx in range(tmp_internal_cons.shape[0]):
            total_infeasibility_gradient[infeasibility_dof+constraint_offset+constraint_idx] = -tmp_internal_cons[constraint_idx]
            for dof_idx in range(num_statevars):
                total_infeasibility_gradient[dof_idx] -= lmul[constraint_offset+constraint_idx] * tmp_internal_cons_jac[constraint_idx, dof_idx]
                for dof_idx_2 in range(dof_idx, num_statevars):
                    total_infeasibility_hess[dof_idx, dof_idx_2] -= lmul[constraint_offset+constraint_idx] * tmp_internal_cons_hess[constraint_idx, dof_idx, dof_idx_2]
                    if dof_idx != dof_idx_2:
                        total_infeasibility_hess[dof_idx_2, dof_idx] -= lmul[constraint_offset+constraint_idx] * tmp_internal_cons_hess[constraint_idx, dof_idx, dof_idx_2]
                for dof_idx_2 in range(compset.phase_record.phase_dof):
                    total_infeasibility_hess[dof_idx, dof_offset + dof_idx_2] -= lmul[constraint_offset+constraint_idx] *  tmp_internal_cons_hess[constraint_idx, dof_idx, num_statevars + dof_idx_2]
                    total_infeasibility_hess[dof_offset + dof_idx_2, dof_idx] = total_infeasibility_hess[dof_idx, dof_offset + dof_idx_2]
                total_infeasibility_hess[dof_idx, infeasibility_dof+constraint_offset+constraint_idx] = -tmp_internal_cons_jac[constraint_idx, dof_idx]
                total_infeasibility_hess[infeasibility_dof+constraint_offset+constraint_idx, dof_idx] = -tmp_internal_cons_jac[constraint_idx, dof_idx]
            for dof_idx in range(compset.phase_record.phase_dof):
                total_infeasibility_gradient[dof_offset+dof_idx] -= lmul[constraint_offset+constraint_idx] * tmp_internal_cons_jac[constraint_idx, num_statevars + dof_idx]
                for dof_idx_2 in range(dof_idx, compset.phase_record.phase_dof):
                    total_infeasibility_hess[dof_offset + dof_idx, dof_offset + dof_idx_2] -= lmul[constraint_offset+constraint_idx] * tmp_internal_cons_hess[constraint_idx, num_statevars + dof_idx, num_statevars + dof_idx_2]
                    if dof_idx != dof_idx_2:
                        total_infeasibility_hess[dof_offset + dof_idx_2, dof_idx] -= lmul[constraint_offset+constraint_idx] * tmp_internal_cons_hess[constraint_idx, num_statevars + dof_idx, num_statevars + dof_idx_2]
                total_infeasibility_hess[dof_offset+dof_idx, infeasibility_dof+constraint_offset+constraint_idx] = -tmp_internal_cons_jac[constraint_idx, num_statevars + dof_idx]
                total_infeasibility_hess[infeasibility_dof+constraint_offset+constraint_idx, dof_offset+dof_idx] = -tmp_internal_cons_jac[constraint_idx, num_statevars + dof_idx]
        dof_offset += compset.phase_record.phase_dof
        constraint_offset += compset.phase_record.num_internal_cons
        masses_tmp[:,:] = 0
        tmp_internal_cons[:] = 0
        tmp_internal_cons_jac[:,:] = 0
        tmp_internal_cons_hess[:,:,:] = 0

    partials = []
    # Compute objective function, gradient and Hessian (sum-squares of mass residual)
    for pei_idx in range(prescribed_element_indices.shape[0]):
        comp_idx = prescribed_element_indices[pei_idx]
        comp_amt = np.dot(all_phase_amounts[:, comp_idx], phase_amt)
        partials.append(comp_amt)
        total_infeasibility += (comp_amt - prescribed_elemental_amounts[pei_idx])**2
        for dof_idx in range(infeasibility_dof):
            total_infeasibility_gradient[dof_idx] += 2 * mass_jac[comp_idx, dof_idx] * (comp_amt - prescribed_elemental_amounts[pei_idx])
            for dof_idx_2 in range(dof_idx, infeasibility_dof):
                total_infeasibility_hess[dof_idx, dof_idx_2] += 2 * (mass_hess[comp_idx, dof_idx, dof_idx_2] *
                                                                     (comp_amt - prescribed_elemental_amounts[pei_idx]) +
                                                                      mass_jac[comp_idx, dof_idx_2] * mass_jac[comp_idx, dof_idx]
                                                                     )
                if dof_idx != dof_idx_2:
                    total_infeasibility_hess[dof_idx_2, dof_idx] += 2 * (mass_hess[comp_idx, dof_idx, dof_idx_2] *
                                                                     (comp_amt - prescribed_elemental_amounts[pei_idx]) +
                                                                      mass_jac[comp_idx, dof_idx_2] * mass_jac[comp_idx, dof_idx]
                                                                     )
    print('pea', np.array(prescribed_elemental_amounts))
    print('partials', partials)
    print('els', [compset.phase_record.nonvacant_elements[i] for i in prescribed_element_indices])

    # infeasibility from phase_amt sum constraint: sum(phase_amt) - prescribed_system_amount = 0
    total_infeasibility_gradient[infeasibility_dof+constraint_offset] = -(np.sum(phase_amt) - prescribed_system_amount)
    for phase_idx in free_stable_compset_indices:
        output_phase_idx = mass_jac.shape[1] - phase_amt.shape[0] + phase_idx
        compset = compsets[phase_idx]
        total_infeasibility_gradient[output_phase_idx] -= lmul[constraint_offset] * 1
        total_infeasibility_hess[output_phase_idx, infeasibility_dof+constraint_offset] -= 1
        total_infeasibility_hess[infeasibility_dof+constraint_offset, output_phase_idx] -= 1

    return total_infeasibility, np.array(total_infeasibility_gradient), np.array(total_infeasibility_hess)

from itertools import chain
def check_infeasibility_grad(compsets, dof, phase_amt, free_stable_compset_indices,
                                                      num_statevars, num_components, prescribed_system_amount,
                                                      free_chemical_potential_indices, prescribed_element_indices,
                                                      prescribed_elemental_amounts):
    cdef CompositionSet compset = compsets[0]
    x0 = np.r_[dof[0][:num_statevars], list(chain.from_iterable([x[num_statevars:] for x in dof])), phase_amt]
    def f(x):
        x = np.array(x)
        new_dof = []
        new_phase_amt = x[-phase_amt.shape[0]:]
        dof_offset = num_statevars
        for phase_idx in range(phase_amt.shape[0]):
            compset = compsets[phase_idx]
            x_complete = np.r_[x[:num_statevars], x[dof_offset:dof_offset + compset.phase_record.phase_dof]]
            new_dof.append(x_complete)
            dof_offset += compset.phase_record.phase_dof
        #fun, gr, h = compute_infeasibility(compsets, new_dof, new_phase_amt, free_stable_compset_indices,
        #                                   num_statevars, num_components, prescribed_system_amount,
        #                                   free_chemical_potential_indices, prescribed_element_indices,
        #                                   prescribed_elemental_amounts)
        #return fun, gr
    return scipy.optimize.approx_fprime(x0, lambda x: f(x)[0], 1e-8) - np.array(f(x0)[1])

from copy import deepcopy
cpdef restore_solution_feasibility(list compsets, object dof, double[::1] phase_amt,
                                   int[::1] free_stable_compset_indices, int num_statevars, int num_components,
                                   double prescribed_system_amount, int[::1] free_chemical_potential_indices,
                                   int[::1] prescribed_element_indices, double[::1] prescribed_elemental_amounts):
    cdef double step_size = 1e-1
    cdef int dof_offset
    cdef int iteration = 0
    cdef CompositionSet compset
    cdef int num_constraints = 0
    cdef double[::1] lmul, new_lmul, new_phase_amt
    for phase_idx in free_stable_compset_indices:
        compset = compsets[phase_idx]
        num_constraints += compset.phase_record.num_internal_cons
    num_constraints += 1 # sum(phase_amt)
    lmul = np.zeros(num_constraints)
    infeasibility, infeas_grad, infeas_hess = compute_infeasibility(compsets, dof, lmul, phase_amt, free_stable_compset_indices,
                                                      num_statevars, num_components, prescribed_system_amount,
                                                      free_chemical_potential_indices, prescribed_element_indices,
                                                      prescribed_elemental_amounts)
    while infeasibility > 1e-6:
        orig_infeasibility = float(infeasibility)
        print('orig_infeasibility', orig_infeasibility, 'step', step_size)
        print(np.testing.assert_equal(infeas_hess, infeas_hess.T))
        print(np.testing.assert_equal(infeas_hess[:num_statevars, :num_statevars], 0.0))
        #dx = -infeas_grad
        dx = np.linalg.lstsq(infeas_hess, -infeas_grad[:], rcond=None)[0]
        #print(check_infeasibility_grad(compsets, dof, phase_amt, free_stable_compset_indices,
        #                                              num_statevars, num_components, prescribed_system_amount,
        #                                              free_chemical_potential_indices, prescribed_element_indices,
        #                                              prescribed_elemental_amounts))
        #print('dx', np.array(dx))
        for step_size in np.logspace(-1, -6, 20):
            print('step_size', step_size)
            print('-infeas_grad', -infeas_grad)
            print('dx', dx)
            new_phase_amt = np.zeros(phase_amt.shape[0])
            new_phase_amt[:] = phase_amt
            new_lmul = np.array(lmul)
            new_dof = deepcopy(dof)
            dof_offset = num_statevars
            # TODO: Update state variables in response to infeasibility (not yet passed as arguments)
            for phase_idx in free_stable_compset_indices:
                compset = compsets[phase_idx]
                x = new_dof[phase_idx]
                print(phase_idx, compset.phase_record.phase_name, phase_idx, new_phase_amt[phase_idx], np.array(x))
                for dof_idx in range(compset.phase_record.phase_dof):
                    x[num_statevars+dof_idx] += step_size * dx[dof_offset+dof_idx]
                    x[num_statevars+dof_idx] = np.clip(x[num_statevars+dof_idx], MIN_SITE_FRACTION, 1.0)
                new_phase_amt[phase_idx] += step_size * dx[dx.shape[0] - new_lmul.shape[0] - new_phase_amt.shape[0] + phase_idx]
                if (new_phase_amt[phase_idx] <= 1e-3):
                    new_phase_amt[phase_idx] = 0
                if new_phase_amt[phase_idx] > prescribed_system_amount:
                    new_phase_amt[phase_idx] = prescribed_system_amount
                dof_offset += compset.phase_record.phase_dof
            for constraint_idx in range(new_lmul.shape[0]):
                new_lmul[constraint_idx] += dx[dx.shape[0] - new_lmul.shape[0] + constraint_idx]
            print('new_lmul ', np.array(new_lmul))
            infeasibility, infeas_grad, infeas_hess = compute_infeasibility(compsets, new_dof, new_lmul, new_phase_amt, free_stable_compset_indices,
                                                              num_statevars, num_components, prescribed_system_amount,
                                                              free_chemical_potential_indices, prescribed_element_indices,
                                                              prescribed_elemental_amounts)
            print('candidate_infeasibility', infeasibility)
            if infeasibility < orig_infeasibility:
                dof[:] = new_dof
                assert phase_amt.shape[0] == new_phase_amt.shape[0]
                phase_amt[:] = new_phase_amt
                lmul[:] = new_lmul
                # Check if the stable set of phases should change
                new_free_stable_compset_indices = np.array(np.nonzero(phase_amt)[0], dtype=np.int32)
                if np.any(new_free_stable_compset_indices != free_stable_compset_indices):
                    print('new_free_stable_compset_indices', new_free_stable_compset_indices)
                    num_constraints = 0
                    for phase_idx in new_free_stable_compset_indices:
                        compset = compsets[phase_idx]
                        num_constraints += compset.phase_record.num_internal_cons
                    num_constraints += 1 # sum(phase_amt)
                    lmul = np.zeros(num_constraints)
                    free_stable_compset_indices = new_free_stable_compset_indices
                    iteration = 0
                break
        iteration += 1
        if iteration > 10:
            print('Unable to restore solution feasibility')
            break
    if infeasibility <= 1e-6:
        print('RESTORED')
    return np.array(lmul), np.array(free_stable_compset_indices, dtype=np.int32)


cpdef find_solution(list compsets, int[::1] free_stable_compset_indices,
                    int num_statevars, int num_components, double prescribed_system_amount,
                    double[::1] initial_chemical_potentials, int[::1] free_chemical_potential_indices,
                    int[::1] fixed_chemical_potential_indices,
                    int[::1] prescribed_element_indices, double[::1] prescribed_elemental_amounts,
                    int[::1] free_statevar_indices, int[::1] fixed_statevar_indices):
    cdef int iteration, idx, idx2, comp_idx, i
    cdef int num_stable_phases, num_fixed_components, num_free_variables
    cdef CompositionSet compset, compset2
    cdef double mass_residual = 1e-30
    cdef double[::1] x, new_y, delta_y
    cdef double[::1] phase_amt = np.array([compset.NP for compset in compsets])
    cdef list dof = [np.array(compset.dof) for compset in compsets]
    cdef list suspended_compsets = []
    cdef int[::1] ipiv = np.zeros(len(compsets) * max([compset.phase_record.phase_dof +
                                                       compset.phase_record.num_internal_cons
                                                       for compset in compsets]), dtype=np.int32)
    cdef double[::1] chemical_potentials = np.array(initial_chemical_potentials)
    cdef double[::1] current_elemental_amounts = np.zeros(chemical_potentials.shape[0])
    cdef int[::1] metastable_phase_iterations = np.zeros(len(compsets), dtype=np.int32)
    cdef int[::1] times_compset_removed = np.zeros(len(compsets), dtype=np.int32)
    cdef double[:,::1] all_phase_energies = np.zeros((len(compsets), 1))
    cdef double[:,::1] all_phase_amounts = np.zeros((len(compsets), chemical_potentials.shape[0]))
    cdef double[:,::1] masses_tmp
    cdef double[::1,:] equilibrium_matrix  # Fortran ordering required by call into lapack
    cdef double[:,::1] phase_matrix  # Fortran ordering required by call into lapack, but this is symmetric
    cdef double[::1] equilibrium_rhs, equilibrium_soln, phase_rhs, soln
    cdef double[::1] delta_statevars = np.zeros(num_statevars)
    cdef double[1] largest_statevar_change, largest_phase_amt_change
    cdef double largest_internal_dof_change, largest_cons_max_residual, largest_internal_cons_max_residual
    cdef bint converged = False
    cdef int max_dof = num_statevars + max([compset.phase_record.phase_dof for compset in compsets])
    iterations = np.arange(100)
    phase_names = [compset.phase_record.phase_name for compset in compsets]
    saved_chemical_potentials = xr.DataArray(np.full((iterations.shape[0], chemical_potentials.shape[0]), np.nan),
                                             dims=('iterations', 'component'),
                                             coords={'iterations': iterations,
                                                     'component': list(compsets[0].phase_record.nonvacant_elements)})
    saved_phase_amounts = xr.DataArray(np.full((iterations.shape[0], len(compsets)), np.nan), dims=('iterations', 'compset'),
                                       coords={'iterations': iterations, 'compset': np.arange(len(compsets)), 'phase': ('compset', phase_names)})
    saved_phase_compositions = xr.DataArray(np.full((iterations.shape[0], len(compsets), chemical_potentials.shape[0]), np.nan),
                                            dims=('iterations', 'compset', 'component'),
                                            coords={'iterations': iterations,
                                                    'compset': np.arange(len(compsets)), 'phase': ('compset', phase_names),
                                                    'component': list(compsets[0].phase_record.nonvacant_elements)})
    saved_phase_stepsizes = xr.DataArray(np.full((iterations.shape[0], len(compsets)), np.nan), dims=('iterations', 'compset'),
                                       coords={'iterations': iterations, 'compset': np.arange(len(compsets)),
                                               'phase': ('compset', phase_names)})
    saved_phase_energies = xr.DataArray(np.full((iterations.shape[0], len(compsets)), np.nan), dims=('iterations', 'compset'),
                                       coords={'iterations': iterations, 'compset': np.arange(len(compsets)),
                                               'phase': ('compset', phase_names)})
    saved_phase_driving_forces = xr.DataArray(np.full((iterations.shape[0], len(compsets)), np.nan), dims=('iterations', 'compset'),
                                              coords={'iterations': iterations, 'compset': np.arange(len(compsets)),
                                                      'phase': ('compset', phase_names)})
    saved_phase_dof = xr.DataArray(np.full((iterations.shape[0], len(compsets), max_dof), np.nan), dims=('iterations', 'compset', 'dof'),
                                       coords={'iterations': iterations,
                                               'compset': np.arange(len(compsets)), 'phase': ('compset', phase_names),
                                               'dof': np.arange(max_dof)})
    saved_phase_delta_dof = xr.DataArray(np.full((iterations.shape[0], len(compsets), max_dof), np.nan),
                                         dims=('iterations', 'compset', 'dof'),
                                         coords={'iterations': iterations,
                                                 'compset': np.arange(len(compsets)),
                                                 'phase': ('compset', phase_names),
                                                 'dof': np.arange(max_dof)})
    saved_infeasibilities = xr.DataArray(np.full(iterations.shape[0], np.nan),
                                        dims=('iterations',),
                                        coords={'iterations': iterations})

    from datetime import datetime
    stamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    for iteration in range(100):
        current_elemental_amounts[:] = 0
        all_phase_energies[:,:] = 0
        all_phase_amounts[:,:] = 0
        largest_statevar_change[0] = 0
        largest_internal_dof_change = 0
        largest_internal_cons_max_residual = 0
        largest_phase_amt_change[0] = 0
        if (mass_residual > 10) and (np.max(np.abs(chemical_potentials)) > 1.0e10):
            print('Mass residual and chemical potentials too big; resetting chemical potentials')
            chemical_potentials[:] = initial_chemical_potentials
        # FIRST STEP: Update phase internal degrees of freedom
        for idx, compset in enumerate(compsets):
            # TODO: Use better dof storage
            x = dof[idx]
            saved_phase_dof[iteration, idx, :len(x)] = x
            masses_tmp = np.zeros((num_components, 1))
            energy_tmp = np.zeros((1,1))
            internal_cons_tmp = np.zeros(compset.phase_record.num_internal_cons)
            # Compute phase matrix (LHS of Eq. 41, Sundman 2015)
            phase_matrix = np.zeros((compset.phase_record.phase_dof + compset.phase_record.num_internal_cons,
                                     compset.phase_record.phase_dof + compset.phase_record.num_internal_cons))
            soln = np.zeros(compset.phase_record.phase_dof + compset.phase_record.num_internal_cons)
            # RHS copied into soln, then overwritten by solve()
            internal_cons_max_residual = \
                compute_phase_system(phase_matrix, soln, compset, delta_statevars, chemical_potentials, x)
            phase_gradient = -np.array(soln[:compset.phase_record.phase_dof])
            # phase_matrix is symmetric by construction, so we can pass in a C-ordered array
            solve(&phase_matrix[0,0], phase_matrix.shape[0], &soln[0], &ipiv[0])

            delta_y = soln[:compset.phase_record.phase_dof]
            saved_phase_delta_dof[iteration, idx, num_statevars:num_statevars+len(delta_y)] = delta_y
            internal_lagrange = soln[compset.phase_record.phase_dof:]

            for comp_idx in range(num_components):
                compset.phase_record.mass_obj(masses_tmp[comp_idx, :], x, comp_idx)
                all_phase_amounts[idx, comp_idx] = masses_tmp[comp_idx, 0]
            compset.phase_record.internal_cons_func(internal_cons_tmp, x)
            compset.phase_record.obj(energy_tmp[0, :], x)

            largest_internal_cons_max_residual = max(largest_internal_cons_max_residual, internal_cons_max_residual)
            new_y = np.array(x)
            candidate_y_energy = np.zeros((1,1))
            candidate_y_masses = np.zeros((num_components, 1))
            candidate_internal_cons = np.zeros(compset.phase_record.num_internal_cons)
            current_phase_gradient = np.array(phase_gradient)
            step_size = 1.0 / (2 + float(np.max(np.abs(delta_y))))
            minimum_step_size = 1e-6 * step_size
            while step_size >= minimum_step_size:
                exceeded_bounds = False
                for i in range(num_statevars, new_y.shape[0]):
                    new_y[i] = x[i] + step_size * delta_y[i-num_statevars]
                    if new_y[i] > 1:
                        if (new_y[i] - 1) > 1e-6:
                            # Allow some tolerance in the name of progress
                            exceeded_bounds = True
                        new_y[i] = 1
                        if delta_y[i-num_statevars] > 0:
                            current_phase_gradient[i-num_statevars] = 0
                    elif new_y[i] < MIN_SITE_FRACTION:
                        if (MIN_SITE_FRACTION - new_y[i]) > 1e-6:
                            # Allow some tolerance in the name of progress
                            exceeded_bounds = True
                        new_y[i] = MIN_SITE_FRACTION
                        if delta_y[i-num_statevars] < 0:
                            current_phase_gradient[i-num_statevars] = 0
                if exceeded_bounds:
                    step_size /= 2
                    continue
                for comp_idx in range(num_components):
                    compset.phase_record.mass_obj(candidate_y_masses[comp_idx, :], new_y, comp_idx)
                compset.phase_record.internal_cons_func(candidate_internal_cons, new_y)
                compset.phase_record.obj(candidate_y_energy[0,:], new_y)
                saved_phase_energies[iteration, idx] = candidate_y_energy[0, 0]
                saved_phase_driving_forces[iteration, idx] = candidate_y_energy[0,0] - np.dot(chemical_potentials, candidate_y_masses[:,0])
                saved_phase_compositions[iteration, idx, :] = candidate_y_masses[:, 0]
                wolfe_criteria = (candidate_y_energy[0,0] - np.dot(chemical_potentials, candidate_y_masses[:,0]) - \
                                  np.dot(internal_lagrange, internal_cons_tmp)) - \
                                 (energy_tmp[0,0] - np.dot(chemical_potentials, masses_tmp[:,0]) - \
                                  np.dot(internal_lagrange, candidate_internal_cons))
                #print(idx, 'wolfe', wolfe_criteria, (step_size*np.dot(current_phase_gradient, delta_y)))
                if wolfe_criteria <= 1e-6 * step_size * np.dot(current_phase_gradient, delta_y):
                    break
                candidate_y_masses[:,0] = 0
                candidate_y_energy[0,0] = 0
                candidate_internal_cons[:] = 0
                step_size /= 2

            saved_phase_stepsizes[iteration, idx] = step_size
            for i in range(num_statevars, new_y.shape[0]):
                largest_internal_dof_change = max(largest_internal_dof_change, abs(new_y[i] - x[i]))
            x[:] = new_y

            for comp_idx in range(num_components):
                compset.phase_record.mass_obj(masses_tmp[comp_idx, :], x, comp_idx)
                all_phase_amounts[idx, comp_idx] = masses_tmp[comp_idx, 0]
                if phase_amt[idx] > 0:
                    current_elemental_amounts[comp_idx] += phase_amt[idx] * masses_tmp[comp_idx, 0]
            compset.phase_record.obj(all_phase_energies[idx, :], x)

        # INTERMISSION: Suspend composition sets of the same phase where there is no miscibility gap
        if False:
            newly_suspended_compset = False
            for cs_idx in range(len(compsets)):
                compset = compsets[cs_idx]
                for cs_idx2 in range(len(compsets)):
                    if cs_idx == cs_idx2:
                        continue
                    if not ((phase_amt[cs_idx] > 0) and (phase_amt[cs_idx2] > 0)):
                        continue
                    compset2 = compsets[cs_idx2]
                    if compset.phase_record.phase_name != compset2.phase_record.phase_name:
                        continue
                    max_mass_diff = 0
                    for comp_idx in range(num_components):
                        max_mass_diff = max(max_mass_diff, abs(all_phase_amounts[cs_idx, comp_idx] - all_phase_amounts[cs_idx2, comp_idx]))
                    if max_mass_diff < 1e-2:
                        if (all_phase_energies[cs_idx, 0] < all_phase_energies[cs_idx2, 0]):
                            # keep idx; remove idx2
                            if cs_idx2 not in suspended_compsets:
                                suspended_compsets.append(cs_idx2)
                                phase_amt[cs_idx] += phase_amt[cs_idx2]
                                newly_suspended_compset = True
                        else:
                            # keep idx2; remove idx
                            if cs_idx not in suspended_compsets:
                                suspended_compsets.append(cs_idx)
                                phase_amt[cs_idx2] += phase_amt[cs_idx]
                                newly_suspended_compset = True
            #print('all_phase_amounts', np.array(all_phase_amounts))
            #print('suspended_compsets', suspended_compsets)
            if newly_suspended_compset:
                for idx in suspended_compsets:
                    phase_amt[idx] = 0
                free_stable_compset_indices = np.array([i for i in range(phase_amt.shape[0])
                                            if phase_amt[i] > MIN_SITE_FRACTION], dtype=np.int32)

        infeasibility = 0
        if False:
            # Restoration phase
            lmul, free_stable_compset_indices = restore_solution_feasibility(compsets, dof, phase_amt,
                                            free_stable_compset_indices, num_statevars, num_components,
                                            prescribed_system_amount, free_chemical_potential_indices,
                                            prescribed_element_indices, prescribed_elemental_amounts)

            infeasibility, _, _ = compute_infeasibility(compsets, dof, lmul, phase_amt,
                                    free_stable_compset_indices, num_statevars, num_components,
                                    prescribed_system_amount, free_chemical_potential_indices,
                                    prescribed_element_indices, prescribed_elemental_amounts)
            free_stable_compset_indices = np.array(np.nonzero(phase_amt)[0], dtype=np.int32)

        # SECOND STEP: Update potentials and phase amounts, according to conditions
        num_stable_phases = free_stable_compset_indices.shape[0]
        print('num_stable_phases', num_stable_phases)
        num_fixed_components = len(prescribed_elemental_amounts)
        num_free_variables = free_chemical_potential_indices.shape[0] + num_stable_phases + \
                             free_statevar_indices.shape[0]
        equilibrium_matrix = np.zeros((num_stable_phases + num_fixed_components + 1, num_free_variables), order='F')
        equilibrium_soln = np.zeros(num_stable_phases + num_fixed_components + 1)
        if (num_stable_phases + num_fixed_components + 1) != num_free_variables:
            raise ValueError('Conditions do not obey Gibbs Phase Rule')

        mass_residual = fill_equilibrium_system(equilibrium_matrix, equilibrium_soln, compsets, chemical_potentials,
                                                current_elemental_amounts, phase_amt, free_chemical_potential_indices,
                                                free_statevar_indices, free_stable_compset_indices,
                                                fixed_chemical_potential_indices,
                                                prescribed_element_indices,
                                                prescribed_elemental_amounts, num_statevars, prescribed_system_amount,
                                                dof)
        lstsq(&equilibrium_matrix[0,0], equilibrium_matrix.shape[0], equilibrium_matrix.shape[1],
              &equilibrium_soln[0], 1e-21)

        extract_equilibrium_solution(chemical_potentials, phase_amt, delta_statevars,
                                     free_chemical_potential_indices, free_statevar_indices,
                                     free_stable_compset_indices, equilibrium_soln,
                                     largest_statevar_change, largest_phase_amt_change, dof)
        saved_chemical_potentials[iteration, :] = chemical_potentials
        saved_phase_amounts[iteration, :] = phase_amt
        saved_infeasibilities[iteration] = infeasibility
        #print('new_chemical_potentials', np.array(chemical_potentials))
        #print('new_phase_amt', np.array(phase_amt))
        #print('times_compset_removed', np.array(times_compset_removed))
        # Wait for mass balance to be satisfied before changing phases
        # Phases that "want" to be removed will keep having their phase_amt set to zero, so mass balance is unaffected
        #print(f'mass_residual {mass_residual} largest_internal_cons_max_residual {largest_internal_cons_max_residual}')
        #print(f'largest_internal_dof_change {largest_internal_dof_change}')
        system_is_feasible = (mass_residual < 1e-05) and (largest_internal_cons_max_residual < 1e-10)
        if system_is_feasible:
            #free_stable_compset_indices = np.array([i for i in range(phase_amt.shape[0])
            #                                        if (phase_amt[i] > MIN_SITE_FRACTION) and \
            #                                        (i not in suspended_compsets)], dtype=np.int32)
            # Check driving forces for metastable phases
            for idx in range(len(compsets)):
                all_phase_energies[idx, 0] -= np.dot(chemical_potentials, all_phase_amounts[idx, :])
            converged, new_free_stable_compset_indices = \
                check_convergence_and_change_phases(phase_amt, free_stable_compset_indices, metastable_phase_iterations,
                                                    times_compset_removed, all_phase_energies,
                                                    largest_internal_dof_change, largest_phase_amt_change,
                                                    largest_statevar_change, iteration > 5)
            free_stable_compset_indices = np.array(new_free_stable_compset_indices, dtype=np.int32)
            if converged:
                converged = True
                break
        for idx in range(len(compsets)):
            if idx in free_stable_compset_indices:
                metastable_phase_iterations[idx] = 0
            else:
                metastable_phase_iterations[idx] += 1

    x = dof[0]
    for cs_dof in dof[1:]:
        x = np.r_[x, cs_dof[num_statevars:]]
    x = np.r_[x, phase_amt]
    saved_debug_ds = xr.Dataset({'MU': saved_chemical_potentials, 'NP': saved_phase_amounts,
                                 'stepsize': saved_phase_stepsizes, 'GM': saved_phase_energies,
                                 'all_dof': saved_phase_dof, 'delta_dof': saved_phase_delta_dof,
                                 'X': saved_phase_compositions, 'DF': saved_phase_driving_forces,
                                 'infeasibility': saved_infeasibilities})
    saved_debug_ds.to_netcdf(f'debug-{stamp}.cdf')
    return converged, x, np.array(chemical_potentials)
