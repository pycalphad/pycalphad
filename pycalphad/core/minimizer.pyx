#cython: unraisable_tracebacks=True
cimport cython
import numpy as np
cimport numpy as np
from pycalphad.core.composition_set cimport CompositionSet
from pycalphad.core.constants import MIN_SITE_FRACTION
from copy import deepcopy
cimport scipy.linalg.cython_lapack as cython_lapack
from libc.stdlib cimport malloc, free

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
                               int num_statevars, double[::1] chemical_potentials, double[::1] phase_dof,
                               int[::1] fixed_phase_dof_indices):
    "Compute the LHS of Eq. 41, Sundman 2015."
    cdef int comp_idx, i, j, cons_idx, fixed_dof_idx
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
        compset.phase_record.formulamole_hess(mass_hess_tmp[comp_idx, :, :], phase_dof, comp_idx)
    for comp_idx in range(num_components):
        for i in range(compset.phase_record.phase_dof):
            for j in range(i, compset.phase_record.phase_dof):
                phase_matrix[i, j] -= chemical_potentials[comp_idx] * mass_hess_tmp[comp_idx,
                                                                                    num_statevars+i,
                                                                                    num_statevars+j]
                if i != j:
                    phase_matrix[j, i] -= chemical_potentials[comp_idx] * mass_hess_tmp[comp_idx,
                                                                                        num_statevars+j,
                                                                                        num_statevars+i]

    phase_matrix[compset.phase_record.phase_dof:compset.phase_record.phase_dof+compset.phase_record.num_internal_cons,
                 :compset.phase_record.phase_dof] = cons_jac_tmp[:, num_statevars:]
    phase_matrix[:compset.phase_record.phase_dof,
                 compset.phase_record.phase_dof:compset.phase_record.phase_dof+compset.phase_record.num_internal_cons] \
        = cons_jac_tmp[:, num_statevars:].T

    for cons_idx in range(fixed_phase_dof_indices.shape[0]):
        fixed_dof_idx = fixed_phase_dof_indices[cons_idx]
        phase_matrix[compset.phase_record.phase_dof + compset.phase_record.num_internal_cons + cons_idx, fixed_dof_idx] = 1
        phase_matrix[fixed_dof_idx, compset.phase_record.phase_dof + compset.phase_record.num_internal_cons] = 1


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
    compset.phase_record.formulahess(hess_tmp, phase_dof)
    compset.phase_record.formulagrad(grad_tmp, phase_dof)

    for comp_idx in range(num_components):
        compset.phase_record.formulamole_grad(mass_jac_tmp[comp_idx, :], phase_dof, comp_idx)

    compute_phase_matrix(phase_matrix, hess_tmp, compset, num_statevars, chemical_potentials, phase_dof,
                         np.array([], dtype=np.int32))

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


cdef np.ndarray fill_equilibrium_system_for_phase(double[::1,:] equilibrium_matrix, double[::1] equilibrium_rhs,
                                            double energy, double[::1] grad, double[:, ::1] hess,
                                            double[:, ::1] masses, double[:, ::1] mass_jac,
                                            double [::1] system_mole_fractions, double current_system_amount, int num_phase_dof,
                                            double[:, ::1] full_e_matrix, double[::1] chemical_potentials,
                                            double[::1] phase_amt, int[::1] free_chemical_potential_indices,
                                            int[::1] free_statevar_indices, int[::1] free_stable_compset_indices,
                                            int[::1] fixed_stable_compset_indices,
                                            int[::1] fixed_chemical_potential_indices,
                                            int[::1] prescribed_element_indices, double[::1] prescribed_elemental_amounts,
                                            int idx, int stable_idx, int num_statevars) except +:
    cdef int comp_idx, component_idx, chempot_idx, compset_idx, statevar_idx, fixed_component_idx, component_row_offset, i, j
    cdef int num_components = chemical_potentials.shape[0]
    cdef int num_stable_phases = free_stable_compset_indices.shape[0]
    cdef int num_fixed_phases = fixed_stable_compset_indices.shape[0]
    cdef int num_fixed_components = prescribed_elemental_amounts.shape[0]
    # Eq. 44
    cdef double[::1] c_G = np.zeros(num_phase_dof)
    cdef double[::1] delta_m = np.zeros(num_components)
    cdef double[:, ::1] c_statevars = np.zeros((num_phase_dof, num_statevars))
    cdef double[:, ::1] c_component = np.zeros((num_components, num_phase_dof))
    cdef double moles_normalization = 0
    cdef double[::1] moles_normalization_grad = np.zeros(num_statevars+num_phase_dof)
    for i in range(num_phase_dof):
        for j in range(num_phase_dof):
            c_G[i] -= full_e_matrix[i, j] * grad[num_statevars+j]
    #print('c_G', np.array(c_G))
    for i in range(num_phase_dof):
        for j in range(num_phase_dof):
            for statevar_idx in range(num_statevars):
                c_statevars[i, statevar_idx] -= full_e_matrix[i, j] * hess[num_statevars + j, statevar_idx]
    for comp_idx in range(num_components):
        for i in range(num_phase_dof):
            for j in range(num_phase_dof):
                c_component[comp_idx, i] += mass_jac[comp_idx, num_statevars + j] * full_e_matrix[i, j]
    #print('c_component', np.array(c_component))
    for comp_idx in range(num_components):
        for i in range(num_phase_dof):
            mu_c_sum = np.dot(np.array(c_component)[:, i], chemical_potentials)
            delta_m[comp_idx] += mass_jac[comp_idx, num_statevars + i] * (mu_c_sum + c_G[i])
    #print('delta_m', np.array(delta_m), np.sum(delta_m))
    for comp_idx in range(num_components):
        moles_normalization += masses[comp_idx, 0]
        for i in range(num_phase_dof+num_statevars):
            moles_normalization_grad[i] += mass_jac[comp_idx, i]
    #print('current_system_amount', current_system_amount)
    #print('moles_normalization', moles_normalization)
    #print('moles_normalization_grad', np.array(moles_normalization_grad))
    # KEY STEPS for filling equilibrium matrix
    # 1. Contribute to the row corresponding to this composition set
    # 1a. Loop through potential conditions to fill out each column
    # 2. Contribute to the rows of all fixed components
    # 2a. Loop through potential conditions to fill out each column
    # 3. Contribute to RHS of each component row
    # 4. Add energies to RHS of each stable composition set
    # 5. Subtract contribution from RHS due to any fixed chemical potentials
    # 6. Subtract fixed chemical potentials from each fixed component RHS
    # 7. Subtract fixed chemical potentials from the N=1 row

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

    # 2. Contribute to the row of all fixed components (fixed mole fraction)
    component_row_offset = num_stable_phases + num_fixed_phases
    for fixed_component_idx in range(num_fixed_components):
        component_idx = prescribed_element_indices[fixed_component_idx]
        free_variable_column_offset = 0
        # 2a. This component row: free chemical potentials
        for i in range(free_chemical_potential_indices.shape[0]):
            chempot_idx = free_chemical_potential_indices[i]
            for j in range(c_component.shape[1]):
                equilibrium_matrix[component_row_offset + fixed_component_idx, free_variable_column_offset + i] += \
                    (phase_amt[idx]/current_system_amount) * mass_jac[component_idx, num_statevars+j] * c_component[chempot_idx, j]
            for j in range(c_component.shape[1]):
                equilibrium_matrix[component_row_offset + fixed_component_idx, free_variable_column_offset + i] += \
                    (phase_amt[idx]/current_system_amount) * (-system_mole_fractions[component_idx] * moles_normalization_grad[num_statevars+j]) * c_component[chempot_idx, j]
        free_variable_column_offset += free_chemical_potential_indices.shape[0]
        # 2a. This component row: free stable composition sets
        for i in range(free_stable_compset_indices.shape[0]):
            compset_idx = free_stable_compset_indices[i]
            # Only fill this out if the current idx is equal to a free composition set
            if compset_idx == idx:
                equilibrium_matrix[
                    component_row_offset + fixed_component_idx, free_variable_column_offset + i] = \
                    (1./current_system_amount)*(masses[component_idx, 0] - system_mole_fractions[component_idx] * moles_normalization)
        free_variable_column_offset += free_stable_compset_indices.shape[0]
        # 2a. This component row: free state variables
        for i in range(free_statevar_indices.shape[0]):
            statevar_idx = free_statevar_indices[i]
            # XXX: Isn't this missing a dZ/dT term? Only relevant for T-dependent mass calculations...
            for j in range(c_statevars.shape[0]):
                equilibrium_matrix[component_row_offset + fixed_component_idx, free_variable_column_offset + i] += \
                    (phase_amt[idx]/current_system_amount) * mass_jac[component_idx, num_statevars+j] * c_statevars[j, statevar_idx]
            for j in range(c_statevars.shape[0]):
                equilibrium_matrix[component_row_offset + fixed_component_idx, free_variable_column_offset + i] += \
                    (phase_amt[idx]/current_system_amount) * (-system_mole_fractions[component_idx] * moles_normalization_grad[num_statevars+j]) * c_statevars[j, statevar_idx]
        # 3.
        for j in range(c_G.shape[0]):
            equilibrium_rhs[component_row_offset + fixed_component_idx] += -(phase_amt[idx]/current_system_amount) * \
                mass_jac[component_idx, num_statevars+j] * c_G[j]
        for j in range(c_G.shape[0]):
            equilibrium_rhs[component_row_offset + fixed_component_idx] += -(phase_amt[idx]/current_system_amount) * \
                (-system_mole_fractions[component_idx] * moles_normalization_grad[num_statevars+j]) * c_G[j]

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
    equilibrium_rhs[stable_idx] = energy
    # 5. Subtract fixed chemical potentials from each phase RHS
    for i in range(fixed_chemical_potential_indices.shape[0]):
        chempot_idx = fixed_chemical_potential_indices[i]
        equilibrium_rhs[stable_idx] -= masses[chempot_idx, 0] * chemical_potentials[chempot_idx]
        # 6. Subtract fixed chemical potentials from each fixed component RHS
        for fixed_component_idx in range(num_fixed_components):
            component_idx = prescribed_element_indices[fixed_component_idx]
            for j in range(c_component.shape[1]):
                equilibrium_rhs[component_row_offset + fixed_component_idx] -= (phase_amt[idx]/current_system_amount) * chemical_potentials[
                    chempot_idx] * mass_jac[component_idx, num_statevars+j] * c_component[chempot_idx, j]
            for j in range(c_component.shape[1]):
                equilibrium_rhs[component_row_offset + fixed_component_idx] -= (phase_amt[idx]/current_system_amount) * chemical_potentials[
                    chempot_idx] * (-system_mole_fractions[component_idx] * moles_normalization_grad[num_statevars+j]) * c_component[chempot_idx, j]
        # 7. Subtract fixed chemical potentials from the N=1 row
        for component_idx in range(num_components):
            for j in range(c_component.shape[1]):
                equilibrium_rhs[system_amount_index] -= phase_amt[idx] * chemical_potentials[
                    chempot_idx] * mass_jac[component_idx, num_statevars+j] * c_component[chempot_idx, j]
    return np.array(delta_m)


cdef object fill_equilibrium_system(double[::1,:] equilibrium_matrix, double[::1] equilibrium_rhs,
                                    object compsets, double[::1] chemical_potentials,
                                    double[::1] current_elemental_amounts, double[::1] phase_amt,
                                    int[::1] free_chemical_potential_indices,
                                    int[::1] free_statevar_indices, int[::1] free_stable_compset_indices,
                                    int[::1] fixed_stable_compset_indices,
                                    int[::1] fixed_chemical_potential_indices,
                                    int[::1] prescribed_element_indices, double[::1] prescribed_elemental_amounts,
                                    int num_statevars, double prescribed_system_amount, object dof,
                                    bint flip_residual_sign, bint finalize_chempots) except +:
    cdef int stable_idx, idx, component_row_offset, component_idx, fixed_component_idx, comp_idx, system_amount_index
    cdef CompositionSet compset
    cdef int num_components = chemical_potentials.shape[0]
    cdef int num_stable_phases = free_stable_compset_indices.shape[0]
    cdef int num_fixed_phases = fixed_stable_compset_indices.shape[0]
    cdef int num_fixed_components = len(prescribed_elemental_amounts)
    # Placeholder (output unused)
    cdef int[::1] ipiv = np.empty(10*num_components*num_stable_phases, dtype=np.int32)
    cdef double mass_residual
    cdef double step_size = 1.0
    cdef double current_system_amount = 0
    cdef double[::1] x
    cdef double[::1,:] energy_tmp
    cdef double[::1] grad_tmp
    cdef double[:,::1] hess_tmp
    cdef double[:,::1] masses_tmp
    cdef double[:,::1] mass_jac_tmp
    cdef double[:,::1] phase_matrix
    cdef double[:,::1] e_matrix, full_e_matrix
    cdef double[::1] mole_fractions = np.zeros(num_components)
    cdef np.ndarray[ndim=2,dtype=np.float64_t] all_delta_m = np.zeros((len(compsets), num_components))

    mass_residuals = np.zeros(num_components)
    # Compute normalized global quantities
    for idx in range(len(compsets)):
        if phase_amt[idx] <= 0:
            continue
        x = dof[idx]
        compset = compsets[idx]
        masses_tmp = np.zeros((num_components, 1))
        for comp_idx in range(num_components):
            compset.phase_record.formulamole_obj(masses_tmp[comp_idx, :], x, comp_idx)
            mole_fractions[comp_idx] += phase_amt[idx] * masses_tmp[comp_idx, 0]
            current_system_amount += phase_amt[idx] * masses_tmp[comp_idx, 0]
    for comp_idx in range(mole_fractions.shape[0]):
        mole_fractions[comp_idx] /= current_system_amount
    #print('mole_fractions', np.array(mole_fractions))
    for stable_idx in range(free_stable_compset_indices.shape[0]):
        idx = free_stable_compset_indices[stable_idx]
        compset = compsets[idx]
        # TODO: Use better dof storage
        # Calculate key phase quantities starting here
        x = dof[idx]
        energy_tmp = np.zeros((1, 1))
        masses_tmp = np.zeros((num_components, 1))
        mass_jac_tmp = np.zeros((num_components, num_statevars + compset.phase_record.phase_dof))
        if finalize_chempots:
            fixed_phase_dof_indices = np.array(np.nonzero(np.array(x)[num_statevars:] <= 1.1*MIN_SITE_FRACTION)[0], dtype=np.int32)
        else:
            fixed_phase_dof_indices = np.array([], dtype=np.int32)
        # Compute phase matrix (LHS of Eq. 41, Sundman 2015)
        phase_matrix = np.zeros(
            (compset.phase_record.phase_dof + compset.phase_record.num_internal_cons + fixed_phase_dof_indices.shape[0],
             compset.phase_record.phase_dof + compset.phase_record.num_internal_cons + fixed_phase_dof_indices.shape[0]))
        full_e_matrix = np.eye(compset.phase_record.phase_dof + compset.phase_record.num_internal_cons + fixed_phase_dof_indices.shape[0])
        hess_tmp = np.zeros((num_statevars + compset.phase_record.phase_dof,
                             num_statevars + compset.phase_record.phase_dof))
        grad_tmp = np.zeros(num_statevars + compset.phase_record.phase_dof)

        compset.phase_record.formulaobj(energy_tmp[:, 0], x)
        for comp_idx in range(num_components):
            compset.phase_record.formulamole_grad(mass_jac_tmp[comp_idx, :], x, comp_idx)
            compset.phase_record.formulamole_obj(masses_tmp[comp_idx, :], x, comp_idx)
        compset.phase_record.formulahess(hess_tmp, x)
        compset.phase_record.formulagrad(grad_tmp, x)

        compute_phase_matrix(phase_matrix, hess_tmp, compset, num_statevars, chemical_potentials, x, fixed_phase_dof_indices)

        invert_matrix(&phase_matrix[0,0], phase_matrix.shape[0], &full_e_matrix[0,0], &ipiv[0])

        delta_m = fill_equilibrium_system_for_phase(equilibrium_matrix, equilibrium_rhs, energy_tmp[0, 0], grad_tmp, hess_tmp,
                                                    masses_tmp, mass_jac_tmp, mole_fractions, current_system_amount,
                                                    compset.phase_record.phase_dof, full_e_matrix, chemical_potentials,
                                                    phase_amt, free_chemical_potential_indices, free_statevar_indices,
                                                    free_stable_compset_indices, fixed_stable_compset_indices,
                                                    fixed_chemical_potential_indices,
                                                    prescribed_element_indices, prescribed_elemental_amounts,
                                                    idx, stable_idx, num_statevars)
        all_delta_m[idx, :] = delta_m

    # Handle phases which are fixed to be stable at some amount
    # Example shown in Eq. 60, Sundman et al 2015
    for fixed_idx in range(fixed_stable_compset_indices.shape[0]):
        idx = fixed_stable_compset_indices[fixed_idx]
        compset = compsets[idx]
        # TODO: Use better dof storage
        # Calculate key phase quantities starting here
        x = dof[idx]
        energy_tmp = np.zeros((1, 1))
        masses_tmp = np.zeros((num_components, 1))
        grad_tmp = np.zeros(num_statevars + compset.phase_record.phase_dof)

        compset.phase_record.formulaobj(energy_tmp[:, 0], x)
        equilibrium_rhs[num_stable_phases + fixed_idx] = energy_tmp[0, 0]
        for fcp_idx in range(free_chemical_potential_indices.shape[0]):
            comp_idx = free_chemical_potential_indices[fcp_idx]
            compset.phase_record.formulamole_obj(masses_tmp[comp_idx, :], x, comp_idx)
            equilibrium_matrix[num_stable_phases + fixed_idx, fcp_idx] = masses_tmp[comp_idx, 0]
        compset.phase_record.formulagrad(grad_tmp, x)
        for free_idx in range(free_statevar_indices.shape[0]):
            sv_idx = free_statevar_indices[free_idx]
            equilibrium_matrix[num_stable_phases + fixed_idx,
                               free_chemical_potential_indices.shape[0] + num_stable_phases + free_idx] = -grad_tmp[sv_idx]
        # TODO: Compute actual delta_m for fixed phase
        delta_m = np.zeros(num_components)
        all_delta_m[idx, :] = delta_m

    # Add mass residual to fixed component row RHS, plus N=1 row
    mass_residual = 0.0
    component_row_offset = num_stable_phases + num_fixed_phases
    system_amount_index = component_row_offset + num_fixed_components
    for fixed_component_idx in range(num_fixed_components):
        component_idx = prescribed_element_indices[fixed_component_idx]
        mass_residuals[fixed_component_idx] = (mole_fractions[component_idx] - prescribed_elemental_amounts[fixed_component_idx])
        mass_residual += abs(
            mole_fractions[component_idx] - prescribed_elemental_amounts[fixed_component_idx]) #/ abs(prescribed_elemental_amounts[fixed_component_idx])
        if not flip_residual_sign:
            component_residual = mole_fractions[component_idx] - prescribed_elemental_amounts[fixed_component_idx]
        else:
            component_residual = prescribed_elemental_amounts[fixed_component_idx] - mole_fractions[component_idx]
        equilibrium_rhs[component_row_offset + fixed_component_idx] -= component_residual
    mass_residual += abs(current_system_amount - prescribed_system_amount)
    system_residual = current_system_amount - prescribed_system_amount
    equilibrium_rhs[system_amount_index] -= system_residual
    return mass_residuals, np.array(all_delta_m)


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
        chemical_potentials[chempot_idx] = equilibrium_soln[soln_index_offset + i]
    soln_index_offset += free_chemical_potential_indices.shape[0]
    largest_delta_phase_amt = np.max(np.abs(equilibrium_soln[soln_index_offset:soln_index_offset+free_stable_compset_indices.shape[0]]))
    for i in range(free_stable_compset_indices.shape[0]):
        compset_idx = free_stable_compset_indices[i]
        phase_amt_change = float(phase_amt[compset_idx])
        phase_amt[compset_idx] += equilibrium_soln[soln_index_offset + i]
        #phase_amt[compset_idx] = np.maximum(0.0, phase_amt[compset_idx])
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
        newly_metastable_compsets = set(np.nonzero((np.array(metastable_phase_iterations) < 5))[0]) - \
                                    set(current_free_stable_compset_indices)
        add_criteria = np.logical_and(np.array(driving_forces) > 1e-5, np.array(times_compset_removed) < 4)
        compsets_to_add = set((np.nonzero(add_criteria)[0])) - newly_metastable_compsets
    else:
        compsets_to_add = set()
    #print('compsets_to_add', compsets_to_add)
    compsets_to_remove = set(np.nonzero(np.array(phase_amt) < 1e-9)[0])
    new_free_stable_compset_indices = np.array(sorted((set(current_free_stable_compset_indices) - compsets_to_remove)
                                                      | compsets_to_add
                                                      )
                                               )
    removed_compset_indices = set(current_free_stable_compset_indices) - set(new_free_stable_compset_indices)
    for idx in removed_compset_indices:
        times_compset_removed[idx] += 1
    converged = False
    if set(current_free_stable_compset_indices) == set(new_free_stable_compset_indices):
        # feasible system, and no phases to add or remove
        converged = True
    #print('converged, new_free_stable_compset_indices', converged, new_free_stable_compset_indices)
    return converged, new_free_stable_compset_indices

cdef class SystemSpecification:
    cdef int num_statevars, num_components
    cdef double prescribed_system_amount
    cdef double[::1] initial_chemical_potentials, prescribed_elemental_amounts
    cdef int[::1] prescribed_element_indices
    cdef int[::1] free_chemical_potential_indices, free_statevar_indices
    cdef int[::1] fixed_chemical_potential_indices, fixed_statevar_indices, fixed_stable_compset_indices

    def __init__(self, int num_statevars, int num_components, double prescribed_system_amount,
                   double[::1] initial_chemical_potentials, double[::1] prescribed_elemental_amounts,
                   int[::1] prescribed_element_indices, int[::1] free_chemical_potential_indices,
                   int[::1] free_statevar_indices, int[::1] fixed_chemical_potential_indices,
                   int[::1] fixed_statevar_indices, int[::1] fixed_stable_compset_indices):
        self.num_statevars = num_statevars
        self.num_components = num_components
        self.prescribed_system_amount = prescribed_system_amount
        self.initial_chemical_potentials = initial_chemical_potentials
        self.prescribed_elemental_amounts = prescribed_elemental_amounts
        self.prescribed_element_indices = prescribed_element_indices
        self.free_chemical_potential_indices = free_chemical_potential_indices
        self.free_statevar_indices = free_statevar_indices
        self.fixed_chemical_potential_indices = fixed_chemical_potential_indices
        self.fixed_statevar_indices = fixed_statevar_indices
        self.fixed_stable_compset_indices = fixed_stable_compset_indices
    def __getstate__(self):
        return (self.num_statevars, self.num_components, self.prescribed_system_amount,
                np.array(self.initial_chemical_potentials), np.array(self.prescribed_elemental_amounts),
                np.array(self.prescribed_element_indices), np.array(self.free_chemical_potential_indices),
                np.array(self.free_statevar_indices), np.array(self.fixed_chemical_potential_indices),
                np.array(self.fixed_statevar_indices), np.array(self.fixed_stable_compset_indices))
    def __setstate__(self, state):
        self.__init__(*state)

cdef class SystemState:
    cdef list compsets
    cdef object dof
    cdef int iteration
    cdef double mass_residual, largest_internal_cons_max_residual, largest_internal_dof_change
    cdef double[::1] phase_amt, chemical_potentials, chempot_diff
    cdef double[:, ::1] phase_compositions, delta_ms
    cdef double[1] largest_statevar_change, largest_phase_amt_change
    cdef int[::1] free_stable_compset_indices
    def __init__(self, SystemSpecification spec, list compsets):
        cdef CompositionSet compset
        self.compsets = compsets
        self.dof = [np.array(compset.dof) for compset in compsets]
        self.iteration = 0
        self.mass_residual = 1e10
        self.largest_internal_cons_max_residual = 0
        self.largest_internal_dof_change = 0
        # Phase fractions need to be converted to moles of formula
        self.phase_amt = np.array([compset.NP for compset in compsets])
        self.chemical_potentials = np.zeros(spec.num_components)
        self.chempot_diff = np.zeros(spec.num_components)
        self.chempot_diff[:] = np.inf
        self.delta_ms = np.zeros((len(compsets), spec.num_components))
        self.phase_compositions = np.zeros((len(compsets), spec.num_components))
        self.free_stable_compset_indices = np.array(np.arange(len(compsets)), dtype=np.int32)
        self.largest_statevar_change[0] = 0
        self.largest_phase_amt_change[0] = 0

        for idx in range(self.phase_amt.shape[0]):
            compset = self.compsets[idx]
            masses_tmp = np.zeros((spec.num_components, 1))
            x = self.dof[idx]
            for comp_idx in range(spec.num_components):
                compset.phase_record.formulamole_obj(masses_tmp[comp_idx, :], x, comp_idx)
                self.phase_compositions[idx, comp_idx] = masses_tmp[comp_idx, 0]
                masses_tmp[:,:] = 0
            # Convert phase fractions to formula units
            self.phase_amt[idx] /= np.sum(self.phase_compositions[idx])
    def __getstate__(self):
        return (self.compsets, self.dof, self.iteration, self.mass_residual, self.largest_internal_cons_max_residual,
                self.largest_internal_dof_change, np.array(self.phase_amt), np.array(self.chemical_potentials),
                np.array(self.chempot_diff), np.array(self.delta_ms), np.array(self.phase_compositions),
                self.largest_statevar_change[0], self.largest_phase_amt_change[0],
                np.array(self.free_stable_compset_indices))
    def __setstate__(self, state):
        (self.compsets, self.dof, self.iteration, self.mass_residual, self.largest_internal_cons_max_residual,
            self.largest_internal_dof_change, self.phase_amt, self.chemical_potentials,
            self.chempot_diff, self.delta_ms, self.phase_compositions, self.largest_statevar_change[0],
            self.largest_phase_amt_change[0], self.free_stable_compset_indices) = state


cpdef take_step(SystemSpecification spec, SystemState state, double step_size):
    cdef double largest_internal_cons_max_residual = 0
    cdef double largest_internal_dof_change = 0
    cdef double[::1] delta_statevars = np.zeros(spec.num_statevars)
    cdef double[::1] current_elemental_amounts = np.zeros(spec.num_components)
    cdef double[:,::1] all_phase_amounts = np.array(state.phase_compositions)
    cdef double[:,::1] masses_tmp, mass_gradient, energy_tmp
    cdef double[::1,:] equilibrium_matrix  # Fortran ordering required by call into lapack
    cdef double[:,::1] phase_matrix  # Fortran ordering required by call into lapack, but this is symmetric
    cdef double[::1] equilibrium_rhs, equilibrium_soln, phase_rhs, soln, internal_cons_tmp, old_chemical_potentials
    cdef CompositionSet compset
    cdef int[::1] ipiv = np.zeros(len(state.compsets) * max([compset.phase_record.phase_dof +
                                                       compset.phase_record.num_internal_cons
                                                       for compset in state.compsets]), dtype=np.int32)
    # FIRST STEP: Update phase internal degrees of freedom
    large_internal_cons = False
    for idx, compset in enumerate(state.compsets):
        if state.iteration == 0:
            break
        # TODO: Use better dof storage
        x = state.dof[idx]
        #print('x', np.array(x))
        masses_tmp = np.zeros((spec.num_components, 1))
        energy_tmp = np.zeros((1,1))
        internal_cons_tmp = np.zeros(compset.phase_record.num_internal_cons)
        # Compute phase matrix (LHS of Eq. 41, Sundman 2015)
        phase_matrix = np.zeros((compset.phase_record.phase_dof + compset.phase_record.num_internal_cons,
                                 compset.phase_record.phase_dof + compset.phase_record.num_internal_cons))
        soln = np.zeros(compset.phase_record.phase_dof + compset.phase_record.num_internal_cons)
        compset.phase_record.internal_cons_func(internal_cons_tmp, x)
        internal_cons_tmp[:] = 0
        # RHS copied into soln, then overwritten by solve()
        internal_cons_max_residual = \
            compute_phase_system(phase_matrix, soln, compset, delta_statevars, state.chemical_potentials, x)
        #print('chemical_potentials', np.array(chemical_potentials))
        #print('phase_matrix', np.array(phase_matrix))
        #print('phase_rhs', np.array(soln))
        phase_gradient = -np.array(soln[:compset.phase_record.phase_dof])
        # phase_matrix is symmetric by construction, so we can pass in a C-ordered array
        solve(&phase_matrix[0,0], phase_matrix.shape[0], &soln[0], &ipiv[0])

        delta_y = soln[:compset.phase_record.phase_dof]
        internal_lagrange = soln[compset.phase_record.phase_dof:]
        mass_gradient = np.zeros((spec.num_components, spec.num_statevars + compset.phase_record.phase_dof))

        for comp_idx in range(spec.num_components):
            compset.phase_record.formulamole_obj(masses_tmp[comp_idx, :], x, comp_idx)
            compset.phase_record.formulamole_grad(mass_gradient[comp_idx, :], x, comp_idx)
        compset.phase_record.internal_cons_func(internal_cons_tmp, x)
        compset.phase_record.formulaobj(energy_tmp[0, :], x)
        delta_m = np.dot(mass_gradient[:, spec.num_statevars:], delta_y)
        rel_delta_m = np.array(delta_m) / np.squeeze(masses_tmp)
        #print(idx, 'delta_m', np.array(delta_m))
        #print('rel_delta_m', np.array(rel_delta_m))
        largest_internal_cons_max_residual = max(largest_internal_cons_max_residual, internal_cons_max_residual)
        new_y = np.array(x)
        candidate_y_energy = np.zeros((1,1))
        candidate_y_masses = np.zeros((spec.num_components, 1))
        candidate_internal_cons = np.zeros(compset.phase_record.num_internal_cons)
        current_phase_gradient = np.array(phase_gradient)
        grad_delta_dot = np.dot(current_phase_gradient, delta_y)
        #print(idx, 'grad_delta_dot', grad_delta_dot)
        #print('delta_y', np.array(delta_y))
        #print('grad_delta_over_dm', grad_delta_dot/delta_m)
        #if np.dot(current_phase_gradient, delta_y) > 0:
        #    print('delta_y is not a descent direction!')
        #if np.max(np.abs(delta_y)) < 1e-7:
        #    step_size = 0.5
        #else:
        #    step_size = 1e-7 / np.max(np.abs(delta_y))
        #print('step_size', step_size)
        minimum_step_size = 1e-20 * step_size
        wolfe_passed = False
        while step_size >= minimum_step_size:
            exceeded_bounds = False
            for i in range(spec.num_statevars, new_y.shape[0]):
                new_y[i] = x[i] + step_size * delta_y[i-spec.num_statevars]
                if new_y[i] > 1:
                    if (new_y[i] - 1) > 1e-11:
                        # Allow some tolerance in the name of progress
                        exceeded_bounds = True
                    new_y[i] = 1
                    if delta_y[i-spec.num_statevars] > 0:
                        current_phase_gradient[i-spec.num_statevars] = 0
                elif new_y[i] < MIN_SITE_FRACTION:
                    if (MIN_SITE_FRACTION - new_y[i]) > 1e-11:
                        # Allow some tolerance in the name of progress
                        exceeded_bounds = True
                    # Reduce by two orders of magnitude, or MIN_SITE_FRACTION, whichever is larger
                    new_y[i] = max(x[i]/100, MIN_SITE_FRACTION)
                    if delta_y[i-spec.num_statevars] < 0:
                        current_phase_gradient[i-spec.num_statevars] = 0
            if exceeded_bounds:
                step_size *= 0.5
                continue
            break

        for i in range(spec.num_statevars, new_y.shape[0]):
            largest_internal_dof_change = max(largest_internal_dof_change, abs(new_y[i] - x[i]))
        x[:] = new_y
        #print(idx, 'step_size', step_size)
        #print(idx, 'new_y', np.array(new_y))

    state.largest_internal_cons_max_residual = largest_internal_cons_max_residual
    state.largest_internal_dof_change = largest_internal_dof_change

    # SECOND STEP: Update potentials and phase amounts, according to conditions
    current_elemental_amounts[:] = 0

    for idx in range(state.phase_amt.shape[0]):
        compset = state.compsets[idx]
        masses_tmp = np.zeros((spec.num_components, 1))
        x = state.dof[idx]
        for comp_idx in range(spec.num_components):
            compset.phase_record.formulamole_obj(masses_tmp[comp_idx, :], x, comp_idx)
            state.phase_compositions[idx, comp_idx] = masses_tmp[comp_idx, 0]
            if state.phase_amt[idx] > 0:
                current_elemental_amounts[comp_idx] += state.phase_amt[idx] * masses_tmp[comp_idx, 0]
            masses_tmp[:,:] = 0

    num_stable_phases = state.free_stable_compset_indices.shape[0]
    num_fixed_phases = spec.fixed_stable_compset_indices.shape[0]
    num_fixed_components = len(spec.prescribed_elemental_amounts)
    num_free_variables = spec.free_chemical_potential_indices.shape[0] + num_stable_phases + \
                         spec.free_statevar_indices.shape[0]

    equilibrium_matrix = np.zeros((num_stable_phases + num_fixed_phases + num_fixed_components + 1, num_free_variables), order='F')
    equilibrium_soln = np.zeros(num_stable_phases + num_fixed_phases + num_fixed_components + 1)
    if (num_stable_phases + num_fixed_phases + num_fixed_components + 1) != num_free_variables:
        raise ValueError('Conditions do not obey Gibbs Phase Rule')

    # Internal degrees of freedom at MIN_SITE_FRACTION need to be fixed when computing chemical potentials.
    # This is to ensure that the gradients accurately reflect the boundary condition.
    # However, we do not want to fix dof during the optimization, so we only enable this when close to convergence.
    finalize_chemical_potentials = (state.mass_residual < 1e-11)

    #print('finalize_chemical_potentials', finalize_chemical_potentials)
    equilibrium_matrix[:,:] = 0
    equilibrium_soln[:] = 0
    mass_residuals, delta_ms = fill_equilibrium_system(equilibrium_matrix, equilibrium_soln, state.compsets, state.chemical_potentials,
                                            current_elemental_amounts, state.phase_amt, spec.free_chemical_potential_indices,
                                            spec.free_statevar_indices, state.free_stable_compset_indices,
                                            spec.fixed_stable_compset_indices, spec.fixed_chemical_potential_indices,
                                            spec.prescribed_element_indices,
                                            spec.prescribed_elemental_amounts, spec.num_statevars, spec.prescribed_system_amount,
                                            state.dof, False, finalize_chemical_potentials)
    # In some cases we may have only one stoichiometric phase stable in the system.
    # This will cause the equilibrium matrix to become singular, and the chemical potentials will be nonsensical.
    # This case can be identified by the presence of a row of all zeros in a fixed-mole-fraction row.
    # For this case, we freeze some of the chemical potentials in place.
    for i in range(num_fixed_components):
        if np.all(np.array(equilibrium_matrix[num_stable_phases + num_fixed_phases + i, :]) == 0):
            equilibrium_matrix[num_stable_phases + num_fixed_phases + i, i] = 1
            equilibrium_soln[i] = state.chemical_potentials[spec.prescribed_element_indices[i]]
    #print('equilibrium_matrix', np.array(equilibrium_matrix))
    #print('equilibrium_rhs', np.array(equilibrium_soln))
    state.mass_residual = np.sum(np.abs(mass_residuals))
    state.delta_ms = delta_ms
    lstsq(&equilibrium_matrix[0,0], equilibrium_matrix.shape[0], equilibrium_matrix.shape[1],
          &equilibrium_soln[0], -1)
    old_chemical_potentials = np.array(state.chemical_potentials)
    extract_equilibrium_solution(state.chemical_potentials, state.phase_amt, delta_statevars,
                                 spec.free_chemical_potential_indices, spec.free_statevar_indices,
                                 state.free_stable_compset_indices, equilibrium_soln,
                                 state.largest_statevar_change, state.largest_phase_amt_change, state.dof)
    for idx in range(len(state.compsets)):
        x = state.dof[idx]
        for sv_idx in range(delta_statevars.shape[0]):
            x[sv_idx] += delta_statevars[sv_idx]
        # XXX: Do not merge this temporary hack for the temperature
        # We need real state variable bounds support
        #x[2] = max(300, x[2])
    # Force some chemical potentials to adopt their fixed values
    for cp_idx in range(spec.fixed_chemical_potential_indices.shape[0]):
        comp_idx = spec.fixed_chemical_potential_indices[cp_idx]
        state.chemical_potentials[comp_idx] = spec.initial_chemical_potentials[comp_idx]
    #print('delta_phase_amt', np.array(new_phase_amt) - np.array(phase_amt))
    state.chempot_diff = np.array(state.chemical_potentials) - old_chemical_potentials

cpdef find_solution(list compsets, int[::1] free_stable_compset_indices,
                    int num_statevars, int num_components, double prescribed_system_amount,
                    double[::1] initial_chemical_potentials, int[::1] free_chemical_potential_indices,
                    int[::1] fixed_chemical_potential_indices,
                    int[::1] prescribed_element_indices, double[::1] prescribed_elemental_amounts,
                    int[::1] free_statevar_indices, int[::1] fixed_statevar_indices):
    cdef int iteration, idx, idx2, comp_idx, phase_idx, i
    cdef int num_stable_phases, num_fixed_components, num_free_variables
    cdef CompositionSet compset, compset2
    cdef double mass_residual = 1e-30
    cdef double max_delta_m, delta_energy
    cdef double[::1] x, new_y, delta_y
    cdef double[::1] delta_m = np.zeros(num_components)
    cdef double[::1] chemical_potentials = np.zeros(num_components)
    cdef int[::1] fixed_stable_compset_indices = np.array(np.nonzero([compset.fixed==True for compset in compsets])[0],
                                                          dtype=np.int32)
    cdef list dof = [np.array(compset.dof) for compset in compsets]
    cdef list suspended_compsets = []
    cdef int[::1] stable_phase_iterations = np.zeros(len(compsets), dtype=np.int32)
    cdef int[::1] metastable_phase_iterations = np.zeros(len(compsets), dtype=np.int32)
    cdef int[::1] times_compset_removed = np.zeros(len(compsets), dtype=np.int32)
    cdef bint converged = False
    cdef int max_dof = num_statevars + max([compset.phase_record.phase_dof for compset in compsets])
    cdef SystemSpecification spec = SystemSpecification(num_statevars, num_components, prescribed_system_amount,
                                                        initial_chemical_potentials, prescribed_elemental_amounts,
                                                        prescribed_element_indices,
                                                        free_chemical_potential_indices, free_statevar_indices,
                                                        fixed_chemical_potential_indices, fixed_statevar_indices,
                                                        fixed_stable_compset_indices)
    cdef SystemState state = SystemState(spec, compsets)
    cdef SystemState old_state

    #print('prescribed_element_indices', np.array(prescribed_element_indices))
    #print('prescribed_elemental_amounts', np.array(prescribed_elemental_amounts))

    state.mass_residual = 1e10
    phase_change_counter = 5
    step_size = 1./10
    for iteration in range(1000):
        state.iteration = iteration
        if (state.mass_residual > 10) and (np.max(np.abs(state.chemical_potentials)) > 1.0e10):
            #print('Mass residual and chemical potentials too big; resetting chemical potentials')
            state.chemical_potentials[:] = spec.initial_chemical_potentials

        #take_step(spec, state, step_size)
        old_state = deepcopy(state)
        for backtracking_iteration in range(5):
            take_step(spec, state, step_size)
            print('old_state.phase_amt', np.array(old_state.phase_amt))
            print('state.phase_amt', np.array(state.phase_amt))
            print('old_state.phase_compositions', np.array(old_state.phase_compositions))
            print('state.phase_compositions', np.array(state.phase_compositions))
            print('state.chempot_diff', np.array(state.chempot_diff))
            delta_phase_amt = np.array(state.phase_amt) - np.array(old_state.phase_amt)
            if (state.mass_residual > 1e-2) and (not np.all(np.array(state.chempot_diff) < 1e-4)) and (iteration > 0):
                #if np.max(np.abs(delta_phase_amt)) > 0.1:
                #    state.phase_amt = np.array(old_state.phase_amt) + 1e-6 * delta_phase_amt
                # When mass residual is not satisfied, do not allow phases to leave the system
                # However, if the chemical potentials are changing very little, phases may leave the system
                for j in range(state.phase_amt.shape[0]):
                    if state.phase_amt[j] < 0:
                        state.phase_amt[j] = 1e-6
            delta_m[:] = 0
            max_delta_m = 0
            for idx in range(state.phase_compositions.shape[0]):
                for j in range(state.phase_compositions.shape[1]):
                    delta_m[j] += state.phase_amt[idx] * state.phase_compositions[idx, j] - old_state.phase_amt[idx] * old_state.phase_compositions[idx, j]
            delta_energy = np.abs(np.dot(old_state.chemical_potentials, np.abs(delta_m)))
            print('delta_m', np.array(delta_m))
            print('delta_energy', delta_energy)
            if delta_energy == 0:
                    delta_energy = 1e-10
            if state.mass_residual < 1e-2:
                step_size = min(1, 1./delta_energy)
            else:
                step_size = 1./10
            break
            #if (np.max(np.abs(delta_m)) > 0.1) and (iteration > 0):
            #    state = deepcopy(old_state)
            #    step_size = min(step_size/10, 1./delta_energy)
            #    print('backtracking')
            #    continue
            #else:
            #    step_size = max(min(1./10, 1./delta_energy), 1./1000)
            #    break
        #else:
        #    raise ValueError('Backtracking line search failed')
        #    #delta_composition = np.dot(state.delta_ms.T, state.phase_amt)
        #    #print(f'delta_composition {delta_composition}')
        #    #print(f'chempot_diff {np.array(state.chempot_diff)}')
        #    #print(f'step_size {step_size}')
        #    # Steps are allowed as long as we are seeing decay in at least one key set of variables
        #    sufficient_step_taken = np.max(np.abs(delta_composition)) < 0.95 * np.max(np.abs(old_delta_composition))
        #    sufficient_step_taken |= np.max(np.abs(state.chempot_diff)) < 0.95 * np.max(np.abs(old_state.chempot_diff))
        #    # When steps get really small, we are near convergence, and there is little need for backtracking
        #    tiny_step = (np.max(np.abs(state.chempot_diff)) < 1e-4)
        #    if True:
        #        step_size = min(2*step_size, 1./10)
        #        break
        #    elif tiny_step:
        #        break
        #    else:
        #        step_size /= 10
        #        state = old_state
        #else:
        #    raise ValueError('Backtracking line search failed')

        # Consolidate duplicate phases and remove unstable phases
        compsets_to_remove = set()
        for idx in range(len(state.compsets)):
            compset = state.compsets[idx]
            if idx in compsets_to_remove:
                continue
            if state.phase_amt[idx] < 1e-10:
                compsets_to_remove.add(idx)
                continue
            for idx2 in range(len(compsets)):
                compset2 = compsets[idx2]
                if idx == idx2:
                    continue
                if idx2 in spec.fixed_stable_compset_indices:
                    continue
                if compset.phase_record.phase_name != compset2.phase_record.phase_name:
                    continue
                if idx2 in compsets_to_remove:
                    continue
                compset_distance = np.max(np.abs(np.array(state.phase_compositions[idx]) - np.array(state.phase_compositions[idx2])))
                if compset_distance < 1e-4:
                    compsets_to_remove.add(idx2)
                    if idx not in spec.fixed_stable_compset_indices:
                        state.phase_amt[idx] += state.phase_amt[idx2]
                    state.phase_amt[idx2] = 0
        new_free_stable_compset_indices = np.array(sorted(set(state.free_stable_compset_indices) - set(compsets_to_remove)), dtype=np.int32)
        if len(new_free_stable_compset_indices) == 0:
            # Do not allow all phases to leave the system
            for phase_idx in state.free_stable_compset_indices:
                state.phase_amt[phase_idx] = 1
            state.chemical_potentials[:] = 0
            # Force some chemical potentials to adopt their fixed values
            for cp_idx in range(spec.fixed_chemical_potential_indices.shape[0]):
                comp_idx = spec.fixed_chemical_potential_indices[cp_idx]
                state.chemical_potentials[comp_idx] = spec.initial_chemical_potentials[comp_idx]
        else:
            state.free_stable_compset_indices = new_free_stable_compset_indices
        print('new_chemical_potentials', np.array(state.chemical_potentials))
        for dof_idx in range(state.phase_amt.shape[0]):
            if state.phase_amt[dof_idx] < 0.0:
                state.phase_amt[dof_idx] = 0

        # Only include chemical potential difference if chemical potential conditions were enabled
        # XXX: This really should be a condition defined in terms of delta_m, because chempot_diff is only necessary
        # because mass_residual is no longer driving convergence for partially/fully open systems
        if spec.fixed_chemical_potential_indices.shape[0] > 0:
            chempot_diff = np.max(np.abs(np.array(state.chemical_potentials)/np.array(chemical_potentials) - 1))
        else:
            chempot_diff = 0.0
        largest_moles_change = np.max(np.abs(state.delta_ms))
        chemical_potentials = state.chemical_potentials
        print('new_phase_amt', np.array(state.phase_amt))
        print('statevars', state.dof[0][:spec.num_statevars])
        print('free_stable_compset_indices', np.array(state.free_stable_compset_indices))

        print('new_chemical_potentials', np.array(chemical_potentials))
        #print('new_phase_amt', np.array(phase_amt))
        #print('times_compset_removed', np.array(times_compset_removed))
        # Wait for mass balance to be satisfied before changing phases
        # Phases that "want" to be removed will keep having their phase_amt set to zero, so mass balance is unaffected
        print(f'mass_residual {state.mass_residual} largest_internal_cons_max_residual {state.largest_internal_cons_max_residual}')
        print(f'largest_internal_dof_change {state.largest_internal_dof_change}')
        system_is_feasible = (state.mass_residual < 1e-8) and (state.largest_internal_cons_max_residual < 1e-9) and \
                             (chempot_diff < 1e-12) and (state.iteration > 5) and (largest_moles_change < 1e-11) and (phase_change_counter == 0)
        if system_is_feasible:
            converged = True
            new_free_stable_compset_indices = np.array([i for i in range(state.phase_amt.shape[0])
                                                        if (state.phase_amt[i] > MIN_SITE_FRACTION) and \
                                                        (i not in suspended_compsets)], dtype=np.int32)
            if set(state.free_stable_compset_indices) == set(new_free_stable_compset_indices):
                converged = True
            state.free_stable_compset_indices = new_free_stable_compset_indices
            # Check driving forces for metastable phases
            # This needs to be done per mole of atoms, not per formula unit, since we compare phases to each other
            driving_forces = np.zeros(len(state.compsets))
            phase_energies_per_mole_atoms = np.zeros((len(state.compsets), 1))
            phase_amounts_per_mole_atoms = np.zeros((len(state.compsets), spec.num_components, 1))
            for idx in range(len(state.compsets)):
                compset = state.compsets[idx]
                x = state.dof[idx]
                for comp_idx in range(spec.num_components):
                    compset.phase_record.mass_obj(phase_amounts_per_mole_atoms[idx, comp_idx, :], x, comp_idx)
                compset.phase_record.obj(phase_energies_per_mole_atoms[idx, :], x)
                driving_forces[idx] =  np.dot(chemical_potentials, phase_amounts_per_mole_atoms[idx, :, 0]) - phase_energies_per_mole_atoms[idx, 0]
            print(f'driving_forces {driving_forces}')
            print(f'phase_energies_per_mole_atoms {phase_energies_per_mole_atoms}')
            converged, new_free_stable_compset_indices = \
                check_convergence_and_change_phases(state.phase_amt, state.free_stable_compset_indices, metastable_phase_iterations,
                                                    times_compset_removed, driving_forces,
                                                    state.largest_internal_dof_change, state.largest_phase_amt_change,
                                                    state.largest_statevar_change, iteration > 3)
            # Force some amount of newly stable phases
            for idx in new_free_stable_compset_indices:
                if state.phase_amt[idx] < 1e-10:
                    state.phase_amt[idx] = 1e-10
            # Force unstable phase amounts to zero
            for idx in range(state.phase_amt.shape[0]):
                if state.phase_amt[idx] < 1e-10:
                    state.phase_amt[idx] = 0
            if converged:
                converged = True
                break
            phase_change_counter = 5
            state.free_stable_compset_indices = np.array(new_free_stable_compset_indices, dtype=np.int32)

        for idx in range(len(state.compsets)):
            if idx in state.free_stable_compset_indices:
                metastable_phase_iterations[idx] = 0
                stable_phase_iterations[idx] += 1
            else:
                metastable_phase_iterations[idx] += 1
                stable_phase_iterations[idx] = 0
        if phase_change_counter > 0:
            phase_change_counter -= 1
    #if not converged:
    #    raise ValueError('Not converged')
    # Convert moles of formula units to phase fractions
    phase_amt = np.array(state.phase_amt) * np.sum(state.phase_compositions, axis=1)

    x = state.dof[0]
    for cs_dof in state.dof[1:]:
        x = np.r_[x, cs_dof[num_statevars:]]
    x = np.r_[x, phase_amt]
    return converged, x, np.array(chemical_potentials)
