cimport cython
import numpy as np
cimport numpy as np
from pycalphad.core.composition_set cimport CompositionSet
from pycalphad.core.constants import MIN_SITE_FRACTION
cimport scipy.linalg.cython_lapack as cython_lapack
from libc.stdlib cimport malloc, free
from libc.math cimport isnan

@cython.boundscheck(False)
cdef void lstsq(double *A, int M, int N, double* x, double rcond) nogil:
    # Note: This function will destroy input matrix A
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
    cdef bint isfinite = True

    for i in range(M*N):
        if isnan(A[i]):
            isfinite = False

    if not isfinite:
        for i in range(N):
            x[i] = 0
    else:
        cython_lapack.dgelsd(&M, &N, &NRHS, A, &N, x, &M, singular_values, &rcond, &rank,
                            work, &lwork, &iwork, &info)
    free(singular_values)
    free(work)
    if info != 0:
        for i in range(N):
            x[i] = -1e19

cpdef void lstsq_check_infeasible(double[:,::] A, double[::] b, double[::] out_x):
    # Note: This function will destroy input matrix A
    A_copy = np.copy(A)
    b_copy = np.copy(b)
    lstsq(&A[0,0], A.shape[0], A.shape[1],
        &out_x[0], 1e-16)
    residual = np.sum(np.square(np.dot(A_copy, out_x) - b_copy))
    if residual > 1e-6:
        # lstsq solution is spurious; throw it away
        out_x[:] = np.nan

@cython.boundscheck(False)
cdef void invert_matrix(double *A, int N, int* ipiv) nogil:
    "A will be overwritten."
    cdef int info = 0
    cdef int i
    cdef double* work = <double*>malloc(N * sizeof(double))
    cdef bint isfinite = True

    for i in range(N**2):
        if isnan(A[i]):
            isfinite = False

    if not isfinite:
        for i in range(N**2):
            A[i] = 0
    else:
        cython_lapack.dgetrf(&N, &N, A, &N, ipiv, &info)
        cython_lapack.dgetri(&N, A, &N, ipiv, work, &N, &info)

    free(work)
    if info != 0:
        for i in range(N**2):
            A[i] = -1e19

@cython.boundscheck(False)
cdef void compute_phase_matrix(double[:,::1] phase_matrix, double[:,::1] hess,
                               double[:, ::1] cons_jac_tmp, double[:, ::1] phase_local_jac_tmp,
                               CompositionSet compset, int num_statevars, double[::1] chemical_potentials,
                               double[::1] phase_dof) nogil:
    "Compute the LHS of Eq. 41, Sundman 2015."
    cdef int comp_idx, i, j
    cdef int num_components = chemical_potentials.shape[0]
    compset.phase_record.internal_cons_jac(cons_jac_tmp, phase_dof)
    if compset.num_phase_local_conditions > 0:
        compset.phase_record.phase_local_cons_jac(phase_local_jac_tmp, phase_dof, compset.phase_local_cons_jac)

    for i in range(compset.phase_record.phase_dof):
        for j in range(compset.phase_record.phase_dof):
            phase_matrix[i, j] = hess[num_statevars+i, num_statevars+j]

    for i in range(compset.phase_record.num_internal_cons):
        for j in range(compset.phase_record.phase_dof):
            phase_matrix[compset.phase_record.phase_dof+i, j] = cons_jac_tmp[i, num_statevars+j]
            phase_matrix[j, compset.phase_record.phase_dof+i] = cons_jac_tmp[i, num_statevars+j]

    for i in range(compset.num_phase_local_conditions):
        for j in range(compset.phase_record.phase_dof):
            phase_matrix[compset.phase_record.phase_dof+compset.phase_record.num_internal_cons+i, j] = phase_local_jac_tmp[i, num_statevars+j]
            phase_matrix[j, compset.phase_record.phase_dof+compset.phase_record.num_internal_cons+i] = phase_local_jac_tmp[i, num_statevars+j]


cdef void write_row_stable_phase(double[:] out_row, double* out_rhs, int[::1] free_chemical_potential_indices,
                                 int[::1] free_stable_compset_indices, int[::1] free_statevar_indices,
                                 int[::1] fixed_chemical_potential_indices, double[::1] chemical_potentials,
                                 double[:, ::1] masses, double[::1] grad, double energy):
    # 1a. This phase row: free chemical potentials
    cdef int free_variable_column_offset = 0
    cdef int chempot_idx, statevar_idx, i
    for i in range(free_chemical_potential_indices.shape[0]):
        chempot_idx = free_chemical_potential_indices[i]
        out_row[free_variable_column_offset + i] = masses[chempot_idx, 0]
    free_variable_column_offset += free_chemical_potential_indices.shape[0]
    # 1a. This phase row: free stable composition sets = zero contribution
    free_variable_column_offset += free_stable_compset_indices.shape[0]
    # 1a. This phase row: free state variables
    for i in range(free_statevar_indices.shape[0]):
        statevar_idx = free_statevar_indices[i]
        out_row[free_variable_column_offset + i] = -grad[statevar_idx]
    out_rhs[0] = energy
    # 4. Subtract fixed chemical potentials from each phase RHS
    for i in range(fixed_chemical_potential_indices.shape[0]):
        chempot_idx = fixed_chemical_potential_indices[i]
        out_rhs[0] -= masses[chempot_idx, 0] * chemical_potentials[chempot_idx]

cdef void write_row_fixed_mole_fraction(double[:] out_row, double* out_rhs, int component_idx,
                                        int[::1] free_chemical_potential_indices, int[::1] free_stable_compset_indices,
                                        int[::1] free_statevar_indices, int[::1] fixed_chemical_potential_indices,
                                        double[::1] chemical_potentials,
                                        double [::1] system_mole_fractions, double current_system_amount,
                                        double[:, ::1] mass_jac, double[:, ::1] c_component,
                                        double[:, ::1] c_statevars, double[::1] c_G, double[:, ::1] masses,
                                        double moles_normalization, double[::1] moles_normalization_grad,
                                        double[::1] phase_amt, int idx, double prefactor):
    if prefactor == 0.0:
        return
    cdef int free_variable_column_offset = 0
    cdef int num_statevars = c_statevars.shape[1]
    cdef int chempot_idx, compset_idx, statevar_idx, i, j
    # 2a. This component row: free chemical potentials
    for i in range(free_chemical_potential_indices.shape[0]):
        chempot_idx = free_chemical_potential_indices[i]
        for j in range(c_component.shape[1]):
            out_row[free_variable_column_offset + i] += prefactor * \
                (phase_amt[idx]/current_system_amount) * mass_jac[component_idx, num_statevars+j] * c_component[chempot_idx, j]
        for j in range(c_component.shape[1]):
            out_row[free_variable_column_offset + i] += prefactor * \
                (phase_amt[idx]/current_system_amount) * (-system_mole_fractions[component_idx] * moles_normalization_grad[num_statevars+j]) * c_component[chempot_idx, j]
    free_variable_column_offset += free_chemical_potential_indices.shape[0]
    # 2a. This component row: free stable composition sets
    for i in range(free_stable_compset_indices.shape[0]):
        compset_idx = free_stable_compset_indices[i]
        # Only fill this out if the current idx is equal to a free composition set
        if compset_idx == idx:
            out_row[free_variable_column_offset + i] += prefactor * \
                (1./current_system_amount)*(masses[component_idx, 0] - system_mole_fractions[component_idx] * moles_normalization)
    free_variable_column_offset += free_stable_compset_indices.shape[0]
    # 2a. This component row: free state variables
    for i in range(free_statevar_indices.shape[0]):
        statevar_idx = free_statevar_indices[i]
        for j in range(c_statevars.shape[0]):
            out_row[free_variable_column_offset + i] += prefactor * \
                (phase_amt[idx]/current_system_amount) * mass_jac[component_idx, num_statevars+j] * c_statevars[j, statevar_idx]
        for j in range(c_statevars.shape[0]):
            out_row[free_variable_column_offset + i] += prefactor * \
                (phase_amt[idx]/current_system_amount) * (-system_mole_fractions[component_idx] * moles_normalization_grad[num_statevars+j]) * c_statevars[j, statevar_idx]
    # 3.
    for j in range(c_G.shape[0]):
        out_rhs[0] += -prefactor * (phase_amt[idx]/current_system_amount) * \
            mass_jac[component_idx, num_statevars+j] * c_G[j]
    for j in range(c_G.shape[0]):
        out_rhs[0] += -prefactor * (phase_amt[idx]/current_system_amount) * \
            (-system_mole_fractions[component_idx] * moles_normalization_grad[num_statevars+j]) * c_G[j]
    # 4. Subtract fixed chemical potentials from phase RHS
    for i in range(fixed_chemical_potential_indices.shape[0]):
        chempot_idx = fixed_chemical_potential_indices[i]
        # 5. Subtract fixed chemical potentials from fixed component RHS
        for j in range(c_component.shape[1]):
            out_rhs[0] -= prefactor * (phase_amt[idx]/current_system_amount) * chemical_potentials[
                chempot_idx] * mass_jac[component_idx, num_statevars+j] * c_component[chempot_idx, j]
        for j in range(c_component.shape[1]):
            out_rhs[0] -= prefactor * (phase_amt[idx]/current_system_amount) * chemical_potentials[
                chempot_idx] * (-system_mole_fractions[component_idx] * moles_normalization_grad[num_statevars+j]) * c_component[chempot_idx, j]

cdef void write_row_fixed_mole_amount(double[:] out_row, double* out_rhs, int component_idx,
                                      int[::1] free_chemical_potential_indices, int[::1] free_stable_compset_indices,
                                      int[::1] free_statevar_indices, int[::1] fixed_chemical_potential_indices,
                                      double[::1] chemical_potentials,
                                      double[:, ::1] mass_jac, double[:, ::1] c_component,
                                      double[:, ::1] c_statevars, double[::1] c_G, double[:, ::1] masses,
                                      double[::1] phase_amt, int idx):
    cdef int free_variable_column_offset = 0
    cdef int num_statevars = c_statevars.shape[1]
    cdef int i, j, chempot_idx, compset_idx, statevar_idx
    # 2a. This component row: free chemical potentials
    for i in range(free_chemical_potential_indices.shape[0]):
        chempot_idx = free_chemical_potential_indices[i]
        for j in range(c_component.shape[1]):
            out_row[free_variable_column_offset + i] += \
                phase_amt[idx] * mass_jac[component_idx, num_statevars+j] * c_component[chempot_idx, j]
    free_variable_column_offset += free_chemical_potential_indices.shape[0]
    # 2a. This component row: free stable composition sets
    for i in range(free_stable_compset_indices.shape[0]):
        compset_idx = free_stable_compset_indices[i]
        # Only fill this out if the current idx is equal to a free composition set
        if compset_idx == idx:
            out_row[free_variable_column_offset + i] += masses[component_idx, 0]
    free_variable_column_offset += free_stable_compset_indices.shape[0]
    # 2a. This component row: free state variables
    for i in range(free_statevar_indices.shape[0]):
        statevar_idx = free_statevar_indices[i]
        for j in range(c_statevars.shape[0]):
            out_row[free_variable_column_offset + i] += \
                phase_amt[idx] * mass_jac[component_idx, num_statevars+j] * c_statevars[j, statevar_idx]
    # 3.
    for j in range(c_G.shape[0]):
        out_rhs[0] += -phase_amt[idx] * mass_jac[component_idx, num_statevars+j] * c_G[j]
    # 4. Subtract fixed chemical potentials from each phase RHS
    for i in range(fixed_chemical_potential_indices.shape[0]):
        chempot_idx = fixed_chemical_potential_indices[i]
        # 6. Subtract fixed chemical potentials from the N=1 row
        for j in range(c_component.shape[1]):
            out_rhs[0] -= phase_amt[idx] * chemical_potentials[
                chempot_idx] * mass_jac[component_idx, num_statevars+j] * c_component[chempot_idx, j]


cdef void fill_equilibrium_system(double[::1,:] equilibrium_matrix, double[::1] equilibrium_rhs,
                                  SystemSpecification spec, SystemState state):
    cdef int stable_idx, idx, component_row_offset, component_idx, fixed_idx, free_idx
    cdef int fixed_component_idx, comp_idx, system_amount_index, fixed_molefrac_cond_idx
    cdef CompositionSet compset
    cdef CompsetState csst
    cdef int num_components = state.chemical_potentials.shape[0]
    cdef int num_stable_phases = state.free_stable_compset_indices.shape[0]
    cdef int num_fixed_phases = spec.fixed_stable_compset_indices.shape[0]
    cdef int num_fixed_mole_fraction_conditions = spec.prescribed_mole_fraction_rhs.shape[0]
    cdef double prefactor

    for stable_idx in range(state.free_stable_compset_indices.shape[0]):
        idx = state.free_stable_compset_indices[stable_idx]
        compset = state.compsets[idx]
        csst = state.cs_states[idx]

        write_row_stable_phase(equilibrium_matrix[stable_idx, :], &equilibrium_rhs[stable_idx], spec.free_chemical_potential_indices,
                               state.free_stable_compset_indices, spec.free_statevar_indices, spec.fixed_chemical_potential_indices,
                               state.chemical_potentials, csst.masses, csst.grad, csst.energy)

    # Handle phases which are fixed to be stable at some amount
    # Example shown in Eq. 60, Sundman et al 2015
    for fixed_idx in range(spec.fixed_stable_compset_indices.shape[0]):
        idx = spec.fixed_stable_compset_indices[fixed_idx]
        compset = state.compsets[idx]
        csst = state.cs_states[idx]
        write_row_stable_phase(equilibrium_matrix[num_stable_phases + fixed_idx, :],
                               &equilibrium_rhs[num_stable_phases + fixed_idx], spec.free_chemical_potential_indices,
                               state.free_stable_compset_indices, spec.free_statevar_indices, spec.fixed_chemical_potential_indices,
                               state.chemical_potentials, csst.masses, csst.grad, csst.energy)

    for stable_idx in range(state.free_stable_compset_indices.shape[0]):
        idx = state.free_stable_compset_indices[stable_idx]
        compset = state.compsets[idx]
        csst = state.cs_states[idx]
        # 2. Contribute to the row of all fixed mole fraction conditions
        component_row_offset = num_stable_phases + num_fixed_phases
        for fixed_molefrac_cond_idx in range(num_fixed_mole_fraction_conditions):
            for component_idx in range(spec.prescribed_mole_fraction_coefficients.shape[1]):
                prefactor = spec.prescribed_mole_fraction_coefficients[fixed_molefrac_cond_idx, component_idx]
                write_row_fixed_mole_fraction(equilibrium_matrix[component_row_offset + fixed_molefrac_cond_idx, :],
                                            &equilibrium_rhs[component_row_offset + fixed_molefrac_cond_idx],
                                            component_idx, spec.free_chemical_potential_indices,
                                            state.free_stable_compset_indices,
                                            spec.free_statevar_indices, spec.fixed_chemical_potential_indices,
                                            state.chemical_potentials,
                                            state.mole_fractions, state.system_amount, csst.mass_jac,
                                            csst.c_component, csst.c_statevars,
                                            csst.c_G, csst.masses, csst.moles_normalization,
                                            csst.moles_normalization_grad, state.phase_amt, idx, prefactor)

        system_amount_index = component_row_offset + num_fixed_mole_fraction_conditions
        # 2X. Also handle the N=1 row
        for component_idx in range(num_components):
            write_row_fixed_mole_amount(equilibrium_matrix[system_amount_index, :],
                                        &equilibrium_rhs[system_amount_index], component_idx,
                                        spec.free_chemical_potential_indices, state.free_stable_compset_indices,
                                        spec.free_statevar_indices, spec.fixed_chemical_potential_indices,
                                        state.chemical_potentials, csst.mass_jac, csst.c_component,
                                        csst.c_statevars, csst.c_G, csst.masses,
                                        state.phase_amt, idx)

    for fixed_idx in range(spec.fixed_stable_compset_indices.shape[0]):
        idx = spec.fixed_stable_compset_indices[fixed_idx]
        compset = state.compsets[idx]
        csst = state.cs_states[idx]
        # 2. Contribute to the row of all fixed mole fraction conditions
        component_row_offset = num_stable_phases + num_fixed_phases
        for fixed_molefrac_cond_idx in range(num_fixed_mole_fraction_conditions):
            for component_idx in range(spec.prescribed_mole_fraction_coefficients.shape[1]):
                prefactor = spec.prescribed_mole_fraction_coefficients[fixed_molefrac_cond_idx, component_idx]
                write_row_fixed_mole_fraction(equilibrium_matrix[component_row_offset + fixed_molefrac_cond_idx, :],
                                            &equilibrium_rhs[component_row_offset + fixed_molefrac_cond_idx],
                                            component_idx, spec.free_chemical_potential_indices,
                                            state.free_stable_compset_indices,
                                            spec.free_statevar_indices, spec.fixed_chemical_potential_indices,
                                            state.chemical_potentials,
                                            state.mole_fractions, state.system_amount, csst.mass_jac,
                                            csst.c_component, csst.c_statevars,
                                            csst.c_G, csst.masses, csst.moles_normalization,
                                            csst.moles_normalization_grad, state.phase_amt, idx, prefactor)

        system_amount_index = component_row_offset + num_fixed_mole_fraction_conditions
        # 2X. Also handle the N=1 row
        for component_idx in range(num_components):
            write_row_fixed_mole_amount(equilibrium_matrix[system_amount_index, :],
                                        &equilibrium_rhs[system_amount_index], component_idx,
                                        spec.free_chemical_potential_indices, state.free_stable_compset_indices,
                                        spec.free_statevar_indices, spec.fixed_chemical_potential_indices,
                                        state.chemical_potentials, csst.mass_jac, csst.c_component,
                                        csst.c_statevars, csst.c_G, csst.masses,
                                        state.phase_amt, idx)


    # Add mass residual to fixed component row RHS, plus N=1 row
    component_row_offset = num_stable_phases + num_fixed_phases
    system_amount_index = component_row_offset + num_fixed_mole_fraction_conditions
    for fixed_molefrac_cond_idx in range(num_fixed_mole_fraction_conditions):
        component_residual = np.dot(spec.prescribed_mole_fraction_coefficients[fixed_molefrac_cond_idx, :], state.mole_fractions) - spec.prescribed_mole_fraction_rhs[fixed_molefrac_cond_idx]
        equilibrium_rhs[component_row_offset + fixed_molefrac_cond_idx] -= component_residual
    system_residual = state.system_amount - spec.prescribed_system_amount
    equilibrium_rhs[system_amount_index] -= system_residual


cdef class SystemSpecification:
    def __init__(self, int num_statevars, int num_components, double prescribed_system_amount,
                   double[::1] initial_chemical_potentials, double[:, ::1] prescribed_mole_fraction_coefficients,
                   double[::1] prescribed_mole_fraction_rhs, int[::1] free_chemical_potential_indices,
                   int[::1] free_statevar_indices, int[::1] fixed_chemical_potential_indices,
                   int[::1] fixed_statevar_indices, int[::1] fixed_stable_compset_indices):
        self.num_statevars = num_statevars
        self.num_components = num_components
        self.prescribed_system_amount = prescribed_system_amount
        self.initial_chemical_potentials = initial_chemical_potentials
        self.prescribed_mole_fraction_coefficients = prescribed_mole_fraction_coefficients
        self.prescribed_mole_fraction_rhs = prescribed_mole_fraction_rhs
        self.free_chemical_potential_indices = free_chemical_potential_indices
        self.free_statevar_indices = free_statevar_indices
        self.fixed_chemical_potential_indices = fixed_chemical_potential_indices
        self.fixed_statevar_indices = fixed_statevar_indices
        self.fixed_stable_compset_indices = fixed_stable_compset_indices
        self.max_num_free_stable_phases = num_components + len(free_statevar_indices) - len(fixed_stable_compset_indices)

        # Assuming the prescribed_mole_fraction_rhs doesn't change, this is
        # constant and we can keep extra computation (especially calls into
        # NumPy out of the run loop)
        if self.prescribed_mole_fraction_rhs.shape[0] > 0:
            # With linear combinations of conditions, RHS can now be exactly zero
            # This means the smallest allowed mass residual needs to be limited to prevent instability
            self.ALLOWED_MASS_RESIDUAL = max(1e-12, min(1e-8, np.min(np.abs(self.prescribed_mole_fraction_rhs))/10.0))
            # Also adjust mass residual if we are near the edge of composition space
            self.ALLOWED_MASS_RESIDUAL = min(self.ALLOWED_MASS_RESIDUAL, (1-np.sum(np.abs(self.prescribed_mole_fraction_rhs)))/10.0)
        else:
            self.ALLOWED_MASS_RESIDUAL = 1e-8

    def __getstate__(self):
        return (self.num_statevars, self.num_components, self.prescribed_system_amount,
                np.array(self.initial_chemical_potentials), np.array(self.prescribed_mole_fraction_coefficients),
                np.array(self.prescribed_mole_fraction_rhs), np.array(self.free_chemical_potential_indices),
                np.array(self.free_statevar_indices), np.array(self.fixed_chemical_potential_indices),
                np.array(self.fixed_statevar_indices), np.array(self.fixed_stable_compset_indices))
    def __setstate__(self, state):
        self.__init__(*state)

    cpdef bint check_convergence(self, SystemState state):
        # convergence criteria
        cdef double ALLOWED_DELTA_Y = 5e-09
        cdef double ALLOWED_DELTA_PHASE_AMT = 1e-10
        cdef double ALLOWED_DELTA_STATEVAR = 1e-5  # changes defined as percent change
        cdef bint solution_is_feasible = (
            (state.largest_phase_amt_change[0] < ALLOWED_DELTA_PHASE_AMT) and
            (state.largest_y_change[0] < ALLOWED_DELTA_Y) and
            (state.largest_statevar_change[0] < ALLOWED_DELTA_STATEVAR) and
            (state.mass_residual < self.ALLOWED_MASS_RESIDUAL)
        )
        if solution_is_feasible and (state.iterations_since_last_phase_change >= 5):
            return True
        else:
            return False

    cpdef bint pre_solve_hook(self, SystemState state):
        return True

    cpdef bint post_solve_hook(self, SystemState state):
        return True

    cpdef bint run_loop(self, SystemState state, int max_iterations):
        cdef double step_size = 1.0
        cdef bint converged = False
        cdef bint phases_changed = False
        cdef size_t iteration
        for iteration in range(max_iterations):
            state.iteration = iteration
            if not self.pre_solve_hook(state):
                break
            eq_soln = solve_state(self, state)
            if not self.post_solve_hook(state):
                break
            phases_changed = remove_and_consolidate_phases(self, state)
            converged = self.check_convergence(state)
            if converged:
                phases_changed = phases_changed or change_phases(self, state)
                if phases_changed:
                    # TODO: this preserves old logic about phase changes, but should we
                    # reset the counter `if phases_changed and not converged` -
                    # i.e. phases were changed by remove_and_consolidate_phases?
                    state.iterations_since_last_phase_change = 0
                else:
                    break
            state.iterations_since_last_phase_change += 1
            state.increment_phase_metastability_counters()
            if not phases_changed:
                advance_state(self, state, eq_soln, step_size)
        if state.free_stable_compset_indices.shape[0] > self.max_num_free_stable_phases:
            # Gibbs phase rule violation in solution
            converged = False
        return converged

    cpdef SystemState get_new_state(self, list compsets):
        return SystemState(self, compsets)

cdef class CompsetState:
    cdef double[::1] x
    cdef double energy
    cdef double[::1] grad
    cdef double[:,::1] hess
    cdef double[:,::1] masses
    cdef double[:,::1] mass_jac
    cdef double[:,::1] phase_matrix
    cdef double[:,::1] full_e_matrix
    cdef double[::1] c_G
    cdef double[:, ::1] c_statevars
    cdef double[:, ::1] c_component
    cdef double[::1] delta_y
    cdef double moles_normalization
    cdef double[::1] internal_cons
    cdef double[::1] moles_normalization_grad
    cdef int[::1] fixed_phase_dof_indices
    cdef int[::1] ipiv
    cdef double[:, ::1] cons_jac_tmp
    cdef double[:, ::1] phase_local_jac_tmp

    def __init__(self, SystemSpecification spec, CompositionSet compset):
        self.x = np.zeros(spec.num_statevars + compset.phase_record.phase_dof)
        self.energy = 0
        self.grad = np.zeros(spec.num_statevars + compset.phase_record.phase_dof)
        self.hess = np.zeros((spec.num_statevars + compset.phase_record.phase_dof,
                             spec.num_statevars + compset.phase_record.phase_dof))
        self.masses = np.zeros((spec.num_components, 1))
        self.mass_jac = np.zeros((spec.num_components,
                                  spec.num_statevars + compset.phase_record.phase_dof))
        self.phase_matrix = np.zeros((compset.phase_record.phase_dof + compset.phase_record.num_internal_cons + compset.num_phase_local_conditions,
                                      compset.phase_record.phase_dof + compset.phase_record.num_internal_cons + compset.num_phase_local_conditions))
        self.full_e_matrix = np.zeros((compset.phase_record.phase_dof + compset.phase_record.num_internal_cons + compset.num_phase_local_conditions,
                                       compset.phase_record.phase_dof + compset.phase_record.num_internal_cons + compset.num_phase_local_conditions))
        self.c_G = np.zeros(compset.phase_record.phase_dof)
        self.c_statevars = np.zeros((compset.phase_record.phase_dof, spec.num_statevars))
        self.c_component = np.zeros((spec.num_components, compset.phase_record.phase_dof))
        self.moles_normalization = 0.0
        self.internal_cons = np.zeros(compset.phase_record.num_internal_cons)
        self.moles_normalization_grad = np.zeros(spec.num_statevars+compset.phase_record.phase_dof)
        self.fixed_phase_dof_indices = np.array([], dtype=np.int32)
        self.ipiv = np.empty(self.phase_matrix.shape[0], dtype=np.int32)
        self.delta_y = np.zeros(compset.phase_record.phase_dof)
        self.cons_jac_tmp = np.zeros((compset.phase_record.num_internal_cons, spec.num_statevars + compset.phase_record.phase_dof))
        self.phase_local_jac_tmp = np.zeros((compset.num_phase_local_conditions, spec.num_statevars + compset.phase_record.phase_dof))

    def __getstate__(self):
        return (np.array(self.x), self.energy, np.array(self.grad), np.array(self.hess),
                np.array(self.phase_matrix), np.array(self.full_e_matrix),
                np.array(self.masses), np.array(self.mass_jac), np.array(self.c_G), np.array(self.c_statevars),
                np.array(self.c_component), self.moles_normalization, np.array(self.internal_cons), np.array(self.moles_normalization_grad),
                np.array(self.fixed_phase_dof_indices, dtype=np.int32), np.array(self.ipiv, dtype=np.int32))
    def __setstate__(self, state):
        (self.x, self.energy, self.grad, self.hess, self.phase_matrix, self.full_e_matrix,
         self.masses, self.mass_jac, self.c_G, self.c_statevars,
         self.c_component, self.moles_normalization, self.internal_cons, self.moles_normalization_grad, self.fixed_phase_dof_indices,
         self.ipiv) = state


cdef class SystemState:
    def __init__(self, SystemSpecification spec, list compsets):
        cdef CompositionSet compset
        cdef int idx, comp_idx
        self.compsets = compsets
        cdef double phase_comp_sum
        for compset in compsets:
            compset.fixed = False
        for idx in spec.fixed_stable_compset_indices:
            compset = compsets[idx]
            compset.fixed = True
        self.cs_states = [CompsetState(spec, compset) for compset in compsets]
        self.dof = [np.array(compset.dof) for compset in compsets]
        self.iteration = 0
        self.iterations_since_last_phase_change = 0
        self.metastable_phase_iterations = np.zeros(len(compsets), dtype=np.int32)
        self.times_compset_removed = np.zeros(len(compsets), dtype=np.int32)
        self.mass_residual = 1e10
        # Phase fractions need to be converted to moles of formula
        self.phase_amt = np.array([compset.NP for compset in compsets])
        self.chemical_potentials = np.zeros(spec.num_components)
        self.previous_chemical_potentials = np.zeros(spec.num_components)
        self.largest_chemical_potential_difference = -np.inf
        self.delta_ms = np.zeros((len(compsets), spec.num_components))
        self.delta_statevars = np.zeros(spec.num_statevars)
        self.phase_compositions = np.zeros((len(compsets), spec.num_components))
        self.free_stable_compset_indices = np.array(np.nonzero([((compset.fixed==False) and (compset.NP>0))
                                                                for compset in compsets])[0], dtype=np.int32)
        self.largest_statevar_change[0] = 0
        self.largest_phase_amt_change[0] = 0
        self.largest_y_change[0] = 0
        self.system_amount = 0
        self.mole_fractions = np.zeros(spec.num_components)
        self._driving_forces = np.zeros(len(compsets))
        self._phase_energies_per_mole_atoms = np.zeros((len(compsets), 1))
        self._phase_amounts_per_mole_atoms = np.zeros((len(compsets), spec.num_components, 1))

        cdef double[:, ::1] masses_tmp = np.zeros((spec.num_components, 1))
        for idx in range(self.phase_amt.shape[0]):
            compset = self.compsets[idx]
            x = self.dof[idx]
            phase_comp_sum = 0.0
            for comp_idx in range(spec.num_components):
                compset.phase_record.formulamole_obj(masses_tmp[comp_idx, :], x, comp_idx)
                self.phase_compositions[idx, comp_idx] = masses_tmp[comp_idx, 0]
                phase_comp_sum += self.phase_compositions[idx, comp_idx]
                masses_tmp[:,:] = 0
            # Convert phase fractions to formula units
            self.phase_amt[idx] /= phase_comp_sum

    def __getstate__(self):
        return (self.compsets, self.cs_states, self.dof, self.iteration, self.iterations_since_last_phase_change,
                self.metastable_phase_iterations, self.times_compset_removed, self.mass_residual,
                np.array(self.phase_amt), np.array(self.chemical_potentials), np.array(self.previous_chemical_potentials),
                np.array(self.delta_ms), np.array(self.phase_compositions), self.largest_chemical_potential_difference,
                self.largest_statevar_change[0], self.largest_phase_amt_change[0], self.largest_y_change[0],
                np.array(self.free_stable_compset_indices), self.system_amount, np.array(self.mole_fractions))
    def __setstate__(self, state):
        (self.compsets, self.cs_states, self.dof, self.iteration, self.iterations_since_last_phase_change,
         self.metastable_phase_iterations, self.times_compset_removed, self.mass_residual,
         self.phase_amt, self.chemical_potentials, self.previous_chemical_potentials,
         self.delta_ms, self.phase_compositions, self.largest_chemical_potential_difference, self.largest_statevar_change[0],
         self.largest_phase_amt_change[0], self.largest_y_change[0], self.free_stable_compset_indices, self.system_amount, self.mole_fractions) = state

    @cython.boundscheck(False)
    cpdef void recompute(self, SystemSpecification spec):
        cdef int num_components = spec.num_components
        cdef CompositionSet compset
        cdef CompsetState csst
        cdef double[::1] x
        cdef int idx, comp_idx, cons_idx, i, j, stable_idx, fixed_idx, fixed_molefrac_cond_idx, num_phase_dof
        cdef double mu_c_sum
        cdef double phase_comp_sum
        self.mole_fractions[:] = 0
        self.delta_ms[:, :] = 0
        self.system_amount = 0
        # Compute normalized global quantities
        for idx in range(len(self.compsets)):
            x = self.dof[idx]
            compset = self.compsets[idx]
            csst = self.cs_states[idx]
            csst.masses[:,:] = 0
            for comp_idx in range(num_components):
                compset.phase_record.formulamole_obj(csst.masses[comp_idx, :], x, comp_idx)
                if self.phase_amt[idx] > 0:
                    self.mole_fractions[comp_idx] += self.phase_amt[idx] * csst.masses[comp_idx, 0]
                    self.system_amount += self.phase_amt[idx] * csst.masses[comp_idx, 0]
                self.phase_compositions[idx, comp_idx] = csst.masses[comp_idx, 0]
        if self.system_amount < 1e-10:
            # XXX: Trying to fix stochastic bug
            self.system_amount = 1e-10
        for comp_idx in range(self.mole_fractions.shape[0]):
            self.mole_fractions[comp_idx] /= self.system_amount

        self.mass_residual = 0.0
        for fixed_molefrac_cond_idx in range(spec.prescribed_mole_fraction_rhs.shape[0]):
            self.mass_residual += abs(np.dot(spec.prescribed_mole_fraction_coefficients[fixed_molefrac_cond_idx,:], self.mole_fractions) - spec.prescribed_mole_fraction_rhs[fixed_molefrac_cond_idx])

        for idx in range(len(self.compsets)):
            compset = self.compsets[idx]
            csst = self.cs_states[idx]
            # TODO: Use better dof storage
            # Calculate key phase quantities starting here
            x = self.dof[idx]
            phase_comp_sum = 0.0
            for comp_idx in range(num_components):
                phase_comp_sum += self.phase_compositions[idx, comp_idx]
            compset.update(x[spec.num_statevars:], self.phase_amt[idx] * phase_comp_sum, x[:spec.num_statevars])
            csst.energy = 0
            csst.mass_jac[:,:] = 0
            # Compute phase matrix (LHS of Eq. 41, Sundman 2015)
            csst.phase_matrix[:,:] = 0
            csst.internal_cons[:] = 0
            csst.hess[:,:] = 0
            csst.grad[:] = 0

            compset.phase_record.formulaobj(<double[:1]>&csst.energy, x)
            for comp_idx in range(num_components):
                compset.phase_record.formulamole_grad(csst.mass_jac[comp_idx, :], x, comp_idx)
            compset.phase_record.formulahess(csst.hess, x)
            compset.phase_record.formulagrad(csst.grad, x)
            compset.phase_record.internal_cons_func(csst.internal_cons, x)

            compute_phase_matrix(csst.phase_matrix, csst.hess, csst.cons_jac_tmp, csst.phase_local_jac_tmp, compset, spec.num_statevars, self.chemical_potentials, x)
            # Copy the phase matrix into the e matrix and invert the e matrix
            for i in range(csst.full_e_matrix.shape[0]):
                for j in range(csst.full_e_matrix.shape[1]):
                    csst.full_e_matrix[i,j] = csst.phase_matrix[i,j]
            invert_matrix(&csst.full_e_matrix[0,0], csst.full_e_matrix.shape[0], &csst.ipiv[0])

            num_phase_dof = compset.phase_record.phase_dof
            csst.c_G[:] = 0
            csst.c_statevars[:,:] = 0
            csst.c_component[:,:] = 0
            csst.moles_normalization = 0
            csst.moles_normalization_grad[:] = 0
            for i in range(num_phase_dof):
                for j in range(num_phase_dof):
                    csst.c_G[i] -= csst.full_e_matrix[i, j] * csst.grad[spec.num_statevars+j]
            for i in range(num_phase_dof):
                for j in range(num_phase_dof):
                    for statevar_idx in range(spec.num_statevars):
                        csst.c_statevars[i, statevar_idx] -= csst.full_e_matrix[i, j] * csst.hess[spec.num_statevars + j, statevar_idx]
            for comp_idx in range(num_components):
                for i in range(num_phase_dof):
                    for j in range(num_phase_dof):
                        csst.c_component[comp_idx, i] += csst.mass_jac[comp_idx, spec.num_statevars + j] * csst.full_e_matrix[i, j]
            for comp_idx in range(num_components):
                for i in range(num_phase_dof):
                    mu_c_sum = 0
                    for j in range(self.chemical_potentials.shape[0]):
                        mu_c_sum += csst.c_component[j, i] * self.chemical_potentials[j]
                    self.delta_ms[idx, comp_idx] += csst.mass_jac[comp_idx, spec.num_statevars + i] * (mu_c_sum + csst.c_G[i])
            for comp_idx in range(num_components):
                csst.moles_normalization += csst.masses[comp_idx, 0]
                for i in range(num_phase_dof+spec.num_statevars):
                    csst.moles_normalization_grad[i] += csst.mass_jac[comp_idx, i]

    cdef double[::1] driving_forces(self):
        cdef int idx, comp_idx
        cdef CompositionSet compset
        cdef double[::1] x
        cdef int num_components = self.chemical_potentials.shape[0]
        # This needs to be done per mole of atoms, not per formula unit, since we compare phases to each other
        self._driving_forces[:] = 0
        for idx in range(len(self.compsets)):
            compset = self.compsets[idx]
            x = self.dof[idx]
            for comp_idx in range(num_components):
                compset.phase_record.mass_obj(self._phase_amounts_per_mole_atoms[idx, comp_idx, :], x, comp_idx)
                self._driving_forces[idx] += self.chemical_potentials[comp_idx] * self._phase_amounts_per_mole_atoms[idx, comp_idx, 0]
            compset.phase_record.obj(self._phase_energies_per_mole_atoms[idx, :], x)
            self._driving_forces[idx] -= self._phase_energies_per_mole_atoms[idx, 0]
        return self._driving_forces

    cdef void increment_phase_metastability_counters(self):
        cdef int idx
        for idx in range(len(self.compsets)):
            if idx in self.free_stable_compset_indices or self.compsets[idx].fixed:
                self.metastable_phase_iterations[idx] = 0
            else:
                self.metastable_phase_iterations[idx] += 1

cpdef construct_equilibrium_system(SystemSpecification spec, SystemState state, int num_reserved_rows) except *:
    cdef double[::1,:] equilibrium_matrix  # Fortran ordering required by call into lapack
    cdef double[::1] equilibrium_soln
    cdef int num_stable_phases, num_fixed_phases, num_fixed_mole_fraction_conditions, num_free_variables

    num_stable_phases = state.free_stable_compset_indices.shape[0]
    num_fixed_phases = spec.fixed_stable_compset_indices.shape[0]
    num_fixed_mole_fraction_conditions = spec.prescribed_mole_fraction_rhs.shape[0]
    num_free_variables = spec.free_chemical_potential_indices.shape[0] + num_stable_phases + \
                         spec.free_statevar_indices.shape[0]

    equilibrium_matrix = np.zeros((num_stable_phases + num_fixed_phases + num_fixed_mole_fraction_conditions + num_reserved_rows + 1,
                                   num_free_variables), order='F')
    equilibrium_rhs = np.zeros(equilibrium_matrix.shape[0])
    if (equilibrium_matrix.shape[0] != equilibrium_matrix.shape[1]):
        raise ValueError('Conditions do not obey Gibbs Phase Rule')
    fill_equilibrium_system(equilibrium_matrix, equilibrium_rhs, spec, state)
    return np.asarray(equilibrium_matrix), np.asarray(equilibrium_rhs)

cpdef state_variable_differential(SystemSpecification spec, SystemState state, int target_statevar_index):
    # Sundman et al 2015, Eq. 74
    cdef double[::1,:] equilibrium_matrix  # Fortran ordering required by call into lapack
    cdef double[::1] equilibrium_soln, delta_chemical_potentials, delta_statevars, delta_phase_amounts
    cdef int[::1] orig_fixed_statevar_indices, orig_free_statevar_indices
    cdef int chempot_idx, statevar_idx, cs_idx, i

    orig_fixed_statevar_indices = np.array(spec.fixed_statevar_indices)
    orig_free_statevar_indices = np.array(spec.free_statevar_indices)
    delta_chemical_potentials = np.zeros(spec.num_components)
    delta_statevars = np.zeros(spec.num_statevars)
    delta_phase_amounts = np.zeros(len(state.compsets))
    spec.fixed_statevar_indices = np.setdiff1d(spec.fixed_statevar_indices, np.array(target_statevar_index))
    spec.free_statevar_indices = np.append(spec.free_statevar_indices, target_statevar_index).astype(np.int32)

    try:
        equilibrium_matrix, equilibrium_soln = construct_equilibrium_system(spec, state, 1)
        equilibrium_soln[:] = 0
        # target_statevar_index is the last column of the matrix, by construction
        equilibrium_matrix[-1, -1] = 1
        equilibrium_soln[-1] = 1
        lstsq_check_infeasible(equilibrium_matrix, equilibrium_soln, equilibrium_soln)
        for i in range(spec.free_chemical_potential_indices.shape[0]):
            chempot_idx = spec.free_chemical_potential_indices[i]
            delta_chemical_potentials[chempot_idx] = equilibrium_soln[i]
        for i in range(state.free_stable_compset_indices.shape[0]):
            cs_idx = state.free_stable_compset_indices[i]
            delta_phase_amounts[cs_idx] = equilibrium_soln[spec.free_chemical_potential_indices.shape[0] + i]
        for i in range(spec.free_statevar_indices.shape[0]):
            statevar_idx = spec.free_statevar_indices[i]
            delta_statevars[statevar_idx] = equilibrium_soln[spec.free_chemical_potential_indices.shape[0] + 
                                                             state.free_stable_compset_indices.shape[0] + i]
        return np.asarray(delta_chemical_potentials), np.asarray(delta_statevars), np.asarray(delta_phase_amounts)
    finally:
        spec.fixed_statevar_indices = orig_fixed_statevar_indices
        spec.free_statevar_indices = orig_free_statevar_indices

cpdef fixed_component_differential(SystemSpecification spec, SystemState state, int target_component_index):
    # Based on Sundman et al 2015, Eq. 74, with some modifications
    cdef double[::1,:] equilibrium_matrix  # Fortran ordering required by call into lapack
    cdef double[::1] equilibrium_soln, delta_chemical_potentials, delta_statevars, delta_phase_amounts
    cdef np.ndarray comparison_array = np.zeros(spec.prescribed_mole_fraction_coefficients.shape[1])
    comparison_array[target_component_index] = 1
    cdef int num_stable_phases = state.free_stable_compset_indices.shape[0]
    cdef int num_fixed_phases = spec.fixed_stable_compset_indices.shape[0]
    cdef int num_fixed_mole_fraction_conditions = spec.prescribed_mole_fraction_rhs.shape[0]
    cdef int chempot_idx, statevar_idx, cs_idx, i
    cdef bint component_was_fixed = False

    for i in range(spec.prescribed_mole_fraction_coefficients.shape[0]):
        if np.all(np.asarray(spec.prescribed_mole_fraction_coefficients[i]) == comparison_array):
            component_was_fixed = True
    if not component_was_fixed:
        raise ValueError('Target component was not fixed in the present calculation')

    delta_chemical_potentials = np.zeros(spec.num_components)
    delta_statevars = np.zeros(spec.num_statevars)
    delta_phase_amounts = np.zeros(len(state.compsets))

    equilibrium_matrix, equilibrium_soln = construct_equilibrium_system(spec, state, 0)
    equilibrium_soln[:] = 0

    # delta mole fractions must sum to zero; we have degrees of freedom to decide how to distribute
    # for now, redistribute evenly over all other fixed components
    for i in range(spec.prescribed_mole_fraction_coefficients.shape[0]):
        if np.all(np.asarray(spec.prescribed_mole_fraction_coefficients[i]) == comparison_array):
            equilibrium_soln[num_stable_phases + num_fixed_phases + i] = 1
        else:
            equilibrium_soln[num_stable_phases + num_fixed_phases + i] = -1/(num_fixed_mole_fraction_conditions)
    lstsq_check_infeasible(equilibrium_matrix, equilibrium_soln, equilibrium_soln)
    for i in range(spec.free_chemical_potential_indices.shape[0]):
        chempot_idx = spec.free_chemical_potential_indices[i]
        delta_chemical_potentials[chempot_idx] = equilibrium_soln[i]
    for i in range(state.free_stable_compset_indices.shape[0]):
        cs_idx = state.free_stable_compset_indices[i]
        delta_phase_amounts[cs_idx] = equilibrium_soln[spec.free_chemical_potential_indices.shape[0] + i]
    for i in range(spec.free_statevar_indices.shape[0]):
        statevar_idx = spec.free_statevar_indices[i]
        delta_statevars[statevar_idx] = equilibrium_soln[spec.free_chemical_potential_indices.shape[0] + 
                                                            state.free_stable_compset_indices.shape[0] + i]
    return np.asarray(delta_chemical_potentials), np.asarray(delta_statevars), np.asarray(delta_phase_amounts)

cpdef chemical_potential_differential(SystemSpecification spec, SystemState state, int target_component_index):
    # Sundman et al 2015, Eq. 74
    cdef double[::1,:] equilibrium_matrix  # Fortran ordering required by call into lapack
    cdef double[::1] equilibrium_soln, delta_chemical_potentials, delta_statevars, delta_phase_amounts
    cdef int[::1] orig_fixed_statevar_indices, orig_free_statevar_indices
    cdef int chempot_idx, statevar_idx, cs_idx, i
    cdef bint component_was_fixed = False

    for i in range(spec.fixed_chemical_potential_indices.shape[0]):
        if spec.fixed_chemical_potential_indices[i] == target_component_index:
            component_was_fixed = True
    if not component_was_fixed:
        raise ValueError('Target chemical potential was not fixed in the present calculation')

    # Release chemical potential condition
    orig_fixed_chemical_potential_indices = np.array(spec.fixed_chemical_potential_indices)
    orig_free_chemical_potential_indices = np.array(spec.free_chemical_potential_indices)
    delta_chemical_potentials = np.zeros(spec.num_components)
    delta_statevars = np.zeros(spec.num_statevars)
    delta_phase_amounts = np.zeros(len(state.compsets))
    spec.fixed_chemical_potential_indices = np.setdiff1d(spec.fixed_chemical_potential_indices, np.array(target_component_index))
    spec.free_chemical_potential_indices = np.append(spec.free_chemical_potential_indices, target_component_index).astype(np.int32)

    try:
        equilibrium_matrix, equilibrium_soln = construct_equilibrium_system(spec, state, 1)
        equilibrium_soln[:] = 0
        equilibrium_matrix[-1, target_component_index] = 1
        equilibrium_soln[-1] = 1
        lstsq_check_infeasible(equilibrium_matrix, equilibrium_soln, equilibrium_soln)
        for i in range(spec.free_chemical_potential_indices.shape[0]):
            chempot_idx = spec.free_chemical_potential_indices[i]
            delta_chemical_potentials[chempot_idx] = equilibrium_soln[i]
        for i in range(state.free_stable_compset_indices.shape[0]):
            cs_idx = state.free_stable_compset_indices[i]
            delta_phase_amounts[cs_idx] = equilibrium_soln[spec.free_chemical_potential_indices.shape[0] + i]
        for i in range(spec.free_statevar_indices.shape[0]):
            statevar_idx = spec.free_statevar_indices[i]
            delta_statevars[statevar_idx] = equilibrium_soln[spec.free_chemical_potential_indices.shape[0] + 
                                                             state.free_stable_compset_indices.shape[0] + i]
        return np.asarray(delta_chemical_potentials), np.asarray(delta_statevars), np.asarray(delta_phase_amounts)
    finally:
        spec.fixed_chemical_potential_indices = orig_fixed_chemical_potential_indices
        spec.free_chemical_potential_indices = orig_free_chemical_potential_indices

cpdef site_fraction_differential(CompsetState csst, double[::1] delta_chempots, double[::1] delta_statevars):
    # Sundman et al 2015, Eq. 78
    cdef double[::1] delta_y = np.zeros(csst.delta_y.shape[0])
    cdef int chempot_idx, statevar_idex

    for i in range(delta_y.shape[0]):
        for statevar_idx in range(delta_statevars.shape[0]):
            delta_y[i] += csst.c_statevars[i, statevar_idx] * delta_statevars[statevar_idx]
        for chempot_idx in range(delta_chempots.shape[0]):
            delta_y[i] += csst.c_component[chempot_idx, i] * delta_chempots[chempot_idx]
    return np.asarray(delta_y)

cpdef solve_state(SystemSpecification spec, SystemState state):
    cdef double[::1,:] equilibrium_matrix  # Fortran ordering required by call into lapack
    cdef double[::1] equilibrium_soln
    cdef int chempot_idx, comp_idx

    state.previous_chemical_potentials[:] = state.chemical_potentials[:]
    state.recompute(spec)

    equilibrium_matrix, equilibrium_soln = construct_equilibrium_system(spec, state, 0)

    lstsq(&equilibrium_matrix[0,0], equilibrium_matrix.shape[0], equilibrium_matrix.shape[1],
          &equilibrium_soln[0], 1e-16)

    # set the chemical potentials from the solution
    for i in range(spec.free_chemical_potential_indices.shape[0]):
        chempot_idx = spec.free_chemical_potential_indices[i]
        state.chemical_potentials[chempot_idx] = equilibrium_soln[i]

    # Force some chemical potentials to adopt their fixed values
    for chempot_idx in range(spec.fixed_chemical_potential_indices.shape[0]):
        comp_idx = spec.fixed_chemical_potential_indices[chempot_idx]
        state.chemical_potentials[comp_idx] = spec.initial_chemical_potentials[comp_idx]

    state.largest_chemical_potential_difference = -np.inf
    for comp_idx in range(spec.num_components):
        state.largest_chemical_potential_difference = max(state.largest_chemical_potential_difference, abs(state.chemical_potentials[comp_idx] - state.previous_chemical_potentials[comp_idx]))

    return equilibrium_soln


# TODO: should we store equilibrium_soln in the state(?)
cpdef advance_state(SystemSpecification spec, SystemState state, double[::1] equilibrium_soln, double step_size):
    # Apply linear corrections in phase amounts, state variables and site fractions
    cdef bint exceeded_bounds
    cdef double minimum_step_size, psc, phase_amt_step_size
    cdef int i, idx, cons_idx, compset_idx, statevar_idx, chempot_idx
    cdef int soln_index_offset = spec.free_chemical_potential_indices.shape[0]  # Chemical potentials handled after solving
    cdef double[::1] new_y, x
    cdef CompsetState csst

    cdef double MIN_PHASE_AMOUNT = 1e-16

    # 1. Step in phase amounts
    # Determine largest allowable step size such that the smallest phase amount is zero
    phase_amt_step_size = step_size
    for i in range(state.free_stable_compset_indices.shape[0]):
        compset_idx = state.free_stable_compset_indices[i]
        if state.phase_amt[compset_idx] + equilibrium_soln[soln_index_offset + i] < MIN_PHASE_AMOUNT:
            # Assuming:
            # 1. NP>0 (the phase would not be a free_stable_compset if not) and
            # 2. delta_NP<0 (must be true if assumption #1 is true and this condition is true)
            # The largest allowable step size satisfies the equation: (NP + step_size * delta_NP = MIN_PHASE_AMOUNT)
            if abs(equilibrium_soln[soln_index_offset + i]) > MIN_PHASE_AMOUNT:
                phase_amt_step_size = min(phase_amt_step_size, (MIN_PHASE_AMOUNT - state.phase_amt[compset_idx]) / equilibrium_soln[soln_index_offset + i])
    # Update the phase amounts using the largest allowable step size
    state.largest_phase_amt_change[0] = 0
    for i in range(state.free_stable_compset_indices.shape[0]):
        compset_idx = state.free_stable_compset_indices[i]
        state.phase_amt[compset_idx] += phase_amt_step_size * equilibrium_soln[soln_index_offset + i]
        state.largest_phase_amt_change[0] = max(state.largest_phase_amt_change[0], abs(phase_amt_step_size * equilibrium_soln[soln_index_offset + i]))
    soln_index_offset += state.free_stable_compset_indices.shape[0]

    # 2. Step in state variables
    state.largest_statevar_change[0] = 0
    state.delta_statevars[:] = 0
    for i in range(spec.free_statevar_indices.shape[0]):
        statevar_idx = spec.free_statevar_indices[i]
        state.delta_statevars[statevar_idx] = equilibrium_soln[soln_index_offset + i]
        if state.dof[0][statevar_idx] == 0:
            psc = np.inf
        else:
            psc = abs(state.delta_statevars[statevar_idx] / state.dof[0][statevar_idx])
        state.largest_statevar_change[0] = max(state.largest_statevar_change[0], psc)
    # Update state variables in the `x` array
    for idx in range(len(state.compsets)):
        x = state.dof[idx]
        for statevar_idx in range(state.delta_statevars.shape[0]):
            x[statevar_idx] += state.delta_statevars[statevar_idx]
        # We need real state variable bounds support

    # 3. Step in phase internal degrees of freedom
    for idx in range(len(state.compsets)):
        # TODO: Use better dof storage
        x = state.dof[idx]
        csst = state.cs_states[idx]

        # Construct delta_y from Eq. 43 in Sundman 2015
        csst.delta_y[:] = 0

        for i in range(csst.delta_y.shape[0]):
            csst.delta_y[i] += csst.c_G[i]
            for statevar_idx in range(state.delta_statevars.shape[0]):
                csst.delta_y[i] += csst.c_statevars[i, statevar_idx] * state.delta_statevars[statevar_idx]
            for chempot_idx in range(state.chemical_potentials.shape[0]):
                csst.delta_y[i] += csst.c_component[chempot_idx, i] * state.chemical_potentials[chempot_idx]
            for cons_idx in range(csst.internal_cons.shape[0]):
                csst.delta_y[i] -= csst.full_e_matrix[csst.delta_y.shape[0] + cons_idx, i] * csst.internal_cons[cons_idx]

        new_y = np.array(x)
        minimum_step_size = 1e-20 * step_size
        while step_size >= minimum_step_size:
            exceeded_bounds = False
            for i in range(spec.num_statevars, new_y.shape[0]):
                new_y[i] = x[i] + step_size * csst.delta_y[i - spec.num_statevars]
                if new_y[i] > 1:
                    if (new_y[i] - 1) > 1e-11:
                        # Allow some tolerance in the name of progress
                        exceeded_bounds = True
                    new_y[i] = 1
                elif new_y[i] < MIN_SITE_FRACTION:
                    if (MIN_SITE_FRACTION - new_y[i]) > 1e-11:
                        # Allow some tolerance in the name of progress
                        exceeded_bounds = True
                    # Reduce by two orders of magnitude, or MIN_SITE_FRACTION, whichever is larger
                    new_y[i] = max(x[i]/100, MIN_SITE_FRACTION)
            if exceeded_bounds:
                step_size *= 0.5
                continue
            break
        state.largest_y_change[0] = 0.0
        for i in range(spec.num_statevars, new_y.shape[0]):
            state.largest_y_change[0] = max(state.largest_y_change[0], abs(x[i] - new_y[i]))
        x[:] = new_y


cdef bint remove_and_consolidate_phases(SystemSpecification spec, SystemState state):
    """Remove phases that have become unstable (phase amount <= 0) and consolidate composition sets in an artificial misicbility gap.

    Updates the state in place.
    """
    cdef int i, j, idx, idx2, cp_idx, comp_idx, dof_idx, phase_idx
    cdef CompositionSet compset, compset2
    cdef bint phases_changed = False
    cdef double composition_difference
    cdef double COMPSET_CONSOLIDATE_DISTANCE = 1e-4

    compset_indices_to_remove = set()
    for i in range(len(state.free_stable_compset_indices)):
        idx = state.free_stable_compset_indices[i]
        compset = state.compsets[idx]
        if compset.fixed:
            continue
        if idx in compset_indices_to_remove:
            continue
        # Remove unstable phases
        if state.phase_amt[idx] < 1e-10:
            compset_indices_to_remove.add(idx)
            state.phase_amt[idx] = 0
            continue
        for j in range(len(state.free_stable_compset_indices)):
            idx2 = state.free_stable_compset_indices[j]
            compset2 = state.compsets[idx2]
            if idx == idx2:
                continue
            if compset2.fixed:
                continue
            if compset.phase_record.phase_name != compset2.phase_record.phase_name:
                continue
            if idx2 in compset_indices_to_remove:
                continue
            # Detect if these compsets describe the same internal configuration inside a miscibility gap
            compsets_should_be_consolidated = True
            # Detected based on composition, we may miss gaps that have nearly
            # the same composition, but different site fractions (such as ordering)
            for comp_idx in range(spec.num_components):
                composition_difference = abs(state.phase_compositions[idx, comp_idx] - state.phase_compositions[idx2, comp_idx])
                if composition_difference > COMPSET_CONSOLIDATE_DISTANCE:
                    compsets_should_be_consolidated = False
                    break
            if compsets_should_be_consolidated:
                compset_indices_to_remove.add(idx2)
                if idx not in spec.fixed_stable_compset_indices:
                    # ensure that the consolidated phase is stable
                    state.phase_amt[idx] = max(state.phase_amt[idx] + state.phase_amt[idx2], 1e-8)
                state.phase_amt[idx2] = 0
    if len(compset_indices_to_remove) > 0:
        if len(compset_indices_to_remove) - len(state.free_stable_compset_indices) == 0:
            # Do not allow all phases to leave the system
            for phase_idx in state.free_stable_compset_indices:
                state.phase_amt[phase_idx] = 1
            state.chemical_potentials[:] = 0
            # Force some chemical potentials to adopt their fixed values
            for cp_idx in range(spec.fixed_chemical_potential_indices.shape[0]):
                comp_idx = spec.fixed_chemical_potential_indices[cp_idx]
                state.chemical_potentials[comp_idx] = spec.initial_chemical_potentials[comp_idx]
        else:
            state.free_stable_compset_indices = np.array(sorted(set(state.free_stable_compset_indices) - compset_indices_to_remove), dtype=np.int32)
            phases_changed = True
    return phases_changed

cdef bint change_phases(SystemSpecification spec, SystemState state):
    cdef int idx, i, cs_idx, least_removed_cs_idx, smallest_df_cs_idx
    cdef double[::1] driving_forces = state.driving_forces()
    cdef double MIN_PHASE_AMOUNT = 1e-9
    cdef int MIN_REQUIRED_METASTABLE_PHASE_ITERATIONS_TO_ADD = 5
    cdef double MIN_DRIVING_FORCE_TO_ADD = 1e-5
    cdef int MAX_ALLOWED_TIMES_COMPSET_REMOVED = 4
    if state.free_stable_compset_indices.shape[0] > spec.max_num_free_stable_phases:
        # Gibbs phase rule is currently being violated
        # Try forcing phases with small amounts out of the equilibrium
        MIN_PHASE_AMOUNT = 1e-4
    phase_amt = state.phase_amt
    current_free_stable_compset_indices = state.free_stable_compset_indices
    compsets_to_remove = set()
    for i in range(current_free_stable_compset_indices.shape[0]):
        cs_idx = current_free_stable_compset_indices[i]
        if phase_amt[cs_idx] < MIN_PHASE_AMOUNT:
            compsets_to_remove.add(cs_idx)

    # Only add phases with positive driving force which have been metastable for at least 5 iterations, which have been removed fewer than 4 times
    compsets_to_add = set()
    for cs_idx in range(state.metastable_phase_iterations.shape[0]):
        should_add_compset = (
            (state.metastable_phase_iterations[cs_idx] >= MIN_REQUIRED_METASTABLE_PHASE_ITERATIONS_TO_ADD)
            and (driving_forces[cs_idx] > MIN_DRIVING_FORCE_TO_ADD)
            and (state.times_compset_removed[cs_idx] < MAX_ALLOWED_TIMES_COMPSET_REMOVED)
        )
        if should_add_compset:
            compsets_to_add.add(cs_idx)
    # Finally, remove all currently stable compsets as candidates
    compsets_to_add -= set(current_free_stable_compset_indices)
    max_allowed_to_add = spec.max_num_free_stable_phases + len(compsets_to_remove) - len(current_free_stable_compset_indices)
    # We must obey the Gibbs phase rule
    if len(compsets_to_add) > 0:
        if max_allowed_to_add < 1:
            # We are at the maximum number of allowed phases, yet there is still positive driving force
            # Destabilize one phase and add only one phase
            possible_phases_to_destabilize = sorted(set(current_free_stable_compset_indices) - compsets_to_add - compsets_to_remove)
            # Destabilize the one that has been removed the least
            least_removed_cs_idx = possible_phases_to_destabilize[0]
            for i in range(1, len(possible_phases_to_destabilize)):
                cs_idx = possible_phases_to_destabilize[i]
                if state.times_compset_removed[cs_idx] < state.times_compset_removed[least_removed_cs_idx]:
                    least_removed_cs_idx = cs_idx
            compsets_to_remove.add(least_removed_cs_idx)
            phase_amt[least_removed_cs_idx] = 0
        # Add the compset with least amount (but still positive) driving force
        possible_phases_to_add = sorted(compsets_to_add)
        smallest_df_cs_idx = possible_phases_to_add[0]
        for i in range(1, len(possible_phases_to_add)):
            cs_idx = possible_phases_to_add[i]
            if driving_forces[cs_idx] < driving_forces[smallest_df_cs_idx]:
                smallest_df_cs_idx = cs_idx
        compsets_to_add = {smallest_df_cs_idx}
    new_free_stable_compset_indices = np.array(sorted((set(current_free_stable_compset_indices) - compsets_to_remove)
                                                      | compsets_to_add
                                                      ),
                                               dtype=np.int32)
    removed_compset_indices = set(current_free_stable_compset_indices) - set(new_free_stable_compset_indices)
    for idx in removed_compset_indices:
        state.times_compset_removed[idx] += 1
    for idx in range(len(state.compsets)):
        if idx in new_free_stable_compset_indices:
            # Force some amount of newly stable phases
            if state.phase_amt[idx] < 1e-10:
                state.phase_amt[idx] = 1e-10
        # Force unstable phase amounts to zero
        else:
            state.phase_amt[idx] = 0
    state.free_stable_compset_indices = new_free_stable_compset_indices
    if set(current_free_stable_compset_indices) == set(new_free_stable_compset_indices):
        # feasible system, and no phases to add or remove
        phases_changed = False
    else:
        phases_changed = True
    return phases_changed
