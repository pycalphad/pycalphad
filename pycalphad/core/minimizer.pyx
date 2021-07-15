cimport cython
import numpy as np
cimport numpy as np
from pycalphad.core.composition_set cimport CompositionSet
from pycalphad.core.constants import MIN_SITE_FRACTION
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
                                        double[::1] phase_amt, int idx):
    cdef int free_variable_column_offset = 0
    cdef int num_statevars = c_statevars.shape[1]
    cdef int chempot_idx, compset_idx, statevar_idx, i, j
    # 2a. This component row: free chemical potentials
    for i in range(free_chemical_potential_indices.shape[0]):
        chempot_idx = free_chemical_potential_indices[i]
        for j in range(c_component.shape[1]):
            out_row[free_variable_column_offset + i] += \
                (phase_amt[idx]/current_system_amount) * mass_jac[component_idx, num_statevars+j] * c_component[chempot_idx, j]
        for j in range(c_component.shape[1]):
            out_row[free_variable_column_offset + i] += \
                (phase_amt[idx]/current_system_amount) * (-system_mole_fractions[component_idx] * moles_normalization_grad[num_statevars+j]) * c_component[chempot_idx, j]
    free_variable_column_offset += free_chemical_potential_indices.shape[0]
    # 2a. This component row: free stable composition sets
    for i in range(free_stable_compset_indices.shape[0]):
        compset_idx = free_stable_compset_indices[i]
        # Only fill this out if the current idx is equal to a free composition set
        if compset_idx == idx:
            out_row[free_variable_column_offset + i] = \
                (1./current_system_amount)*(masses[component_idx, 0] - system_mole_fractions[component_idx] * moles_normalization)
    free_variable_column_offset += free_stable_compset_indices.shape[0]
    # 2a. This component row: free state variables
    for i in range(free_statevar_indices.shape[0]):
        statevar_idx = free_statevar_indices[i]
        for j in range(c_statevars.shape[0]):
            out_row[free_variable_column_offset + i] += \
                (phase_amt[idx]/current_system_amount) * mass_jac[component_idx, num_statevars+j] * c_statevars[j, statevar_idx]
        for j in range(c_statevars.shape[0]):
            out_row[free_variable_column_offset + i] += \
                (phase_amt[idx]/current_system_amount) * (-system_mole_fractions[component_idx] * moles_normalization_grad[num_statevars+j]) * c_statevars[j, statevar_idx]
    # 3.
    for j in range(c_G.shape[0]):
        out_rhs[0] += -(phase_amt[idx]/current_system_amount) * \
            mass_jac[component_idx, num_statevars+j] * c_G[j]
    for j in range(c_G.shape[0]):
        out_rhs[0] += -(phase_amt[idx]/current_system_amount) * \
            (-system_mole_fractions[component_idx] * moles_normalization_grad[num_statevars+j]) * c_G[j]
    # 4. Subtract fixed chemical potentials from phase RHS
    for i in range(fixed_chemical_potential_indices.shape[0]):
        chempot_idx = fixed_chemical_potential_indices[i]
        # 5. Subtract fixed chemical potentials from fixed component RHS
        for j in range(c_component.shape[1]):
            out_rhs[0] -= (phase_amt[idx]/current_system_amount) * chemical_potentials[
                chempot_idx] * mass_jac[component_idx, num_statevars+j] * c_component[chempot_idx, j]
        for j in range(c_component.shape[1]):
            out_rhs[0] -= (phase_amt[idx]/current_system_amount) * chemical_potentials[
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
    cdef int fixed_component_idx, comp_idx, system_amount_index, sv_idx
    cdef CompositionSet compset
    cdef CompsetState csst
    cdef int num_components = state.chemical_potentials.shape[0]
    cdef int num_stable_phases = state.free_stable_compset_indices.shape[0]
    cdef int num_fixed_phases = spec.fixed_stable_compset_indices.shape[0]
    cdef int num_fixed_components = spec.prescribed_elemental_amounts.shape[0]

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
        # 2. Contribute to the row of all fixed components (fixed mole fraction)
        component_row_offset = num_stable_phases + num_fixed_phases
        for fixed_component_idx in range(num_fixed_components):
            component_idx = spec.prescribed_element_indices[fixed_component_idx]
            write_row_fixed_mole_fraction(equilibrium_matrix[component_row_offset + fixed_component_idx, :],
                                          &equilibrium_rhs[component_row_offset + fixed_component_idx],
                                          component_idx, spec.free_chemical_potential_indices,
                                          state.free_stable_compset_indices,
                                          spec.free_statevar_indices, spec.fixed_chemical_potential_indices,
                                          state.chemical_potentials,
                                          state.mole_fractions, state.system_amount, csst.mass_jac,
                                          csst.c_component, csst.c_statevars,
                                          csst.c_G, csst.masses, csst.moles_normalization,
                                          csst.moles_normalization_grad, state.phase_amt, idx)

        system_amount_index = component_row_offset + num_fixed_components
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
        # 2. Contribute to the row of all fixed components (fixed mole fraction)
        component_row_offset = num_stable_phases + num_fixed_phases
        for fixed_component_idx in range(num_fixed_components):
            component_idx = spec.prescribed_element_indices[fixed_component_idx]
            write_row_fixed_mole_fraction(equilibrium_matrix[component_row_offset + fixed_component_idx, :],
                                          &equilibrium_rhs[component_row_offset + fixed_component_idx],
                                          component_idx, spec.free_chemical_potential_indices,
                                          state.free_stable_compset_indices,
                                          spec.free_statevar_indices, spec.fixed_chemical_potential_indices,
                                          state.chemical_potentials,
                                          state.mole_fractions, state.system_amount, csst.mass_jac,
                                          csst.c_component, csst.c_statevars,
                                          csst.c_G, csst.masses, csst.moles_normalization,
                                          csst.moles_normalization_grad, state.phase_amt, idx)

        system_amount_index = component_row_offset + num_fixed_components
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
    system_amount_index = component_row_offset + num_fixed_components
    for fixed_component_idx in range(num_fixed_components):
        component_idx = spec.prescribed_element_indices[fixed_component_idx]
        component_residual = state.mole_fractions[component_idx] - spec.prescribed_elemental_amounts[fixed_component_idx]
        equilibrium_rhs[component_row_offset + fixed_component_idx] -= component_residual
    system_residual = state.system_amount - spec.prescribed_system_amount
    equilibrium_rhs[system_amount_index] -= system_residual


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
    for i in range(free_stable_compset_indices.shape[0]):
        compset_idx = free_stable_compset_indices[i]
        phase_amt_change = float(phase_amt[compset_idx])
        phase_amt[compset_idx] += equilibrium_soln[soln_index_offset + i]
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

cdef class CompsetState:
    cdef double[::1] x
    cdef double energy
    cdef double[::1] grad
    cdef double[:,::1] hess
    cdef double[:,::1] masses
    cdef double[:,::1] mass_jac
    cdef double[:,::1] phase_matrix
    cdef double[::1] phase_rhs
    cdef double[:,::1] full_e_matrix
    cdef double[::1] c_G
    cdef double[:, ::1] c_statevars
    cdef double[:, ::1] c_component
    cdef double moles_normalization
    cdef double[::1] internal_cons
    cdef double[::1] moles_normalization_grad
    cdef int[::1] fixed_phase_dof_indices
    cdef int[::1] ipiv
    def __init__(self, SystemSpecification spec, CompositionSet compset):
        self.x = np.zeros(spec.num_statevars + compset.phase_record.phase_dof)
        self.energy = 0
        self.grad = np.zeros(spec.num_statevars + compset.phase_record.phase_dof)
        self.hess = np.zeros((spec.num_statevars + compset.phase_record.phase_dof,
                             spec.num_statevars + compset.phase_record.phase_dof))
        self.masses = np.zeros((spec.num_components, 1))
        self.mass_jac = np.zeros((spec.num_components,
                                  spec.num_statevars + compset.phase_record.phase_dof))
        self.phase_matrix = np.zeros((compset.phase_record.phase_dof + compset.phase_record.num_internal_cons,
                                      compset.phase_record.phase_dof + compset.phase_record.num_internal_cons))
        self.phase_rhs = np.zeros(compset.phase_record.phase_dof + compset.phase_record.num_internal_cons)
        self.full_e_matrix = np.zeros((compset.phase_record.phase_dof + compset.phase_record.num_internal_cons,
                                       compset.phase_record.phase_dof + compset.phase_record.num_internal_cons))
        self.c_G = np.zeros(compset.phase_record.phase_dof)
        self.c_statevars = np.zeros((compset.phase_record.phase_dof, spec.num_statevars))
        self.c_component = np.zeros((spec.num_components, compset.phase_record.phase_dof))
        self.moles_normalization = 0.0
        self.internal_cons = np.zeros(compset.phase_record.num_internal_cons)
        self.moles_normalization_grad = np.zeros(spec.num_statevars+compset.phase_record.phase_dof)
        self.fixed_phase_dof_indices = np.array([], dtype=np.int32)
        self.ipiv = np.empty(self.phase_matrix.shape[0], dtype=np.int32)
    def __getstate__(self):
        return (np.array(self.x), self.energy, np.array(self.grad), np.array(self.hess),
                np.array(self.phase_matrix), np.array(self.phase_rhs), np.array(self.full_e_matrix),
                np.array(self.masses), np.array(self.mass_jac), np.array(self.c_G), np.array(self.c_statevars),
                np.array(self.c_component), self.moles_normalization, np.array(self.internal_cons), np.array(self.moles_normalization_grad),
                np.array(self.fixed_phase_dof_indices, dtype=np.int32), np.array(self.ipiv, dtype=np.int32))
    def __setstate__(self, state):
        (self.x, self.energy, self.grad, self.hess, self.phase_matrix, self.phase_rhs, self.full_e_matrix,
         self.masses, self.mass_jac, self.c_G, self.c_statevars,
         self.c_component, self.moles_normalization, self.internal_cons, self.moles_normalization_grad, self.fixed_phase_dof_indices,
         self.ipiv) = state


cdef class SystemState:
    cdef list compsets
    cdef list cs_states
    cdef object dof
    cdef int iteration, num_statevars
    cdef double mass_residual, largest_internal_cons_max_residual, largest_internal_dof_change
    cdef double[::1] phase_amt, chemical_potentials, chempot_diff, delta_statevars
    cdef double[:, ::1] phase_compositions, delta_ms
    cdef double[1] largest_statevar_change, largest_phase_amt_change
    cdef int[::1] free_stable_compset_indices
    cdef double system_amount
    cdef double[::1] mole_fractions
    def __init__(self, SystemSpecification spec, list compsets):
        cdef CompositionSet compset
        self.compsets = compsets
        self.cs_states = [CompsetState(spec, compset) for compset in compsets]
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
        self.delta_statevars = np.zeros(spec.num_statevars)
        self.phase_compositions = np.zeros((len(compsets), spec.num_components))
        self.free_stable_compset_indices = np.array(np.nonzero([((compset.fixed==False) and (compset.NP>0))
                                                                for compset in compsets])[0], dtype=np.int32)
        self.largest_statevar_change[0] = 0
        self.largest_phase_amt_change[0] = 0
        self.system_amount = 0
        self.mole_fractions = np.zeros(spec.num_components)

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
        return (self.compsets, self.cs_states, self.dof, self.iteration, self.mass_residual, self.largest_internal_cons_max_residual,
                self.largest_internal_dof_change, np.array(self.phase_amt), np.array(self.chemical_potentials),
                np.array(self.chempot_diff), np.array(self.delta_ms), np.array(self.phase_compositions),
                self.largest_statevar_change[0], self.largest_phase_amt_change[0],
                np.array(self.free_stable_compset_indices), self.system_amount, np.array(self.mole_fractions))
    def __setstate__(self, state):
        (self.compsets, self.cs_states, self.dof, self.iteration, self.mass_residual, self.largest_internal_cons_max_residual,
         self.largest_internal_dof_change, self.phase_amt, self.chemical_potentials,
         self.chempot_diff, self.delta_ms, self.phase_compositions, self.largest_statevar_change[0],
         self.largest_phase_amt_change[0], self.free_stable_compset_indices, self.system_amount, self.mole_fractions) = state

    cdef void recompute(self, SystemSpecification spec):
        cdef int num_components = spec.num_components
        cdef CompositionSet compset
        cdef CompsetState csst
        cdef double[::1] x
        cdef int idx, comp_idx, cons_idx, i, j, stable_idx, fixed_idx, component_idx, fixed_component_idx
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
        for comp_idx in range(self.mole_fractions.shape[0]):
            self.mole_fractions[comp_idx] /= self.system_amount

        self.mass_residual = 0.0
        for fixed_component_idx in range(spec.prescribed_elemental_amounts.shape[0]):
            component_idx = spec.prescribed_element_indices[fixed_component_idx]
            self.mass_residual += abs(self.mole_fractions[component_idx] - spec.prescribed_elemental_amounts[fixed_component_idx])

        for idx in range(len(self.compsets)):
            compset = self.compsets[idx]
            csst = self.cs_states[idx]
            # TODO: Use better dof storage
            # Calculate key phase quantities starting here
            x = self.dof[idx]
            csst.energy = 0
            csst.mass_jac[:,:] = 0
            # Compute phase matrix (LHS of Eq. 41, Sundman 2015)
            csst.phase_matrix[:,:] = 0
            csst.phase_rhs[:] = 0
            csst.internal_cons[:] = 0
            csst.full_e_matrix[:,:] = 0
            for i in range(csst.full_e_matrix.shape[0]):
                csst.full_e_matrix[i,i] = 1
            csst.hess[:,:] = 0
            csst.grad[:] = 0

            compset.phase_record.formulaobj(<double[:1]>&csst.energy, x)
            for comp_idx in range(num_components):
                compset.phase_record.formulamole_grad(csst.mass_jac[comp_idx, :], x, comp_idx)
            compset.phase_record.formulahess(csst.hess, x)
            compset.phase_record.formulagrad(csst.grad, x)
            compset.phase_record.internal_cons_func(csst.internal_cons, x)

            compute_phase_matrix(csst.phase_matrix, csst.hess, compset, spec.num_statevars, self.chemical_potentials, x,
                                 csst.fixed_phase_dof_indices)

            # Compute right-hand side of Eq. 41, Sundman 2015
            for i in range(compset.phase_record.phase_dof):
                csst.phase_rhs[i] = -csst.grad[spec.num_statevars+i]
                for sv_idx in range(spec.num_statevars):
                    csst.phase_rhs[i] -= csst.hess[spec.num_statevars + i, sv_idx] * self.delta_statevars[sv_idx]
                for comp_idx in range(num_components):
                    csst.phase_rhs[i] += self.chemical_potentials[comp_idx] * csst.mass_jac[comp_idx, spec.num_statevars + i]

            for cons_idx in range(compset.phase_record.num_internal_cons):
                csst.phase_rhs[compset.phase_record.phase_dof + cons_idx] = -csst.internal_cons[cons_idx]

            invert_matrix(&csst.phase_matrix[0,0], csst.phase_matrix.shape[0], &csst.full_e_matrix[0,0], &csst.ipiv[0])

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


cpdef take_step(SystemSpecification spec, SystemState state, double step_size):
    cdef double largest_internal_cons_max_residual = 0
    cdef double largest_internal_dof_change = 0
    cdef double internal_cons_max_residual, minimum_step_size
    cdef double[::1] cons_tmp
    cdef double[::1] delta_statevars = np.zeros(spec.num_statevars)
    cdef double[::1] delta_y
    cdef double[::1,:] equilibrium_matrix  # Fortran ordering required by call into lapack
    cdef double[::1] equilibrium_rhs, equilibrium_soln, old_chemical_potentials, new_y, x
    cdef CompositionSet compset
    cdef CompsetState csst
    cdef bint exceeded_bounds
    cdef int i, comp_idx, cp_idx, idx, num_stable_phases, num_fixed_phases, num_fixed_components, num_free_variables

    # STEP 1: Solve the equilibrium matrix (chemical potentials, corrections to phase amounts and state variables)
    state.largest_internal_cons_max_residual = largest_internal_cons_max_residual
    state.largest_internal_dof_change = largest_internal_dof_change
    state.recompute(spec)

    num_stable_phases = state.free_stable_compset_indices.shape[0]
    num_fixed_phases = spec.fixed_stable_compset_indices.shape[0]
    num_fixed_components = len(spec.prescribed_elemental_amounts)
    num_free_variables = spec.free_chemical_potential_indices.shape[0] + num_stable_phases + \
                         spec.free_statevar_indices.shape[0]

    equilibrium_matrix = np.zeros((num_stable_phases + num_fixed_phases + num_fixed_components + 1, num_free_variables), order='F')
    equilibrium_soln = np.zeros(num_stable_phases + num_fixed_phases + num_fixed_components + 1)
    if (num_stable_phases + num_fixed_phases + num_fixed_components + 1) != num_free_variables:
        raise ValueError('Conditions do not obey Gibbs Phase Rule')

    equilibrium_matrix[:,:] = 0
    equilibrium_soln[:] = 0
    fill_equilibrium_system(equilibrium_matrix, equilibrium_soln, spec, state)

    lstsq(&equilibrium_matrix[0,0], equilibrium_matrix.shape[0], equilibrium_matrix.shape[1],
          &equilibrium_soln[0], -1)
    old_chemical_potentials = np.array(state.chemical_potentials)

    # STEP 2: Advance the system state
    # Extract chemical potentials and update phase amounts
    extract_equilibrium_solution(state.chemical_potentials, state.phase_amt, delta_statevars,
                                 spec.free_chemical_potential_indices, spec.free_statevar_indices,
                                 state.free_stable_compset_indices, equilibrium_soln,
                                 state.largest_statevar_change, state.largest_phase_amt_change, state.dof)

    # Force some chemical potentials to adopt their fixed values
    for cp_idx in range(spec.fixed_chemical_potential_indices.shape[0]):
        comp_idx = spec.fixed_chemical_potential_indices[cp_idx]
        state.chemical_potentials[comp_idx] = spec.initial_chemical_potentials[comp_idx]
    state.chempot_diff = np.array(state.chemical_potentials) - old_chemical_potentials

    # Update phase internal degrees of freedom
    for idx, compset in enumerate(state.compsets):
        # TODO: Use better dof storage
        x = state.dof[idx]
        csst = state.cs_states[idx]
        internal_cons_max_residual = 0
        cons_tmp = np.zeros(compset.phase_record.num_internal_cons)
        compset.phase_record.internal_cons_func(cons_tmp, x)
        for cons_idx in range(compset.phase_record.num_internal_cons):
            internal_cons_max_residual = max(internal_cons_max_residual, abs(cons_tmp[cons_idx]))

        # Construct delta_y from Eq. 43 in Sundman 2015
        # TODO: needs charge balance contribution
        delta_y = np.zeros(compset.phase_record.phase_dof)
        for i in range(delta_y.shape[0]):
            delta_y[i] += csst.c_G[i]
            for sv_idx in range(delta_statevars.shape[0]):
                delta_y[i] += csst.c_statevars[i, sv_idx] * delta_statevars[sv_idx]
            for cp_idx in range(state.chemical_potentials.shape[0]):
                delta_y[i] += csst.c_component[cp_idx, i] * state.chemical_potentials[cp_idx]

        largest_internal_cons_max_residual = max(largest_internal_cons_max_residual, internal_cons_max_residual)
        new_y = np.array(x)
        minimum_step_size = 1e-20 * step_size
        while step_size >= minimum_step_size:
            exceeded_bounds = False
            for i in range(spec.num_statevars, new_y.shape[0]):
                new_y[i] = x[i] + step_size * delta_y[i - spec.num_statevars]
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

        for i in range(spec.num_statevars, new_y.shape[0]):
            largest_internal_dof_change = max(largest_internal_dof_change, abs(new_y[i] - x[i]))
        x[:] = new_y

    # Update state variables
    for idx in range(len(state.compsets)):
        x = state.dof[idx]
        for sv_idx in range(delta_statevars.shape[0]):
            x[sv_idx] += delta_statevars[sv_idx]
        # We need real state variable bounds support


cdef void prune_phases(SystemSpecification spec, SystemState state):
    """Remove phases that have become unstable (phase amount <= 0) and consolidate composition sets in an artificial misicbility gap.

    Updates the state in place.
    """
    cdef int idx, idx2, cp_idx, comp_idx, dof_idx, phase_idx
    cdef CompositionSet compset, compset2

    compsets_to_remove = set()
    for idx in range(len(state.compsets)):
        compset = state.compsets[idx]
        if compset.fixed:
            continue
        if idx in compsets_to_remove:
            continue
        # Remove unstable phases
        if state.phase_amt[idx] < 1e-10:
            compsets_to_remove.add(idx)
            continue
        for idx2 in range(len(state.compsets)):  # TODO: used to be compsets instead of state.compsets, is this valid?
            compset2 = state.compsets[idx2]  # TODO: used to be compsets instead of state.compsets, is this valid?
            if idx == idx2:
                continue
            if idx2 in spec.fixed_stable_compset_indices:
                continue
            if compset.phase_record.phase_name != compset2.phase_record.phase_name:
                continue
            if idx2 in compsets_to_remove:
                continue
            # Detect if these compsets describe the same internal configuration inside a miscibility gap
            compset_distances = np.abs(np.array(state.phase_compositions[idx]) - np.array(state.phase_compositions[idx2]))
            if np.all(compset_distances < 1e-4):
                compsets_to_remove.add(idx2)
                if idx not in spec.fixed_stable_compset_indices:
                    # ensure that the consolidated phase is stable
                    state.phase_amt[idx] = max(state.phase_amt[idx] + state.phase_amt[idx2], 1e-8)
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
    for dof_idx in range(state.phase_amt.shape[0]):
        if state.phase_amt[dof_idx] < 0.0:
            state.phase_amt[dof_idx] = 0


cdef bint change_phases(SystemSpecification spec, SystemState state, metastable_phase_iterations, times_compset_removed, driving_forces, can_add_phases):
    cdef int idx
    phase_amt = state.phase_amt
    current_free_stable_compset_indices = state.free_stable_compset_indices
    # Only add phases with positive driving force which have been metastable for at least 5 iterations, which have been removed fewer than 4 times
    if can_add_phases:
        newly_metastable_compsets = set(np.nonzero((np.array(metastable_phase_iterations) < 5))[0]) - \
                                    set(current_free_stable_compset_indices)
        add_criteria = np.logical_and(np.array(driving_forces) > 1e-5, np.array(times_compset_removed) < 4)
        compsets_to_add = set((np.nonzero(add_criteria)[0])) - newly_metastable_compsets
    else:
        compsets_to_add = set()
    compsets_to_remove = set(np.nonzero(np.array(phase_amt) < 1e-9)[0])
    new_free_stable_compset_indices = np.array(sorted((set(current_free_stable_compset_indices) - compsets_to_remove)
                                                      | compsets_to_add
                                                      ),
                                               dtype=np.int32)
    removed_compset_indices = set(current_free_stable_compset_indices) - set(new_free_stable_compset_indices)
    for idx in removed_compset_indices:
        times_compset_removed[idx] += 1
    if set(current_free_stable_compset_indices) == set(new_free_stable_compset_indices):
        # feasible system, and no phases to add or remove
        phases_changed = False
    else:
        phases_changed = True
    state.free_stable_compset_indices = new_free_stable_compset_indices
    for idx in range(len(state.compsets)):
        if idx in new_free_stable_compset_indices:
            # Force some amount of newly stable phases
            if state.phase_amt[idx] < 1e-10:
                state.phase_amt[idx] = 1e-10
        # Force unstable phase amounts to zero
        else:
            state.phase_amt[idx] = 0
    return phases_changed


cpdef find_solution(list compsets, int num_statevars, int num_components,
                    double prescribed_system_amount, double[::1] initial_chemical_potentials,
                    int[::1] free_chemical_potential_indices, int[::1] fixed_chemical_potential_indices,
                    int[::1] prescribed_element_indices, double[::1] prescribed_elemental_amounts,
                    int[::1] free_statevar_indices, int[::1] fixed_statevar_indices):
    cdef int iteration, idx, idx2, comp_idx, phase_idx, i
    cdef int num_stable_phases, num_fixed_components, num_free_variables
    cdef CompositionSet compset, compset2
    cdef double mass_residual = 1e-30
    cdef double delta_energy, allowed_mass_residual
    cdef double[::1] x, new_y, delta_y
    cdef double[::1] chemical_potentials = np.zeros(num_components)
    cdef double[::1] previous_chemical_potentials = np.empty(num_components)
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
    cdef double[::1] previous_phase_amt = np.empty((state.phase_amt.shape[0],))
    cdef double[:, ::1] previous_phase_compositions = np.empty((state.phase_compositions.shape[0], state.phase_compositions.shape[1]))


    if spec.prescribed_elemental_amounts.shape[0] > 0:
        allowed_mass_residual = min(1e-8, np.min(spec.prescribed_elemental_amounts)/10)
        # Also adjust mass residual if we are near the edge of composition space
        allowed_mass_residual = min(allowed_mass_residual, (1-np.sum(spec.prescribed_elemental_amounts))/10)
    else:
        allowed_mass_residual = 1e-8
    state.mass_residual = 1e10
    phase_change_counter = 5
    step_size = 1.0
    for iteration in range(1000):
        state.iteration = iteration
        if (state.mass_residual > 10) and (np.any(np.abs(state.chemical_potentials) > 1.0e10)):
            state.chemical_potentials[:] = spec.initial_chemical_potentials

        previous_chemical_potentials[:] = state.chemical_potentials[:]
        previous_phase_amt[:] = state.phase_amt[:]
        previous_phase_compositions[:, :] = state.phase_compositions[:, :]

        take_step(spec, state, step_size)

        if ((state.mass_residual > 1e-2) and (np.any(np.asarray(state.chempot_diff) > 1.0))) or (iteration == 0):
            # When mass residual is not satisfied, do not allow phases to leave the system
            # However, if the chemical potentials are changing very little, phases may leave the system
            for j in range(state.phase_amt.shape[0]):
                if state.phase_amt[j] < 0:
                    state.phase_amt[j] = 1e-8

        prune_phases(spec, state) # Consolidate duplicate phases and remove unstable phases

        # Only include chemical potential difference if chemical potential conditions were enabled
        # XXX: This really should be a condition defined in terms of delta_m, because chempot_diff is only necessary
        # because mass_residual is no longer driving convergence for partially/fully open systems
        if spec.fixed_chemical_potential_indices.shape[0] > 0:
            chempot_diff = np.max(np.abs(np.array(state.chemical_potentials)/np.array(chemical_potentials) - 1))
        else:
            chempot_diff = 0.0
        chemical_potentials = state.chemical_potentials
        # Wait for mass balance to be satisfied before changing phases
        # Phases that "want" to be removed will keep having their phase_amt set to zero, so mass balance is unaffected
        system_is_feasible = (state.mass_residual < allowed_mass_residual) and (state.largest_internal_cons_max_residual < 1e-9) and \
                             np.all(chempot_diff < 1e-12) and (state.iteration > 5) and np.all(np.abs(state.delta_ms) < 1e-9) and (phase_change_counter == 0)
        if system_is_feasible:
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
            phases_were_changed = change_phases(spec, state, metastable_phase_iterations, times_compset_removed, driving_forces, iteration > 3)
            if not phases_were_changed:
                converged = True
                break
            phase_change_counter = 5

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
