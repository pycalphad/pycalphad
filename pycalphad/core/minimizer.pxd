cdef class SystemState:
    cdef list compsets
    cdef list cs_states
    cdef object dof
    cdef int iteration, num_statevars, iterations_since_last_phase_change
    cdef int[::1] metastable_phase_iterations
    cdef int[::1] times_compset_removed
    cdef double mass_residual, largest_chemical_potential_difference
    cdef double[::1] phase_amt, chemical_potentials, previous_chemical_potentials, delta_statevars
    cdef double[:, ::1] phase_compositions, delta_ms
    cdef double[1] largest_statevar_change, largest_phase_amt_change, largest_y_change
    cdef int[::1] free_stable_compset_indices
    cdef double system_amount
    cdef double[::1] mole_fractions
    cdef double[::1] _driving_forces
    cdef double[:, ::1] _phase_energies_per_mole_atoms
    cdef double[:, :, ::1] _phase_amounts_per_mole_atoms
    cdef void recompute(self, SystemSpecification spec)
    cdef double[::1] driving_forces(self)

cdef class SystemSpecification:
    cdef int num_statevars, num_components, max_num_free_stable_phases
    cdef double prescribed_system_amount
    cdef double[::1] initial_chemical_potentials, prescribed_elemental_amounts
    cdef int[::1] prescribed_element_indices
    cdef int[::1] free_chemical_potential_indices, free_statevar_indices
    cdef int[::1] fixed_chemical_potential_indices, fixed_statevar_indices, fixed_stable_compset_indices
    cpdef bint check_convergence(self, SystemState state, bint phases_changed)
    cpdef bint pre_solve_hook(self, SystemState state)
    cpdef bint post_solve_hook(self, SystemState state)
    cpdef bint run_loop(self, SystemState state, int max_iterations)
    cpdef SystemState get_new_state(self, list compsets)