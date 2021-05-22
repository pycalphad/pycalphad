# distutils: language = c++
cdef class Problem:
    cdef public int num_fixed_dof_constraints
    cdef public int num_internal_constraints
    cdef public int[::1] fixed_dof_indices
    cdef public int[::1] fixed_chempot_indices
    cdef public double[::1] fixed_chempot_values
    cdef public object composition_sets
    cdef public object conditions
    cdef public object pure_elements
    cdef public object nonvacant_elements
    cdef public int num_phases
    cdef public int num_vars
