cdef class Problem:
    cdef public int num_constraints
    cdef public object composition_sets
    cdef public object conditions
    cdef public object pure_elements
    cdef public object nonvacant_elements
    cdef public int num_phases
    cdef public int num_vars
    cdef public double temperature
    cdef public double pressure
    cdef public double[::1] xl
    cdef public double[::1] xu
    cdef public double[::1] x0
    cdef public double[::1] cl
    cdef public double[::1] cu
