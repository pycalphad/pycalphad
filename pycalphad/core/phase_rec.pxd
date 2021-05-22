# distutils: language = c++

cimport cython
import numpy
cimport numpy

ctypedef void (*math_function_t)(double*, const double*, void* user_data) nogil

cdef class FastFunction:
    cdef readonly object _objref
    cdef math_function_t f_ptr
    cdef void *func_data
    cdef void call(self, double *out, double *inp) nogil

@cython.final
cdef public class PhaseRecord(object)[type PhaseRecordType, object PhaseRecordObject]:
    cdef FastFunction _obj
    cdef FastFunction _formulaobj
    cdef FastFunction _formulagrad
    cdef FastFunction _formulahess
    cdef FastFunction _internal_cons_func
    cdef FastFunction _internal_cons_jac
    cdef FastFunction _internal_cons_hess
    cdef numpy.ndarray _masses
    cdef void** _masses_ptr
    cdef numpy.ndarray _formulamoles
    cdef void** _formulamoles_ptr
    cdef numpy.ndarray _formulamolegrads
    cdef void** _formulamolegrads_ptr
    cdef numpy.ndarray _formulamolehessians
    cdef void** _formulamolehessians_ptr
    cdef public size_t num_internal_cons
    cdef public object variables
    cdef public object state_variables
    cdef public object components
    cdef public object pure_elements
    cdef public object nonvacant_elements
    cdef public double[::1] parameters
    cdef public int phase_dof
    cdef public int num_statevars
    cdef public unicode phase_name
    cpdef void obj(self, double[::1] out, double[::1] dof) nogil
    cpdef void formulaobj(self, double[::1] out, double[::1] dof) nogil
    cpdef void obj_2d(self, double[::1] out, double[:, ::1] dof) nogil
    cpdef void obj_parameters_2d(self, double[:, ::1] out, double[:, ::1] dof, double[:, ::1] parameters) nogil
    cpdef void formulagrad(self, double[::1] out, double[::1] dof) nogil
    cpdef void formulahess(self, double[:,::1] out, double[::1] dof) nogil
    cpdef void internal_cons_func(self, double[::1] out, double[::1] dof) nogil
    cpdef void internal_cons_jac(self, double[:,::1] out, double[::1] dof) nogil
    cpdef void internal_cons_hess(self, double[:,:,::1] out, double[::1] dof) nogil
    cpdef void mass_obj(self, double[::1] out, double[::1] dof, int comp_idx) nogil
    cpdef void mass_obj_2d(self, double[::1] out, double[:, ::1] dof, int comp_idx) nogil
    cpdef void formulamole_obj(self, double[::1] out, double[::1] dof, int comp_idx) nogil
    cpdef void formulamole_grad(self, double[::1] out, double[::1] dof, int comp_idx) nogil
    cpdef void formulamole_hess(self, double[:,::1] out, double[::1] dof, int comp_idx) nogil
    # Used only to reconstitute if pickled (i.e. via __reduce__)
    cdef public object ofunc_
    cdef public object formulaofunc_
    cdef public object formulagfunc_
    cdef public object formulahfunc_
    cdef public object internal_cons_func_
    cdef public object internal_cons_jac_
    cdef public object internal_cons_hess_
    cdef public object massfuncs_
    cdef public object formulamolefuncs_
    cdef public object formulamolegradfuncs_
    cdef public object formulamolehessianfuncs_
