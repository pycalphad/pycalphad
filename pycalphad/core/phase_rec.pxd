# distutils: language = c++

cimport cython
import numpy
cimport numpy

ctypedef void (*math_function_t)(double*, const double*, void* user_data) nogil

cdef class FastFunction:
    cdef readonly object _objref, _addr1, _addr2
    cdef math_function_t f_ptr
    cdef void *func_data
    cdef void call(self, double *out, double *inp) nogil

@cython.final
cdef public class PhaseRecord(object)[type PhaseRecordType, object PhaseRecordObject]:
    cdef FastFunction _obj
    cdef FastFunction _grad
    cdef FastFunction _hess
    cdef FastFunction _internal_cons
    cdef FastFunction _internal_jac
    cdef FastFunction _internal_cons_hess
    cdef FastFunction _multiphase_cons
    cdef FastFunction _multiphase_jac
    cdef FastFunction _multiphase_cons_hess
    cdef numpy.ndarray _masses
    cdef void** _masses_ptr
    cdef numpy.ndarray _massgrads
    cdef void** _massgrads_ptr
    cdef numpy.ndarray _masshessians
    cdef void** _masshessians_ptr
    cdef public size_t num_internal_cons
    cdef public size_t num_multiphase_cons
    cdef public object variables
    cdef public object state_variables
    cdef public object components
    cdef public object pure_elements
    cdef public object nonvacant_elements
    cdef public double[::1] parameters
    cdef public int phase_dof
    cdef public unicode phase_name
    cpdef void obj(self, double[::1] out, double[:, ::1] dof) nogil
    cpdef void grad(self, double[::1] out, double[::1] dof) nogil
    cpdef void hess(self, double[:,::1] out, double[::1] dof) nogil
    cpdef void internal_constraints(self, double[::1] out, double[::1] dof) nogil
    cpdef void internal_jacobian(self, double[:,::1] out, double[::1] dof) nogil
    cpdef void internal_cons_hessian(self, double[:,:,::1] out, double[::1] dof) nogil
    cpdef void multiphase_constraints(self, double[::1] out, double[::1] dof_with_phasefrac) nogil
    cpdef void multiphase_jacobian(self, double[:,::1] out, double[::1] dof_with_phasefrac) nogil
    cpdef void multiphase_cons_hessian(self, double[:, :, ::1] out, double[::1] dof) nogil
    cpdef void mass_obj(self, double[::1] out, double[:, ::1] dof, int comp_idx) nogil
    cpdef void mass_grad(self, double[::1] out, double[::1] dof, int comp_idx) nogil
    cpdef void mass_hess(self, double[:,::1] out, double[::1] dof, int comp_idx) nogil

