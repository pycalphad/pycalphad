# distutils: language = c++

cimport cython
cimport symengine
from libcpp.vector cimport vector

@cython.final
cdef public class PhaseRecord(object)[type PhaseRecordType, object PhaseRecordObject]:
    cdef symengine.LLVMDoubleVisitor _obj
    cdef symengine.LLVMDoubleVisitor _grad
    cdef symengine.LLVMDoubleVisitor _hess
    cdef symengine.LLVMDoubleVisitor _internal_cons
    cdef symengine.LLVMDoubleVisitor _internal_jac
    cdef symengine.LLVMDoubleVisitor _internal_cons_hess
    cdef symengine.LLVMDoubleVisitor _multiphase_cons
    cdef symengine.LLVMDoubleVisitor _multiphase_jac
    cdef vector[symengine.LLVMDoubleVisitor] _masses
    cdef vector[symengine.LLVMDoubleVisitor] _massgrads
    cdef vector[symengine.LLVMDoubleVisitor] _masshessians
    cdef public object _ofunc
    cdef public object _gfunc
    cdef public object _hfunc
    cdef public object _intconsfunc
    cdef public object _intjacfunc
    cdef public object _intconshessfunc
    cdef public object _mpconsfunc
    cdef public object _mpjacfunc
    cdef public size_t num_internal_cons
    cdef public size_t num_multiphase_cons
    cdef public object _massfuncs
    cdef public object _massgradfuncs
    cdef public object _masshessianfuncs
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
    cpdef void mass_obj(self, double[::1] out, double[:, ::1] dof, int comp_idx) nogil
    cpdef void mass_grad(self, double[::1] out, double[::1] dof, int comp_idx) nogil
    cpdef void mass_hess(self, double[:,::1] out, double[::1] dof, int comp_idx) nogil

