# distutils: language = c++

cimport cython
cimport symengine
from libcpp.vector cimport vector

@cython.final
cdef public class PhaseRecord(object)[type PhaseRecordType, object PhaseRecordObject]:
    cdef symengine.LambdaRealDoubleVisitor* _obj
    cdef symengine.LambdaRealDoubleVisitor* _grad
    cdef symengine.LambdaRealDoubleVisitor* _hess
    cdef symengine.LambdaRealDoubleVisitor* _internal_cons
    cdef symengine.LambdaRealDoubleVisitor* _internal_jac
    cdef symengine.LambdaRealDoubleVisitor* _internal_cons_hess
    cdef symengine.LambdaRealDoubleVisitor* _multiphase_cons
    cdef symengine.LambdaRealDoubleVisitor* _multiphase_jac
    cdef symengine.LambdaRealDoubleVisitor* _multiphase_cons_hess
    cdef vector[symengine.LambdaRealDoubleVisitor*] _masses
    cdef vector[symengine.LambdaRealDoubleVisitor*] _massgrads
    cdef vector[symengine.LambdaRealDoubleVisitor*] _masshessians
    cdef public object _ofunc
    cdef public object _gfunc
    cdef public object _hfunc
    cdef public object _intconsfunc
    cdef public object _intjacfunc
    cdef public object _intconshessfunc
    cdef public object _mpconsfunc
    cdef public object _mpjacfunc
    cdef public object _mpconshessfunc
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
    cpdef void multiphase_cons_hessian(self, double[:, :, ::1] out, double[::1] dof) nogil
    cpdef void mass_obj(self, double[::1] out, double[:, ::1] dof, int comp_idx) nogil
    cpdef void mass_grad(self, double[::1] out, double[::1] dof, int comp_idx) nogil
    cpdef void mass_hess(self, double[:,::1] out, double[::1] dof, int comp_idx) nogil

