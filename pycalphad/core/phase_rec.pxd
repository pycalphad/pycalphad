ctypedef void func_t(double *out, double *dof, double* params, int *bounds) nogil
ctypedef void func_novec_t(double *dof, double* params, double *out) nogil
ctypedef void func_simple_t(double *out, double *dof, double* params, int *bounds) nogil
ctypedef void func_novec_simple_t(double *dof, double* params, double *out) nogil

cdef public class PhaseRecord(object)[type PhaseRecordType, object PhaseRecordObject]:
    cdef func_t* _obj
    cdef func_novec_t* _grad
    cdef func_novec_t* _hess
    cdef func_simple_t* _mass_obj
    cdef func_novec_simple_t* _mass_grad
    cdef func_novec_simple_t* _mass_hess
    cdef public object variables
    cdef public double[::1] parameters

cdef inline void obj(PhaseRecord prx, double[::1] out, double[::1,:] dof, int bounds) nogil
cdef inline void grad(PhaseRecord prx, double[::1] out, double[::1] dof) nogil
cdef inline void hess(PhaseRecord prx, double[::1,:] out, double[::1] dof) nogil
cdef inline void mass_obj(PhaseRecord prx, double[::1] out, double[::1,:] dof, int bounds) nogil
cdef inline void mass_grad(PhaseRecord prx, double[::1] out, double[::1] dof) nogil
cdef inline void mass_hess(PhaseRecord prx, double[::1,:] out, double[::1] dof) nogil