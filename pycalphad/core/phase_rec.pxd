ctypedef void func_t(double *out, double *dof, double* params, int *bounds) nogil
ctypedef void func_novec_t(double *dof, double* params, double *out) nogil
ctypedef void func_simple_t(double *out, double *dof, double* params, int *comp_idx) nogil
ctypedef void func_novec_simple_t(double *dof, double* params, double *out) nogil

cdef public class PhaseRecord(object)[type PhaseRecordType, object PhaseRecordObject]:
    cdef func_t* _obj
    cdef func_novec_t* _grad
    cdef func_novec_t* _hess
    cdef public object variables
    cdef public double[::1] parameters
    cdef public double[::1] num_sites
    cdef public double[:,:,::1] composition_matrices
    cdef int vacancy_index

cdef void obj(PhaseRecord prx, double[::1] out, double[::1,:] dof, int bounds) nogil
cdef void grad(PhaseRecord prx, double[::1] out, double[::1] dof) nogil
cdef void hess(PhaseRecord prx, double[::1,:] out, double[::1] dof) nogil
cdef void mass_obj(PhaseRecord prx, double[::1] out, double[::1] dof, int comp_idx) nogil
cdef void mass_grad(PhaseRecord prx, double[::1] out, double[::1] dof, int comp_idx) nogil
cdef void mass_hess(PhaseRecord prx, double[::1,:] out, double[::1] dof, int comp_idx) nogil