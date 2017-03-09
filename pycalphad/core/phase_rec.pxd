ctypedef void func_t(double *out, double *dof, double* params, int *bounds) nogil
ctypedef void func_novec_t(double *dof, double* params, double *out) nogil
ctypedef void func_simple_t(double *out, double *dof, double* params, int *comp_idx) nogil
ctypedef void func_novec_simple_t(double *dof, double* params, double *out) nogil
cimport cython

@cython.final
cdef public class PhaseRecord(object)[type PhaseRecordType, object PhaseRecordObject]:
    cdef func_t* _obj
    cdef func_novec_t* _grad
    cdef func_novec_t* _hess
    cdef public object variables
    cdef public double[::1] parameters
    cdef public double[::1] num_sites
    cdef public int[::1] sublattice_dof
    cdef public int phase_dof
    cdef public unicode name
    cdef public double[:,:,::1] composition_matrices
    cdef public int vacancy_index
    cpdef void obj(self, double[::1] out, double[::1,:] dof) nogil
    cpdef void grad(self, double[::1] out, double[::1] dof) nogil
    cpdef void hess(self, double[::1,:] out, double[::1] dof) nogil
    cpdef void mass_obj(self, double[::1] out, double[::1] dof, int comp_idx) nogil
    cpdef void mass_grad(self, double[::1] out, double[::1] dof, int comp_idx) nogil
    cpdef void mass_hess(self, double[::1,:] out, double[::1] dof, int comp_idx) nogil