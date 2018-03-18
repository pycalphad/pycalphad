ctypedef void func_t(double *out, double *dof, double *params, int bounds) nogil
ctypedef void func_novec_t(double *dof, double* params, double *out) nogil
ctypedef void func_simple_t(double *out, double *dof, double* params, int *comp_idx) nogil
ctypedef void func_novec_simple_t(double *dof, double* params, double *out) nogil
cimport cython

@cython.final
cdef public class PhaseRecord(object)[type PhaseRecordType, object PhaseRecordObject]:
    cdef func_t* _obj
    cdef func_novec_t* _grad
    cdef func_novec_t* _hess
    cdef func_t** _masses
    cdef func_novec_t** _massgrads
    cdef public object _ofunc
    cdef public object _gfunc
    cdef public object _hfunc
    cdef public object _massfuncs
    cdef public object _massgradfuncs
    cdef public object variables
    cdef public double[::1] parameters
    cdef public double[::1] num_sites
    cdef public int[::1] sublattice_dof
    cdef public int phase_dof
    cdef public unicode phase_name
    cdef public double[:,:,::1] composition_matrices
    cdef public int vacancy_index
    cpdef void obj(self, double[::1] out, double[:,::1] dof) nogil
    cpdef void grad(self, double[::1] out, double[::1] dof) nogil
    cpdef void hess(self, double[:,::1] out, double[::1] dof) nogil
    cpdef void mass_obj(self, double[::1] out, double[:, ::1] dof, int comp_idx) nogil
    cpdef void mass_grad(self, double[::1] out, double[::1] dof, int comp_idx) nogil
    cpdef void mass_hess(self, double[:,:] out, double[::1] dof, int comp_idx) nogil

cpdef PhaseRecord PhaseRecord_from_cython(object comps, object variables, double[::1] num_sites, double[::1] parameters,
                                          object ofunc, object gfunc, object hfunc,
                                          object massfuncs, object massgradfuncs)