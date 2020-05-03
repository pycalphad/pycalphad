# distutils: language = c++

cimport cython
import numpy
cimport numpy
from libc.stdlib cimport malloc, free

ctypedef void (*math_function_t)(double*, const double*, void* user_data) nogil

cdef struct FastFunctionInfo:
    math_function_t f_ptr
    void *func_data

cdef class FastFunction:
    cdef readonly object _objref
    cdef math_function_t f_ptr
    cdef void *func_data
    cdef void call(self, double *out, double *inp) nogil
    cdef FastFunctionInfo get_info(self) nogil

@cython.final
cdef public class PhaseRecord(object)[type PhaseRecordType, object PhaseRecordObject]:
    cdef FastFunction _obj
    cdef FastFunction _grad
    cdef FastFunction _hess
    cdef FastFunction _internal_cons_func
    cdef FastFunction _internal_cons_jac
    cdef FastFunction _internal_cons_hess
    cdef FastFunction _multiphase_cons_func
    cdef FastFunction _multiphase_cons_jac
    cdef FastFunction _multiphase_cons_hess
    cdef numpy.ndarray _masses
    cdef void** _masses_ptr
    cdef numpy.ndarray _massgrads
    cdef void** _massgrads_ptr
    cdef numpy.ndarray _masshessians
    cdef void** _masshessians_ptr
    cdef int num_nonvacant_elements
    cdef public size_t num_internal_cons
    cdef public size_t num_multiphase_cons
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
    cpdef void obj_2d(self, double[::1] out, double[:, ::1] dof) nogil
    cpdef void obj_parameters_2d(self, double[:, ::1] out, double[:, ::1] dof, double[:, ::1] parameters) nogil
    cpdef void grad(self, double[::1] out, double[::1] dof) nogil
    cpdef void hess(self, double[:,::1] out, double[::1] dof) nogil
    cpdef void internal_cons_func(self, double[::1] out, double[::1] dof) nogil
    cpdef void internal_cons_jac(self, double[:,::1] out, double[::1] dof) nogil
    cpdef void internal_cons_hess(self, double[:,:,::1] out, double[::1] dof) nogil
    cpdef void multiphase_cons_func(self, double[::1] out, double[::1] dof_with_phasefrac) nogil
    cpdef void multiphase_cons_jac(self, double[:,::1] out, double[::1] dof_with_phasefrac) nogil
    cpdef void multiphase_cons_hess(self, double[:, :, ::1] out, double[::1] dof_with_phasefrac) nogil
    cpdef void mass_obj(self, double[::1] out, double[::1] dof, int comp_idx) nogil
    cpdef void mass_obj_2d(self, double[::1] out, double[:, ::1] dof, int comp_idx) nogil
    cpdef void mass_grad(self, double[::1] out, double[::1] dof, int comp_idx) nogil
    cpdef void mass_hess(self, double[:,::1] out, double[::1] dof, int comp_idx) nogil
    cdef LowLevelPhaseRecord get_lowlevel(self) nogil
    # Used only to reconstitute if pickled (i.e. via __reduce__)
    cdef public object ofunc_
    cdef public object gfunc_
    cdef public object hfunc_
    cdef public object internal_cons_func_
    cdef public object internal_cons_jac_
    cdef public object internal_cons_hess_
    cdef public object multiphase_cons_func_
    cdef public object multiphase_cons_jac_
    cdef public object multiphase_cons_hess_
    cdef public object massfuncs_
    cdef public object massgradfuncs_
    cdef public object masshessianfuncs_

cdef double* alloc_dof_with_parameters(double[::1] dof, double[::1] parameters) nogil
cdef double* alloc_dof_with_parameters_vectorized(double[:, ::1] dof, double[::1] parameters) nogil

# A nogil-compatible object which keeps copies of nogil function pointers
# This facilitates nogil access by downstream solvers
cdef cppclass LowLevelPhaseRecord:
    int num_statevars
    int phase_dof
    double[::1] parameters
    FastFunctionInfo _obj
    FastFunctionInfo _grad
    FastFunctionInfo _hess
    FastFunctionInfo _internal_cons_func
    FastFunctionInfo _internal_cons_jac
    FastFunctionInfo _internal_cons_hess
    FastFunctionInfo _multiphase_cons_func
    FastFunctionInfo _multiphase_cons_jac
    FastFunctionInfo _multiphase_cons_hess
    FastFunctionInfo* _masses_ptr
    FastFunctionInfo* _massgrads_ptr
    FastFunctionInfo* _masshessians_ptr
    LowLevelPhaseRecord() nogil
    # TODO: This is a test; probably we want a different initialization scheme

    @staticmethod
    void call_function(double *out, double *inp, FastFunctionInfo ffinfo) nogil:
        if ffinfo.f_ptr != NULL:
            ffinfo.f_ptr(out, inp, ffinfo.func_data)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    void obj(double[::1] outp, double[::1] dof) nogil:
        # dof.shape[0] may be oversized by the caller; do not trust it
        cdef double* dof_concat = alloc_dof_with_parameters(dof[:this.num_statevars+this.phase_dof], this.parameters)
        cdef int num_dof = this.num_statevars + this.phase_dof + this.parameters.shape[0]
        this.call_function(&outp[0], &dof_concat[0], this._obj)
        if this.parameters.shape[0] > 0:
            free(dof_concat)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    void grad(double[::1] out, double[::1] dof) nogil:
        # dof.shape[0] may be oversized by the caller; do not trust it
        cdef double* dof_concat = alloc_dof_with_parameters(dof[:this.num_statevars+this.phase_dof], this.parameters)
        this.call_function(&out[0], &dof_concat[0], this._grad)
        if this.parameters.shape[0] > 0:
            free(dof_concat)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    void hess(double[:, ::1] out, double[::1] dof) nogil:
        # dof.shape[0] may be oversized by the caller; do not trust it
        cdef double* dof_concat = alloc_dof_with_parameters(dof[:this.num_statevars+this.phase_dof], this.parameters)
        this.call_function(&out[0,0], &dof_concat[0], this._hess)
        if this.parameters.shape[0] > 0:
            free(dof_concat)


    @cython.boundscheck(False)
    @cython.wraparound(False)
    void internal_cons_func(double[::1] out, double[::1] dof) nogil:
        # dof.shape[0] may be oversized by the caller; do not trust it
        cdef double* dof_concat = alloc_dof_with_parameters(dof[:this.num_statevars+this.phase_dof], this.parameters)
        this.call_function(&out[0], &dof_concat[0], this._internal_cons_func)
        if this.parameters.shape[0] > 0:
            free(dof_concat)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    void internal_cons_jac(double[:, ::1] out, double[::1] dof) nogil:
        # dof.shape[0] may be oversized by the caller; do not trust it
        cdef double* dof_concat = alloc_dof_with_parameters(dof[:this.num_statevars+this.phase_dof], this.parameters)
        this.call_function(&out[0, 0], &dof_concat[0], this._internal_cons_jac)
        if this.parameters.shape[0] > 0:
            free(dof_concat)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    void internal_cons_hess(double[:, :, ::1] out, double[::1] dof) nogil:
        # dof.shape[0] may be oversized by the caller; do not trust it
        cdef double* dof_concat = alloc_dof_with_parameters(dof[:this.num_statevars+this.phase_dof], this.parameters)
        this.call_function(&out[0, 0, 0], &dof_concat[0], this._internal_cons_hess)
        if this.parameters.shape[0] > 0:
            free(dof_concat)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    void multiphase_cons_func(double[::1] out, double[::1] dof) nogil:
        # dof.shape[0] may be oversized by the caller; do not trust it
        cdef double* dof_concat = alloc_dof_with_parameters(dof[:this.num_statevars+this.phase_dof+1], this.parameters)
        this.call_function(&out[0], &dof_concat[0], this._multiphase_cons_func)
        if this.parameters.shape[0] > 0:
            free(dof_concat)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    void multiphase_cons_jac(double[:, ::1] out, double[::1] dof) nogil:
        # dof.shape[0] may be oversized by the caller; do not trust it
        cdef double* dof_concat = alloc_dof_with_parameters(dof[:this.num_statevars+this.phase_dof+1], this.parameters)
        this.call_function(&out[0, 0], &dof_concat[0], this._multiphase_cons_jac)
        if this.parameters.shape[0] > 0:
            free(dof_concat)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    void multiphase_cons_hess(double[:, :, ::1] out, double[::1] dof) nogil:
        # dof.shape[0] may be oversized by the caller; do not trust it
        cdef double* dof_concat = alloc_dof_with_parameters(dof[:this.num_statevars+this.phase_dof+1], this.parameters)
        this.call_function(&out[0, 0, 0], &dof_concat[0], this._multiphase_cons_hess)
        if this.parameters.shape[0] > 0:
            free(dof_concat)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    void mass_obj(double[::1] out, double[::1] dof, int comp_idx) nogil:
        # dof.shape[0] may be oversized by the caller; do not trust it
        cdef double* dof_concat = alloc_dof_with_parameters(dof[:this.num_statevars+this.phase_dof], this.parameters)
        this.call_function(&out[0], &dof_concat[0], this._masses_ptr[comp_idx])
        if this.parameters.shape[0] > 0:
            free(dof_concat)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    void mass_grad(double[::1] out, double[::1] dof, int comp_idx) nogil:
        # dof.shape[0] may be oversized by the caller; do not trust it
        cdef double* dof_concat = alloc_dof_with_parameters(dof[:this.num_statevars+this.phase_dof], this.parameters)
        this.call_function(&out[0], &dof_concat[0], this._massgrads_ptr[comp_idx])
        if this.parameters.shape[0] > 0:
            free(dof_concat)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    void mass_hess(double[:,::1] out, double[::1] dof, int comp_idx) nogil:
        # dof.shape[0] may be oversized by the caller; do not trust it
        cdef double* dof_concat = alloc_dof_with_parameters(dof[:this.num_statevars+this.phase_dof], this.parameters)
        this.call_function(&out[0,0], &dof_concat[0], this._masshessians_ptr[comp_idx])
        if this.parameters.shape[0] > 0:
            free(dof_concat)
