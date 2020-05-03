# distutils: language = c++

cimport cython
from libc.stdlib cimport malloc, free
import numpy as np
cimport numpy as np
import pycalphad.variables as v
import ctypes

cdef class FastFunction:
    """``FastFunction`` provides a stable(-ish) interface that encapsulates SymEngine function pointers.
    """
    def __cinit__(self, object func):
        if func is None:
            self.f_ptr = NULL
            self.func_data = NULL
            return
        # Preserve reference to object to prevent garbage collection
        self._objref = func
        addr1, addr2 = func.as_ctypes()
        self.f_ptr = (<math_function_t*><size_t>ctypes.addressof(addr1))[0]
        self.func_data =  (<void**><size_t>ctypes.addressof(addr2))[0]
    def __reduce__(self):
        return FastFunction, (self._objref,)
    cdef void call(self, double *out, double *inp) nogil:
        if self.f_ptr != NULL:
            self.f_ptr(out, inp, self.func_data)
    cdef FastFunctionInfo get_info(self) nogil:
        cdef FastFunctionInfo ffinfo = FastFunctionInfo()
        ffinfo.f_ptr = self.f_ptr
        ffinfo.func_data = self.func_data
        return ffinfo

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double* alloc_dof_with_parameters(double[::1] dof, double[::1] parameters) nogil:
    """Remember to free() if parameters.shape[0] > 0"""
    cdef double* dof_concat
    cdef int j
    cdef int num_dof = dof.shape[0] + parameters.shape[0]
    if parameters.shape[0] == 0:
        dof_concat = &dof[0]
    else:
        dof_concat = <double *> malloc(num_dof * sizeof(double))
        for j in range(0,dof.shape[0]):
            dof_concat[j] = dof[j]
        for j in range(dof.shape[0], num_dof):
            dof_concat[j] = parameters[j - dof.shape[0]]
    return dof_concat

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double* alloc_dof_with_parameters_vectorized(double[:, ::1] dof, double[::1] parameters) nogil:
    """Remember to free() if parameters.shape[0] > 0"""
    cdef double* dof_concat
    cdef int i, j
    cdef int num_inps = dof.shape[0]
    cdef int num_dof = dof.shape[1] + parameters.shape[0]

    if parameters.shape[0] == 0:
        dof_concat = &dof[0, 0]
    else:
        dof_concat = <double *> malloc(num_inps * num_dof * sizeof(double))
        for i in range(num_inps):
            for j in range(0,dof.shape[1]):
                dof_concat[i * num_dof + j] = dof[i, j]
            for j in range(dof.shape[1], num_dof):
                dof_concat[i * num_dof + j] = parameters[j - dof.shape[1]]
    return dof_concat


cdef public class PhaseRecord(object)[type PhaseRecordType, object PhaseRecordObject]:
    """
    This object exposes a common API to the solver so it doesn't need to know about the differences
    between Model implementations. PhaseRecords are immutable after initialization.
    """
    def __reduce__(self):
            return PhaseRecord, (self.components, self.state_variables, self.variables, np.array(self.parameters),
                                 self.ofunc_, self.gfunc_, self.hfunc_,
                                 self.massfuncs_, self.massgradfuncs_, self.masshessianfuncs_,
                                 self.internal_cons_func_, self.internal_cons_jac_, self.internal_cons_hess_,
                                 self.multiphase_cons_func_, self.multiphase_cons_jac_, self.multiphase_cons_hess_,
                                 self.num_internal_cons, self.num_multiphase_cons)

    def __cinit__(self, object comps, object state_variables, object variables,
                  double[::1] parameters, object ofunc, object gfunc, object hfunc,
                  object massfuncs, object massgradfuncs, object masshessianfuncs,
                  object internal_cons_func, object internal_cons_jac, object internal_cons_hess,
                  object multiphase_cons_func, object multiphase_cons_jac, object multiphase_cons_hess,
                  size_t num_internal_cons, size_t num_multiphase_cons):
        cdef:
            int var_idx, el_idx
        self.components = comps
        desired_active_pure_elements = [list(x.constituents.keys()) for x in self.components]
        desired_active_pure_elements = [el.upper() for constituents in desired_active_pure_elements for el in constituents]
        pure_elements = sorted(set(desired_active_pure_elements))
        nonvacant_elements = sorted([x for x in set(desired_active_pure_elements) if x != 'VA'])

        self.variables = variables
        self.state_variables = state_variables
        self.num_statevars = len(state_variables)
        self.pure_elements = pure_elements
        self.nonvacant_elements = nonvacant_elements
        self.num_nonvacant_elements = len(nonvacant_elements)
        self.phase_dof = 0
        self.parameters = parameters
        self.num_internal_cons = num_internal_cons
        self.num_multiphase_cons = num_multiphase_cons

        for variable in variables:
            if not isinstance(variable, v.SiteFraction):
                continue
            self.phase_name = <unicode>variable.phase_name
            self.phase_dof += 1

        # Used only to reconstitute if pickled (i.e. via __reduce__)
        self.ofunc_ = ofunc
        self.gfunc_ = gfunc
        self.hfunc_ = hfunc
        self.internal_cons_func_ = internal_cons_func
        self.internal_cons_jac_ = internal_cons_jac
        self.internal_cons_hess_ = internal_cons_hess
        self.multiphase_cons_func_ = multiphase_cons_func
        self.multiphase_cons_jac_ = multiphase_cons_jac
        self.multiphase_cons_hess_ = multiphase_cons_hess
        self.massfuncs_ = massfuncs
        self.massgradfuncs_ = massgradfuncs
        self.masshessianfuncs_ = masshessianfuncs

        if ofunc is not None:
            self._obj = FastFunction(ofunc)
        if gfunc is not None:
            self._grad = FastFunction(gfunc)
        if hfunc is not None:
            self._hess = FastFunction(hfunc)
        if internal_cons_func is not None:
            self._internal_cons_func = FastFunction(internal_cons_func)
        if internal_cons_jac is not None:
            self._internal_cons_jac = FastFunction(internal_cons_jac)
        if internal_cons_hess is not None:
            self._internal_cons_hess = FastFunction(internal_cons_hess)
        if multiphase_cons_func is not None:
            self._multiphase_cons_func = FastFunction(multiphase_cons_func)
        if multiphase_cons_jac is not None:
            self._multiphase_cons_jac = FastFunction(multiphase_cons_jac)
        if multiphase_cons_hess is not None:
            self._multiphase_cons_hess = FastFunction(multiphase_cons_hess)
        if massfuncs is not None:
            self._masses = np.empty(len(nonvacant_elements), dtype='object')
            for el_idx in range(len(nonvacant_elements)):
                self._masses[el_idx] = FastFunction(massfuncs[el_idx])
            self._masses_ptr = <void**> self._masses.data
        if massgradfuncs is not None:
            self._massgrads = np.empty(len(nonvacant_elements), dtype='object')
            for el_idx in range(len(nonvacant_elements)):
                self._massgrads[el_idx] = FastFunction(massgradfuncs[el_idx])
            self._massgrads_ptr = <void**> self._massgrads.data
        if masshessianfuncs is not None:
            self._masshessians = np.empty(len(nonvacant_elements), dtype='object')
            for el_idx in range(len(nonvacant_elements)):
                self._masshessians[el_idx] = FastFunction(masshessianfuncs[el_idx])
            self._masshessians_ptr = <void**> self._masshessians.data

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void obj(self, double[::1] outp, double[::1] dof) nogil:
        # dof.shape[0] may be oversized by the caller; do not trust it
        cdef double* dof_concat = alloc_dof_with_parameters(dof[:self.num_statevars+self.phase_dof], self.parameters)
        cdef int num_dof = self.num_statevars + self.phase_dof + self.parameters.shape[0]
        self._obj.call(&outp[0], &dof_concat[0])
        if self.parameters.shape[0] > 0:
            free(dof_concat)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void obj_2d(self, double[::1] outp, double[:, ::1] dof) nogil:
        # dof.shape[1] may be oversized by the caller; do not trust it
        cdef double* dof_concat = alloc_dof_with_parameters_vectorized(dof[:, :self.num_statevars+self.phase_dof], self.parameters)
        cdef int i
        cdef int num_inps = dof.shape[0]
        cdef int num_dof = self.num_statevars + self.phase_dof + self.parameters.shape[0]
        for i in range(num_inps):
            self._obj.call(&outp[i], &dof_concat[i * num_dof])
        if self.parameters.shape[0] > 0:
            free(dof_concat)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void obj_parameters_2d(self, double[:, ::1] outp, double[:, ::1] dof, double[:, ::1] parameters) nogil:
        """
        Calculate objective function using custom parameters.
        Note dof and parameters are vectorized separately, i.e., broadcast against each other.
        Let dof.shape[0] = M and parameters.shape[0] = N
        Then outp.shape = (M,N)
        """
        # dof.shape[1] may be oversized by the caller; do not trust it
        cdef size_t i, j, dof_idx, param_idx
        cdef size_t num_dof_inps = dof.shape[0]
        cdef size_t num_param_inps = parameters.shape[0]
        # We are trusting parameters.shape[1] to be sized correctly here
        cdef size_t num_params = parameters.shape[1]
        cdef size_t num_dof = self.num_statevars + self.phase_dof + num_params
        cdef double* dof_concat = <double *> malloc(num_param_inps * num_dof * sizeof(double))
        for i in range(num_dof_inps):
            # Initialize all parameter arrays with current dof
            for j in range(num_param_inps):
                for dof_idx in range(num_dof-num_params):
                    dof_concat[j * num_dof + dof_idx] = dof[i, dof_idx]
                for param_idx in range(num_params):
                    dof_concat[j * num_dof + (num_dof-num_params)+param_idx] = parameters[j, param_idx]
            for j in range(num_param_inps):
                self._obj.call(&outp[i,j], &dof_concat[j * num_dof])
        free(dof_concat)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void grad(self, double[::1] out, double[::1] dof) nogil:
        # dof.shape[0] may be oversized by the caller; do not trust it
        cdef double* dof_concat = alloc_dof_with_parameters(dof[:self.num_statevars+self.phase_dof], self.parameters)
        self._grad.call(&out[0], &dof_concat[0])
        if self.parameters.shape[0] > 0:
            free(dof_concat)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void hess(self, double[:, ::1] out, double[::1] dof) nogil:
        # dof.shape[0] may be oversized by the caller; do not trust it
        cdef double* dof_concat = alloc_dof_with_parameters(dof[:self.num_statevars+self.phase_dof], self.parameters)
        self._hess.call(&out[0,0], &dof_concat[0])
        if self.parameters.shape[0] > 0:
            free(dof_concat)


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void internal_cons_func(self, double[::1] out, double[::1] dof) nogil:
        # dof.shape[0] may be oversized by the caller; do not trust it
        cdef double* dof_concat = alloc_dof_with_parameters(dof[:self.num_statevars+self.phase_dof], self.parameters)
        self._internal_cons_func.call(&out[0], &dof_concat[0])
        if self.parameters.shape[0] > 0:
            free(dof_concat)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void internal_cons_jac(self, double[:, ::1] out, double[::1] dof) nogil:
        # dof.shape[0] may be oversized by the caller; do not trust it
        cdef double* dof_concat = alloc_dof_with_parameters(dof[:self.num_statevars+self.phase_dof], self.parameters)
        self._internal_cons_jac.call(&out[0, 0], &dof_concat[0])
        if self.parameters.shape[0] > 0:
            free(dof_concat)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void internal_cons_hess(self, double[:, :, ::1] out, double[::1] dof) nogil:
        # dof.shape[0] may be oversized by the caller; do not trust it
        cdef double* dof_concat = alloc_dof_with_parameters(dof[:self.num_statevars+self.phase_dof], self.parameters)
        self._internal_cons_hess.call(&out[0, 0, 0], &dof_concat[0])
        if self.parameters.shape[0] > 0:
            free(dof_concat)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void multiphase_cons_func(self, double[::1] out, double[::1] dof) nogil:
        # dof.shape[0] may be oversized by the caller; do not trust it
        cdef double* dof_concat = alloc_dof_with_parameters(dof[:self.num_statevars+self.phase_dof+1], self.parameters)
        self._multiphase_cons_func.call(&out[0], &dof_concat[0])
        if self.parameters.shape[0] > 0:
            free(dof_concat)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void multiphase_cons_jac(self, double[:, ::1] out, double[::1] dof) nogil:
        # dof.shape[0] may be oversized by the caller; do not trust it
        cdef double* dof_concat = alloc_dof_with_parameters(dof[:self.num_statevars+self.phase_dof+1], self.parameters)
        self._multiphase_cons_jac.call(&out[0, 0], &dof_concat[0])
        if self.parameters.shape[0] > 0:
            free(dof_concat)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void multiphase_cons_hess(self, double[:, :, ::1] out, double[::1] dof) nogil:
        # dof.shape[0] may be oversized by the caller; do not trust it
        cdef double* dof_concat = alloc_dof_with_parameters(dof[:self.num_statevars+self.phase_dof+1], self.parameters)
        self._multiphase_cons_hess.call(&out[0, 0, 0], &dof_concat[0])
        if self.parameters.shape[0] > 0:
            free(dof_concat)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void mass_obj(self, double[::1] out, double[::1] dof, int comp_idx) nogil:
        # dof.shape[0] may be oversized by the caller; do not trust it
        cdef double* dof_concat = alloc_dof_with_parameters(dof[:self.num_statevars+self.phase_dof], self.parameters)
        (<FastFunction>self._masses_ptr[comp_idx]).call(&out[0], &dof_concat[0])
        if self.parameters.shape[0] > 0:
            free(dof_concat)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void mass_obj_2d(self, double[::1] out, double[:, ::1] dof, int comp_idx) nogil:
        # dof.shape[1] may be oversized by the caller; do not trust it
        cdef double* dof_concat = alloc_dof_with_parameters_vectorized(dof[:, :self.num_statevars+self.phase_dof], self.parameters)
        cdef int i
        cdef int num_inps = dof.shape[0]
        cdef int num_dof = self.num_statevars + self.phase_dof + self.parameters.shape[0]
        for i in range(num_inps):
            (<FastFunction>self._masses_ptr[comp_idx]).call(&out[i], &dof_concat[i * num_dof])
        if self.parameters.shape[0] > 0:
            free(dof_concat)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void mass_grad(self, double[::1] out, double[::1] dof, int comp_idx) nogil:
        # dof.shape[0] may be oversized by the caller; do not trust it
        cdef double* dof_concat = alloc_dof_with_parameters(dof[:self.num_statevars+self.phase_dof], self.parameters)
        (<FastFunction>self._massgrads_ptr[comp_idx]).call(&out[0], &dof_concat[0])
        if self.parameters.shape[0] > 0:
            free(dof_concat)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void mass_hess(self, double[:,::1] out, double[::1] dof, int comp_idx) nogil:
        # dof.shape[0] may be oversized by the caller; do not trust it
        cdef double* dof_concat = alloc_dof_with_parameters(dof[:self.num_statevars+self.phase_dof], self.parameters)
        (<FastFunction>self._masshessians_ptr[comp_idx]).call(&out[0,0], &dof_concat[0])
        if self.parameters.shape[0] > 0:
            free(dof_concat)
    cdef LowLevelPhaseRecord get_lowlevel(self) nogil:
        cdef int i
        cdef LowLevelPhaseRecord llprx = LowLevelPhaseRecord()
        llprx.num_statevars = self.num_statevars
        llprx.phase_dof = self.phase_dof
        llprx.parameters = self.parameters
        llprx._obj = self._obj.get_info()
        llprx._grad = self._grad.get_info()
        llprx._hess = self._hess.get_info()
        llprx._internal_cons_func = self._internal_cons_func.get_info()
        llprx._internal_cons_jac = self._internal_cons_jac.get_info()
        llprx._internal_cons_hess = self._internal_cons_hess.get_info()
        llprx._multiphase_cons_func = self._multiphase_cons_func.get_info()
        llprx._multiphase_cons_jac = self._multiphase_cons_jac.get_info()
        llprx._multiphase_cons_hess = self._multiphase_cons_hess.get_info()
        llprx._masses_ptr = <FastFunctionInfo*>malloc(self.num_nonvacant_elements * sizeof(FastFunctionInfo))
        llprx._massgrads_ptr = <FastFunctionInfo*>malloc(self.num_nonvacant_elements * sizeof(FastFunctionInfo))
        llprx._masshessians_ptr = <FastFunctionInfo*>malloc(self.num_nonvacant_elements * sizeof(FastFunctionInfo))
        for i in range(self.num_nonvacant_elements):
            llprx._masses_ptr[i] = (<FastFunction>self._masses_ptr[i]).get_info()
            llprx._massgrads_ptr[i] = (<FastFunction>self._massgrads_ptr[i]).get_info()
            llprx._masshessians_ptr[i] = (<FastFunction>self._masshessians_ptr[i]).get_info()
        return llprx
