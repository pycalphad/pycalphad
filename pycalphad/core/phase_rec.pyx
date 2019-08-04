# distutils: language = c++

cimport cython
from libc.stdlib cimport malloc, free
import numpy as np
cimport numpy as np
import pycalphad.variables as v
from symengine.lib.symengine_wrapper cimport LLVMDouble, LambdaDouble

cdef class FastFunction:
    def __cinit__(self, object func):
        if isinstance(func, LambdaDouble):
            self.lambda_double = func
        elif isinstance(func, LLVMDouble):
            self.llvm_double = func
        elif func is None:
            pass
        else:
            raise ValueError('Unknown callable function type: {}'.format(func.__class__))
    cdef void call(self, double *out, double *inp) nogil:
        if self.llvm_double is not None:
            self.llvm_double.unsafe_real_ptr(inp, out)
        elif self.lambda_double is not None:
            self.lambda_double.unsafe_real_ptr(inp, out)

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
                                 self._obj, self._grad, self._hess, self._masses, self._massgrads,
                                 self._masshessians, self._internal_cons, self._internal_jac, self._internal_cons_hess,
                                 self._multiphase_cons, self._multiphase_jac, self._multiphase_cons_hess,
                                 self.num_internal_cons, self.num_multiphase_cons)

    def __cinit__(self, object comps, object state_variables, object variables,
                  double[::1] parameters, object ofunc, object gfunc, object hfunc,
                  object massfuncs, object massgradfuncs, object masshessianfuncs,
                  object internal_cons_func, object internal_jac_func, object internal_cons_hess_func,
                  object multiphase_cons_func, object multiphase_jac_func, object multiphase_cons_hess_func,
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
        self.pure_elements = pure_elements
        self.nonvacant_elements = nonvacant_elements
        self.phase_dof = 0
        self.parameters = parameters
        self.num_internal_cons = num_internal_cons
        self.num_multiphase_cons = num_multiphase_cons

        for variable in variables:
            if not isinstance(variable, v.SiteFraction):
                continue
            self.phase_name = <unicode>variable.phase_name
            self.phase_dof += 1

        if ofunc is not None:
            self._obj = FastFunction(ofunc)
        if gfunc is not None:
            self._grad = FastFunction(gfunc)
        if hfunc is not None:
            self._hess = FastFunction(hfunc)
        if internal_cons_func is not None:
            self._internal_cons = FastFunction(internal_cons_func)
        if internal_jac_func is not None:
            self._internal_jac = FastFunction(internal_jac_func)
        if internal_cons_hess_func is not None:
            self._internal_cons_hess = FastFunction(internal_cons_hess_func)
        if multiphase_cons_func is not None:
            self._multiphase_cons = FastFunction(multiphase_cons_func)
        if multiphase_jac_func is not None:
            self._multiphase_jac = FastFunction(multiphase_jac_func)
        if multiphase_cons_hess_func is not None:
            self._multiphase_cons_hess = FastFunction(multiphase_cons_hess_func)
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
    cpdef void obj(self, double[::1] outp, double[:, ::1] dof) nogil:
        cdef double* dof_concat = alloc_dof_with_parameters_vectorized(dof, self.parameters)
        cdef int i
        cdef int num_inps = dof.shape[0]
        cdef int num_dof = dof.shape[1] + self.parameters.shape[0]

        for i in range(num_inps):
            self._obj.call(&outp[i], &dof_concat[i * num_dof])
        if self.parameters.shape[0] > 0:
            free(dof_concat)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void grad(self, double[::1] out, double[::1] dof) nogil:
        cdef double* dof_concat = alloc_dof_with_parameters(dof, self.parameters)
        self._grad.call(&out[0], &dof_concat[0])
        if self.parameters.shape[0] > 0:
            free(dof_concat)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void hess(self, double[:, ::1] out, double[::1] dof) nogil:
        cdef double* dof_concat = alloc_dof_with_parameters(dof, self.parameters)
        self._hess.call(&out[0,0], &dof_concat[0])
        if self.parameters.shape[0] > 0:
            free(dof_concat)


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void internal_constraints(self, double[::1] out, double[::1] dof) nogil:
        cdef double* dof_concat = alloc_dof_with_parameters(dof, self.parameters)
        self._internal_cons.call(&out[0], &dof_concat[0])
        if self.parameters.shape[0] > 0:
            free(dof_concat)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void internal_jacobian(self, double[:, ::1] out, double[::1] dof) nogil:
        cdef double* dof_concat = alloc_dof_with_parameters(dof, self.parameters)
        self._internal_jac.call(&out[0, 0], &dof_concat[0])
        if self.parameters.shape[0] > 0:
            free(dof_concat)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void internal_cons_hessian(self, double[:, :, ::1] out, double[::1] dof) nogil:
        cdef double* dof_concat = alloc_dof_with_parameters(dof, self.parameters)
        self._internal_cons_hess.call(&out[0, 0, 0], &dof_concat[0])
        if self.parameters.shape[0] > 0:
            free(dof_concat)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void multiphase_constraints(self, double[::1] out, double[::1] dof) nogil:
        cdef double* dof_concat = alloc_dof_with_parameters(dof, self.parameters)
        self._multiphase_cons.call(&out[0], &dof_concat[0])
        if self.parameters.shape[0] > 0:
            free(dof_concat)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void multiphase_jacobian(self, double[:, ::1] out, double[::1] dof) nogil:
        cdef double* dof_concat = alloc_dof_with_parameters(dof, self.parameters)
        self._multiphase_jac.call(&out[0, 0], &dof_concat[0])
        if self.parameters.shape[0] > 0:
            free(dof_concat)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void multiphase_cons_hessian(self, double[:, :, ::1] out, double[::1] dof) nogil:
        cdef double* dof_concat = alloc_dof_with_parameters(dof, self.parameters)
        self._multiphase_cons_hess.call(&out[0, 0, 0], &dof_concat[0])
        if self.parameters.shape[0] > 0:
            free(dof_concat)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void mass_obj(self, double[::1] out, double[:, ::1] dof, int comp_idx) nogil:
        cdef double* dof_concat = alloc_dof_with_parameters_vectorized(dof, self.parameters)
        cdef int i
        cdef int num_inps = dof.shape[0]
        cdef int num_dof = dof.shape[1] + self.parameters.shape[0]
        for i in range(num_inps):
            (<FastFunction>self._masses_ptr[comp_idx]).call(&out[i], &dof_concat[i * num_dof])
        if self.parameters.shape[0] > 0:
            free(dof_concat)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void mass_grad(self, double[::1] out, double[::1] dof, int comp_idx) nogil:
        cdef double* dof_concat = alloc_dof_with_parameters(dof, self.parameters)
        (<FastFunction>self._massgrads_ptr[comp_idx]).call(&out[0], &dof_concat[0])
        if self.parameters.shape[0] > 0:
            free(dof_concat)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void mass_hess(self, double[:,::1] out, double[::1] dof, int comp_idx) nogil:
        cdef double* dof_concat = alloc_dof_with_parameters(dof, self.parameters)
        (<FastFunction>self._masshessians_ptr[comp_idx]).call(&out[0,0], &dof_concat[0])
        if self.parameters.shape[0] > 0:
            free(dof_concat)
