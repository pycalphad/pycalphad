# distutils: language = c++

cimport cython
from libcpp.map cimport map
from libcpp.utility cimport pair
from libcpp.string cimport string
from cython.operator cimport dereference as deref
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
        if isinstance(func, FastFunction):
            func = func._objref
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

cdef class FastFunctionFactory:
    def __cinit__(self, phase_record_factory, phase_name):
        cdef int INITIAL_CACHE_SIZE = 100
        self.phase_record_factory = phase_record_factory
        self.phase_name = phase_name
        self._cache = np.empty(INITIAL_CACHE_SIZE, dtype='object')
        self._cache_ptr = <void**> self._cache.data
        self._cache_cur_idx = -1

    cdef void* get_func(self, string property_name) nogil except *:
        cdef pair[string, string] cache_key = pair[string, string](string(<char*>'func'), property_name)
        cdef map[pair[string, string], int].iterator it
        it = self._cache_property_map.find(cache_key)
        if it == self._cache_property_map.end():
            with gil:
                self._cache_cur_idx += 1
                if self._cache_cur_idx > self._cache.shape[0]:
                    raise ValueError('Cache error')
                self._cache[self._cache_cur_idx] = FastFunction(self.phase_record_factory.get_phase_property(self.phase_name, (<bytes>property_name).decode('utf-8'), include_grad=False, include_hess=False).func)
                self._cache_property_map[cache_key] = self._cache_cur_idx
            it = self._cache_property_map.find(cache_key)
        return <void*>self._cache_ptr[deref(it).second]

    cdef void* get_grad(self, string property_name) nogil except *:
        cdef pair[string, string] cache_key = pair[string, string](string(<char*>'grad'), property_name)
        cdef map[pair[string, string], int].iterator it
        it = self._cache_property_map.find(cache_key)
        if it == self._cache_property_map.end():
            with gil:
                self._cache_cur_idx += 1
                if self._cache_cur_idx > self._cache.shape[0]:
                    raise ValueError('Cache error')
                self._cache[self._cache_cur_idx] = FastFunction(self.phase_record_factory.get_phase_property(self.phase_name, (<bytes>property_name).decode('utf-8'), include_grad=True, include_hess=False).grad)
                self._cache_property_map[cache_key] = self._cache_cur_idx
            it = self._cache_property_map.find(cache_key)
        return <void*>self._cache_ptr[deref(it).second]

    cdef void* get_hess(self, string property_name) nogil except *:
        cdef pair[string, string] cache_key = pair[string, string](string(<char*>'hess'), property_name)
        cdef map[pair[string, string], int].iterator it
        it = self._cache_property_map.find(cache_key)
        if it == self._cache_property_map.end():
            with gil:
                self._cache_cur_idx += 1
                if self._cache_cur_idx > self._cache.shape[0]:
                    raise ValueError('Cache error')
                self._cache[self._cache_cur_idx] = FastFunction(self.phase_record_factory.get_phase_property(self.phase_name, (<bytes>property_name).decode('utf-8'), include_grad=False, include_hess=True).hess)
                self._cache_property_map[cache_key] = self._cache_cur_idx
            it = self._cache_property_map.find(cache_key)
        return <void*>self._cache_ptr[deref(it).second]

    cpdef FastFunction get_cons_func(self):
        return FastFunction(self.phase_record_factory.get_phase_constraints(self.phase_name).internal_cons_func)

    cpdef FastFunction get_cons_jac(self):
        return FastFunction(self.phase_record_factory.get_phase_constraints(self.phase_name).internal_cons_jac)

    cpdef FastFunction get_cons_hess(self):
        return FastFunction(self.phase_record_factory.get_phase_constraints(self.phase_name).internal_cons_hess)

    cpdef int get_cons_len(self):
        return self.phase_record_factory.get_phase_constraints(self.phase_name).num_internal_cons

    cpdef FastFunction get_mole_fraction_func(self, unicode element_name):
        return FastFunction(self.phase_record_factory.get_phase_formula_moles_element(self.phase_name,
                                element_name, per_formula_unit=False).func)

    cpdef FastFunction get_mole_fraction_grad(self, unicode element_name):
        return FastFunction(self.phase_record_factory.get_phase_formula_moles_element(self.phase_name,
                                element_name, per_formula_unit=False).grad)

    cpdef FastFunction get_mole_fraction_hess(self, unicode element_name):
        return FastFunction(self.phase_record_factory.get_phase_formula_moles_element(self.phase_name,
                                element_name, per_formula_unit=False).hess)

    cpdef FastFunction get_mole_formula_func(self, unicode element_name):
        return FastFunction(self.phase_record_factory.get_phase_formula_moles_element(self.phase_name,
                                element_name, per_formula_unit=True).func)

    cpdef FastFunction get_mole_formula_grad(self, unicode element_name):
        return FastFunction(self.phase_record_factory.get_phase_formula_moles_element(self.phase_name,
                                element_name, per_formula_unit=True).grad)

    cpdef FastFunction get_mole_formula_hess(self, unicode element_name):
        return FastFunction(self.phase_record_factory.get_phase_formula_moles_element(self.phase_name,
                                element_name, per_formula_unit=True).hess)

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
            return PhaseRecord, (self.phase_record_factory, self.phase_name)

    def __cinit__(self, object phase_record_factory, str phase_name):
        cdef:
            int el_idx
        self.phase_record_factory = phase_record_factory
        self.components = phase_record_factory.comps
        self.phase_name = phase_name
        self.variables = phase_record_factory.models[phase_name].site_fractions
        self.state_variables = phase_record_factory.state_variables
        self.num_statevars = len(phase_record_factory.state_variables)
        self.pure_elements = phase_record_factory.pure_elements
        self.nonvacant_elements = phase_record_factory.nonvacant_elements
        self.molar_masses = phase_record_factory.molar_masses
        self.parameters = phase_record_factory.param_values
        
        self.phase_dof = len(phase_record_factory.models[phase_name].site_fractions)

        self.function_factory = FastFunctionFactory(phase_record_factory, phase_name)

        self._internal_cons_func = self.function_factory.get_cons_func()
        self._internal_cons_jac = self.function_factory.get_cons_jac()
        self._internal_cons_hess = self.function_factory.get_cons_hess()
        self.num_internal_cons = self.function_factory.get_cons_len()
        self._masses = np.empty(len(self.nonvacant_elements), dtype='object')
        for el_idx, el in enumerate(self.nonvacant_elements):
            self._masses[el_idx] = self.function_factory.get_mole_fraction_func(el)
        self._masses_ptr = <void**> self._masses.data
        self._formulamoles = np.empty(len(self.nonvacant_elements), dtype='object')
        for el_idx, el in enumerate(self.nonvacant_elements):
            self._formulamoles[el_idx] = self.function_factory.get_mole_formula_func(el)
        self._formulamoles_ptr = <void**> self._formulamoles.data
        self._formulamolegrads = np.empty(len(self.nonvacant_elements), dtype='object')
        for el_idx, el in enumerate(self.nonvacant_elements):
            self._formulamolegrads[el_idx] = self.function_factory.get_mole_formula_grad(el)
        self._formulamolegrads_ptr = <void**> self._formulamolegrads.data
        self._formulamolehessians = np.empty(len(self.nonvacant_elements), dtype='object')
        for el_idx, el in enumerate(self.nonvacant_elements):
            self._formulamolehessians[el_idx] = self.function_factory.get_mole_formula_hess(el)
        self._formulamolehessians_ptr = <void**> self._formulamolehessians.data

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void prop(self, double[::1] outp, double[::1] dof, string property_name) nogil except *:
        # dof.shape[0] may be oversized by the caller; do not trust it
        cdef double* dof_concat = alloc_dof_with_parameters(dof[:self.num_statevars+self.phase_dof], self.parameters)
        cdef int num_dof = self.num_statevars + self.phase_dof + self.parameters.shape[0]
        (<FastFunction>self.function_factory.get_func(property_name)).call(&outp[0], &dof_concat[0])
        if self.parameters.shape[0] > 0:
            free(dof_concat)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void prop_2d(self, double[::1] outp, double[:, ::1] dof, string property_name) nogil except *:
        # dof.shape[1] may be oversized by the caller; do not trust it
        cdef double* dof_concat = alloc_dof_with_parameters_vectorized(dof[:, :self.num_statevars+self.phase_dof], self.parameters)
        cdef int i
        cdef int num_inps = dof.shape[0]
        cdef int num_dof = self.num_statevars + self.phase_dof + self.parameters.shape[0]
        for i in range(num_inps):
           (<FastFunction>self.function_factory.get_func(property_name)).call(&outp[i], &dof_concat[i * num_dof])
        if self.parameters.shape[0] > 0:
            free(dof_concat)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void prop_parameters_2d(self, double[:, ::1] outp, double[:, ::1] dof,
                                  double[:, ::1] parameters, string property_name) nogil except *:
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
        cdef size_t dof_offset = self.num_statevars + self.phase_dof
        cdef double* dof_concat = <double *> malloc(num_param_inps * num_dof * sizeof(double))
        for i in range(num_dof_inps):
            # Initialize all parameter arrays with current dof
            for j in range(num_param_inps):
                for dof_idx in range(num_dof-num_params):
                    dof_concat[j * num_dof + dof_idx] = dof[i, dof_idx]
                for param_idx in range(num_params):
                    dof_concat[j * num_dof + dof_offset + param_idx] = parameters[j, param_idx]
            for j in range(num_param_inps):
                (<FastFunction>self.function_factory.get_func(property_name)).call(&outp[i,j], &dof_concat[j * num_dof])
        free(dof_concat)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void prop_grad(self, double[::1] out, double[::1] dof, string property_name) nogil except *:
        # dof.shape[0] may be oversized by the caller; do not trust it
        cdef double* dof_concat = alloc_dof_with_parameters(dof[:self.num_statevars+self.phase_dof], self.parameters)
        (<FastFunction>self.function_factory.get_grad(property_name)).call(&out[0], &dof_concat[0])
        if self.parameters.shape[0] > 0:
            free(dof_concat)

    cpdef void obj(self, double[::1] outp, double[::1] dof) nogil:
        self.prop(outp, dof, <char*>'GM')

    cpdef void formulaobj(self, double[::1] outp, double[::1] dof) nogil:
        self.prop(outp, dof, <char*>'G')

    cpdef void obj_2d(self, double[::1] outp, double[:, ::1] dof) nogil:
        self.prop_2d(outp, dof, <char*>'GM')

    cpdef void obj_parameters_2d(self, double[:, ::1] outp, double[:, ::1] dof, double[:, ::1] parameters) nogil:
        self.prop_parameters_2d(outp, dof, parameters, <char*>'GM')

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void formulagrad(self, double[::1] out, double[::1] dof) nogil:
        # dof.shape[0] may be oversized by the caller; do not trust it
        cdef double* dof_concat = alloc_dof_with_parameters(dof[:self.num_statevars+self.phase_dof], self.parameters)
        (<FastFunction>self.function_factory.get_grad(<char*>'G')).call(&out[0], &dof_concat[0])
        if self.parameters.shape[0] > 0:
            free(dof_concat)


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void formulahess(self, double[:, ::1] out, double[::1] dof) nogil:
        # dof.shape[0] may be oversized by the caller; do not trust it
        cdef double* dof_concat = alloc_dof_with_parameters(dof[:self.num_statevars+self.phase_dof], self.parameters)
        (<FastFunction>self.function_factory.get_hess(<char*>'G')).call(&out[0,0], &dof_concat[0])
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
    cpdef void formulamole_obj(self, double[::1] out, double[::1] dof, int comp_idx) nogil:
        # dof.shape[0] may be oversized by the caller; do not trust it
        cdef double* dof_concat = alloc_dof_with_parameters(dof[:self.num_statevars+self.phase_dof], self.parameters)
        (<FastFunction>self._formulamoles_ptr[comp_idx]).call(&out[0], &dof_concat[0])
        if self.parameters.shape[0] > 0:
            free(dof_concat)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void formulamole_grad(self, double[::1] out, double[::1] dof, int comp_idx) nogil:
        # dof.shape[0] may be oversized by the caller; do not trust it
        cdef double* dof_concat = alloc_dof_with_parameters(dof[:self.num_statevars+self.phase_dof], self.parameters)
        (<FastFunction>self._formulamolegrads_ptr[comp_idx]).call(&out[0], &dof_concat[0])
        if self.parameters.shape[0] > 0:
            free(dof_concat)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void formulamole_hess(self, double[:,::1] out, double[::1] dof, int comp_idx) nogil:
        # dof.shape[0] may be oversized by the caller; do not trust it
        cdef double* dof_concat = alloc_dof_with_parameters(dof[:self.num_statevars+self.phase_dof], self.parameters)
        (<FastFunction>self._formulamolehessians_ptr[comp_idx]).call(&out[0,0], &dof_concat[0])
        if self.parameters.shape[0] > 0:
            free(dof_concat)

