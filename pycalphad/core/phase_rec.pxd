# distutils: language = c++

cimport cython
from libcpp.map cimport map
from libcpp.utility cimport pair
from libcpp.string cimport string
import numpy
cimport numpy

ctypedef void (*math_function_t)(double*, const double*, void* user_data) nogil

cdef class FastFunction:
    cdef readonly object _objref
    cdef math_function_t f_ptr
    cdef void *func_data
    cdef void call(self, double *out, double *inp) nogil

cdef class FastFunctionFactory:
    cdef object phase_record_factory
    cdef unicode phase_name
    cdef numpy.ndarray _cache
    cdef map[pair[string, string], int] _cache_property_map
    cdef int _cache_cur_idx
    cdef void** _cache_ptr
    cdef void* get_func(self, string property_name) except * nogil
    cdef void* get_grad(self, string property_name) except * nogil
    cdef void* get_hess(self, string property_name) except * nogil
    cpdef FastFunction get_cons_func(self)
    cpdef FastFunction get_cons_jac(self)
    cpdef FastFunction get_cons_hess(self)
    cpdef int get_cons_len(self)
    cpdef FastFunction get_mole_fraction_func(self, unicode element_name)
    cpdef FastFunction get_mole_fraction_grad(self, unicode element_name)
    cpdef FastFunction get_mole_fraction_hess(self, unicode element_name)
    cpdef FastFunction get_mole_formula_func(self, unicode element_name)
    cpdef FastFunction get_mole_formula_grad(self, unicode element_name)
    cpdef FastFunction get_mole_formula_hess(self, unicode element_name)

@cython.final
cdef public class PhaseRecord(object)[type PhaseRecordType, object PhaseRecordObject]:
    cdef FastFunction _obj
    cdef FastFunction _formulaobj
    cdef FastFunction _formulagrad
    cdef FastFunction _formulahess
    cdef FastFunction _internal_cons_func
    cdef FastFunction _internal_cons_jac
    cdef FastFunction _internal_cons_hess
    cdef numpy.ndarray _masses
    cdef void** _masses_ptr
    cdef numpy.ndarray _formulamoles
    cdef void** _formulamoles_ptr
    cdef numpy.ndarray _formulamolegrads
    cdef void** _formulamolegrads_ptr
    cdef numpy.ndarray _formulamolehessians
    cdef void** _formulamolehessians_ptr
    cdef public size_t num_internal_cons
    cdef public object phase_record_factory
    cdef public FastFunctionFactory function_factory
    cdef public object variables
    cdef public object state_variables
    cdef public object components
    cdef public object pure_elements
    cdef public object nonvacant_elements
    cdef public double[::1] molar_masses
    cdef public double[::1] parameters
    cdef public int phase_dof
    cdef public int num_statevars
    cdef public unicode phase_name 
    cpdef void prop(self, double[::1] out, double[::1] dof, string property_name) except * nogil
    cpdef void prop_2d(self, double[::1] out, double[:, ::1] dof, string property_name) except * nogil
    cpdef void prop_parameters_2d(self, double[:, ::1] out, double[:, ::1] dof,
                                  double[:, ::1] parameters, string property_name) except * nogil
    cpdef void prop_grad(self, double[::1] out, double[::1] dof, string property_name) except * nogil
    cpdef void obj(self, double[::1] out, double[::1] dof) nogil
    cpdef void formulaobj(self, double[::1] out, double[::1] dof) nogil
    cpdef void obj_2d(self, double[::1] out, double[:, ::1] dof) nogil
    cpdef void obj_parameters_2d(self, double[:, ::1] out, double[:, ::1] dof, double[:, ::1] parameters) nogil
    cpdef void formulagrad(self, double[::1] out, double[::1] dof) nogil
    cpdef void formulahess(self, double[:,::1] out, double[::1] dof) nogil
    cpdef void internal_cons_func(self, double[::1] out, double[::1] dof) nogil
    cpdef void internal_cons_jac(self, double[:,::1] out, double[::1] dof) nogil
    cpdef void internal_cons_hess(self, double[:,:,::1] out, double[::1] dof) nogil
    cpdef void mass_obj(self, double[::1] out, double[::1] dof, int comp_idx) nogil
    cpdef void mass_obj_2d(self, double[::1] out, double[:, ::1] dof, int comp_idx) nogil
    cpdef void formulamole_obj(self, double[::1] out, double[::1] dof, int comp_idx) nogil
    cpdef void formulamole_grad(self, double[::1] out, double[::1] dof, int comp_idx) nogil
    cpdef void formulamole_hess(self, double[:,::1] out, double[::1] dof, int comp_idx) nogil

