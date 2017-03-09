cimport cython
import numpy as np
cimport numpy as np
from cpython cimport (PY_VERSION_HEX, PyCObject_Check,
    PyCObject_AsVoidPtr, PyCapsule_CheckExact, PyCapsule_GetPointer, Py_INCREF, Py_DECREF)
import pycalphad.variables as v

# From https://gist.github.com/pv/5437087
cdef void* f2py_pointer(obj):
    if PY_VERSION_HEX < 0x03000000:
        if (PyCObject_Check(obj)):
            return PyCObject_AsVoidPtr(obj)
    elif PY_VERSION_HEX >= 0x02070000:
        if (PyCapsule_CheckExact(obj)):
            return PyCapsule_GetPointer(obj, NULL);
    raise ValueError("Not an object containing a void ptr")


cdef public class PhaseRecord(object)[type PhaseRecordType, object PhaseRecordObject]:
    def __cinit__(self, object comps, object variables, double[::1] num_sites, double[::1] parameters, object ofunc, object gfunc,
                  object hfunc):
        cdef:
            int var_idx, subl_index
        # XXX: Doesn't refcounting need to happen here to keep the codegen objects from disappearing?
        self.variables = variables
        self.phase_dof = 0
        self.sublattice_dof = np.zeros(num_sites.shape[0], dtype=np.int32)
        self.parameters = parameters
        self.num_sites = num_sites
        # In the future, this should be bigger than num_sites.shape[0] to allow for multiple species
        # of the same type in the same sublattice for, e.g., same species with different charges
        self.composition_matrices = np.full((len(comps), num_sites.shape[0], 2), -1.)
        if 'VA' in comps:
            self.vacancy_index = comps.index('VA')
        else:
            self.vacancy_index = -1
        var_idx = 0
        for variable in variables:
            if not isinstance(variable, v.SiteFraction):
                continue
            subl_index = variable.sublattice_index
            species = variable.species
            comp_index = comps.index(species)
            self.composition_matrices[comp_index, subl_index, 0] = num_sites[subl_index]
            self.composition_matrices[comp_index, subl_index, 1] = var_idx
            self.sublattice_dof[subl_index] += 1
            var_idx += 1
            self.phase_dof += 1
        # Trigger lazy computation
        if ofunc is not None:
            ofunc.kernel
            self._obj = <func_t*> f2py_pointer(ofunc._kernel._cpointer)
        if gfunc is not None:
            gfunc.kernel
            self._grad = <func_novec_t*> f2py_pointer(gfunc._kernel._cpointer)
        if hfunc is not None:
            hfunc.kernel
            self._hess = <func_novec_t*> f2py_pointer(hfunc._kernel._cpointer)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void obj(self, double[::1] out, double[::1,:] dof, int bounds) nogil:
        self._obj(&out[0], &dof[0,0], &self.parameters[0], &bounds)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void grad(self, double[::1] out, double[::1] dof) nogil:
        self._grad(&dof[0], &self.parameters[0], &out[0])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void hess(self, double[::1,:] out, double[::1] dof) nogil:
        self._hess(&dof[0], &self.parameters[0], &out[0,0])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void mass_obj(self, double[::1] out, double[::1] dof, int comp_idx) nogil:
        cdef double mass_normalization_factor = 0
        out[0] = 0
        for entry_idx in range(self.num_sites.shape[0]):
            if self.composition_matrices[comp_idx, entry_idx, 1] > -1:
                out[0] += self.composition_matrices[comp_idx, entry_idx, 0] * dof[<int>self.composition_matrices[comp_idx, entry_idx, 1]]
        for subl_idx in range(self.num_sites.shape[0]):
            if (self.vacancy_index > -1) and self.composition_matrices[self.vacancy_index, subl_idx, 1] > -1:
                mass_normalization_factor += self.num_sites[subl_idx] * (1-dof[<int>self.composition_matrices[self.vacancy_index, subl_idx, 1]])
            else:
                mass_normalization_factor += self.num_sites[subl_idx]
        if mass_normalization_factor != 0:
            out[0] /= mass_normalization_factor
        else:
            out[0] = 0

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void mass_grad(self, double[::1] out, double[::1] dof, int comp_idx) nogil:
        cdef double mass_normalization_factor = 0
        cdef double mass = 0
        cdef double site_count
        cdef int grad_idx
        for subl_idx in range(self.num_sites.shape[0]):
            if self.composition_matrices[comp_idx, subl_idx, 1] > -1:
                mass += self.num_sites[subl_idx] * dof[<int>self.composition_matrices[comp_idx, subl_idx, 1]]
            if self.vacancy_index > -1 and self.composition_matrices[self.vacancy_index, subl_idx, 1] > -1:
                mass_normalization_factor += self.num_sites[subl_idx] * (1-dof[<int>self.composition_matrices[self.vacancy_index, subl_idx, 1]])
            else:
                mass_normalization_factor += self.num_sites[subl_idx]
        if mass == 0 or mass_normalization_factor == 0:
            return
        if comp_idx != self.vacancy_index:
            for subl_idx in range(self.composition_matrices.shape[1]):
                grad_idx = <int>self.composition_matrices[comp_idx, subl_idx, 1]
                if grad_idx > -1:
                    out[grad_idx] = self.composition_matrices[comp_idx, subl_idx, 0] / mass_normalization_factor
            if self.vacancy_index > -1:
                for subl_idx in range(self.composition_matrices.shape[1]):
                    grad_idx = <int>self.composition_matrices[self.vacancy_index, subl_idx, 1]
                    if grad_idx > -1:
                        out[grad_idx] = (mass * self.composition_matrices[self.vacancy_index, subl_idx, 0]) / (mass_normalization_factor **  2)
        else:
            for subl_idx in range(self.composition_matrices.shape[1]):
                grad_idx = <int>self.composition_matrices[comp_idx, subl_idx, 1]
                site_count = self.composition_matrices[comp_idx, subl_idx, 0]
                if grad_idx > -1:
                    out[grad_idx] = (site_count * mass_normalization_factor + (site_count ** 2) * dof[grad_idx]) / (mass_normalization_factor ** 2)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void mass_hess(self, double[::1,:] out, double[::1] dof, int comp_idx) nogil:
        cdef double mass_normalization_factor = 0
        cdef double mass = 0
        cdef int hess_x_idx, hess_y_comp_idx, hess_y_idx, subl_x_idx, subl_y_idx, subl_idx
        if self.vacancy_index == -1:
            return
        if comp_idx == self.vacancy_index:
            out[:] = -1e100
            return
        for subl_idx in range(self.num_sites.shape[0]):
            if self.composition_matrices[comp_idx, subl_idx, 1] > -1:
                mass += self.num_sites[subl_idx] * dof[<int>self.composition_matrices[comp_idx, subl_idx, 1]]
            if self.vacancy_index > -1 and self.composition_matrices[self.vacancy_index, subl_idx, 1] > -1:
                mass_normalization_factor += self.num_sites[subl_idx] * (1-dof[<int>self.composition_matrices[self.vacancy_index, subl_idx, 1]])
            else:
                mass_normalization_factor += self.num_sites[subl_idx]
        if mass == 0 or mass_normalization_factor == 0:
            return
        for subl_x_idx in range(self.composition_matrices.shape[1]):
            hess_x_idx = <int>self.composition_matrices[self.vacancy_index, subl_x_idx, 1]
            if hess_x_idx > -1:
                for subl_y_idx in range(self.composition_matrices.shape[1]):
                    hess_y_idx = <int>self.composition_matrices[self.vacancy_index, subl_y_idx, 1]
                    hess_y_comp_idx = <int>self.composition_matrices[comp_idx, subl_y_idx, 1]
                    if hess_y_idx > -1:
                        out[hess_x_idx, hess_y_idx] = out[hess_y_idx, hess_x_idx] = 2 * mass * (self.num_sites[subl_x_idx] * self.num_sites[subl_y_idx]) / (mass_normalization_factor**3)
                    if hess_y_comp_idx > -1:
                        out[hess_x_idx, hess_y_comp_idx] = out[hess_y_comp_idx, hess_x_idx] = (self.num_sites[subl_x_idx] * self.num_sites[subl_y_idx]) / mass_normalization_factor**2
