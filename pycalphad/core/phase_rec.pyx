cimport cython
import numpy as np
cimport numpy as np
from cpython cimport (PY_VERSION_HEX, PyCObject_Check,
    PyCObject_AsVoidPtr, PyCapsule_CheckExact, PyCapsule_GetPointer, Py_INCREF, Py_DECREF)
from pycalphad.core.compiled_model cimport CompiledModel
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
    def __reduce__(self):
        if self.cmpmdl is not None:
            return PhaseRecord_from_compiledmodel, (self.cmpmdl, np.asarray(self.parameters))
        else:
            return PhaseRecord_from_f2py_pickle, (self.variables, self.phase_dof, self.sublattice_dof,
                                                  self.parameters, self.num_sites, self.composition_matrices,
                                                  self.vacancy_index, self._ofunc, self._gfunc, self._hfunc)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void obj(self, double[::1] out, double[::1,:] dof) nogil:
        if self._obj != NULL:
            self._obj(&out[0], &dof[0,0], &self.parameters[0], <int*>&out.shape[0])
        else:
            with gil:
                self.cmpmdl.eval_energy(out, dof, self.parameters)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void grad(self, double[::1] out, double[::1] dof) nogil:
        if self._grad != NULL:
            self._grad(&dof[0], &self.parameters[0], &out[0])
        else:
            with gil:
                self.cmpmdl.eval_energy_gradient(out, dof, self.parameters)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void hess(self, double[::1,:] out, double[::1] dof) nogil:
        cdef double[::1] x1,x2
        cdef double[::1] grad1, grad2
        cdef double[::1,:] debugout
        cdef double epsilon = 1e-12
        cdef int grad_idx
        cdef int col_idx
        cdef int total_dof = 2 + self.phase_dof
        if self._hess != NULL:
            self._hess(&dof[0], &self.parameters[0], &out[0,0])
        else:
            with gil:
                grad1 = np.zeros(total_dof)
                grad2 = np.zeros(total_dof)
                x1 = np.array(dof)
                x2 = np.array(dof)

                for grad_idx in range(total_dof):
                    if grad_idx > 1:
                        x1[grad_idx] = max(x1[grad_idx] - epsilon, 1e-12)
                        x2[grad_idx] = min(x2[grad_idx] + epsilon, 1)
                    else:
                        x1[grad_idx] = x1[grad_idx] - 1e6 * epsilon
                        x2[grad_idx] = x2[grad_idx] + 1e6 * epsilon
                    self.cmpmdl.eval_energy_gradient(grad1, x1, self.parameters)
                    self.cmpmdl.eval_energy_gradient(grad2, x2, self.parameters)
                    for col_idx in range(total_dof):
                        out[col_idx,grad_idx] = (grad2[col_idx]-grad1[col_idx])/(x2[grad_idx] - x1[grad_idx])
                    x1[grad_idx] = dof[grad_idx]
                    x2[grad_idx] = dof[grad_idx]
                    grad1[:] = 0
                    grad2[:] = 0
                for grad_idx in range(total_dof):
                    for col_idx in range(grad_idx, total_dof):
                        out[grad_idx,col_idx] = out[col_idx,grad_idx] = (out[grad_idx,col_idx]+out[col_idx,grad_idx])/2
                if self.cmpmdl._debug:
                    debugout = np.asfortranarray(np.zeros_like(out))
                    if self.parameters.shape[0] == 0:
                        self.cmpmdl._debughess(&dof[0], NULL, &debugout[0,0])
                    else:
                        self.cmpmdl._debughess(&dof[0], &self.parameters[0], &debugout[0,0])
                    try:
                        np.testing.assert_allclose(out,debugout)
                    except AssertionError as e:
                        print('--')
                        print('Hessian mismatch')
                        print(e)
                        print(np.array(debugout)-np.array(out))
                        print('DOF:', np.array(dof))
                        print(self.cmpmdl.constituents)
                        print('--')

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

# cdef classmethods are not yet supported, otherwise we would use that
# it's not a big deal since we declare PhaseRecord final to allow cpdef nogil functions
cpdef PhaseRecord PhaseRecord_from_compiledmodel(CompiledModel cmpmdl, double[::1] parameters):
    cdef PhaseRecord inst
    inst = PhaseRecord()
    inst.cmpmdl = cmpmdl
    inst.variables = cmpmdl.variables
    inst.sublattice_dof = cmpmdl.sublattice_dof
    inst.phase_dof = sum(cmpmdl.sublattice_dof)
    inst.parameters = parameters
    inst.num_sites = cmpmdl.site_ratios
    inst.composition_matrices = cmpmdl.composition_matrices
    inst.vacancy_index = cmpmdl.vacancy_index
    inst._obj = NULL
    inst._grad = NULL
    inst._hess = NULL
    return inst

cpdef PhaseRecord PhaseRecord_from_f2py(object comps, object variables, double[::1] num_sites, double[::1] parameters,
              object ofunc, object gfunc, object hfunc):
    cdef:
        int var_idx, subl_index
        PhaseRecord inst
    inst = PhaseRecord()
    # XXX: Doesn't refcounting need to happen here to keep the codegen objects from disappearing?
    inst.variables = variables
    inst.phase_dof = 0
    inst.sublattice_dof = np.zeros(num_sites.shape[0], dtype=np.int32)
    inst.parameters = parameters
    inst.num_sites = num_sites
    # In the future, this should be bigger than num_sites.shape[0] to allow for multiple species
    # of the same type in the same sublattice for, e.g., same species with different charges
    inst.composition_matrices = np.full((len(comps), num_sites.shape[0], 2), -1.)
    if 'VA' in comps:
        inst.vacancy_index = comps.index('VA')
    else:
        inst.vacancy_index = -1
    var_idx = 0
    for variable in variables:
        if not isinstance(variable, v.SiteFraction):
            continue
        subl_index = variable.sublattice_index
        species = variable.species
        comp_index = comps.index(species)
        inst.composition_matrices[comp_index, subl_index, 0] = num_sites[subl_index]
        inst.composition_matrices[comp_index, subl_index, 1] = var_idx
        inst.sublattice_dof[subl_index] += 1
        var_idx += 1
        inst.phase_dof += 1
    # Trigger lazy computation
    if ofunc is not None:
        inst._ofunc = ofunc
        ofunc.kernel
        inst._obj = <func_t*> f2py_pointer(ofunc._kernel._cpointer)
    if gfunc is not None:
        inst._gfunc = gfunc
        gfunc.kernel
        inst._grad = <func_novec_t*> f2py_pointer(gfunc._kernel._cpointer)
    if hfunc is not None:
        inst._hfunc = hfunc
        hfunc.kernel
        inst._hess = <func_novec_t*> f2py_pointer(hfunc._kernel._cpointer)
    return inst

def PhaseRecord_from_f2py_pickle(variables, phase_dof, sublattice_dof, parameters, num_sites, composition_matrices,
                                 vacancy_index, ofunc, gfunc, hfunc):
    inst = PhaseRecord()
    # XXX: Doesn't refcounting need to happen here to keep the codegen objects from disappearing?
    inst.variables = variables
    inst.phase_dof = 0
    inst.sublattice_dof = sublattice_dof
    inst.parameters = parameters
    inst.num_sites = num_sites
    inst.composition_matrices = composition_matrices
    inst.vacancy_index = vacancy_index
    inst.phase_dof = phase_dof
    # Trigger lazy computation
    if ofunc is not None:
        ofunc.kernel
        inst._obj = <func_t*> f2py_pointer(ofunc._kernel._cpointer)
    if gfunc is not None:
        gfunc.kernel
        inst._grad = <func_novec_t*> f2py_pointer(gfunc._kernel._cpointer)
    if hfunc is not None:
        hfunc.kernel
        inst._hess = <func_novec_t*> f2py_pointer(hfunc._kernel._cpointer)
    return inst
