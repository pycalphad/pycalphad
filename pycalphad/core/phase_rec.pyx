cimport cython
from cpython.mem cimport PyMem_Malloc, PyMem_Free
import numpy as np
cimport numpy as np
from cpython cimport PyCapsule_CheckExact, PyCapsule_GetPointer
import pycalphad.variables as v

# From https://gist.github.com/pv/5437087
cdef void* cython_pointer(obj):
    if PyCapsule_CheckExact(obj):
        return PyCapsule_GetPointer(obj, NULL);
    raise ValueError("Not an object containing a void ptr")


cdef public class PhaseRecord(object)[type PhaseRecordType, object PhaseRecordObject]:
    """
    This object exposes a common API to the solver so it doesn't need to know about the differences
    between Model and CompiledModel. Each PhaseRecord holds a reference to its own Model or CompiledModel;
    these objects are pickleable. PhaseRecords are immutable after initialization.
    """
    def __reduce__(self):
            return PhaseRecord_from_cython_pickle, (self.variables, self.phase_dof, self.sublattice_dof,
                                                  self.parameters, self.num_sites, self.composition_matrices,
                                                  self.vacancy_index, self._ofunc, self._gfunc, self._hfunc)

    def __dealloc__(self):
        PyMem_Free(self._masses)
        PyMem_Free(self._massgrads)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void obj(self, double[::1] out, double[:,::1] dof) nogil:
        if self._obj != NULL:
            self._obj(&out[0], &dof[0,0], &self.parameters[0], <int>out.shape[0])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void grad(self, double[::1] out, double[::1] dof) nogil:
        if self._grad != NULL:
            self._grad(&dof[0], &self.parameters[0], &out[0])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void hess(self, double[:,::1] out, double[::1] dof) nogil:
        if self._hess != NULL:
            self._hess(&dof[0], &self.parameters[0], &out[0,0])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void mass_obj(self, double[::1] out, double[:, ::1] dof, int comp_idx) nogil:
        if self._masses != NULL:
            self._masses[comp_idx](&out[0], &dof[0,0], &self.parameters[0], <int>out.shape[0])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void mass_grad(self, double[::1] out, double[::1] dof, int comp_idx) nogil:
        if self._massgrads != NULL:
            self._massgrads[comp_idx](&dof[0], &self.parameters[0], &out[0])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void mass_hess(self, double[:,:] out, double[::1] dof, int comp_idx) nogil:
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


cpdef PhaseRecord PhaseRecord_from_cython(object comps, object variables, double[::1] num_sites, double[::1] parameters,
              object ofunc, object gfunc, object hfunc, object massfuncs, object massgradfuncs):
    cdef:
        int var_idx, subl_index, el_idx
        PhaseRecord inst
    desired_active_pure_elements = [list(x.constituents.keys()) for x in comps]
    desired_active_pure_elements = [el.upper() for constituents in desired_active_pure_elements for el in constituents]
    pure_elements = sorted(set(desired_active_pure_elements))
    nonvacant_elements = sorted([x for x in set(desired_active_pure_elements) if x != 'VA'])
    inst = PhaseRecord()
    # XXX: Missing inst.phase_name
    # XXX: Doesn't refcounting need to happen here to keep the codegen objects from disappearing?
    inst.variables = variables
    inst.phase_dof = 0
    inst.sublattice_dof = np.zeros(num_sites.shape[0], dtype=np.int32)
    inst.parameters = parameters
    inst.num_sites = num_sites
    inst.composition_matrices = np.full((len(pure_elements), num_sites.shape[0], 2), -1.)
    if 'VA' in pure_elements:
        inst.vacancy_index = pure_elements.index('VA')
    else:
        inst.vacancy_index = -1
    var_idx = 0
    for variable in variables:
        if not isinstance(variable, v.SiteFraction):
            continue
        inst.phase_name = <unicode>variable.phase_name
        subl_index = variable.sublattice_index
        inst.sublattice_dof[subl_index] += 1
        var_idx += 1
        inst.phase_dof += 1
    # Trigger lazy computation
    if ofunc is not None:
        inst._ofunc = ofunc
        ofunc.kernel
        inst._obj = <func_t*> cython_pointer(ofunc._cpointer)
    if gfunc is not None:
        inst._gfunc = gfunc
        gfunc.kernel
        inst._grad = <func_novec_t*> cython_pointer(gfunc._cpointer)
    if hfunc is not None:
        inst._hfunc = hfunc
        hfunc.kernel
        inst._hess = <func_novec_t*> cython_pointer(hfunc._cpointer)
    if massfuncs is not None:
        inst._massfuncs = massfuncs
        inst._masses = <func_t**>PyMem_Malloc(len(pure_elements) * sizeof(func_t*))
        for el_idx in range(len(nonvacant_elements)):
            massfuncs[el_idx].kernel
            inst._masses[el_idx] = <func_t*> cython_pointer(massfuncs[el_idx]._cpointer)
    if massgradfuncs is not None:
        inst._massgradfuncs = massgradfuncs
        inst._massgrads = <func_novec_t**>PyMem_Malloc(len(nonvacant_elements) * sizeof(func_novec_t*))
        for el_idx in range(len(nonvacant_elements)):
            massgradfuncs[el_idx].kernel
            inst._massgrads[el_idx] = <func_novec_t*> cython_pointer(massgradfuncs[el_idx]._cpointer)
    return inst

def PhaseRecord_from_cython_pickle(variables, phase_dof, sublattice_dof, parameters, num_sites, composition_matrices,
                                 vacancy_index, ofunc, gfunc, hfunc):
    inst = PhaseRecord()
    # XXX: Missing inst.phase_name
    # XXX: Doesn't refcounting need to happen here to keep the codegen objects from disappearing?
    inst.variables = variables
    for variable in variables:
        if not isinstance(variable, v.SiteFraction):
            continue
        inst.phase_name = <unicode>variable.phase_name
        break
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
        inst._obj = <func_t*> cython_pointer(ofunc._cpointer)
    if gfunc is not None:
        gfunc.kernel
        inst._grad = <func_novec_t*> cython_pointer(gfunc._cpointer)
    if hfunc is not None:
        hfunc.kernel
        inst._hess = <func_novec_t*> cython_pointer(hfunc._cpointer)
    return inst
