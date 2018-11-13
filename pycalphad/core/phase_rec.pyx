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
            return PhaseRecord_from_cython_pickle, (self.variables, self.phase_dof,
                                                    np.array(self.sublattice_dof, dtype=np.int32),
                                                    np.array(self.parameters), np.array(self.num_sites),
                                                    np.array(self.composition_matrices),
                                                    self.vacancy_index, self._ofunc, self._gfunc,
                                                    self._massfuncs, self._massgradfuncs)

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
    cpdef void mass_obj(self, double[::1] out, double[:, ::1] dof, int comp_idx) nogil:
        if self._masses != NULL:
            self._masses[comp_idx](&out[0], &dof[0,0], &self.parameters[0], <int>out.shape[0])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void mass_grad(self, double[::1] out, double[::1] dof, int comp_idx) nogil:
        if self._massgrads != NULL:
            self._massgrads[comp_idx](&dof[0], &self.parameters[0], &out[0])

cpdef PhaseRecord PhaseRecord_from_cython(object comps, object variables, double[::1] num_sites, double[::1] parameters,
                                          object ofunc, object gfunc, object massfuncs, object massgradfuncs):
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
                                 vacancy_index, ofunc, gfunc, massfuncs, massgradfuncs):
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
    if massfuncs is not None:
        inst._massfuncs = massfuncs
        inst._masses = <func_t**>PyMem_Malloc(len(massfuncs) * sizeof(func_t*))
        for el_idx in range(len(massfuncs)):
            massfuncs[el_idx].kernel
            inst._masses[el_idx] = <func_t*> cython_pointer(massfuncs[el_idx]._cpointer)
    if massgradfuncs is not None:
        inst._massgradfuncs = massgradfuncs
        inst._massgrads = <func_novec_t**>PyMem_Malloc(len(massgradfuncs) * sizeof(func_novec_t*))
        for el_idx in range(len(massgradfuncs)):
            massgradfuncs[el_idx].kernel
            inst._massgrads[el_idx] = <func_novec_t*> cython_pointer(massgradfuncs[el_idx]._cpointer)
    return inst
