cimport cython
import numpy as np
cimport numpy as np
from cpython cimport (PY_VERSION_HEX, PyCObject_Check,
    PyCObject_AsVoidPtr, PyCapsule_CheckExact, PyCapsule_GetPointer, Py_INCREF, Py_DECREF)

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
    def __cinit__(self, object variables, double[::1] parameters, object ofunc, object gfunc,
                  object hfunc, object mofunc, object mgfunc, object mhfunc):
        # XXX: Doesn't refcounting need to happen here to keep the codegen objects from disappearing?
        self.variables = variables
        self.parameters = parameters
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
        if mofunc is not None:
            mofunc.kernel
            self._mass_obj = <func_simple_t*> f2py_pointer(mofunc._kernel._cpointer)
        if mgfunc is not None:
            mgfunc.kernel
            self._mass_grad = <func_novec_simple_t*> f2py_pointer(mgfunc._kernel._cpointer)
        if mhfunc is not None:
            mhfunc.kernel
            self._mass_hess = <func_novec_simple_t*> f2py_pointer(mhfunc._kernel._cpointer)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void obj(PhaseRecord prx, double[::1] out, double[::1,:] dof, int bounds) nogil:
    prx._obj(&out[0], &dof[0,0], &prx.parameters[0], &bounds)

def obj_python(PhaseRecord prx, double[::1] out, double[::1,:] dof):
    obj(prx, out, dof, dof.shape[0])

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void grad(PhaseRecord prx, double[::1] out, double[::1] dof) nogil:
    prx._grad(&dof[0], &prx.parameters[0], &out[0])

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void hess(PhaseRecord prx, double[::1,:] out, double[::1] dof) nogil:
    prx._hess(&dof[0], &prx.parameters[0], &out[0,0])

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void mass_obj(PhaseRecord prx, double[::1] out, double[::1,:] dof, int bounds) nogil:
    prx._mass_obj(&out[0], &dof[0,0], &prx.parameters[0], &bounds)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void mass_grad(PhaseRecord prx, double[::1] out, double[::1] dof) nogil:
    prx._mass_grad(&dof[0], &prx.parameters[0], &out[0])

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void mass_hess(PhaseRecord prx, double[::1,:] out, double[::1] dof) nogil:
    prx._mass_hess(&dof[0], &prx.parameters[0], &out[0,0])