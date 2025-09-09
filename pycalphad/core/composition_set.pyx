# distutils: language = c++
from pycalphad.core.phase_rec cimport PhaseRecord, FastFunction
cimport numpy as np
import numpy as np
from libc.string cimport memset
cimport cython
from pycalphad.core.constraints import build_phase_local_constraints

cdef public class CompositionSet(object)[type CompositionSetType, object CompositionSetObject]:
    """
    This is the primary object the solver interacts with. It keeps the state of a phase (P, T, y...) at a
    particular solver iteration and can be updated using the update() member function. Every CompositionSet
    has a reference to a particular PhaseRecord which describes the prototype of the phase. These objects
    can be created and destroyed by the solver as needed to describe the stable set of phases. Multiple
    CompositionSets can point to the same PhaseRecord for the case of miscibility gaps. CompositionSets are
    not pickleable. They are used in miscibility gap deteciton.
    """
    def __cinit__(self, PhaseRecord prx):
        self.phase_record = prx
        self.dof = np.zeros(len(self.phase_record.variables)+len(self.phase_record.state_variables))
        self.X = np.zeros(len(self.phase_record.nonvacant_elements))
        self._X_2d_view = <double[:self.X.shape[0],:1]>&self.X[0]
        self.energy = 0
        self.NP = 0
        self.fixed = False
        self._energy_2d_view = <double[:1]>&self.energy

    def __deepcopy__(self, memodict=None):
        cdef CompositionSet other
        memodict = {} if memodict is None else memodict
        other = CompositionSet(self.phase_record)
        other.phase_record = self.phase_record
        other.dof[:] = self.dof
        other.X[:] = self.X
        other.energy = 1.0*self.energy
        other._energy_2d_view = <double[:1]>&other.energy
        other.NP = 1.0*self.NP
        other.fixed = bool(self.fixed)
        other.phase_local_cons_func = self.phase_local_cons_func
        other.phase_local_cons_jac = self.phase_local_cons_jac
        other.num_phase_local_conditions = int(self.num_phase_local_conditions)
        return other

    def __repr__(self):
        return str(self.__class__.__name__) + "({0}, {1}, NP={2}, GM={3})".format(self.phase_record.phase_name,
                                                                          np.asarray(self.X), self.NP, self.energy)

    cpdef void set_local_conditions(self, dict phase_local_conditions):
        mod = self.phase_record.phase_record_factory.models[self.phase_record.phase_name]
        cfuncs = build_phase_local_constraints(mod, self.phase_record.state_variables + self.phase_record.variables,
                                               phase_local_conditions,
                                               parameters=self.phase_record.phase_record_factory.param_symbols)
        self.phase_local_cons_func = FastFunction(cfuncs.internal_cons_func)
        self.phase_local_cons_jac = FastFunction(cfuncs.internal_cons_jac)
        self.num_phase_local_conditions = cfuncs.num_internal_cons

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void update(self, double[::1] site_fracs, double phase_amt, double[::1] state_variables):
        cdef int comp_idx
        for comp_idx in range(state_variables.shape[0]):
            self.dof[comp_idx] = state_variables[comp_idx]
        for comp_idx in range(site_fracs.shape[0]):
            self.dof[state_variables.shape[0] + comp_idx] = site_fracs[comp_idx]
        self.NP = phase_amt
        self.energy = 0
        memset(&self.X[0], 0, self.X.shape[0] * sizeof(double))
        self.phase_record.obj(self._energy_2d_view, self.dof)
        for comp_idx in range(self.X.shape[0]):
            self.phase_record.mass_obj(self._X_2d_view[comp_idx], self.dof, comp_idx)
