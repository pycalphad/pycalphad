# distutils: language = c++
from pycalphad.core.phase_rec cimport PhaseRecord, FastFunction

cdef public class CompositionSet(object)[type CompositionSetType, object CompositionSetObject]:
    cdef public PhaseRecord phase_record
    cdef readonly double[::1] dof, X
    cdef double[:,::1] _X_2d_view
    cdef public double NP
    cdef public bint fixed
    cdef readonly double energy
    cdef double[::1] _energy_2d_view
    cdef FastFunction phase_local_cons_func
    cdef FastFunction phase_local_cons_jac
    cdef public int num_phase_local_conditions
    cpdef void set_local_conditions(self, dict phase_local_conditions)
    cpdef void update(self, double[::1] site_fracs, double phase_amt, double[::1] state_variables)
