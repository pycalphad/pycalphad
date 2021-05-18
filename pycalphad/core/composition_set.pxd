# distutils: language = c++
from pycalphad.core.phase_rec cimport PhaseRecord

cdef public class CompositionSet(object)[type CompositionSetType, object CompositionSetObject]:
    cdef public PhaseRecord phase_record
    cdef readonly double[::1] dof, X
    cdef double[:,::1] _X_2d_view
    cdef public double NP
    cdef public bint fixed
    cdef readonly double energy
    cdef double[::1] _energy_2d_view
    cpdef void update(self, double[::1] site_fracs, double phase_amt, double[::1] state_variables)
