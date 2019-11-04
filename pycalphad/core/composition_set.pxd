# distutils: language = c++
from pycalphad.core.phase_rec cimport PhaseRecord

cdef public class CompositionSet(object)[type CompositionSetType, object CompositionSetObject]:
    cdef public PhaseRecord phase_record
    cdef readonly double[::1] dof, X
    cdef double[:,::1] _X_2d_view
    cdef public double NP
    cdef public int zero_seen
    cdef readonly double energy
    cdef double[::1] _energy_2d_view
    cdef readonly double[::1] grad
    cdef readonly double[::1] _prev_dof
    cdef readonly double _prev_energy
    cdef readonly double[::1] _prev_grad
    cdef readonly bint _first_iteration
    cdef void reset(self)
    cdef void update(self, double[::1] site_fracs, double phase_amt, double[::1] state_variables, bint skip_derivatives) nogil
