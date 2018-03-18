from pycalphad.core.phase_rec cimport PhaseRecord

cdef public class CompositionSet(object)[type CompositionSetType, object CompositionSetObject]:
    cdef public PhaseRecord phase_record
    cdef readonly double[::1] dof, X
    cdef double[:,::1] _dof_2d_view
    cdef double[:,::1] _X_2d_view
    cdef public double NP
    cdef public int zero_seen
    cdef readonly double energy
    cdef double[::1] _energy_2d_view
    cdef readonly double[::1] grad
    cdef readonly double[:,::1] hess
    cdef readonly double[::1] _prev_dof
    cdef readonly double _prev_energy
    cdef readonly double[::1] _prev_grad
    cdef readonly double[:,::1] _prev_hess
    cdef readonly bint _first_iteration
    cdef void reset(self)
    cdef void _hessian_update(self, double[::1] dof, double[:] prev_dof, double[:,::1] current_hess,
                              double[:,:] prev_hess,  double[:] current_grad, double[:] prev_grad,
                              double* energy, double* prev_energy) nogil
    cdef void update(self, double[::1] site_fracs, double phase_amt, double pressure, double temperature, bint skip_derivatives) nogil