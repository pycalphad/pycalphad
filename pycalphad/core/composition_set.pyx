from pycalphad.core.phase_rec cimport PhaseRecord
cimport numpy as np
import numpy as np
from libc.stdlib cimport malloc, free
from libc.string cimport memset
from libc.math cimport fabs
cimport cython

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
        cdef int has_va = <int>(prx.vacancy_index > -1)
        self.phase_record = prx
        self.zero_seen = 0
        self.dof = np.zeros(len(self.phase_record.variables)+2)
        self.X = np.zeros(self.phase_record.composition_matrices.shape[0]-has_va)
        self._dof_2d_view = <double[:1,:self.dof.shape[0]]>&self.dof[0]
        self._X_2d_view = <double[:self.X.shape[0],:1]>&self.X[0]
        self.energy = 0
        self.NP = 0
        self._energy_2d_view = <double[:1]>&self.energy
        self.grad = np.zeros(self.dof.shape[0])
        self.hess = np.zeros((self.dof.shape[0], self.dof.shape[0]))
        self._prev_energy = 0
        self._prev_dof = np.zeros(self.dof.shape[0])
        self._prev_grad = np.zeros(self.dof.shape[0])
        self._prev_hess = np.zeros((self.dof.shape[0], self.dof.shape[0]))
        self._first_iteration = True

    def __deepcopy__(self, memodict=None):
        cdef int has_va = <int>(self.phase_record.vacancy_index > -1)
        cdef CompositionSet other
        memodict = {} if memodict is None else memodict
        other = CompositionSet(self.phase_record)
        other.phase_record = self.phase_record
        other.zero_seen = 0
        other.dof[:] = self.dof
        other.X[:] = self.X
        other.mass_grad[:,:] = self.mass_grad
        other.mass_hess[:,:,:] = self.mass_hess
        other.energy = 1.0*self.energy
        other._energy_2d_view = <double[:1]>&other.energy
        other.NP = 1.0*self.NP
        other.grad[:] = self.grad
        other.hess[:,:] = self.hess
        return other

    def __repr__(self):
        return str(self.__class__.__name__) + "({0}, {1}, NP={2}, GM={3})".format(self.phase_record.phase_name,
                                                                          np.asarray(self.X), self.NP, self.energy)

    cdef void reset(self):
        self.zero_seen = 0
        self._prev_energy = 0
        self._prev_dof[:] = 0
        self._prev_grad[:] = 0
        self._prev_hess[:,:] = 0
        self._first_iteration = True

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void _hessian_update(self, double[::1] dof, double[:] prev_dof, double[:,::1] current_hess,
                              double[:,:] prev_hess,  double[:] current_grad, double[:] prev_grad,
                              double* energy, double* prev_energy) nogil:
        # Notation from Nocedal and Wright, 2006, Equation 8.19
        cdef int dof_idx, dof_idx_2
        cdef int dof_len = dof.shape[0]
        cdef double *sk = <double*>malloc(dof_len * sizeof(double))
        cdef double *yk = <double*>malloc(dof_len * sizeof(double))
        cdef double *bk_sk = <double*>malloc(dof_len * sizeof(double))
        cdef double *ybk = <double*>malloc(dof_len * sizeof(double))
        cdef double ybk_norm = 0
        cdef double denominator = 0

        for dof_idx in range(dof_len):
            sk[dof_idx] = dof[dof_idx] - prev_dof[dof_idx]
            yk[dof_idx] = current_grad[dof_idx] - prev_grad[dof_idx]
            prev_dof[dof_idx] = dof[dof_idx]
            prev_grad[dof_idx] = current_grad[dof_idx]
        for dof_idx in range(dof_len):
            bk_sk[dof_idx] = 0
            for dof_idx_2 in range(dof_len):
                bk_sk[dof_idx] += prev_hess[dof_idx, dof_idx_2] * sk[dof_idx_2]
            ybk[dof_idx] = yk[dof_idx] - bk_sk[dof_idx]
            ybk_norm += ybk[dof_idx] ** 2
            denominator += ybk[dof_idx] * sk[dof_idx]
        ybk_norm = ybk_norm ** 0.5
        # Fall back to finite difference approximation unless it's a "medium-size" step
        # This is a really conservative approach and could probably be improved for performance
        if fabs(denominator) < 1e-2 or (ybk_norm < 1e-2) or (ybk_norm > 10):
            self.phase_record.hess(current_hess, dof)
        else:
            # Symmetric Rank 1 (SR1) update
            for dof_idx in range(dof_len):
                for dof_idx_2 in range(dof_idx, dof_len):
                    current_hess[dof_idx, dof_idx_2] = current_hess[dof_idx_2, dof_idx] = \
                        prev_hess[dof_idx, dof_idx_2] + (ybk[dof_idx] * ybk[dof_idx_2] / denominator)
            prev_hess[:,:] = current_hess
        prev_energy[0] = energy[0]
        free(sk)
        free(yk)
        free(bk_sk)
        free(ybk)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void update(self, double[::1] site_fracs, double phase_amt, double pressure, double temperature, bint skip_derivatives) nogil:
        cdef int comp_idx
        cdef int past_va = 0
        self.dof[0] = pressure
        self.dof[1] = temperature
        self.dof[2:] = site_fracs
        self.NP = phase_amt
        self.energy = 0
        memset(&self.grad[0], 0, self.grad.shape[0] * sizeof(double))
        memset(&self.hess[0,0], 0, self.hess.shape[0] * self.hess.shape[1] * sizeof(double))
        memset(&self.X[0], 0, self.X.shape[0] * sizeof(double))
        self.phase_record.obj(self._energy_2d_view, self._dof_2d_view)
        if not skip_derivatives:
            self.phase_record.grad(self.grad, self.dof)
        for comp_idx in range(self.X.shape[0]):
            self.phase_record.mass_obj(self._X_2d_view[comp_idx-past_va], self._dof_2d_view, comp_idx)
        if not skip_derivatives:
            if self._first_iteration == True:
                self.phase_record.hess(self.hess, self.dof)
                self._prev_dof[:] = self.dof
                self._prev_energy = self.energy
                self._prev_grad[:] = self.grad
                self._prev_hess[:,:] = self.hess
                self._first_iteration = False
            else:
                self._hessian_update(self.dof, self._prev_dof, self.hess, self._prev_hess, self.grad, self._prev_grad,
                                     &self.energy, &self._prev_energy)