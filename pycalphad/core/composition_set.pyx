from pycalphad.core.phase_rec cimport PhaseRecord
cimport numpy as np
import numpy as np

cdef public class CompositionSet(object)[type CompositionSetType, object CompositionSetObject]:
    def __cinit__(self, PhaseRecord prx):
        cdef int has_va = <int>(prx.vacancy_index > -1)
        self.phase_record = prx
        self.dof = np.zeros(len(self.phase_record.variables)+2)
        self.X = np.zeros(self.phase_record.composition_matrices.shape[0]-has_va)
        self.mass_grad = np.zeros((self.X.shape[0]+has_va, self.phase_record.phase_dof))
        self.mass_hess = np.zeros((self.X.shape[0]+has_va, self.phase_record.phase_dof, self.phase_record.phase_dof))
        self._dof_2d_view = <double[:1,:self.dof.shape[0]]>&self.dof[0]
        self._X_2d_view = <double[:self.X.shape[0],:1]>&self.X[0]
        self.energy = 0
        self._energy_2d_view = <double[:1]>&self.energy
        self.grad = np.zeros(self.dof.shape[0])
        self.hess = np.zeros((self.dof.shape[0], self.dof.shape[0]))
        self._prev_energy = 0
        self._prev_dof = np.zeros(self.dof.shape[0])
        self._prev_grad = np.zeros(self.dof.shape[0])
        self._prev_hess = np.zeros((self.dof.shape[0], self.dof.shape[0]))
        self._first_iteration = True

    def __repr__(self):
        return str(self.__class__.__name__) + "({0}, {1})".format(self.phase_record.phase_name, np.asarray(self.X))

    cdef void reset(self):
        self._prev_energy = 0
        self._prev_dof[:] = 0
        self._prev_grad[:] = 0
        self._prev_hess[:,:] = 0
        self._first_iteration = True

    cdef void _hessian_update(self, double[::1] dof, double[:] prev_dof, double[:,::1] current_hess,
                              double[:,:] prev_hess,  double[:] current_grad, double[:] prev_grad,
                              double* energy, double* prev_energy):
        # Notation from Nocedal and Wright, 2006, Equation 8.19
        cdef int dof_idx, dof_idx_2
        cdef int dof_len = dof.shape[0]
        cdef double[:] sk = np.empty(dof_len)
        cdef double[:] yk = np.empty(dof_len)
        cdef double[:] bk_sk = np.empty(dof_len)
        cdef double[:] ybk = np.empty(dof_len)
        cdef double ybk_norm
        cdef denominator = 0

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
            denominator += ybk[dof_idx] * sk[dof_idx]
        ybk_norm = np.linalg.norm(ybk)
        # Fall back to finite difference approximation unless it's a "medium-size" step
        # This is a really conservative approach and could probably be improved for performance
        if abs(denominator) < 1e-2 or (ybk_norm < 1e-2) or (ybk_norm > 10):
            self.phase_record.hess(current_hess, dof)
        else:
            # Symmetric Rank 1 (SR1) update
            for dof_idx in range(dof_len):
                for dof_idx_2 in range(dof_idx, dof_len):
                    current_hess[dof_idx, dof_idx_2] = current_hess[dof_idx_2, dof_idx] = \
                        prev_hess[dof_idx, dof_idx_2] + (ybk[dof_idx] * ybk[dof_idx_2] / denominator)
            prev_hess[:,:] = current_hess
        prev_energy[0] = energy[0]

    cdef void update(self, double[::1] site_fracs, double phase_amt, double pressure, double temperature):
        cdef int comp_idx
        cdef int past_va = 0
        self.dof[0] = pressure
        self.dof[1] = temperature
        self.dof[2:] = site_fracs
        self.NP = phase_amt
        self.energy = 0
        self.grad[:] = 0
        self.hess[:,:] = 0
        self.X[:] = 0
        self.mass_grad[:,:] = 0
        self.mass_hess[:,:,:] = 0
        self.phase_record.obj(self._energy_2d_view, self._dof_2d_view)
        self.phase_record.grad(self.grad, self.dof)
        for comp_idx in range(self.mass_grad.shape[0]):
            if comp_idx == self.phase_record.vacancy_index:
                past_va = 1
                continue
            self.phase_record.mass_obj(self._X_2d_view[comp_idx-past_va], site_fracs, comp_idx)
            self.phase_record.mass_grad(self.mass_grad[comp_idx], site_fracs, comp_idx)
            self.phase_record.mass_hess(self.mass_hess[comp_idx], site_fracs, comp_idx)
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