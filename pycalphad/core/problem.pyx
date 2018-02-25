from pycalphad.core.composition_set cimport CompositionSet
cimport numpy as np
import numpy as np
from pycalphad.core.constants import MIN_SITE_FRACTION, MIN_PHASE_FRACTION
import pycalphad.variables as v

cdef class Problem:
    def __init__(self, comp_sets, comps, conditions):
        cdef CompositionSet compset
        cdef int num_sitefrac_bals = sum([compset.phase_record.sublattice_dof.shape[0] for compset in comp_sets])
        cdef int num_mass_bals = len([i for i in conditions.keys() if i.startswith('X_')]) + 1
        cdef int num_constraints = num_sitefrac_bals + num_mass_bals
        cdef int constraint_idx = 0
        cdef int var_idx = 0
        cdef int phase_idx = 0
        cdef double indep_sum = sum([float(val) for i, val in conditions.items() if i.startswith('X_')])
        cdef object dependent_comp
        if len(comp_sets) == 0:
            raise ValueError('Number of phases is zero')
        self.composition_sets = comp_sets
        self.conditions = conditions
        desired_active_pure_elements = [list(x.constituents.keys()) for x in comps]
        desired_active_pure_elements = [el.upper() for constituents in desired_active_pure_elements for el in constituents]
        self.pure_elements = sorted(set(desired_active_pure_elements))
        self.nonvacant_elements = [x for x in self.pure_elements if x != 'VA']
        dependent_comp = set(self.pure_elements) - set([i[2:] for i in conditions.keys() if i.startswith('X_')]) - {'VA'}
        dependent_comp = list(dependent_comp)[0]
        self.num_phases = len(self.composition_sets)
        self.num_vars = sum(compset.phase_record.phase_dof for compset in comp_sets) + self.num_phases
        self.num_constraints = num_constraints
        # TODO: No more special-casing T and P conditions
        self.temperature = self.conditions['T']
        self.pressure = self.conditions['P']
        self.xl = np.r_[np.full(self.num_vars - self.num_phases, MIN_SITE_FRACTION),
                        np.full(self.num_phases, MIN_PHASE_FRACTION)]
        self.xu = np.r_[np.ones(self.num_vars - self.num_phases)*2e19,
                        np.ones(self.num_phases)*2e19]
        self.x0 = np.zeros(self.num_vars)
        for compset in self.composition_sets:
            self.x0[var_idx:var_idx+compset.phase_record.phase_dof] = compset.dof[2:]
            self.x0[self.num_vars-self.num_phases+phase_idx] = compset.NP
            var_idx += compset.phase_record.phase_dof
            phase_idx += 1
        self.cl = np.zeros(num_constraints)
        self.cu = np.zeros(num_constraints)
        # Site fraction balance constraints
        self.cl[:num_sitefrac_bals] = 1
        self.cu[:num_sitefrac_bals] = 1
        # Mass balance constraints
        for constraint_idx, comp in enumerate(self.nonvacant_elements):
            if comp == dependent_comp:
                # TODO: Only handles N=1
                self.cl[num_sitefrac_bals+constraint_idx] = 1-indep_sum
                self.cu[num_sitefrac_bals+constraint_idx] = 1-indep_sum
            else:
                self.cl[num_sitefrac_bals+constraint_idx] = self.conditions['X_' + str(comp)]
                self.cu[num_sitefrac_bals+constraint_idx] = self.conditions['X_' + str(comp)]

    def objective(self, x_in):
        cdef CompositionSet compset
        cdef int phase_idx = 0
        cdef double total_obj = 0
        cdef int var_offset = 0
        cdef double[::1] x = np.r_[self.pressure, self.temperature, np.array(x_in)]
        cdef double tmp = 0
        cdef double[:,::1] dof_2d_view = <double[:1,:x.shape[0]]>&x[0]
        cdef double[::1] energy_2d_view = <double[:1]>&tmp

        for compset in self.composition_sets:
            x = np.r_[self.pressure, self.temperature, x_in[var_offset:var_offset+compset.phase_record.phase_dof]]
            dof_2d_view = <double[:1,:x.shape[0]]>&x[0]
            compset.phase_record.obj(energy_2d_view, dof_2d_view)
            total_obj += x_in[self.num_vars-self.num_phases+phase_idx] * tmp
            phase_idx += 1
            var_offset += compset.phase_record.phase_dof
            tmp = 0

        return total_obj

    def gradient(self, x_in):
        cdef CompositionSet compset
        cdef int phase_idx = 0
        cdef int var_offset = 0
        cdef int dof_x_idx
        cdef double total_obj = 0
        cdef double[::1] x = np.r_[self.pressure, self.temperature, np.array(x_in)]
        cdef double tmp = 0
        cdef double[:,::1] dof_2d_view = <double[:1,:x.shape[0]]>&x[0]
        cdef double[::1] energy_2d_view = <double[:1]>&tmp
        cdef double[::1] grad_tmp = np.zeros(x.shape[0])
        cdef np.ndarray[ndim=1, dtype=np.float64_t] gradient_term = np.zeros(self.num_vars)

        for compset in self.composition_sets:
            x = np.r_[self.pressure, self.temperature, x_in[var_offset:var_offset+compset.phase_record.phase_dof]]
            dof_2d_view = <double[:1,:x.shape[0]]>&x[0]
            compset.phase_record.obj(energy_2d_view, dof_2d_view)
            compset.phase_record.grad(grad_tmp, x)
            for dof_x_idx in range(compset.phase_record.phase_dof):
                gradient_term[var_offset + dof_x_idx] = \
                    x_in[self.num_vars-self.num_phases+phase_idx] * grad_tmp[2+dof_x_idx]  # Remove P,T grad part
            gradient_term[self.num_vars - self.num_phases + phase_idx] = tmp
            grad_tmp[:] = 0
            tmp = 0
            var_offset += compset.phase_record.phase_dof
            phase_idx += 1

        gradient_term[np.isnan(gradient_term)] = 0
        return gradient_term

    def constraints(self, x_in):
        cdef CompositionSet compset
        cdef int num_sitefrac_bals = sum([compset.phase_record.sublattice_dof.shape[0] for compset in self.composition_sets])
        cdef int num_mass_bals = len([i for i in self.conditions.keys() if i.startswith('X_')]) + 1
        cdef double indep_sum = sum([float(val) for i, val in self.conditions.items() if i.startswith('X_')])
        cdef np.ndarray[ndim=1, dtype=np.float64_t] l_constraints = np.zeros(self.num_constraints)
        cdef int phase_idx, var_offset, constraint_offset, var_idx, iter_idx, grad_idx, \
            hess_idx, comp_idx, idx, sum_idx, spidx, active_in_subl
        cdef double[::1] x = np.r_[self.pressure, self.temperature, np.array(x_in)]
        cdef double[::1] x_tmp
        cdef double[:,::1] dof_2d_view = <double[:1,:x.shape[0]]>&x[0]
        cdef double[::1] tmp_mass = np.atleast_1d(np.zeros(1))

        # First: Site fraction balance constraints
        var_idx = 0
        constraint_offset = 0
        for phase_idx in range(self.num_phases):
            compset = self.composition_sets[phase_idx]
            for idx in range(compset.phase_record.sublattice_dof.shape[0]):
                active_in_subl = compset.phase_record.sublattice_dof[idx]
                for sum_idx in range(active_in_subl):
                    l_constraints[constraint_offset + idx] += x[2+sum_idx+var_idx]
                var_idx += active_in_subl
            constraint_offset += compset.phase_record.sublattice_dof.shape[0]
        # Second: Mass balance of each component
        for comp_idx, comp in enumerate(self.nonvacant_elements):
            var_offset = 0
            for phase_idx in range(self.num_phases):
                compset = self.composition_sets[phase_idx]
                spidx = self.num_vars - self.num_phases + phase_idx
                # TODO: This is a hack until the constraint system is rewritten
                x_tmp = np.r_[self.pressure, self.temperature, x[2+var_offset:2+var_offset+compset.phase_record.phase_dof]]
                dof_2d_view = <double[:1,:x_tmp.shape[0]]>&x_tmp[0]
                compset.phase_record.mass_obj(tmp_mass, dof_2d_view, comp_idx)
                l_constraints[constraint_offset] += x[2+spidx] * tmp_mass[0]
                var_offset += compset.phase_record.phase_dof
                tmp_mass[0] = 0
            constraint_offset += 1
        return l_constraints

    def jacobian(self, x_in):
        cdef CompositionSet compset
        cdef double[::1] x = np.r_[self.pressure, self.temperature, np.array(x_in)]
        cdef double[::1] x_tmp
        cdef double[:,::1] dof_2d_view = <double[:1,:x.shape[0]]>&x[0]
        cdef double[::1] tmp_mass = np.atleast_1d(np.zeros(1))
        cdef double[::1] tmp_mass_grad = np.zeros(2+self.num_vars)
        cdef double[:,::1] constraint_jac = np.zeros((self.num_constraints, self.num_vars))
        cdef int phase_idx, var_offset, constraint_offset, var_idx, iter_idx, grad_idx, \
            hess_idx, comp_idx, idx, sum_idx, spidx, active_in_subl, phase_offset

        # Ordering of constraints by row: sitefrac bal of each phase, then component mass balance
        # Ordering of constraints by column: site fractions of each phase, then phase fractions
        # First: Site fraction balance constraints
        var_idx = 0
        constraint_offset = 0
        for phase_idx in range(self.num_phases):
            compset = self.composition_sets[phase_idx]
            phase_offset = 0
            for idx in range(compset.phase_record.sublattice_dof.shape[0]):
                active_in_subl = compset.phase_record.sublattice_dof[idx]
                constraint_jac[constraint_offset + idx,
                var_idx:var_idx + active_in_subl] = 1
                var_idx += active_in_subl
                phase_offset += active_in_subl
            constraint_offset += compset.phase_record.sublattice_dof.shape[0]
        # Second: Mass balance of each component
        for comp_idx, comp in enumerate(self.nonvacant_elements):
            var_offset = 0
            for phase_idx in range(self.num_phases):
                compset = self.composition_sets[phase_idx]
                spidx = self.num_vars - self.num_phases + phase_idx
                # TODO: This is a hack until the constraint system is rewritten
                x_tmp = np.r_[self.pressure, self.temperature, x[2+var_offset:2+var_offset+compset.phase_record.phase_dof]]
                dof_2d_view = <double[:1,:x_tmp.shape[0]]>&x_tmp[0]
                compset.phase_record.mass_grad(tmp_mass_grad, x_tmp, comp_idx)
                # current phase frac times the comp_grad
                for grad_idx in range(var_offset, var_offset + compset.phase_record.phase_dof):
                    constraint_jac[constraint_offset, grad_idx] = \
                        x[2+spidx] * tmp_mass_grad[2 + grad_idx - var_offset]
                compset.phase_record.mass_obj(tmp_mass, dof_2d_view, comp_idx)
                constraint_jac[constraint_offset, spidx] += tmp_mass[0]
                tmp_mass[0] = 0
                tmp_mass_grad[:] = 0
                var_offset += compset.phase_record.phase_dof
            constraint_offset += 1
        return np.array(constraint_jac)
