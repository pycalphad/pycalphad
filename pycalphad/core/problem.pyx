# distutils: language = c++
from pycalphad.core.composition_set cimport CompositionSet
cimport numpy as np
import numpy as np


cdef class Problem:
    def __init__(self, comp_sets, comps, conditions):
        cdef CompositionSet compset
        cdef int num_internal_cons = sum(compset.phase_record.num_internal_cons for compset in comp_sets)
        cdef object state_variables
        cdef int num_fixed_dof_cons, idx
        cdef int num_constraints
        cdef int constraint_idx = 0
        cdef int var_idx = 0
        cdef int phase_idx = 0
        cdef double indep_sum = sum([float(val) for i, val in conditions.items() if i.startswith('X_')])
        cdef object dependent_comp
        if len(comp_sets) == 0:
            raise ValueError('Number of phases is zero')
        state_variables = comp_sets[0].phase_record.state_variables
        fixed_statevars = [(key, value) for key, value in conditions.items() if key in [str(k) for k in state_variables]]
        num_fixed_dof_cons = len(state_variables)

        self.composition_sets = comp_sets
        self.conditions = conditions
        desired_active_pure_elements = [list(x.constituents.keys()) for x in comps]
        desired_active_pure_elements = [el.upper() for constituents in desired_active_pure_elements for el in constituents]
        self.pure_elements = sorted(set(desired_active_pure_elements))
        self.nonvacant_elements = [x for x in self.pure_elements if x != 'VA']
        self.fixed_chempot_indices = np.array([self.nonvacant_elements.index(key[3:]) for key in conditions.keys() if key.startswith('MU_')], dtype=np.int32)
        self.fixed_chempot_values = np.array([float(value) for key, value in conditions.items() if key.startswith('MU_')])
        self.num_phases = len(self.composition_sets)
        self.num_vars = sum(compset.phase_record.phase_dof for compset in comp_sets) + self.num_phases + len(state_variables)
        self.num_internal_constraints = num_internal_cons
        self.num_fixed_dof_constraints = num_fixed_dof_cons
        self.fixed_dof_indices = np.zeros(self.num_fixed_dof_constraints, dtype=np.int32)
        all_dof = list(str(k) for k in state_variables)
        for compset in comp_sets:
            all_dof.extend(compset.phase_record.variables)
        for idx in range(len(fixed_statevars)):
            k = fixed_statevars[idx][0]
            self.fixed_dof_indices[idx] = all_dof.index(k)
