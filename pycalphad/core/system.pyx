from pycalphad.core.composition_set cimport CompositionSet
cimport numpy as np
from pycalphad.core.constants import MIN_SITE_FRACTION, MIN_PHASE_FRACTION

cdef class System:
    def __init__(self, comp_sets, comps, conditions):
        cdef CompositionSet compset
        cdef int num_sitefrac_bals = sum([compset.phase_record.sublattice_dof.shape[0] for compset in comp_sets])
        cdef int num_mass_bals = len([i for i in conditions.keys() if i.startswith('X_')]) + 1
        cdef int num_constraints = num_sitefrac_bals + num_mass_bals
        cdef int constraint_idx = 0
        cdef int var_idx = 0
        cdef int phase_idx = 0
        cdef double indep_sum = sum([float(val) for i, val in conditions.items() if i.startswith('X_')])
        cdef object dependent_comp = set(comps) - set([i[2:] for i in conditions.keys() if i.startswith('X_')]) - {'VA'}
        dependent_comp = list(dependent_comp)[0]
        self.composition_sets = comp_sets
        self.conditions = conditions
        self.components = sorted(comps)
        self.num_phases = len(self.composition_sets)
        self.num_vars = sum(compset.phase_record.phase_dof for compset in comp_sets) + self.num_phases
        self.num_constraints = num_constraints
        # TODO: No more special-casing T and P conditions
        self.temperature = self.conditions['T']
        self.pressure = self.conditions['P']
        self.xl = np.r_[np.full(self.num_vars - self.num_phases, MIN_SITE_FRACTION),
                        np.full(self.num_phases, MIN_PHASE_FRACTION)]
        self.xu = np.r_[np.ones(self.num_vars - self.num_phases),
                        np.ones(self.num_phases)]
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
        for constraint_idx, comp in enumerate(self.components):
            if comp == dependent_comp:
                # TODO: Only handles N=1
                self.cl[num_sitefrac_bals+constraint_idx] = 1-indep_sum
                self.cu[num_sitefrac_bals+constraint_idx] = 1-indep_sum
            else:
                self.cl[num_sitefrac_bals+constraint_idx] = self.conditions['X_' + comp]
                self.cu[num_sitefrac_bals+constraint_idx] = self.conditions['X_' + comp]

    def objective(self, x_in):
        cdef CompositionSet compset
        cdef int phase_idx = 0
        cdef double total_obj = 0
        cdef double[::1] x = np.r_[self.pressure, self.temperature, np.array(x_in)]
        cdef double tmp = 0
        cdef double[:,::1] dof_2d_view = <double[:1,:x.shape[0]]>&x[0]
        cdef double[::1] energy_2d_view = <double[:1]>&tmp

        for compset in self.composition_sets:
            compset.phase_record.obj(energy_2d_view, dof_2d_view)
            total_obj += x[2+self.num_vars-self.num_phases+phase_idx] * tmp
            phase_idx += 1
            tmp = 0

        return total_obj

    def gradient(self, x):
        #
        # The callback for calculating the gradient
        #
        return np.array([
                    x[0] * x[3] + x[3] * np.sum(x[0:3]),
                    x[0] * x[3],
                    x[0] * x[3] + 1.0,
                    x[0] * np.sum(x[0:3])
                    ])

    def constraints(self, x):
        #
        # The callback for calculating the constraints
        #
        return np.array((np.prod(x), np.dot(x, x)))

    def jacobian(self, x):
        #
        # The callback for calculating the Jacobian
        #
        return np.concatenate((np.prod(x) / x, 2*x))

    def hessian(self, x, lagrange, obj_factor):
        #
        # The callback for calculating the Hessian
        #
        H = obj_factor*np.array((
                (2*x[3], 0, 0, 0),
                (x[3],   0, 0, 0),
                (x[3],   0, 0, 0),
                (2*x[0]+x[1]+x[2], x[0], x[0], 0)))

        H += lagrange[0]*np.array((
                (0, 0, 0, 0),
                (x[2]*x[3], 0, 0, 0),
                (x[1]*x[3], x[0]*x[3], 0, 0),
                (x[1]*x[2], x[0]*x[2], x[0]*x[1], 0)))

        H += lagrange[1]*2*np.eye(4)

        #
        # Note:
        #
        #
        # Needs to return lower triangular matrix
        return H
