#cython: profile=True
#cython: linetrace=True
#cython: binding=True
from collections import defaultdict, OrderedDict
import copy
import itertools
import numpy as np
cimport numpy as np
cimport cython
cdef extern from "_isnan.h":
    bint isnan (double) nogil
import scipy.spatial
from pycalphad.core.phase_rec cimport PhaseRecord, PhaseRecord_from_f2py
from pycalphad.core.constants import MIN_SITE_FRACTION, COMP_DIFFERENCE_TOL, BIGNUM
import pycalphad.variables as v

# Maximum residual driving force (J/mol-atom) allowed for convergence
MAX_SOLVE_DRIVING_FORCE = 1e-4
# Maximum number of multi-phase solver iterations
MAX_SOLVE_ITERATIONS = 300
# Minimum energy (J/mol-atom) difference between iterations before stopping solver
MIN_SOLVE_ENERGY_PROGRESS = 1e-3
# Maximum absolute value of a Lagrange multiplier before it's recomputed with an alternative method
MAX_ABS_LAGRANGE_MULTIPLIER = 1e16
INITIAL_OBJECTIVE_WEIGHT = 1
cdef double MAX_ENERGY = BIGNUM

cdef public class CompositionSet(object)[type CompositionSetType, object CompositionSetObject]:
    cdef public PhaseRecord phase_record
    cdef readonly double[::1] dof, X
    cdef double[::1,:] _dof_2d_view
    cdef double[:,::1] _X_2d_view
    cdef readonly double[:, ::1] mass_grad
    cdef readonly double[:, :, :] mass_hess
    cdef public double NP
    cdef readonly double energy
    cdef double[::1] _energy_2d_view
    cdef readonly double[::1] grad
    cdef readonly double[::1,:] hess
    cdef readonly double[::1] _prev_dof
    cdef readonly double _prev_energy
    cdef readonly double[::1] _prev_grad
    cdef readonly double[::1,:] _prev_hess
    cdef readonly bint _first_iteration

    def __cinit__(self, PhaseRecord prx):
        self.phase_record = prx
        self.dof = np.zeros(len(self.phase_record.variables)+2)
        self.X = np.zeros(self.phase_record.composition_matrices.shape[0])
        self.mass_grad = np.zeros((self.X.shape[0], self.phase_record.phase_dof))
        self.mass_hess = np.zeros((self.X.shape[0], self.phase_record.phase_dof, self.phase_record.phase_dof))
        self._dof_2d_view = <double[:1:1,:self.dof.shape[0]]>&self.dof[0]
        self._X_2d_view = <double[:self.X.shape[0],:1]>&self.X[0]
        self.energy = 0
        self._energy_2d_view = <double[:1]>&self.energy
        self.grad = np.zeros(self.dof.shape[0])
        self.hess = np.zeros((self.dof.shape[0], self.dof.shape[0]), order='F')
        self._prev_energy = 0
        self._prev_dof = np.zeros(self.dof.shape[0])
        self._prev_grad = np.zeros(self.dof.shape[0])
        self._prev_hess = np.zeros((self.dof.shape[0], self.dof.shape[0]), order='F')
        self._first_iteration = True

    cdef void reset(self):
        self._prev_energy = 0
        self._prev_dof[:] = 0
        self._prev_grad[:] = 0
        self._prev_hess[:,:] = 0
        self._first_iteration = True

    cdef void update(self, double[::1] site_fracs, double phase_amt, double pressure, double temperature):
        cdef int comp_idx
        self._prev_dof[:] = self.dof
        self._prev_energy = self.energy
        self._prev_grad[:] = self.grad
        self._prev_hess[:,:] = self.hess
        self.dof[0] = pressure
        self.dof[1] = temperature
        self.dof[2:] = site_fracs
        self.NP = phase_amt
        self.energy = 0
        self.grad[:] = 0
        self.hess[:,:] = 0
        self._first_iteration = False
        self.phase_record.obj(self._energy_2d_view, self._dof_2d_view)
        self.phase_record.grad(self.grad, self.dof)
        self.phase_record.hess(self.hess, self.dof)
        for comp_idx in range(self.X.shape[0]):
            self.phase_record.mass_obj(self._X_2d_view[comp_idx], self.dof, comp_idx)
            self.phase_record.mass_grad(self.mass_grad[comp_idx], self.dof, comp_idx)
            self.phase_record.mass_hess(self.mass_hess[comp_idx], self.dof, comp_idx)


def remove_degenerate_phases(object composition_sets, bint allow_negative_fractions):
    """
    For each phase pair with composition difference below tolerance,
    eliminate phase with largest index.
    Also remove phases with phase fractions close to zero.

    Parameters
    ----------


    """
    cdef double[:,:] comp_matrix
    cdef double[:,:] comp_distances
    cdef double phfsum = 0
    cdef object redundant_phases, kept_phase, removed_phases, saved_indices
    cdef int num_phases = len(composition_sets)
    cdef int phase_idx, sidx
    cdef int[:] indices
    cdef CompositionSet compset
    # Group phases into multiple composition sets
    cdef object phase_indices = defaultdict(lambda: list())
    for phase_idx in range(num_phases):
        name = <unicode>composition_sets[phase_idx].phase_record.name
        if name == "":
            continue
        phase_indices[name].append(phase_idx)
    # Compute pairwise distances between compositions of like phases
    for name, idxs in phase_indices.items():
        indices = np.array(idxs, dtype=np.int32)
        if indices.shape[0] == 1:
            # Phase is unique
            continue
        # All composition sets should have the same X shape (same number of possible components)
        comp_matrix = np.empty((np.max(indices)+1, composition_sets[0].X.shape[0]))
        # The reason we don't do this based on Y fractions is because
        # of sublattice symmetry. It's very easy to detect a "miscibility gap" which is actually
        # symmetry equivalent, i.e., D([A, B] - [B, A]) > tol, but they are the same configuration.
        for idx in range(indices.shape[0]):
            compset = composition_sets[indices[idx]]
            comp_matrix[indices[idx], :] = compset.X
        comp_distances = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(comp_matrix, metric='chebyshev'))
        redundant_phases = set()
        redundant_phases |= {indices[0]}
        for i in range(len(indices)):
            for j in range(i, len(indices)):
                if i == j:
                    continue
                if comp_distances[i, j] < COMP_DIFFERENCE_TOL:
                    redundant_phases |= {indices[i], indices[j]}
        redundant_phases = sorted(redundant_phases)
        kept_phase = redundant_phases[0]
        removed_phases = redundant_phases[1:]
        # Their NP values will be added to the kept phase
        # and they will be nulled out
        for redundant in removed_phases:
            composition_sets[kept_phase].NP += composition_sets[redundant].NP
            composition_sets[redundant].NP = np.nan
    for phase_idx in range(num_phases):
        if abs(composition_sets[phase_idx].NP) <= MIN_SITE_FRACTION:
            composition_sets[phase_idx].NP = MIN_SITE_FRACTION
        elif (composition_sets[phase_idx].NP <= MIN_SITE_FRACTION) and (not allow_negative_fractions):
            composition_sets[phase_idx].NP = np.nan

    entries_to_delete = sorted([idx for idx, compset in enumerate(composition_sets) if np.isnan(compset.NP)],
                               reverse=True)
    for idx in entries_to_delete:
        del composition_sets[idx]


def _compute_phase_dof(dbf, comps, phases):
    """
    Generate a list of the number of each phase's internal phase degrees of freedom.
    """
    phase_dof = []
    for name in phases:
        total = 0
        for idx in range(len(dbf.phases[name].sublattices)):
            active_in_subl = set(dbf.phases[name].constituents[idx]).intersection(comps)
            total += len(active_in_subl)
        phase_dof.append(total)
    return np.array(phase_dof, dtype=np.int)

@cython.boundscheck(False)
@cython.wraparound(False)
def _compute_constraints(object comps, object phases,
                         object cur_conds, double[::1] site_fracs,
                         np.ndarray[dtype=np.float64_t, ndim=1] phase_fracs, object phase_records):
    """
    Compute the constraint vector and constraint Jacobian matrix.
    """
    cdef int num_sitefrac_bals = sum([phase_records[x].sublattice_dof.shape[0] for x in phases])
    cdef int num_mass_bals = len([i for i in cur_conds.keys() if i.startswith('X_')]) + 1
    cdef double indep_sum = sum([float(val) for i, val in cur_conds.items() if i.startswith('X_')])
    cdef double[::1] comp_obj_value = np.atleast_1d(np.zeros(1))
    cdef object dependent_comp = set(comps) - set([i[2:] for i in cur_conds.keys() if i.startswith('X_')]) - {'VA'}
    dependent_comp = list(dependent_comp)[0]
    cdef int num_constraints = num_sitefrac_bals + num_mass_bals
    cdef int num_phases = len(phases)
    cdef int num_vars = site_fracs.shape[0] + num_phases
    cdef int max_phase_dof = max([x.phase_dof for x in phase_records.values()])
    cdef double[::1] l_constraints = np.zeros(num_constraints)
    cdef double[::1,:] constraint_jac = np.zeros((num_constraints, num_vars), order='F')
    cdef np.ndarray[ndim=3, dtype=np.float64_t] constraint_hess = np.zeros((num_constraints, num_vars, num_vars), order='F')
    cdef double[::1] sfview
    cdef double[::1] comp_grad_value = np.zeros(max_phase_dof)
    cdef double[::1,:] comp_hess_value = np.zeros((max_phase_dof, max_phase_dof), order='F')
    cdef int phase_idx, var_offset, constraint_offset, var_idx, iter_idx, grad_idx, \
        hess_idx, comp_idx, idx, sum_idx, ais_len
    cdef PhaseRecord prn
    cdef double phase_frac

    # Ordering of constraints by row: sitefrac bal of each phase, then component mass balance
    # Ordering of constraints by column: site fractions of each phase, then phase fractions
    # First: Site fraction balance constraints
    var_idx = 0
    constraint_offset = 0
    for phase_idx in range(num_phases):
        prn = phase_records[phases[phase_idx]]
        with nogil:
            for idx in range(prn.sublattice_dof.shape[0]):
                ais_len = prn.sublattice_dof[idx]
                constraint_jac[constraint_offset + idx,
                var_idx:var_idx + ais_len] = 1
                l_constraints[constraint_offset + idx] = -1
                for sum_idx in range(var_idx, var_idx + ais_len):
                    l_constraints[constraint_offset + idx] += site_fracs[sum_idx]
                var_idx += ais_len
        constraint_offset += prn.sublattice_dof.shape[0]
    # Second: Mass balance of each component
    for comp_idx, comp in enumerate(comps):
        if comp == 'VA':
            continue
        var_offset = 0
        for phase_idx in range(num_phases):
            prn = phase_records[phases[phase_idx]]
            phase_frac = phase_fracs[phase_idx]
            spidx = site_fracs.shape[0] + phase_idx
            sfview = site_fracs[var_offset:var_offset + prn.phase_dof]
            with nogil:
                comp_obj_value[0] = 0
                for grad_idx in range(prn.phase_dof):
                    comp_grad_value[grad_idx] = 0
                    for hess_idx in range(grad_idx, prn.phase_dof):
                        comp_hess_value[grad_idx, hess_idx] = comp_hess_value[hess_idx, grad_idx] = 0
                prn.mass_obj(comp_obj_value, sfview, comp_idx)
                prn.mass_grad(comp_grad_value, sfview, comp_idx)
                prn.mass_hess(comp_hess_value, sfview, comp_idx)
                # current phase frac times the comp_grad
                for grad_idx in range(var_offset, var_offset + prn.phase_dof):
                    constraint_jac[constraint_offset, grad_idx] = \
                        phase_frac * comp_grad_value[grad_idx - var_offset]
                    constraint_hess[constraint_offset, spidx, grad_idx] = comp_grad_value[grad_idx - var_offset]
                    constraint_hess[constraint_offset, grad_idx, spidx] = comp_grad_value[grad_idx - var_offset]
                    for hess_idx in range(var_offset, var_offset + prn.phase_dof):
                        constraint_hess[constraint_offset, grad_idx, hess_idx] = \
                            phase_frac * comp_hess_value[grad_idx - var_offset, hess_idx - var_offset]
                l_constraints[constraint_offset] += phase_frac * comp_obj_value[0]
                constraint_jac[constraint_offset, spidx] += comp_obj_value[0]
                var_offset += prn.phase_dof
        if comp != dependent_comp:
            l_constraints[constraint_offset] -= float(cur_conds['X_' + comp])
        else:
            # TODO: Assuming N=1 (fixed for dependent component)
            l_constraints[constraint_offset] -= (1 - indep_sum)
        constraint_offset += 1
    return np.array(l_constraints), np.array(constraint_jac), constraint_hess


@cython.boundscheck(False)
@cython.wraparound(False)
cdef _build_multiphase_gradient(int[:] phase_dof, phases, cur_conds, double[::1] site_fracs, double[:] phase_fracs,
                                l_constraints, constraint_jac, l_multipliers, phase_records, double obj_weight):
    cdef double[::1] obj_result = np.zeros(1)
    cdef double[::1] obj_res = np.zeros(1)
    cdef int max_phase_dof = max(phase_dof)
    cdef double[::1] grad_res = np.zeros(2+max_phase_dof)
    cdef double[::1] dof = np.zeros(2+max_phase_dof)
    cdef double[::1,:] dof_2d_view = <double[:1:1, :dof.shape[0]]>&dof[0]
    cdef int num_vars = site_fracs.shape[0] + len(phases)
    cdef double[::1] gradient_term = np.zeros(num_vars)
    cdef int var_offset = 0
    cdef int phase_idx = 0
    cdef int dof_x_idx
    cdef double phase_frac
    cdef PhaseRecord prn
    dof[0] = cur_conds['P']
    dof[1] = cur_conds['T']

    for name, phase_frac in zip(phases, phase_fracs):
        prn = phase_records[name]
        with nogil:
            dof[2:2+prn.phase_dof] = site_fracs[var_offset:var_offset + prn.phase_dof]
            prn.obj(obj_res, dof_2d_view)
            # This can happen for phases with non-physical vacancy content
            if isnan(obj_res[0]):
                obj_res[0] = MAX_ENERGY
            obj_result[0] += obj_weight * phase_frac * obj_res[0]
            prn.grad(grad_res, dof)
            for dof_x_idx in range(prn.phase_dof):
                gradient_term[var_offset + dof_x_idx] = \
                    obj_weight * phase_frac * grad_res[2+dof_x_idx]  # Remove P,T grad part
            gradient_term[site_fracs.shape[0] + phase_idx] = obj_weight * obj_res[0]
            var_offset += prn.phase_dof
            phase_idx += 1
    return np.asarray(obj_result), np.asarray(gradient_term)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef _build_multiphase_system(int[:] phase_dof, phases, cur_conds, double[::1] site_fracs, double[:] phase_fracs,
                              l_constraints, constraint_jac,
                              np.ndarray[ndim=3, dtype=np.float64_t] constraint_hess,
                              np.ndarray[ndim=1, dtype=np.float64_t] l_multipliers,
                              phase_records, double obj_weight):
    cdef double[::1] obj_res = np.empty(1)
    cdef int max_phase_dof = max(phase_dof)
    cdef double[::1] grad_res = np.empty(2+max_phase_dof)
    cdef double[::1,:] tmp_hess = np.empty((2+max_phase_dof, 2+max_phase_dof), order='F')
    cdef double* tmp_hess_ptr = &tmp_hess[0,0]
    cdef double[::1] dof = np.empty(2+max_phase_dof)
    cdef double[::1,:] dof_2d_view = <double[:1:1, :dof.shape[0]]>&dof[0]
    cdef int num_vars = len(site_fracs) + len(phases)
    cdef double[::1,:] l_hessian = np.zeros((num_vars, num_vars), order='F')
    cdef double[::1] gradient_term = np.zeros(num_vars)
    cdef int var_offset = 0
    cdef int phase_idx = 0
    cdef int constraint_idx, dof_x_idx, dof_y_idx, hess_x, hess_y, hess_idx
    cdef double phase_frac
    cdef double total_obj = 0
    cdef PhaseRecord prn
    dof[0] = cur_conds['P']
    dof[1] = cur_conds['T']

    for name, phase_frac in zip(phases, phase_fracs):
        prn = phase_records[name]
        tmp_hess = np.zeros((2+prn.phase_dof, 2+prn.phase_dof), order='F')
        tmp_hess_ptr = &tmp_hess[0,0]
        with nogil:
            dof[2:2+prn.phase_dof] = site_fracs[var_offset:var_offset + prn.phase_dof]
            grad_res[:] = 0
            obj_res[0] = 0
            tmp_hess[:,:] = 0
            prn.obj(obj_res, dof_2d_view)
            # This can happen for phases with non-physical vacancy content
            if isnan(obj_res[0]):
                obj_res[0] = MAX_ENERGY
            total_obj += obj_weight * phase_frac * obj_res[0]
            prn.grad(grad_res, dof[:2+prn.phase_dof])
            prn.hess(tmp_hess, dof[:2+prn.phase_dof])
            for dof_x_idx in range(prn.phase_dof):
                gradient_term[var_offset + dof_x_idx] = \
                    obj_weight * phase_frac * grad_res[2+dof_x_idx]  # Remove P,T grad part
            gradient_term[site_fracs.shape[0] + phase_idx] = obj_weight * obj_res[0]

            for dof_x_idx in range(prn.phase_dof):
                for dof_y_idx in range(dof_x_idx,prn.phase_dof):
                    l_hessian[var_offset+dof_x_idx, var_offset+dof_y_idx] = \
                      obj_weight * phase_frac * tmp_hess_ptr[2+dof_x_idx + (2+prn.phase_dof)*(2+dof_y_idx)]
                    l_hessian[var_offset+dof_y_idx, var_offset+dof_x_idx] = \
                      obj_weight * l_hessian[var_offset+dof_x_idx, var_offset+dof_y_idx]
                # Phase fraction / site fraction cross derivative
                l_hessian[site_fracs.shape[0] + phase_idx, var_offset + dof_x_idx] = \
                     obj_weight * grad_res[2+dof_x_idx] # Remove P,T grad part
                l_hessian[var_offset + dof_x_idx, site_fracs.shape[0] + phase_idx] = obj_weight * grad_res[2+dof_x_idx]
            var_offset += prn.phase_dof
            phase_idx += 1
    l_hessian -= np.einsum('i,ijk->jk', l_multipliers, constraint_hess, order='F')
    return np.asarray(total_obj), np.asarray(l_hessian), np.asarray(gradient_term)

def _solve_eq_at_conditions(dbf, comps, properties, phase_records, conds_keys, verbose, diagnostic, compute_constraints):
    """
    Compute equilibrium for the given conditions.
    This private function is meant to be called from a worker subprocess.
    For that case, usually only a small slice of the master 'properties' is provided.
    Since that slice will be copied, we also return the modified 'properties'.

    Parameters
    ----------
    dbf : Database
        Thermodynamic database containing the relevant parameters.
    comps : list
        Names of components to consider in the calculation.
    properties : Dataset
        Will be modified! Thermodynamic properties and conditions.
    phase_records : dict of PhaseRecord
        Details on phase callables.
    conds_keys : list of str
        List of conditions axes in dimension order.
    verbose : bool
        Print details.
    diagnostic : bool
        Dump convergence details to CSV file.

    Returns
    -------
    properties : Dataset
        Modified with equilibrium values.
    """
    cdef:
        double indep_sum
        int num_phases, num_vars, cur_iter, old_phase_length, new_phase_length, var_idx, sfidx, pfidx, m, n
        int vmax_window_size
        int obj_decreases
        double previous_window_average, obj_weight, vmax
        PhaseRecord prn
        CompositionSet compset
        cdef int[:] phase_dof
        cdef double[::1,:] l_hessian
        cdef double[::1] gradient_term, mass_buf
        double[::1] vmax_averages
        np.ndarray[ndim=1, dtype=np.float64_t] p_y, l_constraints, step
        np.ndarray[ndim=1, dtype=np.float64_t] site_fracs, l_multipliers, phase_fracs
        np.ndarray[ndim=2, dtype=np.float64_t] ymat, zmat, qmat, rmat, constraint_jac
        np.ndarray[ndim=2, dtype=np.float64_t] diagnostic_matrix

    for key, value in phase_records.items():
        if not isinstance(phase_records[key], PhaseRecord):
            phase_records[key] = PhaseRecord_from_f2py(comps, value.variables, np.array(value.num_sites, dtype=np.float),
                                                       value.parameters, value.obj, value.grad, value.hess)
    # Factored out via profiling
    prop_MU_values = properties['MU'].values
    prop_NP_values = properties['NP'].values
    prop_Phase_values = properties['Phase'].values
    prop_X_values = properties['X'].values
    prop_Y_values = properties['Y'].values
    prop_GM_values = properties['GM'].values
    phase_dof_dict = {name: len(set(phase_records[name].variables) - {v.T, v.P})
                      for name in phase_records.keys()}
    it = np.nditer(prop_GM_values, flags=['multi_index'])

    #if verbose:
    #    print('INITIAL CONFIGURATION')
    #    print(properties.MU)
    #    print(properties.Phase)
    #    print(properties.NP)
    #    print(properties.X)
    #    print(properties.Y)
    #    print('---------------------')
    while not it.finished:
        # A lot of this code relies on cur_conds being ordered!
        cur_conds = OrderedDict(zip(conds_keys,
                                    [np.asarray(properties['GM'].coords[b][a], dtype=np.float)
                                     for a, b in zip(it.multi_index, conds_keys)]))
        if len(cur_conds) == 0:
            cur_conds = properties['GM'].coords
        # sum of independently specified components
        indep_sum = np.sum([float(val) for i, val in cur_conds.items() if i.startswith('X_')])
        if indep_sum > 1:
            # Sum of independent component mole fractions greater than one
            # Skip this condition set
            # We silently allow this to make 2-D composition mapping easier
            prop_MU_values[it.multi_index] = np.nan
            prop_NP_values[it.multi_index + np.index_exp[:len(phases)]] = np.nan
            prop_Phase_values[it.multi_index + np.index_exp[:len(phases)]] = ''
            prop_X_values[it.multi_index + np.index_exp[:len(phases)]] = np.nan
            prop_Y_values[it.multi_index] = np.nan
            prop_GM_values[it.multi_index] = np.nan
            it.iternext()
            continue
        dependent_comp = set(comps) - set([i[2:] for i in cur_conds.keys() if i.startswith('X_')]) - {'VA'}
        if len(dependent_comp) == 1:
            dependent_comp = list(dependent_comp)[0]
        else:
            raise ValueError('Number of dependent components different from one')
        composition_sets = []
        for phase_idx, phase_name in enumerate(prop_Phase_values[it.multi_index]):
            if phase_name == '' or phase_name == '_FAKE_':
                continue
            phrec = phase_records[phase_name]
            phrec.reset_model_state()
            sfx = prop_Y_values[it.multi_index + np.index_exp[phase_idx, :phrec.phase_dof]]
            phase_amt = prop_NP_values[it.multi_index + np.index_exp[phase_idx]]
            compset = CompositionSet(phrec)
            compset.update(sfx, phase_amt, cur_conds['P'], cur_conds['T'])
            composition_sets.append(compset)
        diagnostic_matrix_shape = 7
        if diagnostic:
            diagnostic_matrix = np.full((MAX_SOLVE_ITERATIONS, diagnostic_matrix_shape + len(set(comps) - {'VA'})), np.nan)
            debug_fn = 'debug-{}.csv'.format('-'.join([str(x) for x in it.multi_index]))
        vmax_window_size = 10
        previous_window_average = np.inf
        vmax_averages = np.zeros(vmax_window_size)
        obj_decreases = 0
        alpha = 1
        obj_weight = INITIAL_OBJECTIVE_WEIGHT
        allow_negative_fractions = False
        for cur_iter in range(MAX_SOLVE_ITERATIONS):
            # print('CUR_ITER:', cur_iter)
            old_phase_length = len(composition_sets)
            if cur_iter > 0.8 * MAX_SOLVE_ITERATIONS:
                allow_negative_fractions = False
            remove_degenerate_phases(composition_sets, allow_negative_fractions)
            num_phases = len(composition_sets)
            # XXX: 'phases' is here for compatibility until full support for CompositionSet
            phases = prop_Phase_values[it.multi_index + np.index_exp[:num_phases]]
            new_phase_length = len(composition_sets)
            total_dof = sum([compset.phase_record.phase_dof for compset in composition_sets])
            if num_phases == 0:
                raise ValueError('Zero phases are left in the system', cur_conds)
            if (num_phases == 1) and np.all(composition_sets[0].dof[2:] == 1.):
                # Single phase with zero internal degrees of freedom, can't do any refinement
                # TODO: In the future we may be able to refine other degrees of freedom like temperature
                # Chemical potentials have no meaning for this case
                prop_MU_values[it.multi_index] = np.nan
                break
            phase_fracs = np.empty(num_phases)
            for phase_idx in range(num_phases):
                phase_fracs[phase_idx] = composition_sets[phase_idx].NP
            phase_dof = np.array([phase_dof_dict[name] for name in phases], dtype=np.int32)
            dof_idx = 0
            site_fracs = np.empty(total_dof)
            for phase_idx in range(num_phases):
                site_fracs[dof_idx:dof_idx+composition_sets[phase_idx].phase_record.phase_dof] = composition_sets[phase_idx].dof[2:]
                dof_idx += composition_sets[phase_idx].phase_record.phase_dof
            if len(site_fracs) == 0:
                print(properties)
                raise ValueError('Site fractions are invalid')
            l_constraints, constraint_jac, constraint_hess  = \
                compute_constraints(comps, phases, cur_conds, site_fracs, phase_fracs, phase_records)
            # Reset Lagrange multipliers if active set of phases change
            if cur_iter == 0 or (old_phase_length != new_phase_length) or np.any(np.isnan(l_multipliers)):
                l_multipliers = np.zeros(l_constraints.shape[0])
            # Equation 18.14a in Nocedal and Wright
            num_vars = len(site_fracs) + len(phases)
            energy, l_hessian, gradient_term = _build_multiphase_system(phase_dof, phases, cur_conds, site_fracs, phase_fracs,
                                                                l_constraints, constraint_jac, constraint_hess,
                                                                l_multipliers, phase_records, obj_weight)
            if np.any(np.isnan(l_hessian)):
                print('Invalid l_hessian')
                l_hessian = np.asfortranarray(np.eye(l_hessian.shape[0]))
            if np.any(np.isnan(gradient_term)):
                raise ValueError('Invalid gradient_term')
            # Equation 18.10 in Nocedal and Wright
            master_hess = np.zeros((num_vars + l_constraints.shape[0], num_vars + l_constraints.shape[0]))
            master_hess[:num_vars, :num_vars] = l_hessian
            master_hess[:num_vars, num_vars:] = -constraint_jac.T
            master_hess[num_vars:, :num_vars] = constraint_jac
            master_grad = np.zeros(l_hessian.shape[0] + l_constraints.shape[0])
            master_grad[:l_hessian.shape[0]] = -np.array(gradient_term)
            master_grad[l_hessian.shape[0]:] = -l_constraints
            try:
                step = np.linalg.solve(master_hess, master_grad)
            except np.linalg.LinAlgError:
                print(cur_conds)
                raise
            for sfidx in range(site_fracs.shape[0]):
                site_fracs[sfidx] = min(max(site_fracs[sfidx] + alpha * step[sfidx], MIN_SITE_FRACTION), 1)
            for pfidx in range(phase_fracs.shape[0]):
                phase_fracs[pfidx] = min(max(phase_fracs[pfidx] + alpha * step[site_fracs.shape[0] + pfidx], -4), 5)
            dof_idx = 0
            for phase_idx in range(num_phases):
                compset = composition_sets[phase_idx]
                compset.update(site_fracs[dof_idx:dof_idx+compset.phase_record.phase_dof],
                                                   phase_fracs[phase_idx], cur_conds['P'], cur_conds['T'])
                dof_idx += compset.phase_record.phase_dof
            if verbose:
                print('Phases', phases)
                print('step', step)
                print('Site fractions', site_fracs)
                print('Phase fractions', phase_fracs)
            old_energy = copy.deepcopy(prop_GM_values[it.multi_index])
            old_chem_pots = copy.deepcopy(prop_MU_values[it.multi_index])
            l_multipliers = np.array(step[num_vars:])
            np.clip(l_multipliers, -MAX_ABS_LAGRANGE_MULTIPLIER, MAX_ABS_LAGRANGE_MULTIPLIER, out=l_multipliers)
            if np.any(np.isnan(l_multipliers)):
                print('Invalid l_multipliers after recalculation', l_multipliers)
                l_multipliers[:] = 0
            if verbose:
                print('NEW_L_MULTIPLIERS', l_multipliers)
            vmax = np.max(np.abs(l_constraints))
            num_mass_bals = len([i for i in cur_conds.keys() if i.startswith('X_')]) + 1
            chemical_potentials = l_multipliers[sum([len(dbf.phases[i].sublattices) for i in phases]):
                                                sum([len(dbf.phases[i].sublattices) for i in phases]) + num_mass_bals] / obj_weight
            prop_MU_values[it.multi_index] = chemical_potentials
            prop_NP_values[it.multi_index + np.index_exp[:len(phases)]] = phase_fracs
            prop_X_values[it.multi_index + np.index_exp[:len(phases)]] = 0
            prop_GM_values[it.multi_index] = energy / obj_weight
            var_offset = 0
            for phase_idx in range(len(phases)):
                prop_Y_values[it.multi_index + np.index_exp[phase_idx, :phase_dof[phase_idx]]] = \
                    site_fracs[var_offset:var_offset + phase_dof[phase_idx]]
                comp_idx = 0
                # Necessary to fix gh-62 and gh-63
                past_va = False
                for comp_idx, comp in enumerate(comps):
                    if comp == 'VA':
                        past_va = True
                        continue
                    mass_buf = np.zeros(1)
                    prn = phase_records[phases[phase_idx]]
                    prn.mass_obj(mass_buf,
                                 site_fracs[var_offset:var_offset + phase_dof[phase_idx]],
                                 comp_idx)
                    prop_X_values[it.multi_index + np.index_exp[phase_idx, comp_idx-int(past_va)]] = mass_buf[0]
                    comp_idx += 1
                var_offset += phase_dof[phase_idx]

            properties.attrs['solve_iterations'] += 1
            total_comp = np.nansum(prop_NP_values[it.multi_index][..., np.newaxis] * \
                                   prop_X_values[it.multi_index], axis=-2)
            driving_force = (prop_MU_values[it.multi_index] * total_comp).sum(axis=-1) - \
                             prop_GM_values[it.multi_index]
            driving_force = np.squeeze(driving_force)
            if diagnostic:
                diagnostic_matrix[cur_iter, 0] = cur_iter
                diagnostic_matrix[cur_iter, 1] = energy / obj_weight
                diagnostic_matrix[cur_iter, 2] = np.linalg.norm(step)
                diagnostic_matrix[cur_iter, 3] = driving_force
                diagnostic_matrix[cur_iter, 4] = vmax
                diagnostic_matrix[cur_iter, 5] = np.abs(prop_MU_values[it.multi_index] - old_chem_pots).max()
                diagnostic_matrix[cur_iter, 6] = obj_weight
                for iy, mu in enumerate(prop_MU_values[it.multi_index]):
                    diagnostic_matrix[cur_iter, 7+iy] = mu
            if verbose:
                print('Chem pot progress', prop_MU_values[it.multi_index] - old_chem_pots)
                print('Energy progress', prop_GM_values[it.multi_index] - old_energy)
                print('Driving force', driving_force)
                print('obj weight', obj_weight)
            no_progress = np.abs(prop_MU_values[it.multi_index] - old_chem_pots).max() < 0.1
            no_progress &= np.abs(prop_GM_values[it.multi_index] - old_energy) < MIN_SOLVE_ENERGY_PROGRESS
            no_progress &= np.abs(driving_force) < MAX_SOLVE_DRIVING_FORCE
            if no_progress:
                for pfidx in range(phase_fracs.shape[0]):
                    if phase_fracs[pfidx] < 0:
                        no_progress = False
                        allow_negative_fractions = False
            if no_progress and cur_iter == MAX_SOLVE_ITERATIONS-1:
                print('Driving force failed to converge: {}'.format(cur_conds))
                prop_MU_values[it.multi_index] = np.nan
                prop_NP_values[it.multi_index] = np.nan
                prop_X_values[it.multi_index] = np.nan
                prop_Y_values[it.multi_index] = np.nan
                prop_GM_values[it.multi_index] = np.nan
                prop_Phase_values[it.multi_index] = ''
                if diagnostic:
                    np.savetxt(debug_fn, diagnostic_matrix, delimiter=',')
                break
            elif no_progress:
                if verbose:
                    print('No progress')
                num_mass_bals = len([i for i in cur_conds.keys() if i.startswith('X_')]) + 1
                chemical_potentials = l_multipliers[sum([len(dbf.phases[i].sublattices) for i in phases]):
                                                    sum([len(dbf.phases[i].sublattices) for i in phases]) + num_mass_bals] / obj_weight
                prop_MU_values[it.multi_index] = chemical_potentials
                if diagnostic:
                    np.savetxt(debug_fn, diagnostic_matrix, delimiter=',')
                break
            elif (not no_progress) and cur_iter == MAX_SOLVE_ITERATIONS-1:
                if diagnostic:
                    np.savetxt(debug_fn, diagnostic_matrix, delimiter=',')
                print('Failed to converge: {}'.format(cur_conds))
                prop_MU_values[it.multi_index] = np.nan
                prop_NP_values[it.multi_index] = np.nan
                prop_X_values[it.multi_index] = np.nan
                prop_Y_values[it.multi_index] = np.nan
                prop_GM_values[it.multi_index] = np.nan
                prop_Phase_values[it.multi_index] = ''
            if (cur_iter > 0) and cur_iter % vmax_window_size == 0:
                new_window_average = np.median(vmax_averages)
                if (obj_decreases < 2) and (previous_window_average * new_window_average < 1e-20) and (cur_iter < 0.8 * MAX_SOLVE_ITERATIONS):
                    if obj_weight > 1:
                        obj_weight *= 0.1
                        l_multipliers *= 0.1
                        obj_decreases += 1
                        if verbose:
                            print('Decreasing objective weight')
                elif (obj_decreases < 2) and (new_window_average / previous_window_average > 10) and (cur_iter < 0.8 * MAX_SOLVE_ITERATIONS):
                    if obj_weight > 1:
                        obj_weight *= 0.1
                        l_multipliers *= 0.1
                        obj_decreases += 1
                        if verbose:
                            print('Decreasing objective weight')
                elif (new_window_average > 1e-12) or (np.linalg.norm(step) > 1e-5):
                    if obj_weight < 1e6:
                        obj_weight *= 10
                        l_multipliers *= 10
                        if verbose:
                            print('Increasing objective weight')
                previous_window_average = new_window_average
            if (cur_iter > 0.8 * MAX_SOLVE_ITERATIONS) and obj_weight == INITIAL_OBJECTIVE_WEIGHT:
                obj_weight *= 1000
                l_multipliers *= 1000
                if verbose:
                    print('Increasing objective weight to force convergence')
            vmax_averages[cur_iter % vmax_window_size] = vmax
        it.iternext()
    return properties