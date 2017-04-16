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
        cdef int has_va = <int>(prx.vacancy_index > -1)
        self.phase_record = prx
        self.dof = np.zeros(len(self.phase_record.variables)+2)
        self.X = np.zeros(self.phase_record.composition_matrices.shape[0]-has_va)
        self.mass_grad = np.zeros((self.X.shape[0]+has_va, self.phase_record.phase_dof))
        self.mass_hess = np.zeros((self.X.shape[0]+has_va, self.phase_record.phase_dof, self.phase_record.phase_dof))
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

    def __repr__(self):
        return str(self.__class__.__name__) + "({0}, {1})".format(self.phase_record.phase_name, np.asarray(self.X))

    cdef void reset(self):
        self._prev_energy = 0
        self._prev_dof[:] = 0
        self._prev_grad[:] = 0
        self._prev_hess[:,:] = 0
        self._first_iteration = True

    cdef void _hessian_update(self, double[::1] dof, double[:] prev_dof, double[::1,:] current_hess,
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


cdef bint remove_degenerate_phases(object composition_sets, bint allow_negative_fractions, bint verbose):
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
        name = <unicode>composition_sets[phase_idx].phase_record.phase_name
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
        if (composition_sets[phase_idx].NP <= MIN_SITE_FRACTION) and (not allow_negative_fractions):
            composition_sets[phase_idx].NP = np.nan
        elif abs(composition_sets[phase_idx].NP) <= MIN_SITE_FRACTION:
            composition_sets[phase_idx].NP = MIN_SITE_FRACTION


    entries_to_delete = sorted([idx for idx, compset in enumerate(composition_sets) if np.isnan(compset.NP)],
                               reverse=True)
    for idx in entries_to_delete:
        if verbose:
            print('Removing ' + repr(composition_sets[idx]))
        del composition_sets[idx]
    if len(entries_to_delete) > 0:
        return True
    else:
        return False

cdef bint add_new_phases(composition_sets, phase_records, current_grid, chemical_potentials, minimum_df, verbose):
    cdef double[:] driving_forces
    cdef int df_idx = 0
    cdef double largest_df = -np.inf
    cdef double[:] df_comp
    cdef unicode df_phase_name
    cdef CompositionSet compset
    cdef bint distinct = False
    driving_forces = (chemical_potentials * current_grid.X.values).sum(axis=-1) - current_grid.GM.values
    for i in range(driving_forces.shape[0]):
        if driving_forces[i] > largest_df:
            largest_df = driving_forces[i]
            df_idx = i
    if largest_df > minimum_df:
        # To add a phase, must not be within COMP_DIFFERENCE_TOL of composition of the same phase of its type
        df_comp = current_grid.X.values[df_idx]
        df_phase_name = str(current_grid.Phase.values[df_idx])
        for compset in composition_sets:
            if compset.phase_record.phase_name != df_phase_name:
                continue
            distinct = False
            for comp_idx in range(df_comp.shape[0]):
                if abs(df_comp[comp_idx] - compset.X[comp_idx]) > COMP_DIFFERENCE_TOL:
                    distinct = True
            if not distinct:
                if verbose:
                    print('Candidate composition set ' + df_phase_name + ' at ' + str(np.array(df_comp)) + ' is not distinct')
                return False
        # Set all phases to have equal amounts as new phase is added
        for compset in composition_sets:
            compset.NP = 1./(len(composition_sets)+1)
        compset = CompositionSet(phase_records[df_phase_name])
        compset.update(current_grid.Y.values[df_idx, :compset.phase_record.phase_dof], 1./(len(composition_sets)+1),
                       current_grid.coords['P'], current_grid.coords['T'])
        composition_sets.append(compset)
        if verbose:
            print('Adding ' + repr(compset) + ' Driving force: ' + str(largest_df))
        return True
    return False


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
def _compute_constraints(composition_sets, object comps, object cur_conds):
    """
    Compute the constraint vector and constraint Jacobian matrix.
    """
    cdef CompositionSet compset
    cdef int num_sitefrac_bals = sum([compset.phase_record.sublattice_dof.shape[0] for compset in composition_sets])
    cdef int num_mass_bals = len([i for i in cur_conds.keys() if i.startswith('X_')]) + 1
    cdef double indep_sum = sum([float(val) for i, val in cur_conds.items() if i.startswith('X_')])
    cdef double[::1] comp_obj_value = np.atleast_1d(np.zeros(1))
    cdef object dependent_comp = set(comps) - set([i[2:] for i in cur_conds.keys() if i.startswith('X_')]) - {'VA'}
    dependent_comp = list(dependent_comp)[0]
    cdef int num_constraints = num_sitefrac_bals + num_mass_bals
    cdef int num_phases = len(composition_sets)
    cdef int num_vars = sum(compset.phase_record.phase_dof for compset in composition_sets) + num_phases
    cdef double[::1] l_constraints = np.zeros(num_constraints)
    cdef double[::1,:] constraint_jac = np.zeros((num_constraints, num_vars), order='F')
    cdef np.ndarray[ndim=3, dtype=np.float64_t] constraint_hess = np.zeros((num_constraints, num_vars, num_vars), order='F')
    cdef int phase_idx, var_offset, constraint_offset, var_idx, iter_idx, grad_idx, \
        hess_idx, comp_idx, idx, sum_idx, ais_len, phase_offset
    cdef int past_va = 0

    # Ordering of constraints by row: sitefrac bal of each phase, then component mass balance
    # Ordering of constraints by column: site fractions of each phase, then phase fractions
    # First: Site fraction balance constraints
    var_idx = 0
    constraint_offset = 0
    for phase_idx in range(num_phases):
        compset = composition_sets[phase_idx]
        phase_offset = 0
        for idx in range(compset.phase_record.sublattice_dof.shape[0]):
            ais_len = compset.phase_record.sublattice_dof[idx]
            constraint_jac[constraint_offset + idx,
            var_idx:var_idx + ais_len] = 1
            l_constraints[constraint_offset + idx] = -1
            for sum_idx in range(ais_len):
                l_constraints[constraint_offset + idx] += compset.dof[2+sum_idx+phase_offset]
            var_idx += ais_len
            phase_offset += ais_len
        constraint_offset += compset.phase_record.sublattice_dof.shape[0]
    # Second: Mass balance of each component
    for comp_idx, comp in enumerate(comps):
        if comp == 'VA':
            past_va = 1
            continue
        var_offset = 0
        for phase_idx in range(num_phases):
            compset = composition_sets[phase_idx]
            spidx = num_vars - num_phases + phase_idx
            # current phase frac times the comp_grad
            for grad_idx in range(var_offset, var_offset + compset.phase_record.phase_dof):
                constraint_jac[constraint_offset, grad_idx] = \
                    compset.NP * compset.mass_grad[comp_idx, grad_idx - var_offset]
                constraint_hess[constraint_offset, spidx, grad_idx] = compset.mass_grad[comp_idx, grad_idx - var_offset]
                constraint_hess[constraint_offset, grad_idx, spidx] = compset.mass_grad[comp_idx, grad_idx - var_offset]
                for hess_idx in range(var_offset, var_offset + compset.phase_record.phase_dof):
                    constraint_hess[constraint_offset, grad_idx, hess_idx] = \
                        compset.NP * compset.mass_hess[comp_idx, grad_idx - var_offset, hess_idx - var_offset]
            l_constraints[constraint_offset] += compset.NP * compset.X[comp_idx-past_va]
            constraint_jac[constraint_offset, spidx] += compset.X[comp_idx-past_va]
            var_offset += compset.phase_record.phase_dof
        if comp != dependent_comp:
            l_constraints[constraint_offset] -= float(cur_conds['X_' + comp])
        else:
            # TODO: Assuming N=1 (fixed for dependent component)
            l_constraints[constraint_offset] -= (1 - indep_sum)
        constraint_offset += 1
    return np.array(l_constraints), np.array(constraint_jac), constraint_hess

@cython.boundscheck(False)
@cython.wraparound(False)
cdef _build_multiphase_system(composition_sets, l_constraints, constraint_jac,
                              np.ndarray[ndim=3, dtype=np.float64_t] constraint_hess,
                              np.ndarray[ndim=1, dtype=np.float64_t] l_multipliers,
                              double obj_weight):
    cdef CompositionSet compset
    cdef int num_phases = len(composition_sets)
    cdef int num_vars = sum(compset.phase_record.phase_dof for compset in composition_sets) + num_phases
    cdef double[::1,:] l_hessian = np.zeros((num_vars, num_vars), order='F')
    cdef double[::1] gradient_term = np.zeros(num_vars)
    cdef int var_offset = 0
    cdef int phase_idx = 0
    cdef int constraint_idx, dof_x_idx, dof_y_idx, hess_x, hess_y, hess_idx
    cdef double total_obj = 0

    for compset in composition_sets:
        for dof_x_idx in range(compset.phase_record.phase_dof):
            gradient_term[var_offset + dof_x_idx] = \
                obj_weight * compset.NP * compset.grad[2+dof_x_idx]  # Remove P,T grad part
        gradient_term[num_vars - num_phases + phase_idx] = obj_weight * compset.energy
        total_obj += obj_weight * compset.NP * compset.energy

        for dof_x_idx in range(compset.phase_record.phase_dof):
            for dof_y_idx in range(dof_x_idx,compset.phase_record.phase_dof):
                l_hessian[var_offset+dof_x_idx, var_offset+dof_y_idx] = \
                  obj_weight * compset.NP * compset.hess[2+dof_x_idx,2+dof_y_idx]
                l_hessian[var_offset+dof_y_idx, var_offset+dof_x_idx] = \
                  l_hessian[var_offset+dof_x_idx, var_offset+dof_y_idx]
            # Phase fraction / site fraction cross derivative
            l_hessian[num_vars - num_phases + phase_idx, var_offset + dof_x_idx] = \
                 obj_weight * compset.grad[2+dof_x_idx] # Remove P,T grad part
            l_hessian[var_offset + dof_x_idx, num_vars - num_phases + phase_idx] = obj_weight * compset.grad[2+dof_x_idx]
        var_offset += compset.phase_record.phase_dof
        phase_idx += 1
    l_hessian -= np.einsum('i,ijk->jk', l_multipliers, constraint_hess, order='F')
    return np.asarray(total_obj), np.asarray(l_hessian), np.asarray(gradient_term)

def _solve_eq_at_conditions(dbf, comps, properties, phase_records, grid, conds_keys, verbose,
                            diagnostic, compute_constraints):
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
    grid : Dataset
        Sample of energy landscape of the system.
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
        bint converged, changed_phases
        double previous_window_average, obj_weight, vmax, minimum_df
        PhaseRecord prn
        CompositionSet compset
        cdef double[::1,:] l_hessian
        cdef double[::1] gradient_term, mass_buf
        double[::1] vmax_averages
        np.ndarray[ndim=1, dtype=np.float64_t] p_y, l_constraints, step, chemical_potentials
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
        converged = False
        changed_phases = False
        cur_conds = OrderedDict(zip(conds_keys,
                                    [np.asarray(properties['GM'].coords[b][a], dtype=np.float)
                                     for a, b in zip(it.multi_index, conds_keys)]))
        if len(cur_conds) == 0:
            cur_conds = properties['GM'].coords
        current_grid = grid.sel(P=cur_conds['P'], T=cur_conds['T'])
        # sum of independently specified components
        indep_sum = np.sum([float(val) for i, val in cur_conds.items() if i.startswith('X_')])
        if indep_sum > 1:
            # Sum of independent component mole fractions greater than one
            # Skip this condition set
            # We silently allow this to make 2-D composition mapping easier
            prop_MU_values[it.multi_index] = np.nan
            prop_NP_values[it.multi_index + np.index_exp[:]] = np.nan
            prop_Phase_values[it.multi_index + np.index_exp[:]] = ''
            prop_X_values[it.multi_index + np.index_exp[:]] = np.nan
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
            sfx = prop_Y_values[it.multi_index + np.index_exp[phase_idx, :phrec.phase_dof]]
            phase_amt = prop_NP_values[it.multi_index + np.index_exp[phase_idx]]
            compset = CompositionSet(phrec)
            compset.update(sfx, phase_amt, cur_conds['P'], cur_conds['T'])
            composition_sets.append(compset)
        chemical_potentials = prop_MU_values[it.multi_index]
        energy = prop_GM_values[it.multi_index]
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
            if cur_iter > 0.8 * MAX_SOLVE_ITERATIONS:
                allow_negative_fractions = False
            if cur_iter > 0 and cur_iter % 5 == 0:
                if cur_iter == 0:
                    minimum_df = -10
                else:
                    minimum_df = 0
                changed_phases |= add_new_phases(composition_sets, phase_records, current_grid, chemical_potentials, minimum_df, verbose)
            changed_phases |= remove_degenerate_phases(composition_sets, allow_negative_fractions, verbose)
            num_phases = len(composition_sets)
            total_dof = sum([compset.phase_record.phase_dof for compset in composition_sets])
            if num_phases == 0:
                print('Zero phases are left in the system: {}'.format(cur_conds))
                converged = False
                if diagnostic:
                    np.savetxt(debug_fn, diagnostic_matrix, delimiter=',')
                break
            phase_fracs = np.empty(num_phases)
            for phase_idx in range(num_phases):
                phase_fracs[phase_idx] = composition_sets[phase_idx].NP
            dof_idx = 0
            site_fracs = np.empty(total_dof)
            for phase_idx in range(num_phases):
                site_fracs[dof_idx:dof_idx+composition_sets[phase_idx].phase_record.phase_dof] = composition_sets[phase_idx].dof[2:]
                dof_idx += composition_sets[phase_idx].phase_record.phase_dof

            if (num_phases == 1) and np.all(np.asarray(composition_sets[0].dof[2:]) == 1.):
                # Single phase with zero internal degrees of freedom, can't do any refinement
                # TODO: In the future we may be able to refine other degrees of freedom like temperature
                # Chemical potentials have no meaning for this case
                chemical_potentials[:] = np.nan
                converged = True
                break

            l_constraints, constraint_jac, constraint_hess = compute_constraints(composition_sets, comps, cur_conds)
            # Reset Lagrange multipliers if active set of phases change
            if cur_iter == 0 or changed_phases or np.any(np.isnan(l_multipliers)):
                l_multipliers = np.zeros(l_constraints.shape[0])
                changed_phases = False
            num_vars = len(site_fracs) + len(composition_sets)
            old_energy = energy / obj_weight
            old_chem_pots = chemical_potentials.copy()
            energy, l_hessian, gradient_term = _build_multiphase_system(composition_sets, l_constraints,
                                                                        constraint_jac, constraint_hess,
                                                                        l_multipliers, obj_weight)
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
            if verbose:
                print('Phases', composition_sets)
                print('step', step)
                print('Site fractions', site_fracs)
                print('Phase fractions', phase_fracs)
            dof_idx = 0
            total_comp = np.zeros(prop_X_values.shape[-1])
            for phase_idx in range(num_phases):
                compset = composition_sets[phase_idx]
                compset.update(site_fracs[dof_idx:dof_idx+compset.phase_record.phase_dof],
                                                   phase_fracs[phase_idx], cur_conds['P'], cur_conds['T'])
                for comp_idx in range(total_comp.shape[0]):
                    total_comp[comp_idx] += compset.NP * compset.X[comp_idx]
                dof_idx += compset.phase_record.phase_dof
            l_multipliers = np.array(step[num_vars:])
            np.clip(l_multipliers, -MAX_ABS_LAGRANGE_MULTIPLIER, MAX_ABS_LAGRANGE_MULTIPLIER, out=l_multipliers)
            if np.any(np.isnan(l_multipliers)):
                print('Invalid l_multipliers after recalculation', l_multipliers)
                l_multipliers[:] = 0
            if verbose:
                print('NEW_L_MULTIPLIERS', l_multipliers)
            vmax = np.max(np.abs(l_constraints))
            num_mass_bals = len([i for i in cur_conds.keys() if i.startswith('X_')]) + 1
            chemical_potentials[:] = l_multipliers[sum([compset.phase_record.sublattice_dof.shape[0] for compset in composition_sets]):
                                                   sum([compset.phase_record.sublattice_dof.shape[0] for compset in composition_sets]) + num_mass_bals] / obj_weight

            driving_force = (chemical_potentials * total_comp).sum(axis=-1) - \
                             energy / obj_weight
            driving_force = np.squeeze(driving_force)
            if diagnostic:
                diagnostic_matrix[cur_iter, 0] = cur_iter
                diagnostic_matrix[cur_iter, 1] = energy / obj_weight
                diagnostic_matrix[cur_iter, 2] = np.linalg.norm(step)
                diagnostic_matrix[cur_iter, 3] = driving_force
                diagnostic_matrix[cur_iter, 4] = vmax
                diagnostic_matrix[cur_iter, 5] = np.abs(chemical_potentials - old_chem_pots).max()
                diagnostic_matrix[cur_iter, 6] = obj_weight
                for iy, mu in enumerate(chemical_potentials):
                    diagnostic_matrix[cur_iter, 7+iy] = mu
            if verbose:
                print('Chemical potentials', np.asarray(chemical_potentials))
                print('Chem pot progress', chemical_potentials - old_chem_pots)
                print('Energy progress', energy / obj_weight - old_energy)
                print('Driving force', driving_force)
                print('obj weight', obj_weight)
            no_progress = np.abs(chemical_potentials - old_chem_pots).max() < 0.1
            no_progress &= np.abs(energy / obj_weight - old_energy) < MIN_SOLVE_ENERGY_PROGRESS
            no_progress &= np.abs(driving_force) < MAX_SOLVE_DRIVING_FORCE
            no_progress &= num_phases <= prop_Phase_values.shape[-1]
            if no_progress:
                for pfidx in range(phase_fracs.shape[0]):
                    if phase_fracs[pfidx] < 0:
                        no_progress = False
                        allow_negative_fractions = False
            if no_progress and cur_iter == MAX_SOLVE_ITERATIONS-1:
                print('Driving force failed to converge: {}'.format(cur_conds))
                if diagnostic:
                    np.savetxt(debug_fn, diagnostic_matrix, delimiter=',')
                converged = False
                break
            elif no_progress:
                if verbose:
                    print('No progress')
                num_mass_bals = len([i for i in cur_conds.keys() if i.startswith('X_')]) + 1
                chemical_potentials = l_multipliers[sum([compset.phase_record.sublattice_dof.shape[0] for compset in composition_sets]):
                                                sum([compset.phase_record.sublattice_dof.shape[0] for compset in composition_sets]) + num_mass_bals] / obj_weight
                if diagnostic:
                    np.savetxt(debug_fn, diagnostic_matrix, delimiter=',')
                converged = True
                break
            elif (not no_progress) and cur_iter == MAX_SOLVE_ITERATIONS-1:
                if diagnostic:
                    np.savetxt(debug_fn, diagnostic_matrix, delimiter=',')
                print('Failed to converge: {}'.format(cur_conds))
                converged = False
                break
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

        if converged:
            prop_MU_values[it.multi_index] = chemical_potentials
            prop_NP_values[it.multi_index + np.index_exp[:len(composition_sets)]] = phase_fracs
            prop_NP_values[it.multi_index + np.index_exp[len(composition_sets):]] = np.nan
            prop_X_values[it.multi_index + np.index_exp[:]] = 0
            prop_GM_values[it.multi_index] = energy / obj_weight
            for phase_idx in range(len(composition_sets)):
                prop_Phase_values[it.multi_index + np.index_exp[phase_idx]] = composition_sets[phase_idx].phase_record.phase_name
            for phase_idx in range(len(composition_sets), prop_Phase_values.shape[-1]):
                prop_Phase_values[it.multi_index + np.index_exp[phase_idx]] = ''
                prop_X_values[it.multi_index + np.index_exp[phase_idx, :]] = np.nan
            var_offset = 0
            total_comp = np.zeros(prop_X_values.shape[-1])
            for phase_idx in range(num_phases):
                compset = composition_sets[phase_idx]
                prop_Y_values[it.multi_index + np.index_exp[phase_idx, :compset.phase_record.phase_dof]] = \
                    site_fracs[var_offset:var_offset + compset.phase_record.phase_dof]
                prop_X_values[it.multi_index + np.index_exp[phase_idx, :]] = compset.X
                var_offset += compset.phase_record.phase_dof
        else:
            prop_MU_values[it.multi_index] = np.nan
            prop_NP_values[it.multi_index] = np.nan
            prop_X_values[it.multi_index] = np.nan
            prop_Y_values[it.multi_index] = np.nan
            prop_GM_values[it.multi_index] = np.nan
            prop_Phase_values[it.multi_index] = ''
        it.iternext()
    return properties