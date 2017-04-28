from collections import defaultdict, OrderedDict
import numpy as np
cimport numpy as np
cimport cython
cdef extern from "_isnan.h":
    bint isnan (double) nogil
import scipy.spatial
from pycalphad.core.composition_set cimport CompositionSet
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


cdef bint remove_degenerate_phases(object composition_sets, bint allow_negative_fractions, bint verbose):
    """
    For each phase pair with composition difference below tolerance,
    eliminate phase with largest index.
    Also remove phases with phase fractions close to zero.
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
    cdef object phase_indices = defaultdict(list)
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

cdef bint add_new_phases(object composition_sets, object phase_records,
                         object current_grid, np.ndarray[ndim=1, dtype=np.float64_t] chemical_potentials,
                         double minimum_df, bint verbose):
    """
    Attempt to add a new phase with the largest driving force (based on chemical potentials). Candidate phases
    are taken from current_grid and modify the composition_sets object. The function returns a boolean indicating
    whether it modified composition_sets.
    """
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

@cython.boundscheck(False)
@cython.wraparound(False)
def _compute_constraints(object composition_sets, object comps, object cur_conds):
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
        hess_idx, comp_idx, idx, sum_idx, active_in_subl, phase_offset
    cdef int vacancy_offset = 0

    # Ordering of constraints by row: sitefrac bal of each phase, then component mass balance
    # Ordering of constraints by column: site fractions of each phase, then phase fractions
    # First: Site fraction balance constraints
    var_idx = 0
    constraint_offset = 0
    for phase_idx in range(num_phases):
        compset = composition_sets[phase_idx]
        phase_offset = 0
        for idx in range(compset.phase_record.sublattice_dof.shape[0]):
            active_in_subl = compset.phase_record.sublattice_dof[idx]
            constraint_jac[constraint_offset + idx,
            var_idx:var_idx + active_in_subl] = 1
            l_constraints[constraint_offset + idx] = -1
            for sum_idx in range(active_in_subl):
                l_constraints[constraint_offset + idx] += compset.dof[2+sum_idx+phase_offset]
            var_idx += active_in_subl
            phase_offset += active_in_subl
        constraint_offset += compset.phase_record.sublattice_dof.shape[0]
    # Second: Mass balance of each component
    for comp_idx, comp in enumerate(comps):
        if comp == 'VA':
            vacancy_offset = 1
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
            l_constraints[constraint_offset] += compset.NP * compset.X[comp_idx-vacancy_offset]
            constraint_jac[constraint_offset, spidx] += compset.X[comp_idx-vacancy_offset]
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
cdef _build_multiphase_system(object composition_sets, np.ndarray[ndim=1, dtype=np.float64_t] l_constraints,
                              np.ndarray[ndim=2, dtype=np.float64_t] constraint_jac,
                              np.ndarray[ndim=3, dtype=np.float64_t] constraint_hess,
                              np.ndarray[ndim=1, dtype=np.float64_t] l_multipliers):
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
                compset.NP * compset.grad[2+dof_x_idx]  # Remove P,T grad part
        gradient_term[num_vars - num_phases + phase_idx] = compset.energy
        total_obj += compset.NP * compset.energy

        for dof_x_idx in range(compset.phase_record.phase_dof):
            for dof_y_idx in range(dof_x_idx,compset.phase_record.phase_dof):
                l_hessian[var_offset+dof_x_idx, var_offset+dof_y_idx] = \
                  compset.NP * compset.hess[2+dof_x_idx,2+dof_y_idx]
                l_hessian[var_offset+dof_y_idx, var_offset+dof_x_idx] = \
                  l_hessian[var_offset+dof_x_idx, var_offset+dof_y_idx]
            # Phase fraction / site fraction cross derivative
            l_hessian[num_vars - num_phases + phase_idx, var_offset + dof_x_idx] = \
                 compset.grad[2+dof_x_idx] # Remove P,T grad part
            l_hessian[var_offset + dof_x_idx, num_vars - num_phases + phase_idx] = compset.grad[2+dof_x_idx]
        var_offset += compset.phase_record.phase_dof
        phase_idx += 1
    l_hessian -= np.einsum('i,ijk->jk', l_multipliers, constraint_hess, order='F')
    return np.asarray(total_obj), np.asarray(l_hessian), np.asarray(gradient_term)

def _solve_eq_at_conditions(comps, properties, phase_records, grid, conds_keys, verbose):
    """
    Compute equilibrium for the given conditions.
    This private function is meant to be called from a worker subprocess.
    For that case, usually only a small slice of the master 'properties' is provided.
    Since that slice will be copied, we also return the modified 'properties'.

    Parameters
    ----------
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

    Returns
    -------
    properties : Dataset
        Modified with equilibrium values.
    """
    cdef double indep_sum
    cdef int num_phases, num_vars, cur_iter, old_phase_length, new_phase_length, var_idx, sfidx, pfidx, m, n
    cdef int vmax_window_size
    cdef int obj_decreases
    cdef bint converged, changed_phases
    cdef double previous_window_average, vmax, minimum_df
    cdef PhaseRecord prn
    cdef CompositionSet compset
    cdef double[::1,:] l_hessian
    cdef double[::1] gradient_term, mass_buf
    cdef double[::1] vmax_averages
    cdef np.ndarray[ndim=1, dtype=np.float64_t] p_y, l_constraints, step, chemical_potentials
    cdef np.ndarray[ndim=1, dtype=np.float64_t] site_fracs, l_multipliers, phase_fracs
    cdef np.ndarray[ndim=2, dtype=np.float64_t] ymat, zmat, qmat, rmat, constraint_jac

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
        alpha = 1
        allow_negative_fractions = False
        for cur_iter in range(MAX_SOLVE_ITERATIONS):
            if cur_iter > 0.8 * MAX_SOLVE_ITERATIONS:
                allow_negative_fractions = False
            if cur_iter > 0 and cur_iter % 5 == 0:
                minimum_df = 0
                changed_phases |= add_new_phases(composition_sets, phase_records, current_grid, chemical_potentials, minimum_df, verbose)
            changed_phases |= remove_degenerate_phases(composition_sets, allow_negative_fractions, verbose)
            num_phases = len(composition_sets)
            total_dof = sum([compset.phase_record.phase_dof for compset in composition_sets])
            if num_phases == 0:
                print('Zero phases are left in the system: {}'.format(cur_conds))
                converged = False
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

            l_constraints, constraint_jac, constraint_hess = _compute_constraints(composition_sets, comps, cur_conds)
            # Reset Lagrange multipliers if active set of phases change
            if cur_iter == 0 or changed_phases or np.any(np.isnan(l_multipliers)):
                l_multipliers = np.zeros(l_constraints.shape[0])
                changed_phases = False
            num_vars = len(site_fracs) + len(composition_sets)
            old_energy = energy
            old_chem_pots = chemical_potentials.copy()
            energy, l_hessian, gradient_term = _build_multiphase_system(composition_sets, l_constraints,
                                                                        constraint_jac, constraint_hess,
                                                                        l_multipliers)
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
                                                   sum([compset.phase_record.sublattice_dof.shape[0] for compset in composition_sets]) + num_mass_bals]

            driving_force = (chemical_potentials * total_comp).sum(axis=-1) - \
                             energy
            driving_force = np.squeeze(driving_force)
            if verbose:
                print('Chemical potentials', np.asarray(chemical_potentials))
                print('Chem pot progress', chemical_potentials - old_chem_pots)
                print('Energy progress', energy - old_energy)
                print('Driving force', driving_force)
            no_progress = np.abs(chemical_potentials - old_chem_pots).max() < 0.1
            no_progress &= np.abs(energy - old_energy) < MIN_SOLVE_ENERGY_PROGRESS
            no_progress &= np.abs(driving_force) < MAX_SOLVE_DRIVING_FORCE
            no_progress &= num_phases <= prop_Phase_values.shape[-1]
            if no_progress:
                for pfidx in range(phase_fracs.shape[0]):
                    if phase_fracs[pfidx] < 0:
                        no_progress = False
                        allow_negative_fractions = False
            if no_progress and cur_iter == MAX_SOLVE_ITERATIONS-1:
                print('Driving force failed to converge: {}'.format(cur_conds))
                converged = False
                break
            elif no_progress:
                if verbose:
                    print('No progress')
                num_mass_bals = len([i for i in cur_conds.keys() if i.startswith('X_')]) + 1
                chemical_potentials = l_multipliers[sum([compset.phase_record.sublattice_dof.shape[0] for compset in composition_sets]):
                                                sum([compset.phase_record.sublattice_dof.shape[0] for compset in composition_sets]) + num_mass_bals]
                converged = True
                break
            elif (not no_progress) and cur_iter == MAX_SOLVE_ITERATIONS-1:
                print('Failed to converge: {}'.format(cur_conds))
                converged = False
                break

        if converged:
            prop_MU_values[it.multi_index] = chemical_potentials
            prop_NP_values[it.multi_index + np.index_exp[:len(composition_sets)]] = phase_fracs
            prop_NP_values[it.multi_index + np.index_exp[len(composition_sets):]] = np.nan
            prop_X_values[it.multi_index + np.index_exp[:]] = 0
            prop_GM_values[it.multi_index] = energy
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