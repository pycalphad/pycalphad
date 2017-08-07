from collections import defaultdict, OrderedDict
import operator
from itertools import chain
import numpy as np
cimport numpy as np
cimport cython
cdef extern from "_isnan.h":
    bint isnan (double) nogil
import scipy.spatial
import collections
from pycalphad.core.problem cimport Problem
from pycalphad.core.solver import InteriorPointSolver
from pycalphad.core.hyperplane cimport hyperplane
from pycalphad.core.composition_set cimport CompositionSet
from pycalphad.core.phase_rec cimport PhaseRecord, PhaseRecord_from_cython
from pycalphad.core.constants import MIN_SITE_FRACTION, MIN_PHASE_FRACTION, COMP_DIFFERENCE_TOL, BIGNUM
import pycalphad.variables as v

# Maximum residual driving force (J/mol-atom) allowed for convergence
MAX_SOLVE_DRIVING_FORCE = 1e-4
# Maximum number of multi-phase solver iterations
MAX_SOLVE_ITERATIONS = 300
# Minimum energy (J/mol-atom) difference between iterations before stopping solver
MIN_SOLVE_ENERGY_PROGRESS = 1e-3
# Maximum absolute value of a Lagrange multiplier before it's recomputed with an alternative method
MAX_ABS_LAGRANGE_MULTIPLIER = 1e16

cdef bint remove_degenerate_phases(object composition_sets,
                                   double comp_diff_tol, int allowed_zero_seen, bint verbose):
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
    cdef int phase_idx, sidx, idx
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
        comp_matrix = np.full((len(composition_sets), composition_sets[0].X.shape[0]), np.inf)
        # The reason we don't do this based on Y fractions is because
        # of sublattice symmetry. It's very easy to detect a "miscibility gap" which is actually
        # symmetry equivalent, i.e., D([A, B] - [B, A]) > tol, but they are the same configuration.
        for idx in range(num_phases):
            compset = composition_sets[idx]
            comp_matrix[idx, :] = compset.X
        comp_distances = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(comp_matrix, metric='chebyshev'))
        for idx in range(comp_distances.shape[0]):
            if idx not in indices:
                comp_distances[idx,:] = np.inf
                comp_distances[:,idx] = np.inf
        redundant_phases = set()
        for i in range(comp_distances.shape[0]):
            for j in range(i, comp_distances.shape[0]):
                if i == j:
                    continue
                if comp_distances[i, j] < comp_diff_tol:
                    redundant_phases |= {i, j}
        redundant_phases = sorted(redundant_phases)
        if len(redundant_phases) > 1:
            kept_phase = redundant_phases[0]
            removed_phases = redundant_phases[1:]
        else:
            removed_phases = []
        # Their NP values will be added to the kept phase
        # and they will be nulled out
        for redundant in removed_phases:
            composition_sets[kept_phase].NP += composition_sets[redundant].NP
            if verbose:
                print('Redundant phase:', composition_sets[redundant])
            composition_sets[redundant].NP = np.nan
    for phase_idx in range(num_phases):
        if abs(composition_sets[phase_idx].NP) <= MIN_PHASE_FRACTION:
            composition_sets[phase_idx].NP = MIN_PHASE_FRACTION
            composition_sets[phase_idx].zero_seen += 1
            if composition_sets[phase_idx].zero_seen > allowed_zero_seen:
                if verbose:
                    print('Exceeded zero seen:', composition_sets[phase_idx])
                composition_sets[phase_idx].NP = np.nan

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

@cython.boundscheck(False)
cdef bint add_new_phases(object composition_sets, object phase_records,
                         object current_grid, np.ndarray[ndim=1, dtype=np.float64_t] chemical_potentials,
                         double minimum_df, comps, cur_conds, bint verbose) except *:
    """
    Attempt to add a new phase with the largest driving force (based on chemical potentials). Candidate phases
    are taken from current_grid and modify the composition_sets object. The function returns a boolean indicating
    whether it modified composition_sets.
    """
    cdef object solver = InteriorPointSolver(verbose=verbose)
    cdef double[::1] driving_forces
    cdef int max_endmembers = max([np.prod(x.sublattice_dof) for x in phase_records.values()])
    cdef int num_df_candidates = 10
    cdef np.ndarray largest_driving_forces_indices = np.full((len(phase_records), num_df_candidates+max_endmembers), -1, dtype=int)
    cdef int df_idx = 0
    cdef int i, idx, comp_idx, sfidx, dof_idx
    cdef double largest_df = -np.inf
    cdef size_t min_phase_fraction_idx
    cdef double[:] df_comp
    cdef double[::1] sfx
    cdef double[:] step
    cdef double[:,::1] current_grid_Y = current_grid.Y.values
    cdef double[:,::1] current_grid_X = current_grid.X.values
    cdef np.ndarray current_grid_Phase = current_grid.Phase.values
    cdef object phase_indices = [(idx, np.sort(np.argwhere(current_grid_Phase == <unicode>key))[:,0]) for idx, key in enumerate(sorted(phase_records.keys()))]
    cdef unicode df_phase_name
    cdef CompositionSet compset
    cdef CompositionSet candidate_compset
    cdef CompositionSet existing_candidate
    cdef object candidates = defaultdict(list)
    cdef bint distinct = False
    cdef long[::1] part_indices
    cdef double df, sublsum
    driving_forces = (chemical_potentials * current_grid.X.values).sum(axis=-1) - current_grid.GM.values
    # For each phase, choose 'num_df_candidates' points with the most driving force
    for phase_idx, phase_idx_arr in phase_indices:
        df_phase_name = <unicode>current_grid_Phase[phase_idx_arr[0]]
        if phase_idx_arr.shape[0] <= num_df_candidates:
            largest_driving_forces_indices[phase_idx, :phase_idx_arr.shape[0]] = phase_idx_arr
        else:
            part_indices = np.argpartition(driving_forces[phase_idx_arr[0]:phase_idx_arr[-1]], -num_df_candidates)[-num_df_candidates:]
            largest_driving_forces_indices[phase_idx, :num_df_candidates] = phase_idx_arr[part_indices]
            # Force endmembers to be candidates regardless of driving force
            # This addresses an issue with missing candidates due to sampling too few points (gh-103).
            # Doing this helps escape metastable configurations.
            largest_driving_forces_indices[phase_idx,
                                           num_df_candidates:num_df_candidates+np.prod(phase_records[df_phase_name].sublattice_dof)] = \
                np.arange(phase_idx_arr[0],
                          phase_idx_arr[0]+np.prod(phase_records[df_phase_name].sublattice_dof))
    # For each phase's point, generate a CompositionSet and try to refine it at fixed potential
    for phase_idx in range(len(phase_records)):
        df_phase_name = <unicode>current_grid_Phase[phase_indices[phase_idx][1][0]]
        for i in largest_driving_forces_indices[phase_idx]:
            if i < 0:
                continue
            compset = CompositionSet(phase_records[df_phase_name])
            sfx = current_grid_Y[i, :compset.phase_record.phase_dof].copy()
            for solve_iter in range(100):
                compset.update(sfx, 1.0, cur_conds['P'], cur_conds['T'], False)
                if np.sum(compset.X) > 1.01:
                    raise ValueError(repr(compset) + str(np.array(compset.dof)))
                # We assume here that all single-phase constraints are satisfied
                # Potentially not true for charge balance constraints
                l_constraints, constraint_jac, constraint_hess = _compute_constraints([compset], comps, cur_conds)

                # Exclude mass balance constraints in computation of other Lagrange multipliers
                reduced_constraint_jac = constraint_jac[:-len(set(comps) - {'VA'}), :-1]
                # Compute null and range space
                # reduced_constraint_jac is m * n
                m = reduced_constraint_jac.shape[0]
                n = reduced_constraint_jac.shape[1]
                q, r = np.linalg.qr(reduced_constraint_jac.T, mode='complete')
                y = q[:, :m]
                z = q[:, m:]
                #print('refined reduced hessian eigenvalues', np.linalg.eigvals(np.dot(np.dot(z.T, compset.hess[2:,2:]), z)))
                if z.size == 0:
                    break
                reduced_hess = np.dot(np.dot(z.T, compset.hess[2:,2:]), z)
                U, s, V = np.linalg.svd(reduced_hess, full_matrices=False)
                # Constrain eigenvalues to be between these values
                np.clip(s, 1e-4, 1e6)
                S = np.diag(s)
                reduced_hess = np.dot(U, np.dot(S, V))

                try:
                    cons_hess_inv = np.linalg.inv(reduced_hess)
                    a = np.dot(z.T, compset.grad[2:] - np.dot(constraint_jac[-len(set(comps) - {'VA'}):,:-1].T, chemical_potentials))
                    b = np.dot(cons_hess_inv, a)
                    step = -np.dot(z, b)
                except np.linalg.LinAlgError:
                    print(np.array(constraint_jac))
                    break
                if np.max(np.abs(step)) < 1e-12:
                    #print('Breaking at', np.array(step))
                    break
                for sfidx in range(sfx.shape[0]):
                    sfx[sfidx] = min(max(sfx[sfidx] + step[sfidx], MIN_SITE_FRACTION), 1.0)
                dof_idx = 0
                for i in compset.phase_record.sublattice_dof:
                    sublsum = sum(sfx[dof_idx:dof_idx+i])
                    for sfidx in range(dof_idx, dof_idx+i):
                        sfx[sfidx] /= sublsum
                    dof_idx += i
            df = np.multiply(chemical_potentials, compset.X).sum() - compset.energy
            #print('Testing', (compset, df))
            comp_distances = np.array([np.max(np.abs(np.array(existing_candidate.X) - np.array(compset.X)))
                                       for existing_candidate, existing_df in candidates[df_phase_name]])
            if df > -MAX_SOLVE_DRIVING_FORCE and (comp_distances.shape[0] == 0 or np.min(comp_distances) > 1e-4):
                candidates[df_phase_name].append((compset, df))

    if verbose:
        print('Candidates to add: ')
        print(candidates)

    candidates = list(chain(*candidates.values()))
    candidate_dfs = [x[1] for x in candidates]
    candidates = [x[0] for x in candidates]
    phases_changed = False
    if (len(candidates) > 0) and (max(candidate_dfs) > MAX_SOLVE_DRIVING_FORCE):
        phases_changed = True
        # First N points are fictitious
        fict_matrix = np.full((chemical_potentials.shape[0], chemical_potentials.shape[0]), MIN_SITE_FRACTION)
        fict_matrix[np.diag_indices(fict_matrix.shape[0])] = 1
        compositions = np.r_[fict_matrix, np.array([compset.X for compset in chain(composition_sets, candidates)])]
        energies = np.array([compset.energy for compset in chain(composition_sets, candidates)])
        energies = np.r_[np.repeat(np.max(energies)+1e4, chemical_potentials.shape[0]), energies]
        result_fractions = np.zeros(chemical_potentials.shape[0])
        best_guess_simplex = np.array(np.arange(chemical_potentials.shape[0]), dtype=np.int32)
        comp_conds = sorted([x for x in sorted(cur_conds.keys()) if x.startswith('X_')])
        pot_conds = sorted([x for x in sorted(cur_conds.keys()) if x.startswith('MU_')])

        comp_values = [cur_conds[cond] for cond in comp_conds]
        # Insert dependent composition value
        # TODO: Handle W(comp) as well as X(comp) here
        specified_components = {x[2:] for x in comp_conds}
        dependent_component = set(current_grid.coords['component'].values) - specified_components
        dependent_component = list(dependent_component)
        if len(dependent_component) != 1:
            raise ValueError('Number of dependent components is different from one')
        insert_idx = sorted(current_grid.coords['component'].values).index(dependent_component[0])
        comp_values = np.r_[comp_values[:insert_idx], 1 - np.sum(comp_values), comp_values[insert_idx:]]
        # Prevent compositions near an edge from going negative
        np.clip(comp_values, MIN_SITE_FRACTION*10, 1.0, out=comp_values)

        try:
            result_energy = hyperplane(compositions, energies, comp_values,
                                       chemical_potentials, result_fractions, best_guess_simplex)
        except ValueError as e:
            if verbose:
                print(e)
            return False
        new_composition_sets = []
        best_guess_simplex = np.array(best_guess_simplex, dtype=np.int32)
        if verbose:
            print('best_guess_simplex', np.array(best_guess_simplex))
        for idx in range(best_guess_simplex.shape[0]):
            i = best_guess_simplex[idx]
            if i < chemical_potentials.shape[0]:
                # Don't try to add fictitious points
                continue
            elif i < len(composition_sets) + chemical_potentials.shape[0]:
                compset = composition_sets[i-chemical_potentials.shape[0]]
            else:
                compset = candidates[i - len(composition_sets) - chemical_potentials.shape[0]]
            compset.NP = max(MIN_PHASE_FRACTION, result_fractions[idx])
            if verbose:
                print('Adding ' + repr(compset))
            new_composition_sets.append(compset)
        cs_sum = sum(compset.NP for compset in new_composition_sets)
        for compset in new_composition_sets:
            compset.NP /= cs_sum
        if (len(new_composition_sets) > 0) and (composition_sets != new_composition_sets):
            composition_sets[:] = new_composition_sets
        else:
            phases_changed = False
    return phases_changed

@cython.boundscheck(False)
@cython.wraparound(False)
cdef _compute_constraints(object composition_sets, object comps, object cur_conds):
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
    cdef double[:,::1] constraint_jac = np.zeros((num_constraints, num_vars))
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

def _solve_eq_at_conditions(comps, properties, phase_records, grid, conds_keys, verbose,
                            problem=Problem, solver=InteriorPointSolver):
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
    cdef int num_phases, num_vars, cur_iter, old_phase_length, new_phase_length, var_idx, dof_idx, comp_idx, phase_idx, sfidx, pfidx, m, n
    cdef bint converged, changed_phases
    cdef double vmax, minimum_df
    cdef PhaseRecord phase_record
    cdef CompositionSet compset
    cdef double[:,::1] l_hessian
    cdef double[:,:] inv_hess
    cdef double[::1] gradient_term, mass_buf
    cdef np.ndarray[ndim=1, dtype=np.float64_t] p_y, l_constraints, step, chemical_potentials
    cdef np.ndarray[ndim=1, dtype=np.float64_t] site_fracs, l_multipliers, phase_fracs
    cdef np.ndarray[ndim=2, dtype=np.float64_t] constraint_jac

    for key, value in phase_records.items():
        if not isinstance(phase_records[key], PhaseRecord):
            phase_records[key] = PhaseRecord_from_cython(comps, value.variables, np.array(value.num_sites, dtype=np.float),
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
        num_mass_bals = len([i for i in cur_conds.keys() if i.startswith('X_')]) + 1
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
            phase_record = phase_records[phase_name]
            sfx = prop_Y_values[it.multi_index + np.index_exp[phase_idx, :phase_record.phase_dof]]
            phase_amt = prop_NP_values[it.multi_index + np.index_exp[phase_idx]]
            phase_amt = max(phase_amt, MIN_PHASE_FRACTION)
            compset = CompositionSet(phase_record)
            compset.update(sfx, phase_amt, cur_conds['P'], cur_conds['T'], False)
            composition_sets.append(compset)
        chemical_potentials = prop_MU_values[it.multi_index]
        energy = prop_GM_values[it.multi_index]
        # Remove duplicate phases -- we will add them back later
        remove_degenerate_phases(composition_sets, 0.5, 100, verbose)
        iter_solver = solver(verbose=verbose)
        iterations = 0
        while iterations < 10:
            prob = problem(composition_sets, comps, cur_conds)
            result = iter_solver.solve(prob)
            composition_sets = prob.composition_sets
            if result.converged:
                x = result.x
                chemical_potentials = result.chemical_potentials
                var_offset = 0
                phase_idx = 0
                for compset in composition_sets:
                    compset.update(x[var_offset:var_offset + compset.phase_record.phase_dof],
                                   x[prob.num_vars - prob.num_phases + phase_idx], cur_conds['P'], cur_conds['T'], True)
                    var_offset += compset.phase_record.phase_dof
                    phase_idx += 1
            changed_phases = add_new_phases(composition_sets, phase_records,
                                            current_grid, chemical_potentials,
                                            1e-4, comps, cur_conds, verbose)
            iterations += 1
            if not changed_phases:
                chemical_potentials[:] = result.chemical_potentials
                break
        converged = result.converged
        remove_degenerate_phases(composition_sets, 1e-3, 0, verbose)
        if converged:
            prop_MU_values[it.multi_index] = chemical_potentials
            prop_Phase_values[it.multi_index] = ''
            prop_NP_values[it.multi_index + np.index_exp[:len(composition_sets)]] = [compset.NP for compset in composition_sets]
            prop_NP_values[it.multi_index + np.index_exp[len(composition_sets):]] = np.nan
            prop_Y_values[it.multi_index] = np.nan
            prop_X_values[it.multi_index + np.index_exp[:]] = 0
            prop_GM_values[it.multi_index] = 0
            for phase_idx in range(len(composition_sets)):
                prop_Phase_values[it.multi_index + np.index_exp[phase_idx]] = composition_sets[phase_idx].phase_record.phase_name
            for phase_idx in range(len(composition_sets), prop_Phase_values.shape[-1]):
                prop_Phase_values[it.multi_index + np.index_exp[phase_idx]] = ''
                prop_X_values[it.multi_index + np.index_exp[phase_idx, :]] = np.nan
            var_offset = 0
            total_comp = np.zeros(prop_X_values.shape[-1])
            for phase_idx in range(len(composition_sets)):
                compset = composition_sets[phase_idx]
                prop_Y_values[it.multi_index + np.index_exp[phase_idx, :compset.phase_record.phase_dof]] = \
                    compset.dof[2:]
                prop_X_values[it.multi_index + np.index_exp[phase_idx, :]] = compset.X
                prop_GM_values[it.multi_index] += compset.NP * compset.energy
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
