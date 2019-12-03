# distutils: language = c++
from collections import defaultdict, OrderedDict
import numpy as np
cimport numpy as np
cimport cython
cdef extern from "_isnan.h":
    bint isnan (double) nogil
import scipy.spatial
from pycalphad.core.problem cimport Problem
from pycalphad.core.solver import InteriorPointSolver
from pycalphad.core.composition_set cimport CompositionSet
from pycalphad.core.phase_rec cimport PhaseRecord
from pycalphad.core.constants import *
import pycalphad.variables as v


cdef bint remove_degenerate_phases(object composition_sets, object removed_compsets,
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
        if abs(composition_sets[phase_idx].NP) <= 1.1*MIN_PHASE_FRACTION:
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
        removed_compsets.append(composition_sets[idx])
        del composition_sets[idx]
    if len(entries_to_delete) > 0:
        return True
    else:
        return False


cdef bint add_new_phases(object composition_sets, object removed_compsets, object phase_records,
                         object grid, object current_idx, np.ndarray[ndim=1, dtype=np.float64_t] chemical_potentials,
                         double[::1] state_variables, double minimum_df, bint verbose) except *:
    """
    Attempt to add a new phase with the largest driving force (based on chemical potentials). Candidate phases
    are taken from current_grid and modify the composition_sets object. The function returns a boolean indicating
    whether it modified composition_sets.
    """
    cdef double[:] driving_forces
    cdef int df_idx = 0
    cdef double largest_df = -np.inf
    cdef double[:] df_comp
    cdef double[:,::1] current_grid_Y = grid.Y[*current_idx, ...]
    cdef double[:,::1] current_grid_X = grid.X[*current_idx, ...]
    cdef np.ndarray current_grid_Phase = grid.Phase[*current_idx, ...]
    cdef unicode df_phase_name
    cdef CompositionSet compset = composition_sets[0]
    cdef int num_statevars = len(compset.phase_record.state_variables)
    cdef bint distinct = False
    driving_forces = (chemical_potentials * current_grid_X).sum(axis=-1) - grid.GM[*current_idx, ...]
    for i in range(driving_forces.shape[0]):
        if driving_forces[i] > largest_df:
            df_comp = current_grid_Y[i]
            df_phase_name = <unicode>current_grid_Phase[i]
            distinct = True
            for compset in removed_compsets:
                if df_phase_name != compset.phase_record.phase_name:
                    continue
                distinct = False
                for comp_idx in range(compset.phase_record.phase_dof):
                    if abs(df_comp[comp_idx] - compset.dof[num_statevars+comp_idx]) > 10*COMP_DIFFERENCE_TOL:
                        distinct = True
                        break
                if not distinct:
                    break
            if not distinct:
                if verbose:
                    print('Candidate composition set ' + df_phase_name + ' at ' + str(np.array(compset.X)) + ' is not distinct from previously removed phase')
                continue
            largest_df = driving_forces[i]
            df_idx = i
    if largest_df > minimum_df:
        # To add a phase, must not be within COMP_DIFFERENCE_TOL of composition of the same phase of its type
        df_comp = current_grid_X[df_idx]
        df_phase_name = <unicode>current_grid_Phase[df_idx]
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
        compset = CompositionSet(phase_records[df_phase_name])
        compset.update(current_grid_Y[df_idx, :compset.phase_record.phase_dof], 1./(len(composition_sets)+1),
                       state_variables, False)
        composition_sets.append(compset)
        if verbose:
            print('Adding ' + repr(compset) + ' Driving force: ' + str(largest_df))
        return True
    return False

cpdef pointsolve(composition_sets, comps, cur_conds, prob, iter_solver):
    "Mutates composititon_sets with updated values if it converges. Returns SolverResult."
    return _solve_and_update_problem(composition_sets, comps, cur_conds, prob, iter_solver)

cdef _solve_and_update_problem(composition_sets, comps, cur_conds, prob, iter_solver):
    cdef CompositionSet compset
    result = iter_solver.solve(prob)
    composition_sets = prob.composition_sets
    if result.converged:
        x = result.x
        compset = composition_sets[0]
        var_offset = len(compset.phase_record.state_variables)
        phase_idx = 0
        for compset in composition_sets:
            compset.update(x[var_offset:var_offset + compset.phase_record.phase_dof],
                           x[prob.num_vars - prob.num_phases + phase_idx], x[:len(compset.phase_record.state_variables)], True)
            var_offset += compset.phase_record.phase_dof
            phase_idx += 1
    return result


cdef _solve_and_update_if_converged(composition_sets, comps, cur_conds, problem, iter_solver):
    "Mutates composititon_sets with updated values if it converges. Returns SolverResult."
    prob = problem(composition_sets, comps, cur_conds)
    return _solve_and_update_problem(composition_sets, comps, cur_conds, prob, iter_solver)

def _solve_eq_at_conditions(comps, properties, phase_records, grid, conds_keys, state_variables, verbose,
                            problem=Problem, solver=None):
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
    problem : pycalphad.core.problem.Problem
        Problem instance
    solver : pycalphad.core.solver.SolverBase
        Instance of a SolverBase subclass. If None is supplied, defaults to a
        pycalphad.core.solver.InteriorPointSolver

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
    iter_solver = solver if solver is not None else InteriorPointSolver(verbose=verbose)
    for key, value in phase_records.items():
        if not isinstance(phase_records[key], PhaseRecord):
            phase_records[key] = PhaseRecord(comps, value.state_variables, value.variables,
                                             value.parameters, value.obj, value.grad,
                                             value.mass_obj, value.mass_grad,
                                             value.internal_cons_func, value.internal_jac,
                                             value.multiphase_cons, value.multiphase_jac,
                                             value.num_internal_cons, value.num_multiphase_cons)

    pure_elements = set(v.Species(list(spec.constituents.keys())[0])
                                  for spec in comps
                                    if (len(spec.constituents.keys()) == 1 and
                                    list(spec.constituents.keys())[0] == spec.name)
                       )
    pure_elements = sorted(pure_elements)

    # Factored out via profiling
    prop_MU_values = properties.MU
    prop_NP_values = properties.NP
    prop_Phase_values = properties.Phase
    prop_X_values = properties.X
    prop_Y_values = properties.Y
    prop_GM_values = properties.GM
    str_state_variables = [str(k) for k in state_variables if str(k) in grid.coords.keys()]
    it = np.nditer(prop_GM_values, flags=['multi_index'])

    while not it.finished:
        # A lot of this code relies on cur_conds being ordered!
        converged = False
        changed_phases = False
        cur_conds = OrderedDict(zip(conds_keys,
                                    [np.asarray(properties.coords[b][a], dtype=np.float)
                                     for a, b in zip(it.multi_index, conds_keys)]))
        # assume 'points' and other dimensions (internal dof, etc.) always follow
        curr_idx = [it.multi_index[i] for i, key in enumerate(conds_keys) if key in str_state_variables]
        state_variable_values = np.array([value for key, value in sorted(cur_conds.items(), key=lambda x: str(x[0]))
                                          if key in str_state_variables])
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

        composition_sets = []
        removed_compsets = []
        for phase_idx, phase_name in enumerate(prop_Phase_values[it.multi_index]):
            if phase_name == '' or phase_name == '_FAKE_':
                continue
            phase_record = phase_records[phase_name]
            sfx = prop_Y_values[it.multi_index + np.index_exp[phase_idx, :phase_record.phase_dof]]
            phase_amt = prop_NP_values[it.multi_index + np.index_exp[phase_idx]]
            phase_amt = max(phase_amt, MIN_PHASE_FRACTION)
            compset = CompositionSet(phase_record)
            compset.update(sfx, phase_amt, state_variable_values, False)
            composition_sets.append(compset)
        chemical_potentials = prop_MU_values[it.multi_index]
        energy = prop_GM_values[it.multi_index]
        # Remove duplicate phases -- we will add them back later
        remove_degenerate_phases(composition_sets, [], 0.5, 100, verbose)
        iterations = 0
        history = []
        while (iterations < 10) and (not iter_solver.ignore_convergence):
            result = _solve_and_update_if_converged(composition_sets, comps, cur_conds, problem, iter_solver)

            if result.converged:
                chemical_potentials[:] = result.chemical_potentials
            changed_phases = add_new_phases(composition_sets, removed_compsets, phase_records,
                                            grid, curr_idx, chemical_potentials, state_variable_values,
                                            1e-4, verbose)
            changed_phases |= remove_degenerate_phases(composition_sets, removed_compsets, 1e-3, 0, verbose)
            iterations += 1
            if not changed_phases:
                break
        if changed_phases:
            result = _solve_and_update_if_converged(composition_sets, comps, cur_conds, problem, iter_solver)
            chemical_potentials[:] = result.chemical_potentials
        if not iter_solver.ignore_convergence:
            converged = result.converged
            remove_degenerate_phases(composition_sets, [], 1e-3, 0, verbose)
        else:
            converged = True
        if converged:
            if verbose:
                print('Composition Sets', composition_sets)
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
                    compset.dof[len(compset.phase_record.state_variables):]
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
