# distutils: language = c++
from collections import OrderedDict
import numpy as np
cimport numpy as np
cimport cython
cdef extern from "_isnan.h":
    bint isnan (double) nogil
from pycalphad.core.solver import Solver
from pycalphad.core.composition_set cimport CompositionSet
from pycalphad.core.phase_rec cimport PhaseRecord
from pycalphad.core.constants import *


cdef bint add_new_phases(object composition_sets, object removed_compsets, object phase_records,
                         object grid, object current_idx, np.ndarray[ndim=1, dtype=np.float64_t] chemical_potentials,
                         double[::1] state_variables, double minimum_df, bint verbose) except *:
    """
    Attempt to add a new phase with the largest driving force (based on chemical potentials). Candidate phases
    are taken from current_grid and modify the composition_sets object. The function returns a boolean indicating
    whether it modified composition_sets.
    """
    cdef double[:] driving_forces
    cdef int comp_idx
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
    driving_forces = np.dot(current_grid_X, chemical_potentials) - grid.GM[*current_idx, ...]
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
        if df_phase_name == '_FAKE_':
            if verbose:
                print('Chemical potentials are poorly conditioned')
            return False
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
        compset.update(current_grid_Y[df_idx, :compset.phase_record.phase_dof], 1e-6,
                       state_variables)
        composition_sets.append(compset)
        if verbose:
            print('Adding ' + repr(compset) + ' Driving force: ' + str(largest_df))
        return True
    return False

@cython.boundscheck(False)
cdef int argmax(double* a, int a_shape) nogil:
    cdef int i
    cdef int result = 0
    cdef double highest = -1e30
    for i in range(a_shape):
        if a[i] > highest:
            highest = a[i]
            result = i
    return result

def add_nearly_stable(object composition_sets, dict phase_records,
                      object grid, object current_idx, np.ndarray[ndim=1, dtype=np.float64_t] chemical_potentials,
                      double[::1] state_variables, double minimum_df, bint verbose):
    cdef double[::1] driving_forces, driving_forces_for_phase
    cdef double[:,::1] current_grid_Y = grid.Y[*current_idx, ...]
    cdef double[:,::1] current_grid_X = grid.X[*current_idx, ...]
    cdef double[::1] current_grid_GM = grid.GM[*current_idx, ...]
    cdef unicode phase_name
    cdef CompositionSet compset = composition_sets[0]
    cdef set entered_phases = {compset.phase_record.phase_name for compset in composition_sets}
    cdef PhaseRecord phase_record
    cdef int num_statevars = len(compset.phase_record.state_variables)
    cdef int df_idx, minimum_df_idx
    cdef bint phases_added = False
    driving_forces = np.dot(current_grid_X, chemical_potentials) - current_grid_GM
    # Add unrepresented phases as metastable composition sets
    # This should help catch phases around the limit of stability
    for phase_name in sorted(phase_records.keys()):
        if phase_name in entered_phases:
            continue
        phase_record = phase_records[phase_name]
        phase_indices = grid.attrs['phase_indices'][phase_name]
        driving_forces_for_phase = driving_forces[phase_indices.start:phase_indices.stop]
        minimum_df_idx = argmax(&driving_forces_for_phase[0], driving_forces_for_phase.shape[0])
        if driving_forces_for_phase[minimum_df_idx] >= minimum_df:
            phases_added = True
            df_idx = phase_indices.start + minimum_df_idx
            compset = CompositionSet(phase_record)
            compset.update(current_grid_Y[df_idx, :phase_record.phase_dof], 0.0, state_variables)
            if verbose:
                print('Adding metastable ' + repr(compset) + ' Driving force: ' + str(driving_forces_for_phase[minimum_df_idx]))
            composition_sets.append(compset)
    return phases_added

cpdef update_composition_sets(composition_sets, solver_result, remove_metastable=True):
    """
        update_composition_sets(composition_sets, solver_result, remove_metastable=True)

    Parameters
    ----------
    composition_sets : List[CompositionSet]
    solver_result : pycalphad.core.solver.SolverResult
    remove_metastable : Optional[bool]
        If True (the default), remove metastable compsets from the compositions_sets.

    """
    cdef CompositionSet compset
    x = solver_result.x
    compset = composition_sets[0]
    num_compsets = len(composition_sets)
    num_state_variables = len(compset.phase_record.state_variables)
    var_offset = num_state_variables
    num_vars = sum([compset.phase_record.phase_dof for compset in composition_sets]) + num_compsets + num_state_variables
    phase_idx = 0
    compsets_to_remove = []
    for compset in composition_sets:
        phase_amt = x[num_vars - num_compsets + phase_idx]
        # Mark unstable phases for removal
        if phase_amt == 0.0 and not compset.fixed:
            compsets_to_remove.append(int(phase_idx))
        compset.update(x[var_offset:var_offset + compset.phase_record.phase_dof],
                       phase_amt, x[:num_state_variables])
        var_offset += compset.phase_record.phase_dof
        phase_idx += 1
    if remove_metastable:
        # Watch removal order here, as the indices of composition_sets are changing!
        for idx in reversed(compsets_to_remove):
            del composition_sets[idx]


cpdef solve_and_update(composition_sets, conditions, solver, remove_metastable=True):
    """
        solve_and_update(composition_sets, conditions, solver, remove_metastable=True)

    Use the solver to find a solution satisfying the conditions from the starting point
    given by the composition sets.

    Parameters
    ----------
    composition_sets : List[CompositionSet]
    conditions : OrderedDict[str, float]
    solver : pycalphad.core.solver.SolverBase
    remove_metastable : Optional[bool]
        If True (the default), remove metastable compsets from the compositions_sets.

    """
    result = solver.solve(composition_sets, conditions)
    update_composition_sets(composition_sets, result, remove_metastable=remove_metastable)
    return result


def _solve_eq_at_conditions(properties, phase_records, grid, conds_keys, state_variables, verbose, solver=None):
    """
        _solve_eq_at_conditions(properties, phase_records, grid, conds_keys, state_variables, verbose, solver=None)

    Compute equilibrium for the given conditions.
    This private function is meant to be called from a worker subprocess.
    For that case, usually only a small slice of the master 'properties' is provided.
    Since that slice will be copied, we also return the modified 'properties'.

    Parameters
    ----------
    properties : Dataset
        Will be modified! Thermodynamic properties and conditions.
    phase_records : dict of PhaseRecord
        Details on phase callables.
    grid : Dataset
        Sample of energy landscape of the system.
    conds_keys : List[str]
        List of conditions sorted in dimension order.
    state_variables : List[v.StateVariable]
        List of state variables sorted in dimension order.
    verbose : bool
        Print details.
    solver : pycalphad.core.solver.SolverBase
        Instance of a SolverBase subclass. If None is supplied, defaults to a Solver.

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
    iter_solver = solver if solver is not None else Solver(verbose=verbose)

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
                                    [np.asarray(properties.coords[b][a], dtype=np.float_)
                                     for a, b in zip(it.multi_index, conds_keys)]))
        # assume 'points' and other dimensions (internal dof, etc.) always follow
        curr_idx = [it.multi_index[i] for i, key in enumerate(conds_keys) if key in str_state_variables]
        state_variable_values = [cur_conds[key] for key in str_state_variables]
        state_variable_values = np.array(state_variable_values)
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
            compset.update(sfx, phase_amt, state_variable_values)
            composition_sets.append(compset)
        chemical_potentials = prop_MU_values[it.multi_index]
        energy = prop_GM_values[it.multi_index]
        add_nearly_stable(composition_sets, phase_records, grid, curr_idx, chemical_potentials,
                          state_variable_values, -1000, verbose)
        #print('Composition Sets', composition_sets)
        phase_amt_sum = 0.0
        for compset in composition_sets:
            phase_amt_sum += compset.NP
        for compset in composition_sets:
            compset.NP /= phase_amt_sum
        iterations = 0
        history = []
        while (iterations < 10) and (not iter_solver.ignore_convergence):
            if len(composition_sets) == 0:
                changed_phases = False
                break
            result = solve_and_update(composition_sets, cur_conds, iter_solver)

            chemical_potentials[:] = result.chemical_potentials
            changed_phases = add_new_phases(composition_sets, removed_compsets, phase_records,
                                            grid, curr_idx, chemical_potentials, state_variable_values,
                                            1e-4, verbose)
            iterations += 1
            if not changed_phases:
                break
        if changed_phases:
            result = solve_and_update(composition_sets, cur_conds, iter_solver)
            chemical_potentials[:] = result.chemical_potentials
        if not iter_solver.ignore_convergence:
            converged = result.converged
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
            # Copy out any free state variables (P, T, etc.)
            # All CompositionSets should have equal state variable values, so we copy from the first one
            for sv_idx, ssv in enumerate(str_state_variables):
                # If the state variable is listed as a free variable in our results
                # The LightDataset interface is not clear here
                if properties.data_vars.get(ssv, None) is not None:
                    properties.data_vars[ssv][1][it.multi_index] = composition_sets[0].dof[sv_idx]
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
