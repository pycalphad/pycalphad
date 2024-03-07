from pycalphad.mapping.primitives import STATEVARS
from pycalphad import calculate, variables as v
from pycalphad.core.composition_set import CompositionSet
from pycalphad.mapping.primitives import _get_value_for_var, Point, Node
from xarray import Dataset
from pycalphad.core.solver import Solver
import numpy as np
from copy import deepcopy
from pycalphad.core.constants import COMP_DIFFERENCE_TOL


def degrees_of_freedom(point, components, num_potential_conditions):
    '''
    Gets degrees of freedom (F = C + 2 - P - fixed potentials)
    Fixed potentials = 2 - num_potential_conditions (assuming T and P are the only potentials we can set)
    Components assume it contains 'VA'
    '''
    return len(components)-1 + 2 - len(point.stable_composition_sets) - (2-num_potential_conditions)

def _get_conditions_from_eq(eq_result: Dataset):
    '''
    From equilibrium results, extract all conditions - conditions must be scalar
    Conditions include N, P, T, X, W, MU
    '''
    conds = {}
    for coord_key, coord_val in eq_result.coords.items():
        coord_val = coord_val.values
        if coord_key in ("N", "P", "T"):
            assert coord_val.size == 1, f"Condition coordinate must have exactly one value to extract. Got {coord_val} for {coord_key}."
            conds[getattr(v, coord_key)] = float(np.atleast_1d(coord_val)[0])
        elif coord_key.startswith(("X_", "W_", "MU_")):
            assert coord_val.size == 1, f"Condition coordinate must have exactly one value to extract. Got {coord_val} for {coord_key}."
            cond_type_str, cond_arg = coord_key.split("_")  # split e.g. "X_NI" into ("X", "NI")
            conds[getattr(v, cond_type_str)(cond_arg)] = float(np.atleast_1d(coord_val)[0])
    return conds

def _is_a_potential(cond):
    '''
    Checks whether a condition in a potential or not
    '''
    if cond in (v.T, v.P):
        return True
    elif isinstance(cond, v.ChemicalPotential):
        return True
    else:
        return False

def _sort_cs_by_what_to_fix(dbf, comps, models, cs_list):
    '''
    Heuristic for sorting composition sets by the "best" CS to fix
    The fixed CS should be the most ordered. If the ordering is the same, then the CS nearest to the composition axis
    '''
    #Calculate degree of ordering and prod(x*(1-x)) for each phase
    doo_list = []
    prod_list = []
    for cs in cs_list:
        p_cs = np.array([cs.dof[len(STATEVARS):]])
        state_vars = {str(k): _get_value_for_var(cs, k) for k in [v.T, v.P, v.N]}
        doo = calculate(dbf, comps, [cs.phase_record.phase_name], output='DOO', model=models, points=p_cs, **state_vars)
        doo_list.append(doo.DOO.values.ravel()[0])
        #For really small non-zero DOO, set to 0. Disordered phases have a change of outputting a tiny number due to numerical limits
        #When this happens, it can mess up which phase to fix
        if doo_list[-1] < 1e-10:
            doo_list[-1] = 0
        prod_list.append(np.prod(cs.X)*np.prod(1-np.array(cs.X)))

    #Sort composition sets by DOO, if same, then sort by product
    for i in range(len(cs_list)):
        for j in range(i+1, len(cs_list)):
            if doo_list[j] > doo_list[i]:
                temp_cs = cs_list[i]
                cs_list[i] = cs_list[j]
                cs_list[j] = temp_cs
            elif doo_list[i] == doo_list[j]:
                if prod_list[j] < prod_list[i]:
                    temp_cs = cs_list[i]
                    cs_list[i] = cs_list[j]
                    cs_list[j] = temp_cs
    return cs_list


def _extract_point_from_dataset(dbf, comps, models, eq_result, phase_records, num_phases_to_fix=1) -> Point:
    '''
    From equilibrium results, create Point object

    This requires creating a composition set for every stable phase then fixing the first n composition sets
        Number of stable phases must be n+1 (at least 1 free composition set)
    If no number of phases to fix, then all composition sets are free - not really what you want for mapping
    If n number of phases to fix, then fix the first n composition sets
    For all free composition sets, set NP to 1/number of free comp sets - does this make sense? Maybe it's okay since we're gonna solve with set conditions later
    '''
    # NOTE: it is the responsibility of the caller to remove a condition from the returned Point
    # TODO: Validation for a point calculation
    stable_compset_indices = np.nonzero((eq_result.NP.values > 0.0).squeeze())[0]
    num_free_compsets = len(stable_compset_indices) - num_phases_to_fix
    assert num_free_compsets > 0, f"Starting point must have at least {num_free_compsets + 1} stable phases to fix {num_phases_to_fix}. Got {len(stable_compset_indices)} stable phases:\n{eq_result}"
    compsets = []
    for cs_idx, phase_idx in enumerate(stable_compset_indices):
        phase_name = eq_result.Phase.values.squeeze()[phase_idx]
        if phase_name == '':
            continue
        cs = CompositionSet(phase_records[phase_name])
        # we need to reshape to 2D in the case of a unary where the phase dimension has size 1 and would be squeezed out.
        sitefracs = np.asarray(eq_result.Y.values.squeeze().reshape(-1, eq_result.Y.values.shape[-1])[phase_idx, :cs.phase_record.phase_dof], dtype=float)
        statevars = np.asarray([float(np.asarray(eq_result.coords[sv].values).squeeze()) for sv in map(str, sorted(STATEVARS, key=str))], dtype=float)
        cs.update(sitefracs, eq_result.NP.values.squeeze()[phase_idx], statevars)
        compsets.append(cs)
    compsets = _sort_cs_by_what_to_fix(dbf, comps, models, compsets)

    #If num_phases_to_fix is 0, then we'll assume we're stepping and we don't want to adjust phase amounts
    if num_phases_to_fix > 0:
        for i in range(len(compsets)):
            if i < num_phases_to_fix:
                compsets[i].NP = 0
                compsets[i].fixed = True
            else:
                compsets[i].NP = 1/num_free_compsets

    curr_conds = _get_conditions_from_eq(eq_result)
    return Point(curr_conds, compsets[:num_phases_to_fix], compsets[num_phases_to_fix:], [])

def calculate_with_new_conditions(point : Point, new_conds, free_var):
    '''
    Create a new point and recalculate equilibrium with the given conditions
    Store the original composition set to check for phase changes later on

    This serves to replace the take_step function and output the new_point and original composition sets
    '''
    # Get composition sets
    free_cs = deepcopy(point.free_composition_sets)
    fixed_cs = deepcopy(point.fixed_composition_sets)
    meta_cs = deepcopy(point.metastable_composition_sets)
    trial_cs = free_cs + fixed_cs + meta_cs
    copy_conds = {k:val for k,val in new_conds.items()}

    # Update composition sets to new conditions
    np_sum = 0
    for cs in trial_cs:
        if not cs.fixed:
            np_sum += cs.NP
        cs.update(cs.dof[len(STATEVARS):], cs.NP, np.asarray([new_conds[sv] for sv in STATEVARS], dtype=float))

    # If user sets a variable to free, then delete it from the condition
    # For stepping, we don't free any variables
    if free_var is not None:
        del new_conds[free_var]

    # Store shallow copy of composition set list and solve
    # The shallow copy will keep the new solution as well as the original list of composition sets in the event that the solver removes a phase
    solution_cs = [cs for cs in trial_cs]
    solver = Solver(remove_metastable=True, allow_changing_phases=False)
    result = solver.solve(solution_cs, {str(k): val for k, val in new_conds.items()})
    new_point = Point(new_conds, [cs for cs in solution_cs if cs.fixed], [cs for cs in solution_cs if not cs.fixed], meta_cs)

    # Add free variable back in to conditions and store in the new point
    if free_var is not None:
        if free_var in STATEVARS:
            new_conds[free_var] = _get_value_for_var(new_point.stable_composition_sets[0], free_var)
        else:
            new_conds[free_var] = sum(_get_value_for_var(cs, free_var)*cs.NP for cs in new_point.stable_composition_sets)

    new_point.global_conditions = new_conds

    return result, new_point, trial_cs

def check_point_is_valid(point: Point):
    # Get composition sets
    free_cs = deepcopy(point.free_composition_sets)
    fixed_cs = deepcopy(point.fixed_composition_sets)
    meta_cs = deepcopy(point.metastable_composition_sets)
    trial_cs = free_cs + fixed_cs + meta_cs
    conds = {k:val for k,val in point.global_conditions.items()}

    # Update composition sets to new conditions
    for cs in trial_cs:
        cs.update(cs.dof[len(STATEVARS):], 1/len(trial_cs), np.asarray([conds[sv] for sv in STATEVARS], dtype=float))
        cs.fixed = False

    for k in conds:
        if k not in STATEVARS:
            conds[k] = sum(_get_value_for_var(cs, k)*cs.NP for cs in trial_cs)

    solution_cs = [cs for cs in trial_cs]
    solver = Solver(remove_metastable=True, allow_changing_phases=False)
    result = solver.solve(solution_cs, {str(k): val for k, val in conds.items()})

    return result.converged and compare_cs_for_change_in_phases(solution_cs, trial_cs) == 0


def check_point_is_global_min(point: Point, chem_pot, sys_definition, phase_records, tol = 1e-4, pdens=500):
    '''
    Python implementation of custom_add_new_phases
    There was some seg faults issues with cython, so this serves as an alternative

    This calculates a grid of points for each phase, then checks if there is a phase that becomes stable (below hyperplane)
    Three additional checks for this new phase
        Driving force of phase formation must be larger than threshold (tol)
        Name of phase is different than that in composition set list
            If name of phase is the same, then check for composition difference (in the event there's a miscibility gap)
    '''
    dbf = sys_definition['dbf']
    comps = sys_definition['comps']
    phases = sys_definition['phases']
    models = sys_definition['models']
    pres = point.global_conditions[v.P]
    temp = point.global_conditions[v.T]
    state_variables = np.array([point.global_conditions[val] for val in STATEVARS], dtype=float)
    test_points = calculate(dbf, comps, phases, P=pres, T=temp, model=models, phase_records=phase_records, pdens=pdens)
    phase_id = np.squeeze(test_points.Phase.values)
    comps = np.squeeze(test_points.X.values)
    site_fracs = np.squeeze(test_points.Y.values)
    gm = np.squeeze(test_points.GM.values)

    g_chempot = comps * np.array(chem_pot)
    dG = (np.sum(g_chempot, axis=1)) - gm

    # Largest driving force (positive)
    max_id = np.argmax(dG)

    if dG[max_id] < tol:
        return True, point

    # Check if largest driving force is unique
    # Unique phase name
    # Or unique composition if phase name is not unique
    num_sv = len(state_variables)
    distinct_comp = False
    new_phase_name = True
    for cs in point.stable_composition_sets:
        if phase_id[max_id] == cs.phase_record.phase_name:
            new_phase_name = False
            for i in range(cs.phase_record.phase_dof):
                if abs(cs.dof[num_sv + i] - site_fracs[max_id, i]) > 10*COMP_DIFFERENCE_TOL:
                    distinct_comp = True
                    break
    if not distinct_comp and not new_phase_name:
        return True, point

    # Create composition set and create new point, which is the previous point + new phase
    compset = CompositionSet(phase_records[phase_id[max_id]])
    compset.update(site_fracs[max_id, :compset.phase_record.phase_dof], 1e-6, state_variables)

    new_point = Point(point.global_conditions, point.fixed_composition_sets, point.free_composition_sets, point.metastable_composition_sets)
    new_point._free_composition_sets.append(compset)

    return False, new_point

def compare_cs_for_change_in_phases(prev_cs, new_cs):
    num_different_phases = len(set(new_cs).symmetric_difference(set(prev_cs)))
    return num_different_phases

def create_node_from_different_points(prev_point: Point, new_point: Point, axis_vars, axis_lims = None):
    '''
    Given two nearby points, create a node
    This is used after calculate_with_new_conditions or check_point_is_global_min where a composition may be added or removed
        This also must be used only after these two, since we compare composition sets of the before and after
    Check which phase was added or removed, and fix it, then remove all free conditions and solve
    '''
    # Compare the two nodes and check which new/removed phase to fix
    conds = new_point.global_conditions
    prev_cs = [cs for cs in prev_point.stable_composition_sets]
    new_cs = [cs for cs in new_point.stable_composition_sets]
    phases_added = set(new_cs) - set(prev_cs)
    phases_removed = set(prev_cs) - set(new_cs)
    if len(phases_removed) + len(phases_added) != 1:
        return None

    # Fix added or removed phase and update conditions
    if len(phases_added) == 1:
        compset_to_fix = list(phases_added)[0]
    elif len(phases_removed) == 1:
        compset_to_fix = list(phases_removed)[0]
        new_cs += [compset_to_fix]
    compset_to_fix.fixed = True
    compset_to_fix.update(compset_to_fix.dof[len(STATEVARS):], 0.0, np.asarray([conds[sv] for sv in STATEVARS], dtype=float))

    # At least one composition set needs to have an amount, if none do, then set one free CS to 1
    if all(cs.NP == 0.0 for cs in new_cs):
        for cs in new_cs:
            if not cs.fixed:
                cs.NP = 1.0
                break

    # Free all input variables
    # The number of free variables = number of fixed composition sets
    solution_cs = [cs for cs in new_cs]
    new_conds = {k:sv for k,sv in conds.items()}
    for av in axis_vars:
        del new_conds[av]

    # Solve
    try:
        solver = Solver(remove_metastable=True, allow_changing_phases=False)
        result = solver.solve(solution_cs, {str(k):val for k,val in new_conds.items()})
        if not result.converged:
            return None
    except:
        return None

    if set(solution_cs) != set(new_cs):
        return None

    # Add free variables back
    for av in axis_vars:
        if av in STATEVARS:
            new_conds[av] = _get_value_for_var(solution_cs[0], av)
        else:
            new_conds[av] = sum(_get_value_for_var(cs, av)*cs.NP for cs in solution_cs)

    # Check new conditions to axis limits if provided
    if axis_lims is not None:
        for av in axis_vars:
            if new_conds[av] > axis_lims[av][1] or new_conds[av] < axis_lims[av][0]:
                return None

    # Create new node with parent to the previous point at the updated conditions
    parent = Point(new_conds, prev_point.fixed_composition_sets, prev_point.free_composition_sets, prev_point.metastable_composition_sets)
    fixed_cs = [cs for cs in solution_cs if cs.fixed]
    free_cs = [cs for cs in solution_cs if not cs.fixed]
    new_node = Node(new_conds, fixed_cs, free_cs, [], parent)
    return new_node