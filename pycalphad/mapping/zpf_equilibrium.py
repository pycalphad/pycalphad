import copy
import logging

import numpy as np

from pycalphad import calculate, variables as v
from pycalphad.core.solver import Solver
from pycalphad.core.composition_set import CompositionSet
from pycalphad.property_framework.metaproperties import DormantPhase
from pycalphad.core.constants import COMP_DIFFERENCE_TOL
from pycalphad.property_framework.computed_property import JanssonDerivative

from pycalphad.mapping.primitives import Node, Point
import pycalphad.mapping.utils as map_utils

_log = logging.getLogger(__name__)

def update_equilibrium_with_new_conditions(point: Point, new_conditions: dict[v.StateVariable, str], free_var: v.StateVariable = None):
    """
    Updates point with new set of conditions
    Assumes that the new conditions are small distance away from the current conditions

    Parameters
    ----------
    point : Point
        Point to update
    new_conditions : dict[v.StateVariable, str]
        New set of conditions
    free_var : v.StateVariable (optional)
        Variable to free when updating conditions
        This will be the case if there's a fixed composition set, so
        freeing a variable will conserve the Gibbs phase rule

    Returns
    -------
    (new_point, orig_cs)
        new_point - Point with updated composition sets and conditions
        orig_cs - Original list of composition sets
            Degrees of freedom in the composition sets will be updated with the new point conditions
            Will include any composition sets that became unstable after updating

    or

    None - equilibrium failed
    """
    # Update composition sets with new state variables
    comp_sets = copy.deepcopy(point.stable_composition_sets)
    for cs in comp_sets:
        state_variables = cs.phase_record.state_variables
        new_state_conds = map_utils.get_statevars_array(new_conditions, state_variables)
        cs.update(cs.dof[len(state_variables):], cs.NP, new_state_conds)

    # Remove free variable condition if given - this assumes that Gibbs phase rule will be satisfy if done
    if free_var is not None:
        del new_conditions[free_var]

    # Keep track of original composition sets (these will be updated with the solver, but the original list will remain even if a phase becomes unstable)
    orig_cs = [cs for cs in comp_sets]
    try:
        solver = Solver(remove_metastable=True, allow_changing_phases=False)
        results = solver.solve(comp_sets, new_conditions)
        if not results.converged:
            return None
    except Exception:
        return None

    # Add free variable back
    if free_var is not None:
        new_conditions[free_var] = free_var.compute_property(comp_sets, new_conditions, results.chemical_potentials)

    new_point = Point(new_conditions, np.array(results.chemical_potentials), [cs for cs in comp_sets if cs.fixed], [cs for cs in comp_sets if not cs.fixed])
    return new_point, orig_cs

def _find_global_min_cs(point: Point, system_info: dict, pdens = 500, tol = 1e-5, num_candidates = 1):
    """
    Finds potential composition set for global min check

    For each possible phase:
        1. Sample DOF and find CS that maximizes driving force
        2. Create a DormantPhase with CS and compute driving force with potentials at equilibrium
            Or check the top N CS that maximizes driving force and compute driving force and take max
        3. If driving force is positive, then new phase may be stable in when attempting to find a new node

    Parameters
    ----------
    point : Point
        Point to check global min
    system_info : dict
        Dictionary containing information for pycalphad.calculate
    pdens : int
        Sampling density
    tol : float
        Tolerance for whether a new CS is considered stable
    num_candidates : int
        Number of candidate CS to check driving force
        To avoid redundant calculations, this will only check driving force
        for unique phases. So setting this to a high number will not significantly
        affect performance

    Returns
    -------
    (cs, dG) : (potential new composition set, driving force of composition set)

    or

    None : if no composition set was found below the current chemical potential hyperplane
    """
    dbf = system_info["dbf"]
    comps = system_info["comps"]
    phases = system_info["phases"]
    models = system_info["models"]
    phase_records = system_info["phase_records"]

    # Get driving force and find index that maximized driving force
    state_conds = {str(key): point.global_conditions[key] for key in point.global_conditions if map_utils.is_state_variable(key)}
    points = calculate(dbf, comps, phases, model=models, phase_records=phase_records, output="GM", to_xarray=False, pdens=pdens, **state_conds)
    gm = np.squeeze(points.GM)
    x = np.squeeze(points.X)
    y = np.squeeze(points.Y)
    phase_ids = np.squeeze(points.Phase)
    g_chempot = x * point.chemical_potentials
    dGs = np.sum(g_chempot, axis=1) - gm

    sorted_indices = np.argsort(dGs)[::-1]
    cs = None
    dG = 0
    # Record phases that driving force is checked
    # Each phase will be checked once to avoid redundant calculations
    tested_phases = []
    for i in range(np.amin([num_candidates, len(sorted_indices)])):
        index = sorted_indices[i]
        if phase_ids[index] not in tested_phases:
            tested_phases.append(phase_ids[index])
            test_cs = CompositionSet(phase_records[str(phase_ids[index])])
            # Create numpy array for site fractions. There seems to be some models where
            # np.squeeze(points.Y) returns a non-writable array. Not sure why, but when it does,
            # updating the composition set will fail with a ValueError: buffer source array
            # is read only. Creating a new array for the site fractions seems to fix this issue
            site_fracs = np.array(y[index][:test_cs.phase_record.phase_dof], dtype=np.float64)
            test_cs.update(site_fracs, 1.0, map_utils.get_statevars_array(point.global_conditions, test_cs.phase_record.state_variables))
            dormant_phase = DormantPhase(test_cs, None)
            test_dg = point.get_property(dormant_phase.driving_force)
            _log.info(f"Testing phase {phase_ids[index]} with dG={dGs[index]} -> {test_dg} for global min.")
            if test_dg > dG:
                dG = test_dg
                cs = test_cs

    # If driving force is above tolerance, then create a new point with the additional composition set
    if dG < tol:
        return None
    else:
        return cs, dG
    
def find_global_min_point(point: Point, system_info: dict, pdens = 500, tol = 1e-5, num_candidates = 1):
    """
    Checks global min on current point and attempts to find a new point 
    with a new composition set if current point is not global min

    1. Find potential new composition set for global min
    2. If no composition set is found, then return
    3. If composition set is found, check that the new CS doesn't match
       with a currently stable CS
    4. Hope that this works on miscibility gaps

    Parameters
    ----------
    point : Point
        Point to check global min
    system_info : dict
        Dictionary containing information for pycalphad.calculate
    pdens : int
        Sampling density
    tol : float
        Tolerance for whether a new CS is considered stable
    num_candidates : int
        Number of candidate CS to check driving force
        To avoid redundant calculations, this will only check driving force
        for unique phases. So setting this to a high number will not significantly
        affect performance

    Returns
    -------
    new_point : Point with added composition set

    or

    None : if point is global min or global min was unable to be found
    """
    min_cs_result = _find_global_min_cs(point, system_info, pdens, tol, num_candidates)
    if min_cs_result is None:
        return None
    else:
        cs, dG = min_cs_result
        
        _log.info(f'Global min potentially detected. {point.stable_phases} + {cs.phase_record.phase_name} with dG = {dG}')
        if _detect_degenerate_phase(point, cs):
            new_point = Point(point.global_conditions, point.chemical_potentials, point.fixed_composition_sets, point.free_composition_sets)
            map_utils.update_cs_phase_frac(cs, 1e-6)
            new_point._free_composition_sets.append(cs)
            return new_point
        else:
            _log.info('Global min was falsely detected. No global min found')
            return None

def _detect_degenerate_phase(point: Point, new_cs: CompositionSet):
    """
    Check that the new composition set detected during global min check is truely a new CS

    1. Check phase name of new CS against original CS list
    2. If phase name is same, check site fraction constituency between the two CS
    3. If site fraction constituency is different, run a equilibrium calc between
       the two CS to see if they merge. (There are some cases where the new CS is just
       beyond the tolerance for site fraction check, in which solving for equilibrium will
       fix this. In a sense, we can solve equilibrium for all CS pairs with the same name
       and skip step 2, but step 2 will avoid just unecessary calculations)

    Parameters
    ----------
    point : Point
        Point to compare new composition set against
    new_cs : CompositionSet
        Composition set to compare against the point

    Returns
    -------
    bool
        True if new_cs is valid
        False if new_cs is the same CS as one already in the point
    """
    num_sv = new_cs.phase_record.num_statevars
    for cs in point.stable_composition_sets:
        if new_cs.phase_record.phase_name != cs.phase_record.phase_name:
            continue

        if np.allclose(cs.dof[num_sv:], new_cs.dof[num_sv:], atol=10*COMP_DIFFERENCE_TOL):
            return False

        ref_cs_copy = CompositionSet(cs.phase_record)
        ref_cs_copy.update(cs.dof[num_sv:], 1, cs.dof[:num_sv])
        new_cs_copy = CompositionSet(new_cs.phase_record)
        new_cs_copy.update(new_cs.dof[num_sv:], 1e-6, new_cs.dof[:num_sv])
        conds = {key: point.get_property(key) for key in point.global_conditions}
        _log.info(f"Testing free equilibrium with {ref_cs_copy}, {new_cs_copy}")
        try:
            solver = Solver(remove_metastable=True)
            results = solver.solve([ref_cs_copy, new_cs_copy], conds)
            if not results.converged:
                return False
        except Exception:
            return False

        _log.info(f"Equilibrium: {ref_cs_copy}, {new_cs_copy}")
        if np.allclose(ref_cs_copy.dof[num_sv:], new_cs_copy.dof[num_sv:], atol=10*COMP_DIFFERENCE_TOL):
            return False

        if ref_cs_copy.NP < 1e-3:
            cs.update(new_cs_copy.dof[num_sv:], cs.NP, new_cs_copy.dof[:num_sv])
            return True

    return True
    
def create_node_from_different_points(new_point: Point, orig_cs: list[CompositionSet], axis_vars : list[v.StateVariable]):
    """
    Between two points with different composition sets (only 1 different CS)
    Compute the node, freeing up the axis var and solving for it
    The unique CS will be fixed to 0 for the node

    Parameters
    ----------
    new_point : Point
        Point to add CS to
    orig_cs : [CompositionSet]
        List of CS in the previous point
        This allows us to compare what composition set was added or removed
    axis_vars : [v.StateVariable]
        Variables to free when node finding
        One variable per fixed composition set

    Returns
    -------
    new_node : Node
        Node solved with the fixed composition set (either added or removed from previous point)
    
    or

    None - if node could not be found or CS change 
           is invalid for finding new node (either 0 change in CS or more than 1)
    """
    prev_cs = [cs for cs in orig_cs]
    new_cs = [cs for cs in new_point.stable_composition_sets]

    # Find the unique CS between the prev and new point
    phases_added = list(set(new_cs) - set(prev_cs))
    phases_removed = list(set(prev_cs) - set(new_cs))

    if len(phases_added) + len(phases_removed) != 1:
        return None

    # Fix the unique CS
    if len(phases_added) == 1:
        fixed_cs = phases_added[0]
    elif len(phases_removed) == 1:
        fixed_cs = phases_removed[0]
        new_cs.append(fixed_cs)

    fixed_cs.fixed = True
    map_utils.update_cs_phase_frac(fixed_cs, 0.0)

    # Set one free CS to NP=1 if all CS are NP=0
    if (all(cs.NP == 0.0 for cs in new_cs)):
        for cs in new_cs:
            if not cs.fixed:
                map_utils.update_cs_phase_frac(cs, 1.0)
                break

    # Setup conditions and remove axis variable to free it
    solution_cs = [cs for cs in new_cs]
    new_conditions = copy.deepcopy(new_point.global_conditions)
    for av in axis_vars:
        del new_conditions[av]

    # Solve equilibrium with fixed CS
    try:
        solver = Solver(remove_metastable=True)
        results = solver.solve(solution_cs, new_conditions)
        if not results.converged:
            return None
    except Exception:
        return None

    # Add axis var back
    for av in axis_vars:
        new_conditions[av] = av.compute_property(solution_cs, new_conditions, results.chemical_potentials)

    # Create node with parent between thr previous point
    parent = Point(new_conditions, np.array(results.chemical_potentials), [cs for cs in orig_cs if cs.fixed], [cs for cs in orig_cs if not cs.fixed])
    new_node = Node(new_conditions, np.array(results.chemical_potentials), [cs for cs in solution_cs if cs.fixed], [cs for cs in solution_cs if not cs.fixed], parent)
    return new_node

def compute_derivative(point: Point, v_num: v.StateVariable, v_den: v.StateVariable, free_den = True):
    """
    Computes dot derivative of d v_num / d v_den, which handles removing the numerator variable
        free_den is an unintuitive name, but it refers to the mapping code to state that the denominator variable
        is the free variable that we plan on stepping in, with the numerator being the dependent variable

    Parameters
    ----------
    point : Point
        Point to compure derivative on
    v_num : v.StateVariable
        Variable in numerator of derivative
    v_den : v.StateVariable
        Variable in denominator
    free_den : bool
        Whether to free the numerator variable
        Name refers to the denominator being the variable we're stepping in

    Returns
    -------
    derivative : float
        Note: can be nan if the phase boundary is vertical and the denominator is composition
    """
    # Should be able to use point.stable_composition_sets, which puts the fixed composition sets first
    # This was originally here to account for an indexing issue, which is fixed now
    comp_sets = point.free_composition_sets + point.fixed_composition_sets
    chem_pots = point.chemical_potentials
    conds = copy.deepcopy(point.global_conditions)

    if free_den:
        del conds[v_num]

    derivative_property = JanssonDerivative(v_num, v_den)
    derivative = derivative_property.compute_property(comp_sets, conds, chem_pots)
    return derivative
