import copy
import logging

import numpy as np

from pycalphad import calculate, variables as v
from pycalphad.core.solver import Solver
from pycalphad.core.composition_set import CompositionSet
from pycalphad.property_framework.metaproperties import DormantPhase
from pycalphad.core.constants import COMP_DIFFERENCE_TOL

from pycalphad.mapping.primitives import STATEVARS, Node, Point
import pycalphad.mapping.utils as map_utils

_log = logging.getLogger(__name__)

def update_equilibrium_with_new_conditions(point: Point, new_conditions: dict[v.StateVariable, str], free_var: v.StateVariable = None):
    """
    Pretty much a copy of the old update with new conditions

    Returns
    -------
    new_point - Point with updated composition sets and conditions
        Checking whether the number of composition sets has changed will be checked later
    None - equilibrium failed
    """
    #Update composition sets with new state variables
    new_state_conds = map_utils.get_statevars_array(new_conditions)
    comp_sets = copy.deepcopy(point.stable_composition_sets)
    for cs in comp_sets:
        cs.update(cs.dof[len(STATEVARS):], cs.NP, new_state_conds)

    #Remove free variable condition if given - this assumes that Gibbs phase rule will be satisfy if done
    if free_var is not None:
        del new_conditions[free_var]

    #Keep track of original composition sets (these will be updated with the solver, but the original list will remain even if a phase becomes unstable)
    orig_cs = [cs for cs in comp_sets]
    try:
        solver = Solver(remove_metastable=True, allow_changing_phases=False)
        results = solver.solve(comp_sets, new_conditions)
        if not results.converged:
            return None
    except Exception as e:
        return None

    #Add free variable back
    if free_var is not None:
        new_conditions[free_var] = free_var.compute_property(comp_sets, new_conditions, results.chemical_potentials)

    new_point = Point(new_conditions, np.array(results.chemical_potentials), [cs for cs in comp_sets if cs.fixed], [cs for cs in comp_sets if not cs.fixed])
    return new_point, orig_cs

def find_global_min_point(point: Point, system_info: dict, pdens = 500, tol = 1e-5, num_candidates = 1):
    """
    For each possible phase:
        1. Sample DOF and find CS that minimizes driving force
        2. Create a DormantPhase with CS and compute driving force with potentials at equilibrium
        3. If driving force is negative, then new phase is stable
        4. Check that the new CS doesn"t match with a currently stable CS
        4. Hope that this works on miscibility gaps

    This should take care of the DLASCLS error since we compute the new phase separately so if
    the composition clashes with a fixed phase, we check that afterwards before attempting to
    run equilibrium on two CS with the same composition
    """
    dbf, comps, phases, models, phase_records = system_info["dbf"], system_info["comps"], system_info["phases"], system_info["models"], system_info["phase_records"]
    #Get driving force and find index that maximized driving force
    state_conds = {str(key): point.global_conditions[key] for key in STATEVARS}
    points = calculate(dbf, comps, phases, model=models, phase_records=phase_records, output="GM", to_xarray=False, pdens=pdens, **state_conds)
    gm = np.squeeze(points.GM)
    x = np.squeeze(points.X)
    y = np.squeeze(points.Y)
    phase_ids = np.squeeze(points.Phase)
    g_chempot = x * point.chemical_potentials
    dGs = np.sum(g_chempot, axis=1) - gm

    # max_id = np.argmax(dGs)

    # #Create composition set and create DormantPhase and solve for driving force
    # cs = CompositionSet(phase_records[phase_ids[max_id]])
    # cs.update(y[max_id, :cs.phase_record.phase_dof], 1.0, map_utils.get_statevars_array(point.global_conditions))
    # dormantPhase = DormantPhase(cs, None)
    # dG = point.get_property(dormantPhase.driving_force)

    sorted_indices = np.argsort(dGs)[::-1]
    cs = None
    dG = 0
    tested_phases = []
    for i in range(np.amin([num_candidates, len(sorted_indices)])):
        index = sorted_indices[i]
        if phase_ids[index] not in tested_phases:
            tested_phases.append(phase_ids[index])
            test_cs = CompositionSet(phase_records[phase_ids[index]])
            test_cs.update(y[index][:test_cs.phase_record.phase_dof], 1.0, map_utils.get_statevars_array(point.global_conditions))
            dormantPhase = DormantPhase(test_cs, None)
            test_dg = point.get_property(dormantPhase.driving_force)
            _log.info(f"Testing phase {phase_ids[index]} with dG={dGs[index]}->{test_dg} for global min.")
            if test_dg > dG:
                dG = test_dg
                cs = test_cs
    
    #If driving force is above tolerance, then create a new point with the additional composition set
    if dG < tol:
        return None
    else:
        _log.info(f'Global min potentially detected. {point.stable_phases} + {cs.phase_record.phase_name} with dG = {dG}')
        if _detect_degenerate_phase(point, cs):
            new_point = Point(point.global_conditions, point.chemical_potentials, point.fixed_composition_sets, point.free_composition_sets)
            map_utils.update_cs_phase_frac(cs, 1e-6)
            new_point._free_composition_sets.append(cs)
            return new_point
        else:
            _log.info(f'Global min was falsely detected. No global min found')
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
    """
    num_sv = len(STATEVARS)
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
            solver = Solver(remove_metastable=True, allow_changing_phases=False)
            results = solver.solve([ref_cs_copy, new_cs_copy], conds)
            if not results.converged:
                return False
        except Exception as e:
            return False
        
        _log.info(f"Equilibrium: {ref_cs_copy}, {new_cs_copy}")
        if np.allclose(ref_cs_copy.dof[num_sv:], new_cs_copy.dof[num_sv:], atol=10*COMP_DIFFERENCE_TOL):
            return False
        
        if ref_cs_copy.NP < 1e-3:
            cs.update(new_cs_copy.dof[num_sv:], cs.NP, new_cs_copy.dof[:num_sv])
            return True
        
    return True
    
def create_node_from_different_points(new_point: Point, orig_cs: list[CompositionSet], axis_vars):
    """
    Between two points with different composition sets (only 1 different CS)
    Compute the node, freeing up the axis var and solving for it
    The unique CS will be fixed to 0 for the node
    """
    prev_cs = [cs for cs in orig_cs]
    new_cs = [cs for cs in new_point.stable_composition_sets]

    #Find the unique CS between the prev and new point
    phases_added = list(set(new_cs) - set(prev_cs))
    phases_removed = list(set(prev_cs) - set(new_cs))

    if len(phases_added) + len(phases_removed) != 1:
        return None
    
    #Fix the unique CS
    if len(phases_added) == 1:
        fixed_cs = phases_added[0]
    elif len(phases_removed) == 1:
        fixed_cs = phases_removed[0]
        new_cs.append(fixed_cs)

    fixed_cs.fixed = True
    map_utils.update_cs_phase_frac(fixed_cs, 0.0)

    #Set one free CS to NP=1 if all CS are NP=0
    if (all(cs.NP == 0.0 for cs in new_cs)):
        for cs in new_cs:
            if not cs.fixed:
                map_utils.update_cs_phase_frac(cs, 1.0)
                break

    #Setup conditions and remove axis variable to free it
    solution_cs = [cs for cs in new_cs]
    new_conditions = copy.deepcopy(new_point.global_conditions)
    for av in axis_vars:
        del new_conditions[av]

    #Solve equilibrium with fixed CS
    try:
        solver = Solver(remove_metastable=True, allow_changing_phases=False)
        results = solver.solve(solution_cs, new_conditions)
        if not results.converged:
            return None
    except Exception as e:
        return None
    
    #Add axis var back
    for av in axis_vars:
        new_conditions[av] = av.compute_property(solution_cs, new_conditions, results.chemical_potentials)
    
    #Create node with parent between thr previous point
    parent = Point(new_conditions, np.array(results.chemical_potentials), [cs for cs in orig_cs if cs.fixed], [cs for cs in orig_cs if not cs.fixed])
    new_node = Node(new_conditions, np.array(results.chemical_potentials), [cs for cs in solution_cs if cs.fixed], [cs for cs in solution_cs if not cs.fixed], parent)
    return new_node

def compute_derivative(point: Point, v_num: v.StateVariable, v_den: v.StateVariable, free_den = True):
    """
    Computes dot derivative of d v_num / d v_den

    Pycalphad workspace will support an API for dot derivatives, but I wasn't
    sure how it works
    """
    # Ideally we should be able to use point.stable_composition_sets, but this is
    # a dirty workaround since the deltas only seem to store values for unfixed phases,
    # which messes up the indexing 
    comp_sets = point.free_composition_sets + point.fixed_composition_sets
    chem_pots = point.chemical_potentials
    conds = copy.deepcopy(point.global_conditions)

    if free_den:
        del conds[v_num]

    # Get deltas (denominator of derivative)
    solver = Solver()
    spec = solver.get_system_spec(comp_sets, conds)
    state = spec.get_new_state(comp_sets)
    state.chemical_potentials[:] = chem_pots
    state.recompute(spec)
    deltas = v_den.dot_deltas(spec, state)

    # Get derivative
    # Currently, we'll only support state variables and composition (mole)
    # For state variable, we can simply get the dot derivative
    if v_num in STATEVARS:
        der = v_num.dot_derivative(comp_sets, conds, chem_pots, deltas)
        return der

    # For mole fraction, we take
    #   x = sum(n_alpha * x_alpha)
    # and compute
    #   dx = sum(n_a * dx_a + x_a * dn_a)
    # This is the workaround to indexing issues when there are fixed composition sets
    # The solution in pycalphad would be to store the deltas for all stable phases where it would be 0 for fixed phases
    #   but this is something I should discuss later
    elif isinstance(v_num, v.X):
        if v_num.phase_name is None:
            der = 0
            for ph in point.free_phases:
                vphx = v.X(ph, v_num.species.escaped_name.upper())
                vnp = v.NP(ph)
                x = np.squeeze(vphx.compute_property(comp_sets, conds, chem_pots))
                n = np.squeeze(vnp.compute_property(comp_sets, conds, chem_pots))
                dx = vphx.dot_derivative(comp_sets, conds, chem_pots, deltas)
                dnp = vnp.dot_derivative(comp_sets, conds, chem_pots, deltas)
                der += n*dx + x*dnp
            return der
        else:
            der = v_num.dot_derivative(comp_sets, conds, chem_pots, deltas)
            return der
        
    # We currently don't support other types of variables
    # Would be cool though!
    return None


    