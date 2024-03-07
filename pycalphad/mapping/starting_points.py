import numpy as np
import itertools
from typing import Dict, Iterator, Tuple, List, Mapping
from pycalphad import equilibrium, variables as v
from pycalphad.core.utils import instantiate_models, filter_phases, unpack_components
from pycalphad.codegen.callables import build_phase_records
from pycalphad.core.equilibrium import _adjust_conditions
from pycalphad.core.solver import Solver
from pycalphad_mapping.primitives import _get_value_for_var
from pycalphad.core.constants import MIN_SITE_FRACTION

from .primitives import STATEVARS, Point, Direction

from pycalphad_mapping.utils import _extract_point_from_dataset, _is_a_potential, _get_conditions_from_eq

def starting_point_from_equilibrium(dbf, comps, phases, conditions, condition_to_drop, **eq_kwargs):
    """
    Point-equilibrium-like API for generating a starting point for mapping

    Computes point equilibrium
    Creates Point object from equilibrium results
        If user does not specify a condition to drop, then no fixed phases in Point
        Else, Point will contain 1 fixed phase (first phase that shows up in equilibrium result)
    """
    # condition_to_drop is the one to remove when fixing a phase
    try:
        conditions = {key: float(val) for key, val in conditions.items()}
    except TypeError:
        # depends on leaked scope to debug key/value
        raise TypeError("starting_point_from_equilibrium(): all values must be scalar. Got value {val} for key {key} cannot be coerced to float.")

    phase_records = eq_kwargs.get("phase_records")
    if phase_records is None:
        phases = filter_phases(dbf, unpack_components(dbf, comps), phases)
        models = eq_kwargs.get("model")
        if models is None:
            models = instantiate_models(dbf, comps, phases)
            eq_kwargs["model"] = models
        phase_records = build_phase_records(dbf, comps, phases, STATEVARS, models)
        eq_kwargs["phase_records"] = phase_records

    eq_res = equilibrium(dbf, comps, phases, conditions, **eq_kwargs)
    if condition_to_drop is not None:
        point = _extract_point_from_dataset(dbf, comps, models, eq_res, phase_records)
        del point.global_conditions[condition_to_drop]
    else:
        point = _extract_point_from_dataset(dbf, comps, models, eq_res, phase_records, num_phases_to_fix=0)
    return point


def _generate_fixed_variable_conditions(conditions: Mapping[v.StateVariable, "ArrayLike"], original_conditions, comp_limit) -> Iterator[Dict[v.StateVariable, float]]:
    """Generate all permuations of fixed conditions at their bounds

    Scalar conditions contribute one condition to the fixed variables.
    Non-scalar conditions contribute one lower and one upper bound condition.

    Assumes non-scalar conditions are sorted (the smallest and largest values are the endpoints).
    """
    comp_offset = sum([original_conditions[v] for v in original_conditions if (not isinstance(original_conditions[v], tuple) and v not in STATEVARS)])
    condition_configurations = []
    for cond_key, cond_vals in conditions.items():
        if len(cond_vals) == 1:
            configs = [{cond_key: cond_vals[0]}]
        else:
            #The unpack_conditions function will sometimes not reach the upper axis limit due to the step size (this seems to be due to how np.arange works)
            #In this case, we should check the upper value and adjust to be closer to the upper axis limit within numerical limits
            lower_val = cond_vals[0]
            if cond_key in STATEVARS:
                thres = 0
                upper_val = cond_vals[-1]
            else:
                thres = comp_limit
                upper_val = np.amin([1-comp_offset, cond_vals[-1]])
            if np.abs(lower_val) < thres:
                lower_val = comp_limit
            if np.abs(original_conditions[cond_key][1] - upper_val) > thres:
                upper_val = original_conditions[cond_key][1] - thres
            configs = [{cond_key: lower_val}, {cond_key: upper_val}]
        condition_configurations.append(configs)
    return itertools.product(*condition_configurations)


# TODO:
def _get_unique_starting_points_from_along_potential_axis(dbf, comps, models, eq_result, axis_var, phase_records) -> List[Point]:
    # eq_result is 1D in the axis_var coordinate
    found_starting_points = []

    for coord_idx in range(1, len(eq_result.coords[str(axis_var)])):
        if any(np.isnan(eq_result.isel(**{str(axis_var): coord_idx}).GM.values)) or any(np.isnan(eq_result.isel(**{str(axis_var): coord_idx-1}).GM.values)):
            continue
        p1 = _extract_point_from_dataset(dbf, comps, models, eq_result.isel(**{str(axis_var): coord_idx-1}), phase_records, num_phases_to_fix=0)
        p2 = _extract_point_from_dataset(dbf, comps, models, eq_result.isel(**{str(axis_var): coord_idx}), phase_records, num_phases_to_fix=0)
        #Only look for cases where the number of composition sets in p1 and p2 is 1
        #This could miss two phase regions, but if we're at the composition limits, this may be okay, since only 1 phase should be stable
        #We're really looking for really thin phase changes
        #TODO: This will of course be an issue if the user sets the composition axis limits to not be between 0 and 1 or if the user is doing an isopleth
        if len(p1.stable_composition_sets) == 1 and len(p2.stable_composition_sets) == 1 and p1.stable_composition_sets[0].phase_record.phase_name != p2.stable_composition_sets[0].phase_record.phase_name:
            #Fix one composition set and create a new point and solve
            #    The fixed composition set should be the one further from the axis
            cs_1 = p1.stable_composition_sets[0]
            cs_2 = p2.stable_composition_sets[0]
            cs_2.fixed = True
            cs_2.NP = 0
            cs_2.update(cs_2.dof[len(STATEVARS):], cs_2.NP, cs_1.dof[:len(STATEVARS)])  #Update cs_2 to have the same starting temperature
            curr_conds = _get_conditions_from_eq(eq_result.isel(**{str(axis_var): coord_idx}))
            point = Point(curr_conds, [cs_2], [cs_1], [])


            solution_compsets = [x for x in point.stable_composition_sets]
            for cs in solution_compsets:
                cs.update(cs.dof[len(STATEVARS):], cs.NP, np.asarray([curr_conds[sv] for sv in STATEVARS], dtype=float))

            try:
                sub_conds = {k:v for k,v in curr_conds.items()}
                del sub_conds[axis_var]
                solver = Solver(remove_metastable=True)
                result = solver.solve(solution_compsets, {str(ky): vl for ky, vl in sub_conds.items()})
                if result.converged and len(solution_compsets) == 2:
                    point.global_conditions[axis_var] = _get_value_for_var(cs_1, axis_var)
                    found_starting_points.append(point)
            except Exception as e:
                print(e)

    return found_starting_points


def _get_unique_starting_points_from_along_molar_quantity_axis(dbf, comps, models, eq_result, axis_var, phase_records) -> List[Point]:
    '''
    Go through list of equilibrium results and attempt to extract Point from each result
        Individual eq result must be multi-phase to be able to store
        Only add Point if it is not already in the list - since we're going from min to max along the axis variable, we only have to compare the last Point in the list
    '''
    # eq_result is 1D in the axis_var coordinate
    found_starting_points = []
    for coord_idx in range(len(eq_result.coords[str(axis_var)])):
        # If the point is not equal to the last point, add it to the list of starting points
        try:
            point = _extract_point_from_dataset(dbf, comps, models, eq_result.isel(**{str(axis_var): coord_idx}), phase_records)
        except AssertionError:
            # Point not in a multi-phase region
            continue
        if len(found_starting_points) == 0 or point != found_starting_points[-1]:
            found_starting_points.append(point)
    return found_starting_points


def automatic_starting_points_from_axis_limits(dbf, comps, phases, conditions, **eq_kwargs) -> List[Tuple[Point, Direction]]:
    """
    Search the edges of the space to be mapped for all axis variables.

    Get all conditions that are a range - e.g. (min, max, interval)

    For each free axis variable
        Generate conditions (this will include all scalar conditions, then min + max for all other free conditions)
        Add free axis to condition
        Calculate equilibrium and generate list of Points - differs if free condition is a potential (temperature, pressure) or other (composition)
        Add Points to overall starting point list
    """

    #Grab all free conditions
    original_conditions = {k:v for k,v in conditions.items()}
    conditions = _adjust_conditions(conditions)
    axis_cond_keys = [key for key, val in conditions.items() if len(val) > 1]
    assert len(axis_cond_keys) > 1, f"There should be at least one free variable in the conditions. Got {conditions}"
    has_free_statevar = any([fv in STATEVARS for fv in axis_cond_keys])

    #Dirty workaround, I noticed the solver can have issues converging when we're stepping along temperature and the composition is near the numerical limit
    #   Setting the limit to 1e-6 seemed to work pretty well. I don't want to increase it too much since it could affect how the plot will look at the end
    #On the other hand, if we're stepping along two composition axis, then we want to be at the numerical limits
    #   The equilibrium method is okay with this and this prevents a bunch of unnecessary starting points from generating
    #I think the more stable solution here would be to build phase records with the 0 composition element removed,
    #   then somehow make the composition sets for the full list of elements
    COMPOSITION_LIMIT = 1e-6 if has_free_statevar else 1e-10

    #Generate phase records - should this be an external function?
    phase_records = eq_kwargs.get("phase_records")
    if phase_records is None:
        phases = filter_phases(dbf, unpack_components(dbf, comps), phases)
        models = eq_kwargs.get("model")
        if models is None:
            models = instantiate_models(dbf, comps, phases)
            eq_kwargs["model"] = models
        phase_records = build_phase_records(dbf, comps, phases, STATEVARS, models)
        eq_kwargs["phase_records"] = phase_records

    found_starting_points = []
    start_directions = []   #Starting direction for each point will be the other axis variable
    # For each axis variable
    for axis_var in axis_cond_keys:
        # Fix all other variables to either their upper or lower limit (exhaustively)
        _other_conditions = {key: val for key, val in conditions.items() if key is not axis_var}
        for fixed_conds_tup in _generate_fixed_variable_conditions(_other_conditions, original_conditions, COMPOSITION_LIMIT):
            # fixed_conds_tup is a tuple of single element dicts, we need to combine them into a single dict
            fixed_conds = {axis_var: conditions[axis_var]}
            for fcd in fixed_conds_tup:
                fixed_conds.update(fcd)
            # TODO: should these be step calculations instead of equilibrium?
            eq_res = equilibrium(dbf, comps, phases, fixed_conds, **eq_kwargs)
            if _is_a_potential(axis_var):
                new_points_for_axis = _get_unique_starting_points_from_along_potential_axis(dbf, comps, models, eq_res, axis_var, phase_records)
            else:
                new_points_for_axis = _get_unique_starting_points_from_along_molar_quantity_axis(dbf, comps, models, eq_res, axis_var, phase_records)
            #Releasin the axis var happens during the step process, so it isn't needed here
            #for point in new_points_for_axis:
            #    # Release the current axis variable in exchange for the fixed phase
            #    # That is, axis variable that will be used for mapping from this
            #    # starting point _cannot_ be the current axis_var (i.e. it will be
            #    # orthogonal)
            #    del point.global_conditions[axis_var]
            # TODO: consider that we hit the corner of each 1D edge multiple times, we don't want to double count
            found_starting_points.extend(new_points_for_axis)
            start_directions.extend([axis_cond_keys[1-axis_cond_keys.index(axis_var)] for _ in range(len(new_points_for_axis))])
            #if axis_var not in STATEVARS:
            #    start_directions.extend([axis_cond_keys[1-axis_cond_keys.index(axis_var)] for _ in range(len(new_points_for_axis))])
            #else:
            #    start_directions.extend([axis_var for _ in range(len(new_points_for_axis))])
    return found_starting_points, start_directions
