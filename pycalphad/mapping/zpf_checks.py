from typing import Mapping
import logging

import numpy as np

from pycalphad import variables as v
from pycalphad.core.constants import COMP_DIFFERENCE_TOL
from pycalphad.core.composition_set import CompositionSet

from pycalphad.mapping.primitives import STATEVARS, ZPFLine, Point, MIN_COMPOSITION, ZPFState
import pycalphad.mapping.zpf_equilibrium as zeq

_log = logging.getLogger(__name__)

def simple_check_valid_point(step_results: tuple[Point, list[CompositionSet]], **kwargs):
    """
    Returns True or False for whether step result was successful equilibria
    """
    if step_results is None:
        _log.info("Invalid equilibrium results")

    return step_results is not None

def simple_check_change_in_phases(step_results: tuple[Point, list[CompositionSet]], **kwargs):
    """
    Returns True or False for whether step result resulted in same number of phases
    """
    if step_results is None:
        return False
    new_point, orig_cs = step_results
    num_different_phases = len(set(orig_cs).symmetric_difference(set(new_point.stable_composition_sets)))

    if num_different_phases != 0:
        _log.info(f"Number of stable phases changed from {[cs.phase_record.phase_name for cs in orig_cs]} to {new_point.stable_phases}")

    return num_different_phases == 0

def simple_check_global_min(step_results: tuple[Point, list[CompositionSet]], **kwargs):
    """
    Returns True or False for whether step result is still global min
    """
    if step_results is None:
        return False

    pdens = kwargs.get("pdens", 500)
    tol = kwargs.get("tol", 1e-4)
    system_info = kwargs.get("system_info", None)
    num_candidates = kwargs.get("global_num_candidates", 1)
    new_point, orig_cs = step_results
    global_test_point = zeq.find_global_min_point(new_point, system_info, pdens, tol, num_candidates)

    if global_test_point is not None:
        _log.info(f"Point is not global minimum. Current CS: {new_point.stable_phases}, new CS: {global_test_point.stable_phases}")

    return global_test_point is None

def check_valid_point(zpf_line: ZPFLine, step_results: tuple[Point, list[CompositionSet]], axis_data: Mapping, **kwargs):
    """
    3 possible outcomes
        a) Converged equilibrium -> pass
        b) Failed equilibrium, reduce axis delta -> don"t add point and attempt stepping again
        c) Failed equilibrium and axis delta reached minimum -> end zpf line with unexpected ending
    """
    axis_vars, axis_delta, axis_lims = axis_data["axis_vars"], axis_data["axis_delta"], axis_data["axis_lims"]
    delta_scale = kwargs.get("delta_scale", 0.5)
    min_delta_ratio = kwargs.get("min_delta_ratio", 0.1)
    if step_results is None:
        _log.info(f"Invalid equilibrium result, reducing step size from {zpf_line.current_delta} -> {zpf_line.current_delta*delta_scale}")

        zpf_line.current_delta *= delta_scale
        if zpf_line.current_delta / axis_delta[zpf_line.axis_var] < min_delta_ratio:
            _log.info(f"Step size has reached minimum {zpf_line.current_delta}/{axis_delta[zpf_line.axis_var]} = {min_delta_ratio}. Failing ZPF line")

            zpf_line.status = ZPFState.FAILED
            return None
        zpf_line.status = ZPFState.ATTEMPT_NEW_STEP
    return None

def _check_axis_values_within_limit(zpf_line: ZPFLine, prev_point_vars: dict[v.StateVariable, float], new_point_vars: dict[v.StateVariable, float], axis_data: Mapping, **kwargs):
    """
    Checks that axis values are within the axis limits

    TODO: should check if we really need x_offset, this could affect phases like GRAPHITE which has a composition of X(C)=1
    """
    axis_vars, axis_delta, axis_lims = axis_data["axis_vars"], axis_data["axis_delta"], axis_data["axis_lims"]

    # for av in [zpf_line.axis_var]:
    for av in axis_vars:
        statevar_offset = 0
        x_offset = 0 if zpf_line.axis_var in STATEVARS else MIN_COMPOSITION
        offset = statevar_offset if av in STATEVARS else x_offset
        offset = 0 if av != zpf_line.axis_var else offset

        if new_point_vars[av] <= min(axis_lims[av]) + offset or new_point_vars[av] >= max(axis_lims[av]) - offset:
        # if np.isclose(new_point_vars[av], min(axis_lims[av]) + offset, rtol=1e-5) or np.isclose(new_point_vars[av], max(axis_lims[av]) - offset):
            _log.info(f"New point outside axis limits. {av} = {new_point_vars[av]}. Limits = {axis_lims[av]}")
            return False
    return True

def _check_composition_within_limit(zpf_line: ZPFLine, step_results: tuple[Point, list[CompositionSet]]):
    new_point, orig_cs = step_results
    comp_sum = sum(new_point.get_property(var) for var in new_point.global_conditions if isinstance(var, v.X))
    if comp_sum < MIN_COMPOSITION or comp_sum > 1-MIN_COMPOSITION:
        return False
    return True

def _check_axis_values_by_distance(zpf_line: ZPFLine, prev_point_vars: dict[v.StateVariable, float], new_point_vars: dict[v.StateVariable, float], axis_data: Mapping, **kwargs):
    """
    Checks that the normalized distance between the previous point and the new point is within reasonable values

    Note: a threshold of 3 is quite large since the axis swapping should limit this to 1 (give/take some leeway if the swapping hadn"t occured yet)
    """
    axis_vars, axis_delta, axis_lims = axis_data["axis_vars"], axis_data["axis_delta"], axis_data["axis_lims"]
    normalize_factor = kwargs.get("normalize_factor", {av: 1 for av in axis_vars})

    dist_threshold = kwargs.get("distance_threshold", 3)
    # Squeezing here since v.MoleFraction.compute_property will return an array instead of a scalar
    distances = [np.squeeze(np.abs((new_point_vars[av] - prev_point_vars[av])/normalize_factor[av])) for av in axis_vars]
    dist = np.amax(distances)
    if dist > dist_threshold:
        av = axis_vars[np.argmax(distances)]
        _log.info(f"Axis variable moved more than threshold distance. {av} = {prev_point_vars[av]} -> {new_point_vars[av]}. Distance = {dist*normalize_factor[av]} > {dist_threshold}*{normalize_factor[av]}")

    return dist < dist_threshold

def check_axis_values(zpf_line: ZPFLine, step_results: tuple[Point, list[CompositionSet]], axis_data: Mapping, **kwargs):
    """
    Checks both if axis values are within axis limits and if the axis values hadn"t moved to far from previous point

    The two checks are separated from here so that we can use them in _check_global_min and _check_change_in_phases

    3 possible outcomes
        a) Axes are within limits and minimal distance change -> pass
        b) Axes are outside limits -> end zpf line with graceful ending
        c) Distance changed too much -> end zpf line with unexpected ending
    """
    if step_results is None:
        return None

    axis_vars, axis_delta, axis_lims = axis_data["axis_vars"], axis_data["axis_delta"], axis_data["axis_lims"]

    new_point, orig_cs = step_results
    prev_point = zpf_line.points[-1]
    new_point_vars = {av: new_point.get_property(av) for av in axis_vars}
    prev_point_vars = {av: prev_point.get_property(av) for av in axis_vars}

    if not _check_axis_values_by_distance(zpf_line, prev_point_vars, new_point_vars, axis_data, **kwargs):
        _log.info("Variable more than distance threshold. Failing ZPF line")
        zpf_line.status = ZPFState.FAILED
        return None

    if not _check_axis_values_within_limit(zpf_line, prev_point_vars, new_point_vars, axis_data, **kwargs):
        _log.info("Variable reach axis limit. Ending ZPF line")
        zpf_line.status = ZPFState.REACHED_LIMIT
        return None
    # if not _check_composition_within_limit(zpf_line, step_results):
    #    _log.info("Variable reach composition limit. Ending ZPF line")
    #    zpf_line.status = ZPFState.REACHED_LIMIT
    #    return None

def check_change_in_phases(zpf_line: ZPFLine, step_results: tuple[Point, list[CompositionSet]], axis_data: Mapping, **kwargs):
    """
    If the number of phases changed upon stepping, then attempt to create a new node
        If making the new node was unsuccessful, then we end the zpf line anyways

    3 possible outcomes
        a) No change in phases -> pass
        b) Change in phases, node successfully found -> process new node and end zpf line with graceful ending
        c) Change in phases, node not found -> end zpf line with unexpected ending
    """
    if step_results is None:
        return None

    axis_vars, axis_delta, axis_lims = axis_data["axis_vars"], axis_data["axis_delta"], axis_data["axis_lims"]
    do_not_create_node = kwargs.get("do_not_create_node", False)    #For testing/debugging purposes

    new_point, orig_cs = step_results
    new_point_vars = {av: new_point.get_property(av) for av in axis_vars}
    num_different_phases = len(set(orig_cs).symmetric_difference(set(new_point.stable_composition_sets)))
    if num_different_phases != 0:
        _log.info("Number of phases changed")
        # By default, assumed the zpf failed. When we successfully find a new node, then we change this
        zpf_line.status = ZPFState.FAILED

        # Make sure the set of phases only change by one
        # Since we step along a single axis variable, we can only fix one additional phase to satisfy Gibbs phase rule
        if num_different_phases > 1:
            _log.info("Number of phases changes by more than one. Failing ZPF line")
            return None

        if do_not_create_node:
            new_node = None
        else:
            new_node = zeq.create_node_from_different_points(new_point, orig_cs, axis_vars)

        # If the new node was successfully created, then process the node, otherwise, we unexpectedly ended
        if new_node is not None:
            new_node_vars = {av: new_node.get_property(av) for av in axis_vars}
            # Check that new node satisfy axis limits and distance between nodes
            # If not, then the zpf line ends unexpectedly
            check_axis = _check_axis_values_within_limit(zpf_line, new_point_vars, new_node_vars, axis_data, **kwargs)
            check_dist =_check_axis_values_by_distance(zpf_line, new_point_vars, new_node_vars, axis_data, **kwargs)
            if check_axis and check_dist:
                _log.info(f"New node found successfully. {new_node.global_conditions}, {new_node.fixed_phases}, {new_node.free_phases}")
                zpf_line.status = ZPFState.NEW_NODE_FOUND
                return new_node

        _log.info("New node could not be found. Failing ZPF line")

    return None

def check_global_min(zpf_line: ZPFLine, step_results: tuple[Point, list[CompositionSet]], axis_data: Mapping, **kwargs):
    """
    Check if the point is global minimum
    1. Check if a new composition set can be added that can lower free energy
    2. Create a new node with the additional composition set
    3. Check that the new node is valid

    3 possible outcomes
        a) No change in phases -> pass
        b) Change in phases, node successfully found -> process new node and end zpf line with graceful ending
        c) Change in phases, node not found -> end zpf line with unexpected ending
    """
    if step_results is None:
        return None

    axis_vars, axis_delta, axis_lims = axis_data["axis_vars"], axis_data["axis_delta"], axis_data["axis_lims"]
    global_check_interval = kwargs.get("global_check_interval", 1)
    pdens = kwargs.get("pdens", 500)
    tol = kwargs.get("tol", 1e-4)
    num_candidates = kwargs.get("global_num_candidates", 1)
    system_info = kwargs.get("system_info", None)
    do_not_create_node = kwargs.get("do_not_create_node", False)    #For testing/debugging purposes

    new_point, orig_cs = step_results
    new_point_vars = {av: new_point.get_property(av) for av in axis_vars}
    if len(zpf_line.points) % global_check_interval == 0:
        global_test_point = zeq.find_global_min_point(new_point, system_info, pdens, tol, num_candidates)
        if global_test_point is not None:
            _log.info(f"Global min detected. {new_point.stable_phases} -> {global_test_point.stable_phases}")
            # By default, assumed the zpf failed. When we successfully find a new node, then we change this
            zpf_line.status = ZPFState.FAILED

            if do_not_create_node:
                new_node = None
            else:
                new_node = zeq.create_node_from_different_points(global_test_point, new_point.stable_composition_sets, axis_vars)

            if new_node is not None:
                new_node_vars = {av: new_node.get_property(av) for av in axis_vars}
                # Check that new node satisfy axis limits and distance between nodes
                # If not, then the zpf line ends unexpectedly
                check_axis = _check_axis_values_within_limit(zpf_line, new_point_vars, new_node_vars, axis_data, **kwargs)
                check_dist =_check_axis_values_by_distance(zpf_line, new_point_vars, new_node_vars, axis_data, **kwargs)
                # check_dist = True
                if check_axis and check_dist:
                    zpf_line.status = ZPFState.NEW_NODE_FOUND
                    _log.info(f"New node found successfully. {new_node.global_conditions}, {new_node.fixed_phases}, {new_node.free_phases}")
                    return new_node

            _log.info("New node could not be found. Failing ZPF line")

def check_similar_phase_composition(zpf_line: ZPFLine, step_results: tuple[Point, list[CompositionSet]], axis_data: Mapping, **kwargs):
    """
    If two composition sets are close in composition, then we stop zpf line pre-maturely so that they don"t go on top of each other

    If the composition are the same, it can result in ill-defined matrices in the solver

    2 possible outcomes
        a) Composition sets are separate -> pass
        b) Composition sets are similar -> end zpf line with unexpected ending
    """
    if step_results is None:
        return None

    new_point, orig_cs = step_results
    comp_sets = new_point.stable_composition_sets
    for i in range(len(comp_sets)):
        for j in range(i+1, len(comp_sets)):
            same_comp = np.allclose(comp_sets[i].X, comp_sets[j].X, atol=10*COMP_DIFFERENCE_TOL)
            same_phase = comp_sets[i].phase_record.phase_name == comp_sets[j].phase_record.phase_name
            if same_comp and same_phase:
                zpf_line.status = ZPFState.REACHED_LIMIT
                _log.info(f"Two composition sets have the same composition. Ending ZPF line. {comp_sets[i]} = {comp_sets[j]}")
                return None
    return None
