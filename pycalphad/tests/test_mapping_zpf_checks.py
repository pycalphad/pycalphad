import copy

import numpy as np

from pycalphad import Database, variables as v
from pycalphad.codegen.phase_record_factory import PhaseRecordFactory
from pycalphad.core.utils import instantiate_models

from pycalphad.mapping.starting_points import point_from_equilibrium
from pycalphad.mapping.primitives import ZPFLine, ZPFState, Direction
import pycalphad.mapping.zpf_equilibrium as zeq
import pycalphad.mapping.zpf_checks as zchk
from pycalphad.tests.fixtures import select_database, load_database


"""
Tests for the different checks when stepping a zpf line

Each test should be made to encounter all possible routes of the check functions
    Some check functions will have to be designed to force a condition (ex. check_change_in_phases and check_global_min 
    requires forcing a new node to not be found)
"""

def _create_test_point(dbf, comps, phases, conditions):
    point = point_from_equilibrium(dbf, comps, phases, conditions)
    return point

def _create_default_arguments(axes, delta, lims):
    axis_data = {
        "axis_vars": axes,
        "axis_delta": {ax: d for ax, d in zip(axes, delta)},
        "axis_lims": {ax: lim for ax, lim in zip(axes, lims)}
    }
    extra_args = {
        "delta_scale": 0.5,
        "min_delta_ratio": 0.1,
        "global_check_interval": 1,
        "normalize_factor": {av: axis_data["axis_delta"][av] for av in axis_data["axis_vars"]}
    }
    return axis_data, extra_args

@select_database("alzn_mey.tdb")
def test_check_valid_point(load_database):
    #Create initial point
    dbf = load_database()
    comps = ["AL", "ZN", "VA"]
    phases = ["HCP_A3", "LIQUID", "FCC_A1"]
    conditions = {v.T: 500, v.P: 101325, v.N: 1, v.X("AL"): 0.5}
    point = _create_test_point(dbf, comps, phases, conditions)

    #Start zpf line with initial point
    delta_T = 10
    zpf_line = ZPFLine(point.fixed_phases, point.free_phases)
    zpf_line.points.append(point)
    zpf_line.axis_var = v.T
    zpf_line.axis_direction = Direction.POSITIVE
    zpf_line.current_delta = delta_T

    #Create new point with adjusted condition
    new_conditions = copy.deepcopy(conditions)
    new_conditions[v.T] += delta_T*zpf_line.axis_direction.value
    step_result = zeq.update_equilibrium_with_new_conditions(point, new_conditions)

    axis_data, extra_args = _create_default_arguments([v.T], [10], [(300, 1000)])

    # check_valid_point when step_result is valid equilibrium -> no change in zpf line
    new_node = zchk.check_valid_point(zpf_line, step_result, axis_data, **extra_args)
    assert zpf_line.status == ZPFState.NOT_FINISHED
    assert zpf_line.current_delta == delta_T

    # check_valid_point when step_result is invalid (None) -> zpf line delta should reduce
    new_node = zchk.check_valid_point(zpf_line, None, axis_data, **extra_args)
    assert zpf_line.status == ZPFState.ATTEMPT_NEW_STEP
    assert zpf_line.current_delta == delta_T * extra_args["delta_scale"]

    # check_valid_point when step_result is invalid and current_delta is too small -> zpf line ends
    extra_args["min_delta_ratio"] = 1
    new_node = zchk.check_valid_point(zpf_line, None, axis_data, **extra_args)
    assert zpf_line.status == ZPFState.FAILED

@select_database("alzn_mey.tdb")
def test_check_axis_values(load_database):
    #Create initial point
    dbf = load_database()
    comps = ["AL", "ZN", "VA"]
    phases = ["HCP_A3", "LIQUID", "FCC_A1"]
    conditions = {v.T: 500, v.P: 101325, v.N: 1, v.X("AL"): 0.5}
    point = _create_test_point(dbf, comps, phases, conditions)

    #Start zpf line with initial point
    delta_T = 10
    zpf_line = ZPFLine(point.fixed_phases, point.free_phases)
    zpf_line.points.append(point)
    zpf_line.axis_var = v.T
    zpf_line.axis_direction = Direction.POSITIVE
    zpf_line.current_delta = delta_T

    #Create new point with adjusted condition
    new_conditions = copy.deepcopy(conditions)
    new_conditions[v.T] += delta_T*zpf_line.axis_direction.value
    step_result = zeq.update_equilibrium_with_new_conditions(point, new_conditions)

    axis_data, extra_args = _create_default_arguments([v.T, v.X("AL")], [10, 0.1], [(300,1000), (0,1)])

    # check_axis_values for variables within limit -> no change
    new_node = zchk.check_axis_values(zpf_line, step_result, axis_data, **extra_args)
    assert zpf_line.status == ZPFState.NOT_FINISHED

    # check_axis_values for variable outsite limit
    axis_data["axis_lims"][v.T] = (300, 505)
    new_node = zchk.check_axis_values(zpf_line, step_result, axis_data, **extra_args)
    assert zpf_line.status == ZPFState.REACHED_LIMIT

    # check_axis_values for change in value above distance threshold
    axis_data["axis_lims"][v.T] = (300, 1000)
    extra_args["normalize_factor"][v.T] = 0.1
    new_node = zchk.check_axis_values(zpf_line, step_result, axis_data, **extra_args)
    assert zpf_line.status == ZPFState.FAILED

@select_database("alzn_mey.tdb")
def test_check_similar_phase_composition(load_database):
    #Create initial point
    dbf = load_database()
    comps = ["AL", "ZN", "VA"]
    phases = ["HCP_A3", "LIQUID", "FCC_A1"]
    conditions = {v.T: 500, v.P: 101325, v.N: 1, v.X("AL"): 0.5}
    point = _create_test_point(dbf, comps, phases, conditions)

    #Start zpf line with initial point
    delta_T = 10
    zpf_line = ZPFLine(point.fixed_phases, point.free_phases)
    zpf_line.points.append(point)
    zpf_line.axis_var = v.T
    zpf_line.axis_direction = Direction.POSITIVE
    zpf_line.current_delta = delta_T

    #Create new point with adjusted condition
    new_conditions = copy.deepcopy(conditions)
    new_conditions[v.T] += delta_T*zpf_line.axis_direction.value
    step_result = zeq.update_equilibrium_with_new_conditions(point, new_conditions)

    axis_data, extra_args = _create_default_arguments([v.T, v.X("AL")], [10, 0.1], [(300,1000), (0,1)])

    # check_similar_phase_composition for different phases -> no change
    new_node = zchk.check_similar_phase_composition(zpf_line, step_result, axis_data, **extra_args)
    assert zpf_line.status == ZPFState.NOT_FINISHED

    # check_similar_phase_composition after adding a similar composition set -> zpf fails
    step_result[0]._free_composition_sets.append(step_result[0]._free_composition_sets[0])
    new_node = zchk.check_similar_phase_composition(zpf_line, step_result, axis_data, **extra_args)
    assert zpf_line.status == ZPFState.REACHED_LIMIT

@select_database("alzn_mey.tdb")
def test_check_change_in_phases(load_database):
    #Create initial point
    dbf = load_database()
    comps = ["AL", "ZN", "VA"]
    phases = ["HCP_A3", "LIQUID", "FCC_A1"]
    conditions = {v.T: 610, v.P: 101325, v.N: 1, v.X("AL"): 0.6}
    point = _create_test_point(dbf, comps, phases, conditions)

    #Start zpf line with initial point
    delta_T = 10
    zpf_line = ZPFLine(point.fixed_phases, point.free_phases)
    zpf_line.points.append(point)
    zpf_line.axis_var = v.T
    zpf_line.axis_direction = Direction.POSITIVE
    zpf_line.current_delta = delta_T

    #Create new point with adjusted condition
    new_conditions = copy.deepcopy(conditions)
    new_conditions[v.T] += delta_T*zpf_line.axis_direction.value
    step_result = zeq.update_equilibrium_with_new_conditions(zpf_line.points[-1], new_conditions)

    axis_data, extra_args = _create_default_arguments([v.T], [10], [(300,1000)])

    # check_change_in_phases for same set of phases -> no change
    new_node = zchk.check_change_in_phases(zpf_line, step_result, axis_data, **extra_args)
    assert zpf_line.status == ZPFState.NOT_FINISHED

    zpf_line.points.append(step_result[0])

    #Create new point with adjusted condition
    new_conditions[v.T] += delta_T*zpf_line.axis_direction.value
    step_result = zeq.update_equilibrium_with_new_conditions(zpf_line.points[-1], new_conditions)

    # check_change_in_phases for same phase becoming unstable -> new node found with same set of phases as previous point
    new_node = zchk.check_change_in_phases(zpf_line, step_result, axis_data, **extra_args)
    if zpf_line.status == ZPFState.FAILED:
        assert new_node is None
    elif zpf_line.status == ZPFState.NEW_NODE_FOUND:
        assert new_node is not None
        assert len(new_node.stable_composition_sets) == 2
        assert np.isclose(new_node.get_property(v.T), 622.456, rtol=1e-3)
    
    # check_change_in_phases for same phase becoming unstable and new node could not be found -> zpf line failed
    extra_args["do_not_create_node"] = True
    new_node = zchk.check_change_in_phases(zpf_line, step_result, axis_data, **extra_args)
    assert zpf_line.status == ZPFState.FAILED

@select_database("alzn_mey.tdb")
def test_check_global_min(load_database):
    #Create initial point
    dbf = load_database()
    comps = ["AL", "ZN", "VA"]
    phases = ["HCP_A3", "LIQUID", "FCC_A1"]
    conditions = {v.T: 640, v.P: 101325, v.N: 1, v.X("AL"): 0.6}
    point = _create_test_point(dbf, comps, phases, conditions)

    #Start zpf line with initial point
    delta_T = 10
    zpf_line = ZPFLine(point.fixed_phases, point.free_phases)
    zpf_line.points.append(point)
    zpf_line.axis_var = v.T
    zpf_line.axis_direction = Direction.NEGATIVE
    zpf_line.current_delta = delta_T

    #Create new point with adjusted condition
    new_conditions = copy.deepcopy(conditions)
    new_conditions[v.T] += delta_T*zpf_line.axis_direction.value
    step_result = zeq.update_equilibrium_with_new_conditions(zpf_line.points[-1], new_conditions)

    axis_data, extra_args = _create_default_arguments([v.T], [10], [(300,1000)])
    system_info = {
        "dbf": dbf,
        "comps": comps,
        "phases": phases,
    }
    system_info["models"] = instantiate_models(system_info["dbf"], comps, phases)
    system_info["phase_records"] = PhaseRecordFactory(system_info["dbf"], comps, {v.N, v.P, v.T}, system_info["models"])
    extra_args["system_info"] = system_info

    # check_change_in_phases for same set of phases -> no change
    new_node = zchk.check_global_min(zpf_line, step_result, axis_data, **extra_args)
    assert zpf_line.status == ZPFState.NOT_FINISHED

    zpf_line.points.append(step_result[0])

    #Create new point with adjusted condition
    new_conditions[v.T] += delta_T*zpf_line.axis_direction.value
    step_result = zeq.update_equilibrium_with_new_conditions(zpf_line.points[-1], new_conditions)

    # check_change_in_phases for same phase becoming unstable -> new node found with same set of phases as previous point
    new_node = zchk.check_global_min(zpf_line, step_result, axis_data, **extra_args)
    if zpf_line.status == ZPFState.FAILED:
        assert new_node is None
    elif zpf_line.status == ZPFState.NEW_NODE_FOUND:
        assert new_node is not None
        assert len(new_node.stable_composition_sets) == 2
        assert np.isclose(new_node.get_property(v.T), 622.456, rtol=1e-3)

    # check_change_in_phases for same phase becoming unstable and new node could not be found -> zpf line failed
    extra_args["do_not_create_node"] = True
    new_node = zchk.check_global_min(zpf_line, step_result, axis_data, **extra_args)
    assert zpf_line.status == ZPFState.FAILED
