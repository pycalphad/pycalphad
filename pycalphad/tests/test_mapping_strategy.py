import copy
from importlib.resources import files

import numpy as np
import matplotlib.pyplot as plt

from pycalphad import binplot, ternplot, Database, variables as v
from pycalphad.tests.fixtures import select_database, load_database
from pycalphad.core.utils import instantiate_models
from pycalphad.codegen.phase_record_factory import PhaseRecordFactory

from pycalphad.mapping import StepStrategy, IsoplethStrategy, plot_step, plot_isopleth
from pycalphad.mapping.starting_points import point_from_equilibrium
from pycalphad.mapping.zpf_equilibrium import find_global_min_point

import pycalphad.tests.databases

"""
These tests goes through the full binary, ternary, step and isopleth process to check if all the intended phase regions show up

For specific edge cases, it would be better to initialize a strategy near the edge case and step into it rather than doing the full map

NOTES:
    The isopleth test does not cover the invariant node exit finding with the current database
    Some order-disorder and ionic liquid models run into a ValueError: buffer source array is read-only when trying
        update a composition set. The databases here avoids these cases, but we'll need to address this
    The current tests will not check if mapping created more phase regions than expected
"""

@select_database("alcocrni.tdb")
def test_binary_strategy(load_database):
    dbf = load_database()

    ax, strategy = binplot(dbf, ["CR", "NI", "VA"], None, conditions={v.T: (1000, 2500, 40), v.X("CR"): (0, 1, 0.01), v.P: 101325}, return_strategy=True)
    
    # Two-phase regions intended to show up in the Cr-Ni system
    desired_zpf_sets = [{"BCC_B2", "L12_FCC"}, {"BCC_B2", "LIQUID"}, {"L12_FCC", "LIQUID"}]
    desired_node_sets = [{"BCC_B2", "L12_FCC", "LIQUID"}]

    # All two-phase regions and invariants from mapping
    # NOTE: phase regions that start at terminal phases may have duplicates
    mapping_sets = [set(zpf_line.stable_phases_with_multiplicity) for zpf_line in strategy.zpf_lines]
    node_sets = [set(node.stable_phases_with_multiplicity) for node in strategy.node_queue.nodes]
    
    # Make sure that the phase regions from mapping contains all the desired regions
    # NOTE: this will not test for extra phase regions that mapping may produce
    for dzs in desired_zpf_sets:
        assert dzs in mapping_sets

    for dnz in desired_node_sets:
        assert dnz in node_sets

@select_database("crtiv_ghosh.tdb")
def test_ternary_strategy(load_database):
    dbf = load_database()

    ax, strategy = ternplot(dbf, ["CR", "TI", "V", "VA"], None, conds={v.X("CR"): (0, 1, 0.01), v.X("TI"): (0, 1, 0.01), v.T: 923, v.P: 101325}, return_strategy=True)
    
    # Two-phase regions intended to show up in the Cr-Ni system
    desired_zpf_sets = [{"BCC_A2", "LAVES_C15"}, {"BCC_A2", "HCP_A3"}, {"HCP_A3", "LAVES_C15"}]
    desired_node_sets = [{"BCC_A2", "HCP_A3", "LAVES_C15"}]

    # All two- and three-phase regions from mapping
    # NOTE: phase regions that start at terminal phases may have duplicates
    mapping_sets = [set(zpf_line.stable_phases_with_multiplicity) for zpf_line in strategy.zpf_lines]
    node_sets = [set(node.stable_phases_with_multiplicity) for node in strategy.node_queue.nodes]

    # Make sure that the phase regions from mapping contains all the desired regions
    # NOTE: this will not test for extra phase regions that mapping may produce
    for dzs in desired_zpf_sets:
        assert dzs in mapping_sets

    for dnz in desired_node_sets:
        assert dnz in node_sets

@select_database("alcocrni.tdb")
def test_step_strategy(load_database):
    dbf = load_database()

    strategy = StepStrategy(dbf, ["CR", "NI", "VA"], None, conditions={v.T: (1000, 2500, 40), v.X("CR"): 0.8, v.P: 101325})
    strategy.initialize()
    strategy.do_map()

    # Just check that plot_step runs without failing
    plot_step(strategy)

    # Two-phase regions intended to show up in the Cr-Ni system
    desired_zpf_sets = [{"BCC_B2", "L12_FCC"}, {"BCC_B2"}, {"BCC_B2", "LIQUID"}, {"LIQUID"}]
    desired_node_sets = [{"BCC_B2", "L12_FCC"}, {"BCC_B2", "LIQUID"}]

    # All unique phase regions
    mapping_sets = [set(zpf_line.stable_phases_with_multiplicity) for zpf_line in strategy.zpf_lines]
    node_sets = [set(node.stable_phases_with_multiplicity) for node in strategy.node_queue.nodes]
    
    # Make sure that the phase regions from mapping contains all the desired regions
    # NOTE: this will not test for extra phase regions that mapping may produce
    for dzs in desired_zpf_sets:
        assert dzs in mapping_sets

    for dnz in desired_node_sets:
        assert dnz in node_sets

@select_database("crtiv_ghosh.tdb")
def test_isopleth_strategy(load_database):
    dbf = load_database()

    strategy = IsoplethStrategy(dbf, ["CR", "TI", "V", "VA"], None, conditions={v.T: (1073, 2073, 20), v.X("TI"): (0, 0.8, 0.01), v.X("V"): 0.2, v.P: 101325})
    strategy.initialize()
    strategy.do_map()

    # Check that plot_isopleth runs without fail
    plot_isopleth(strategy)

    # Two-phase regions intended to show up in the Cr-Ni system
    desired_zpf_sets = [{"BCC_A2", "LAVES_C15"}, {"BCC_A2", "LIQUID"}]

    # All unique phase regions
    mapping_sets = [set(zpf_line.stable_phases_with_multiplicity) for zpf_line in strategy.zpf_lines]

    # Make sure that the phase regions from mapping contains all the desired regions
    # NOTE: this will not test for extra phase regions that mapping may produce
    for dzs in desired_zpf_sets:
        assert dzs in mapping_sets

@select_database("femns.tdb")
def test_global_min_check_writable_array(load_database):
    """
    The femns database was one that failed during the global min check during mapping
    This was due to the site fractions from calculate being read-only after squeezing
    which created a 'ValueError: buffer source array is read-only' error when updating
    the composition sets
    Fix was to create a new np array for the site fractions, and use the new array
    to update the composition sets

    This test just makes sures that find_global_min_point can run without crashing on the femns.tdb database
    """
    dbf = load_database()
    comps = ['FE', 'MN', 'S', 'VA']
    phases = list(dbf.phases.keys())
    conds = {v.T: 1000, v.P: 101325, v.X('FE'): 0.3, v.X('S'): 0.2, v.N: 1}
    models = instantiate_models(dbf, comps, phases)
    phase_records = PhaseRecordFactory(dbf, comps, {v.N, v.P, v.T}, models)

    point = point_from_equilibrium(dbf, comps, phases, conds)

    sys_info = {
        "dbf": dbf,
        "comps": comps,
        "phases": phases,
        "models": models,
        "phase_records": phase_records,
    }

    find_global_min_point(point, sys_info)

def test_strategy_adjust_composition_limits():
    """
    This tests that the strategy will adjust the condition limits
    to prevent the map from unnecessarily going to compositions summing to > 1

    The adjustment happens during initialization in MapStrategy so we just need to create a strategy
    and check the axis limits
    """
    TDB = """
    ELEMENT /-   ELECTRON_GAS              0 0 0!
    ELEMENT VA   VACUUM                    0 0 0!
    ELEMENT A   VACUUM                    0 0 0!
    ELEMENT B   VACUUM                    0 0 0!
    ELEMENT C   VACUUM                    0 0 0!
    ELEMENT D   VACUUM                    0 0 0!

    PHASE TEST_PH % 1 1 !
    CONSTITUENT TEST_PH :A,B,C,D: !
    """
    dbf = Database(TDB)

    # Stepping in B at A=0.1
    # v.X('B'): (0, 1, 0.01) -> v.X('B'): (0, 0.9, 0.01)
    comps = ['A', 'B', 'C']
    conds = {v.T: 1000, v.P: 101325, v.X('A'): 0.1, v.X('B'): (0, 1, 0.01)}
    strategy = StepStrategy(dbf, comps, None, conds)

    assert np.isclose(strategy.axis_lims[v.X('B')][0], 0)
    assert np.isclose(strategy.axis_lims[v.X('B')][1], 0.9)

    # Stepping in T at A=0.1
    # v.T should not change
    comps = ['A', 'B']
    conds = {v.T: (1, 2, 0.01), v.P: 101325, v.X('A'): 0.1}
    strategy = StepStrategy(dbf, comps, None, conds)

    assert np.isclose(strategy.axis_lims[v.T][0], 1)
    assert np.isclose(strategy.axis_lims[v.T][1], 2)

    # Isopleth in A-B-C-D at A=0.1, D=0.2
    # v.X('B'): (0, 1, 0.01) -> v.X('B'): (0, 0.7, 0.01)
    # v.X('C'): (0, 1, 0.01) -> v.X('C'): (0, 0.7, 0.01)
    comps = ['A', 'B', 'C', 'D']
    conds = {v.T: 1000, v.P: 101325, v.X('A'): 0.1, v.X('D'): 0.2, v.X('B'): (0, 1, 0.01), v.X('C'): (0, 1, 0.01)}
    strategy = IsoplethStrategy(dbf, comps, None, conds)

    assert np.isclose(strategy.axis_lims[v.X('B')][0], 0)
    assert np.isclose(strategy.axis_lims[v.X('B')][1], 0.7)
    assert np.isclose(strategy.axis_lims[v.X('C')][0], 0)
    assert np.isclose(strategy.axis_lims[v.X('C')][1], 0.7)

    # Isopleth in A-B-C-D at A=0.1, D=0.2 and non-zero minimum limits on B and C
    # v.X('B'): (0.15, 1, 0.01) -> v.X('B'): (0, 0.45, 0.01)
    # v.X('C'): (0.25, 1, 0.01) -> v.X('C'): (0, 0.55, 0.01)
    comps = ['A', 'B', 'C', 'D']
    conds = {v.T: 1000, v.P: 101325, v.X('A'): 0.1, v.X('D'): 0.2, v.X('B'): (0.15, 1, 0.01), v.X('C'): (0.25, 1, 0.01)}
    strategy = IsoplethStrategy(dbf, comps, None, conds)
    
    assert np.isclose(strategy.axis_lims[v.X('B')][0], 0.15)
    assert np.isclose(strategy.axis_lims[v.X('B')][1], 0.45)
    assert np.isclose(strategy.axis_lims[v.X('C')][0], 0.25)
    assert np.isclose(strategy.axis_lims[v.X('C')][1], 0.55)