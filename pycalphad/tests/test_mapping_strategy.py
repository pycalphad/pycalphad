import copy
from importlib.resources import files

import numpy as np
import matplotlib.pyplot as plt
import pytest

from pycalphad import binplot, ternplot, Database, Model, variables as v
from pycalphad.tests.fixtures import select_database, load_database
from pycalphad.core.utils import instantiate_models, get_state_variables
from pycalphad.codegen.phase_record_factory import PhaseRecordFactory
from pycalphad.core.composition_set import CompositionSet
from pycalphad.property_framework import as_property
from pycalphad.property_framework.units import Q_, unit_conversion_context, to_display_units

from pycalphad.mapping import StepStrategy, IsoplethStrategy, TielineStrategy, BinaryStrategy, TernaryStrategy, plot_step, plot_isopleth, plot_ternary
from pycalphad.mapping.starting_points import point_from_equilibrium
from pycalphad.mapping.zpf_equilibrium import find_global_min_point
from pycalphad.mapping.primitives import Point, Node, Direction, ZPFLine, ZPFState, _get_phase_specific_variable
from pycalphad.mapping.plotting import get_label

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

    ax, strategy = binplot(dbf, ["CR", "NI", "VA"], ['BCC_A2', 'FCC_A1', 'LIQUID'], conditions={v.T: (1500, 2200, 50), v.X("CR"): (0, 1, 0.05), v.P: 101325}, return_strategy=True)
    #plt.show()

    # Two-phase regions intended to show up in the Cr-Ni system
    desired_zpf_sets = [{"BCC_A2", "FCC_A1"}, {"BCC_A2", "LIQUID"}, {"FCC_A1", "LIQUID"}]
    desired_node_sets = [{"BCC_A2", "FCC_A1", "LIQUID"}]

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

    num_nodes = len(strategy.node_queue.nodes)
    # Attempting to add node in single phase region will not add a new node
    strategy.add_nodes_from_conditions({v.T: 1600, v.P: 101325, v.X('CR'): 0.3})
    new_num_nodes = len(strategy.node_queue.nodes)
    assert new_num_nodes == num_nodes

    # Attempt to add node in two phase region. 2 will be created for positive and negative direction
    strategy.add_nodes_from_conditions({v.T: 1600, v.P: 101325, v.X('CR'): 0.6})
    new_num_nodes = len(strategy.node_queue.nodes)
    assert new_num_nodes == num_nodes + 2

@select_database("crtiv_ghosh.tdb")
def test_ternary_strategy(load_database):
    dbf = load_database()

    ax, strategy = ternplot(dbf, ["CR", "TI", "V", "VA"], ['BCC_A2', 'HCP_A3', 'LAVES_C15'], conds={v.X("V"): (0, 0.2, 0.05), v.X("TI"): (0, 1, 0.05), v.T: 923, v.P: 101325}, return_strategy=True, label_nodes=True)
    #plt.show()
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

    # Attempt to add node in single-phase region - no nodes added
    num_nodes = len(strategy.node_queue.nodes)
    strategy.add_nodes_from_conditions({v.T: 923, v.P: 101325, v.X('CR'): 0.2, v.X('TI'): 0.2})
    new_num_nodes = len(strategy.node_queue.nodes)
    assert new_num_nodes == num_nodes

    # Attempt to add node in two-phase region - two nodes added for pos/neg direction
    strategy.add_nodes_from_conditions({v.T: 923, v.P: 101325, v.X('CR'): 0.4, v.X('TI'): 0.4})
    new_num_nodes = len(strategy.node_queue.nodes)
    assert new_num_nodes == num_nodes + 2

    # Attempt to add node in three-phase region (force adding) - one node is added where directions are determined from the node
    num_nodes = len(strategy.node_queue.nodes)
    strategy.add_nodes_from_conditions({v.T: 923, v.P: 101325, v.X('CR'): 0.129, v.X('TI'): 0.861}, force_add=True)
    new_num_nodes = len(strategy.node_queue.nodes)
    assert new_num_nodes == num_nodes + 1


@select_database("crtiv_ghosh.tdb")
def test_plot_ternary_without_tielines(load_database):
    dbf = load_database()
    comps = ["CR", "TI", "V", "VA"]
    phases = ["LIQUID"]
    conds = {
        v.X("V"): (0, 1, 0.2),
        v.X("TI"): (0, 1, 0.2),
        v.T: 2000,
        v.P: 101325,
    }
    strategy = TielineStrategy(dbf, comps, phases, conds)

    assert strategy.get_tieline_data(v.X("V"), v.X("TI")) == []
    ax = plot_ternary(strategy)
    assert ax.name == "triangular"
    plt.close(ax.figure)


@select_database("alcocrni.tdb")
def test_step_strategy_through_single_phase(load_database):
    dbf = load_database()

    # Step strategy through single phase regions
    strategy = StepStrategy(dbf, ["CR", "NI", "VA"], ["BCC_A2", "FCC_A1", "LIQUID"], conditions={v.T: (1300, 2000, 10), v.X("CR"): 0.8, v.P: 101325})
    strategy.do_map()

    # Just check that plot_step runs without failing
    plot_step(strategy)

    # Two-phase regions intended to show up in the Cr-Ni system
    desired_zpf_sets = [{"BCC_A2", "FCC_A1"}, {"BCC_A2"}, {"BCC_A2", "LIQUID"}, {"LIQUID"}]
    desired_node_sets = [{"BCC_A2", "FCC_A1"}, {"BCC_A2", "LIQUID"}]

    # All unique phase regions
    mapping_sets = [set(zpf_line.stable_phases_with_multiplicity) for zpf_line in strategy.zpf_lines]
    node_sets = [set(node.stable_phases_with_multiplicity) for node in strategy.node_queue.nodes]


    # Make sure that the phase regions from mapping contains all the desired regions
    # NOTE: this will not test for extra phase regions that mapping may produce
    for dzs in desired_zpf_sets:
        assert dzs in mapping_sets

    for dnz in desired_node_sets:
        assert dnz in node_sets

    # Test behavior of data outputs

    # For T vs. CPM, x and y refers to properties of the entire system -> phases = ['SYSTEM']
    data = strategy.get_data(v.T, 'CPM')
    assert len(data.data) == 1 and data.data[0].phase == 'SYSTEM'

    # v.X('CR') has phase wildcard '*' implicitly added, so phases = ['BCC_A2', 'FCC_A1', 'LIQUID']
    data = strategy.get_data(v.T, v.X('CR'))
    assert len(set(data.phases).symmetric_difference({'BCC_A2', 'FCC_A1', 'LIQUID'})) == 0

    # We force y to be global and x is already global, so phases = ['SYSTEM']
    data = strategy.get_data(v.T, v.X('CR'), global_y=True)
    assert len(data.data) == 1 and data.data[0].phase == 'SYSTEM'

    # x is phase specific, so both x and y are global -> phases = ['SYSTEM']
    data = strategy.get_data(v.X('BCC_A2', 'CR'), v.T)
    assert len(data.data) == 1 and data.data[0].phase == 'SYSTEM'

    # We force x to be global -> phases = ['SYSTEM']
    data = strategy.get_data(v.X('CR'), v.T, global_x=True)
    assert len(data.data) == 1 and data.data[0].phase == 'SYSTEM'

@select_database("pbsn.tdb")
def test_step_strategy_through_node(load_database):
    dbf = load_database()

    # Step strategy through single phase regions
    strategy = StepStrategy(dbf, ["PB", "SN", "VA"], None, conditions={v.T: (425, 550, 5), v.X("SN"): 0.5, v.P: 101325})
    strategy.do_map()

    # Just check that plot_step runs without failing
    plot_step(strategy)

    # Two-phase regions intended to show up in the Pb-Sn system
    desired_zpf_sets = [{"BCT_A5", "FCC_A1"}, {"FCC_A1", "LIQUID"}, {"LIQUID"}]
    desired_node_sets = [{"BCT_A5", "FCC_A1", "LIQUID"}, {"FCC_A1", "LIQUID"}]

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
def test_unary_strategy(load_database):
    """
    Tests that strategy works on unary system
    The strategy needs to maintain certain array shapes for site fractions, composition,
    chemical potentials, etc. when working with unaries, since squeezing arrays can remove
    a needed dimension from an array. More details are given in the _find_global_min_cs function
    in pycalphad.mapping.zpf_equilibrium
    """
    dbf = load_database()
    strategy = StepStrategy(dbf, ["CR", "VA"], ["BCC_A2", "LIQUID"], conditions={v.T: (2150, 2250, 10), v.P: 101325})
    strategy.do_map()
    plot_step(strategy, v.T, 'CPM')

@select_database("crtiv_ghosh.tdb")
def test_isopleth_strategy(load_database):
    dbf = load_database()
    X_V = 0.2

    strategy = IsoplethStrategy(dbf, ["CR", "TI", "V", "VA"], ["BCC_A2", "LIQUID"], conditions={v.T: (1500, 2100, 40), v.X("TI"): (0, 0.2, 0.05), v.X("V"): X_V, v.P: 101325})
    strategy.do_map()

    # Check that plot_isopleth runs without fail
    ax = plot_isopleth(strategy)
    #ax.figure.show()

    # Two-phase regions intended to show up in the Cr-Ti-V system
    desired_zpf_sets = [{"BCC_A2", "LIQUID"}]

    # All unique phase regions
    mapping_sets = [set(zpf_line.stable_phases_with_multiplicity) for zpf_line in strategy.zpf_lines]

    # Make sure that the phase regions from mapping contains all the desired regions
    # NOTE: this will not test for extra phase regions that mapping may produce
    for dzs in desired_zpf_sets:
        assert dzs in mapping_sets

    # Check plotting isopleth with different set of units
    # We test that the conversions work by just looking at the axes limits
    xlims_moles = ax.get_xlim()
    ylims_kelvin = ax.get_ylim()
    # Composition partially hard coded based on
    expected_xlims_mass = [v.get_mass_fractions({v.X("TI"): xl, v.X("V"): X_V}, "CR", dbf)[v.W("TI")] for xl in xlims_moles]
    expected_ylims_celsius = [yl - 273.15 for yl in ylims_kelvin]

    ax2 = plot_isopleth(strategy, x=v.W("TI"), y=v.T["degC"])
    #ax2.figure.show()
    np.testing.assert_allclose(ax2.get_xlim(), expected_xlims_mass)
    np.testing.assert_allclose(ax2.get_ylim(), expected_ylims_celsius)

    plt.close(ax.figure)
    plt.close(ax2.figure)

def test_isopleth_strategy_node_exit():
    """
    Creates simulated zpf lines and nodes in A-B-C system to test exit strategy for isopleths
    """
    TDB = """
    ELEMENT /-   ELECTRON_GAS              0 0 0!
    ELEMENT VA   VACUUM                    0 0 0!
    ELEMENT A   VACUUM                    0 0 0!
    ELEMENT B   VACUUM                    0 0 0!
    ELEMENT C   VACUUM                    0 0 0!

    PHASE ALPHA % 1 1 !
    CONSTITUENT ALPHA :A,B,C: !

    PHASE BETA % 1 1 !
    CONSTITUENT BETA :A,B,C: !

    PHASE GAMMA % 1 1 !
    CONSTITUENT GAMMA :A,B,C: !

    PHASE DELTA % 1 1 !
    CONSTITUENT DELTA :A,B,C: !
    """
    dbf = Database(TDB)
    phases = list(dbf.phases.keys())

    strategy = IsoplethStrategy(dbf, ['A', 'B', 'C', 'VA'], phases,
                                conditions={v.T: (500, 1000, 10), v.P: 101325, v.X('A'): 0.2, v.X('B'): (0, 0.8, 0.01)},
                                initialize=False)

    phase_comps = {
        'ALPHA': [0.9, 0.05, 0.05],
        'BETA': [0.05, 0.9, 0.05],
        'GAMMA': [0.05, 0.05, 0.9],
        'DELTA': [0.3, 0.3, 0.4],
    }
    comp_sets = []
    for p in phases:
        cs = CompositionSet(strategy.phase_records[p])
        cs.update(np.array(phase_comps[p], dtype=np.float64), 0.25, np.array([1, 101325, 700], dtype=np.float64))
        comp_sets.append(cs)
    comp_sets[0].fixed = True
    comp_sets[1].fixed = True
    comp_sets[2].fixed = False
    comp_sets[3].fixed = False


    # Invariant node with 8 total exits
    conds = {v.T: 700, v.P: 101325, v.N: 1, v.X('A'): 0.2, v.X('B'): 0.4}
    node = Node(conds, [0, 0, 0], [comp_sets[0], comp_sets[1]], [comp_sets[2], comp_sets[3]], None)
    exits, exit_dirs = strategy._find_exits_from_node(node)
    assert len(exits) == 8

    # Drawing a line along v.X('A'): 0.2, v.X('B'): (0, 0.8, 0.01), these are the two-phase lines that it will intersect
    desired_free_phases = [{'ALPHA', 'BETA'}, {'ALPHA', 'GAMMA'}, {'BETA', 'DELTA'}, {'GAMMA', 'DELTA'}]
    for point in exits:
        assert set(point.free_phases) in desired_free_phases

    # Invariant node with 6 total exits
    strategy = IsoplethStrategy(dbf, ['A', 'B', 'C', 'VA'], phases,
                                conditions={v.T: (500, 1000, 10), v.P: 101325, v.X('A'): 0.5, v.X('B'): (0, 0.5, 0.01)},
                                initialize=False)
    conds = {v.T: 700, v.P: 101325, v.N: 1, v.X('A'): 0.5, v.X('B'): 0.2}
    node = Node(conds, [0, 0, 0], [comp_sets[0], comp_sets[1]], [comp_sets[2], comp_sets[3]], None)
    exits, exit_dirs = strategy._find_exits_from_node(node)
    # Drawing a line along v.X('A'): 0.5, v.X('B'): (0, 0.5, 0.01), these are the two-phase lines that it will intersect
    desired_free_phases = [{'ALPHA', 'BETA'}, {'ALPHA', 'GAMMA'}, {'ALPHA', 'DELTA'}]
    for point in exits:
        assert set(point.free_phases) in desired_free_phases

    # Non-invariant node with 3 exits
    # ZPF line with fixed ALPHA and free GAMMA
    parent = Point(conds, [0, 0, 0], [comp_sets[0]], [comp_sets[2]])
    # Node with fixed ALPHA, BETA and free GAMMA
    node = Node(conds, [0, 0, 0], [comp_sets[0], comp_sets[1]], [comp_sets[2]], parent)
    exits, exit_dirs = strategy._find_exits_from_node(node)
    assert len(exits) == 3
    desired_exits = [({'BETA'}, {'ALPHA', 'GAMMA'}), ({'ALPHA'}, {'BETA', 'GAMMA'}), ({'BETA'}, {'GAMMA'})]
    for point in exits:
        has_exit_type = False
        for de in desired_exits:
            if set(point.fixed_phases) == de[0] and set(point.free_phases) == de[1]:
                has_exit_type = True
        assert has_exit_type


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
    strategy = StepStrategy(dbf, comps, None, conds, initialize=False)

    assert np.isclose(strategy.axis_lims[v.X('B')][0], 0)
    assert np.isclose(strategy.axis_lims[v.X('B')][1], 0.9)

    # Stepping in T at A=0.1
    # v.T should not change
    comps = ['A', 'B']
    conds = {v.T: (1, 2, 0.01), v.P: 101325, v.X('A'): 0.1}
    strategy = StepStrategy(dbf, comps, None, conds, initialize=False)

    assert np.isclose(strategy.axis_lims[v.T][0], 1)
    assert np.isclose(strategy.axis_lims[v.T][1], 2)

    # Isopleth in A-B-C-D at A=0.1, D=0.2
    # v.X('B'): (0, 1, 0.01) -> v.X('B'): (0, 0.7, 0.01)
    # v.X('C'): (0, 1, 0.01) -> v.X('C'): (0, 0.7, 0.01)
    comps = ['A', 'B', 'C', 'D']
    conds = {v.T: 1000, v.P: 101325, v.X('A'): 0.1, v.X('D'): 0.2, v.X('B'): (0, 1, 0.01), v.X('C'): (0, 1, 0.01)}
    strategy = IsoplethStrategy(dbf, comps, None, conds, initialize=False)

    assert np.isclose(strategy.axis_lims[v.X('B')][0], 0)
    assert np.isclose(strategy.axis_lims[v.X('B')][1], 0.7)
    assert np.isclose(strategy.axis_lims[v.X('C')][0], 0)
    assert np.isclose(strategy.axis_lims[v.X('C')][1], 0.7)

    # Isopleth in A-B-C-D at A=0.1, D=0.2 and non-zero minimum limits on B and C
    # v.X('B'): (0.15, 1, 0.01) -> v.X('B'): (0, 0.45, 0.01)
    # v.X('C'): (0.25, 1, 0.01) -> v.X('C'): (0, 0.55, 0.01)
    comps = ['A', 'B', 'C', 'D']
    conds = {v.T: 1000, v.P: 101325, v.X('A'): 0.1, v.X('D'): 0.2, v.X('B'): (0.15, 1, 0.01), v.X('C'): (0.25, 1, 0.01)}
    strategy = IsoplethStrategy(dbf, comps, None, conds, initialize=False)

    assert np.isclose(strategy.axis_lims[v.X('B')][0], 0.15)
    assert np.isclose(strategy.axis_lims[v.X('B')][1], 0.45)
    assert np.isclose(strategy.axis_lims[v.X('C')][0], 0.25)
    assert np.isclose(strategy.axis_lims[v.X('C')][1], 0.55)

@select_database("CrFeNb_Jacob2016.tdb")
def test_ternary_strategy_process_metastable_node(load_database):
    """
    Tests how TielineStrategy deals with nodes that are metastable

    This is done by purposely creating a known metastable node, which
    the TielineStrategy should be able to detect whether a node is metastable
    and perform the following:
        a) if node is metastable, do not add to node queue and end the zpf line
        b) if node is stable, add to node queue
    """
    # Create system
    dbf = load_database()
    comps = ['CR', 'FE', 'NB', 'VA']
    phases = list(dbf.phases.keys())
    map_conds = {v.T: 1323, v.P: 101325, v.N: 1, v.X('CR'): (0, 1, 0.01), v.X('FE'): (0, 1, 0.01)}

    strategy = TielineStrategy(dbf, comps, phases, map_conds, initialize=False)

    # Conditions where BCC_A2 and LAVES_C14 is stable
    # Add this point as a starting zpf line in the strategy
    #    Number of zpf lines = 1
    #    Number of nodes = 0
    eq_conds = {v.T: 1323, v.P: 101325, v.N: 1, v.X('CR'): 0.2, v.X('FE'): 0.2}
    point = point_from_equilibrium(strategy.dbf, strategy.components, strategy.phases, eq_conds, models=strategy.models, phase_record_factory=strategy.phase_records)
    strategy.zpf_lines.append(ZPFLine([], point.stable_phases))
    strategy.zpf_lines[0].axis_var = v.X('FE')
    strategy.zpf_lines[0].axis_direction = Direction.POSITIVE

    # Create node of BCC_A2, MU and LAVES_C15
    # At eq_conds, all three phases will be stable if LAVES_C14 is suspended
    metastable_phases = ['BCC_A2', 'MU_PHASE', 'LAVES_C15']
    models = instantiate_models(dbf, comps, metastable_phases)
    state_vars = get_state_variables(models, eq_conds)
    phase_record_factory = PhaseRecordFactory(dbf, comps, state_vars, models)
    metastable_point = point_from_equilibrium(dbf, comps, metastable_phases, eq_conds, models=models, phase_record_factory=phase_record_factory)
    metastable_node = Node(metastable_point.global_conditions, metastable_point.chemical_potentials, [], metastable_point.stable_composition_sets, None)

    # In _process_new_node, this will fail the global min check and end the zpf line
    num_nodes = len(strategy.node_queue.nodes)
    strategy._process_new_node(strategy.zpf_lines[0], metastable_node)
    # Test that zpf line was kept, but ended
    assert len(strategy.zpf_lines) == 1
    assert strategy.zpf_lines[0].status == ZPFState.FAILED
    assert len(strategy.node_queue.nodes) == num_nodes

    # _process_new_node with correct/stable node
    # This will pass the _check_full_global_equilibrium test and the node will be added
    # to the node queue
    eq_conds = {v.T: 1323, v.P: 101325, v.N: 1, v.X('CR'): 0.1, v.X('FE'): 0.35}
    stable_point = point_from_equilibrium(strategy.dbf, strategy.components, strategy.phases, eq_conds, models=strategy.models, phase_record_factory=strategy.phase_records)
    stable_node = Node(stable_point.global_conditions, stable_point.chemical_potentials, [], stable_point.stable_composition_sets, None)

    strategy.zpf_lines.append(ZPFLine([], point.stable_phases))
    strategy.zpf_lines[-1].axis_var = v.X('FE')
    strategy.zpf_lines[-1].axis_direction = Direction.POSITIVE

    num_nodes = len(strategy.node_queue.nodes)
    num_zpf_lines = len(strategy.zpf_lines)
    strategy._process_new_node(strategy.zpf_lines[-1], stable_node)
    assert len(strategy.zpf_lines) == num_zpf_lines
    assert len(strategy.node_queue.nodes) == 1+num_nodes
    assert strategy.node_queue.nodes[-1] == stable_node

def test_plot_labels():
    assert get_label(v.NP('*')) == 'Phase Fraction'
    assert get_label(v.NP('BCC_A2')) == 'Phase Fraction (BCC_A2)'

    assert get_label(v.X('CR')) == 'X(Cr)'
    assert get_label(v.X('BCC_A2', 'CR')) == 'X(BCC_A2, Cr)'

    assert get_label(v.W('CR')) == 'W(Cr)'
    assert get_label(v.W('BCC_A2', 'CR')) == 'W(BCC_A2, Cr)'

    assert get_label(v.MU('CR')) == 'MU(Cr) (J / mol)'
    # The abbreviated notation in pint converts J/mol/K to J/K/mol. This
    # seems to be some internal canonical ordering
    # While this is still correct, I do not like it
    # TODO: check if there is a way to override the default canonical
    # ordering in pint so that mol comes before K
    assert get_label('CPM') == 'Heat Capacity (J / K / mol)'
    assert get_label(v.T) == 'Temperature (K)'
    assert get_label(v.P) == 'Pressure (Pa)'

    # A string argument that doesn't have built in units will just return the string
    assert get_label('Custom Prop') == 'Custom Prop'

@select_database("crtiv_ghosh.tdb")
def test_primitive_representation(load_database):
    """
    Tests that str and repr of Point and Node show the necessary information to describe each object
    """
    dbf = load_database()
    strategy = StepStrategy(dbf, ["CR", "VA"], ["BCC_A2", "LIQUID"], conditions={v.T: (2150, 2250, 10), v.P: 101325})
    strategy.do_map()

    node_str_keywords = ['Fixed CS', 'Free CS', 'Conditions', 'Chem_pot', 'Axis']
    node_repr_keywords = ['Node', 'global_conditions', 'chemical_potentials',
                          '_fixed_composition_sets', '_free_composition_sets', 'parent',
                          'axis_var', 'axis_direction', 'exit_hint'
                          ]
    point_str_keywords = ['Fixed CS', 'Free CS', 'Conditions', 'Chem_pot']
    point_repr_keywords = ['Point', 'global_conditions', 'chemical_potentials',
                           '_fixed_composition_sets', '_free_composition_sets'
                           ]
    zpf_line_str_keywords = ['points', 'Fixed phases', 'Free phases', 'Start', 'End']
    zpf_line_repr_keywords = [
        'points',
        'Point',
        'Node',
        'status',
        'axis_var',
        'axis_direction',
        'current_delta'
    ]

    # First point in the first zpf line should be a node
    zpf_line = strategy.zpf_lines[0]
    zpf_line_str = str(zpf_line)
    zpf_line_repr = repr(zpf_line)
    print(zpf_line_str)
    print(zpf_line_repr)

    for keyword in zpf_line_str_keywords:
        assert keyword in zpf_line_str
    for keyword in zpf_line_repr_keywords:
        assert keyword in zpf_line_repr


    assert isinstance(zpf_line.points[0], Node)
    node_str = str(zpf_line.points[0])
    node_repr = repr(zpf_line.points[0])

    for keyword in node_str_keywords:
        assert keyword in node_str
    for keyword in node_repr_keywords:
        assert keyword in node_repr

    assert isinstance(zpf_line.points[1], Point)
    point_str = str(zpf_line.points[1])
    point_repr = repr(zpf_line.points[1])

    for keyword in point_str_keywords:
        assert keyword in point_str
    for keyword in point_repr_keywords:
        assert keyword in point_repr

@select_database("Al-Cu-Y.tdb")
def test_issue_638_degenerate_cs_ternary(load_database):
    """
    Checks that a node is not falsely flagged as not being global min
    This can happen if the initial global min check detects a new composition
    set that is the same phase but slightly different DOF to where it is
    below the tolerance required to be detected as a new global min. The fix
    performs an equilibrium between the new CS and the CS in the node that
    has the same phase name to check whether they are the same or if there
    is a miscibility gap

    This is found in issue 638 on the Al-Cu-Y system at 1700K
    """
    temperature = 1700

    dbf = load_database()
    comps = ['AL', 'CU', 'Y', 'VA']
    phases = list(dbf.phases.keys())
    conds = {v.T: temperature, v.P:101325, v.X('AL'): (0,1,0.02), v.X('Y'): (0,1,0.02)}
    strat = TielineStrategy(dbf, comps, phases, conds)

    # this starting point should be in ['LIQUID', 'ALCU5Y']
    # first two nodes should be ['LIQUID', 'ALCU5Y', 'AL7CU2Y3'] and ['LIQUID', 'ALCU5Y', 'ALCUY']
    strat.add_nodes_from_conditions({v.T: temperature, v.P: 101325, v.X('AL'): 0.4, v.X('Y'): 0.1})

    initial_nodes = len(strat.node_queue.nodes)
    # stop until there's 4 nodes or if mapping finished
    while strat.node_queue._current_node_index < (initial_nodes+1):
        if strat.iterate():
            break

    nodes = [set(n.stable_phases) for n in strat.node_queue.nodes]
    assert {'LIQUID', 'ALCU5Y'} in nodes
    assert {'LIQUID', 'ALCU5Y', 'AL7CU2Y3'} in nodes
    assert {'LIQUID', 'ALCU5Y', 'ALCUY'} in nodes

@select_database("AuSn-13Don.tdb")
def test_issue_638_degenerate_cs_binary(load_database):
    """
    Same as test_issue638_degenerate_cs_ternary but for a binary system
    This tests on the AuSn-13Don database which has issues with removing
    the HCP_A3-LIQUID tielines due to being falsly flagged as not global min
    """
    dbf = load_database()
    comps = ['AU', 'SN', 'VA']
    phases = list(dbf.phases.keys())
    conds = {v.T: (400, 1300, 20), v.P:101325, v.X('SN'): (0,1,0.02)}
    strat = TielineStrategy(dbf, comps, phases, conds)

    strat.add_nodes_from_conditions({v.T: 650, v.P: 101325, v.X('SN'): 0.2})
    initial_nodes = len(strat.node_queue.nodes)
    # stop until there's 4 nodes or if mapping finished
    while strat.node_queue._current_node_index < (initial_nodes+1):
        if strat.iterate():
            break

    nodes = [set(n.stable_phases) for n in strat.node_queue.nodes]
    assert {'LIQUID', 'HCP_A3', 'AUSN_B81'} in nodes

def test_mapping_runs_just_in_time_on_data_retrieval():
    """
    Data retrieval maps just in time, and again after starting points are added
    """
    # Miscibility gap in a single phase (critical temperature ~1200 K), so little code is compiled
    TDB = """
    ELEMENT A BLANK 0 0 0 !
    ELEMENT B BLANK 0 0 0 !
    TYPE_DEFINITION % SEQ * !
    PHASE LIQUID % 1 1 !
    CONSTITUENT LIQUID :A,B: !
    PARAMETER G(LIQUID,A,B;0) 1 20000; 10000 N !
    """
    dbf = Database(TDB)
    # Each starting point maps one short ZPF line to a temperature limit
    conds = {v.P: 101325, v.N: 1, v.T: (895, 905, 10), v.X("B"): (0, 1, 0.05)}
    strategy = TielineStrategy(dbf, ["A", "B"], ["LIQUID"], conds)
    # LIQUID+LIQUID#2. Having a starting point skips the automatic starting point search
    assert strategy.add_nodes_from_conditions({v.T: 900, v.P: 101325, v.X("B"): 0.3}, Direction.POSITIVE)

    # No do_map() before the first retrieval
    tieline_data = strategy.get_tieline_data(v.X("B"), v.T)
    assert len(tieline_data) == 1
    assert strategy._mapping_complete

    strategy.get_tieline_data(v.X("B"), v.T)
    assert len(strategy.zpf_lines) == 1

    added = strategy.add_nodes_from_conditions({v.T: 900, v.P: 101325, v.X("B"): 0.3}, Direction.NEGATIVE)
    assert added
    assert not strategy._mapping_complete
    tieline_data = strategy.get_tieline_data(v.X("B"), v.T)
    assert strategy._mapping_complete
    assert len(tieline_data) == 2

def test_mapping_stopped_by_max_iter_finishes_on_data_retrieval():
    """
    Data retrieval finishes a do_map stopped by max_iter
    """
    # Miscibility gap in a single phase (critical temperature ~1200 K), so little code is compiled
    TDB = """
    ELEMENT A BLANK 0 0 0 !
    ELEMENT B BLANK 0 0 0 !
    TYPE_DEFINITION % SEQ * !
    PHASE LIQUID % 1 1 !
    CONSTITUENT LIQUID :A,B: !
    PARAMETER G(LIQUID,A,B;0) 1 20000; 10000 N !
    """
    dbf = Database(TDB)
    # The starting point maps one short ZPF line to a temperature limit
    conds = {v.P: 101325, v.N: 1, v.T: (895, 905, 10), v.X("B"): (0, 1, 0.05)}
    strategy = TielineStrategy(dbf, ["A", "B"], ["LIQUID"], conds)
    # LIQUID+LIQUID#2
    assert strategy.add_nodes_from_conditions({v.T: 900, v.P: 101325, v.X("B"): 0.3}, Direction.POSITIVE)

    # The first iteration only finds the exit of the starting point
    strategy.do_map(max_iter=1)
    assert not strategy._mapping_complete
    assert len(strategy.zpf_lines) == 0

    tieline_data = strategy.get_tieline_data(v.X("B"), v.T)
    assert strategy._mapping_complete
    assert len(tieline_data) == 1

def test_data_retrieval_leaves_shared_axis_variable_unchanged():
    """
    Phase-local data retrieval does not make the cached axis variable, shared by all strategies, a phase wildcard
    """
    dbf = Database("""
    ELEMENT A   VACUUM                    0 0 0!
    ELEMENT B   VACUUM                    0 0 0!

    PHASE ALPHA % 1 1 !
    CONSTITUENT ALPHA :A,B: !
    """)
    tieline = TielineStrategy(dbf, ["A", "B"], ["ALPHA"], {v.P: 101325, v.T: (500, 1000, 10), v.X("B"): (0, 1, 0.05)})
    step = StepStrategy(dbf, ["A", "B"], ["ALPHA"], {v.P: 101325, v.T: 700, v.X("B"): (0, 1, 0.05)})
    # Conditions are deepcopied, which gives every strategy the same cached axis variable
    axis_var = step.axis_vars[0]
    assert tieline.axis_vars[1] is axis_var

    # A finished zpf line stands in for the mapping that data retrieval would otherwise run first
    cs = CompositionSet(step.phase_records["ALPHA"])
    cs.update(np.array([0.7, 0.3]), 1.0, np.array([1, 101325, 700], dtype=np.float64))
    point = Point({v.N: 1, v.P: 101325, v.T: 700, v.X("B"): 0.3}, np.zeros(2), [], [cs])
    for strategy in [tieline, step]:
        strategy.zpf_lines.append(ZPFLine([], ["ALPHA"], [point], ZPFState.REACHED_LIMIT))

    for get_data in [tieline.get_tieline_data, tieline.get_invariant_data, step.get_data]:
        for x, y in [(v.X("B"), v.T), (v.T, v.X("B"))]:
            # X(B) is not global, so each method makes a phase wildcard X(*,B) to get data for every phase
            get_data(x, y)
            assert axis_var.phase_name is None, f"{get_data.__name__}({x}, {y})"
            # Mapping deepcopies its conditions, which used to give a stray X(*,B) condition
            assert set(copy.deepcopy(step.conditions)) == {v.N, v.P, v.T, v.X("B")}, f"{get_data.__name__}({x}, {y})"


@select_database("AlTe-19Shi.tdb")
def test_degenerate_tieline_rejected_above_congruent_melting(load_database):
    """
    The AL2TE3_BETA+LIQUID tie-line is degenerate just above the congruent melting point (~1138.7 K)
    """
    from pycalphad.mapping.zpf_equilibrium import update_equilibrium_with_new_conditions
    from pycalphad.mapping.zpf_checks import _degenerate_zero_width_tieline
    import pycalphad.mapping.utils as map_utils

    dbf = load_database()
    comps = ["AL", "TE", "VA"]
    # Only the phases of the tie-line; the congruent melting of AL2TE3_BETA does not depend on the others
    phases = ["AL2TE3_BETA", "LIQUID"]

    pt = point_from_equilibrium(dbf, comps, phases, {v.P: 101325, v.N: 1, v.T: 1100, v.X("TE"): 0.61})
    assert set(pt.stable_phases) == {"AL2TE3_BETA", "LIQUID"}
    beta = [cs for cs in pt.stable_composition_sets if cs.phase_record.phase_name == "AL2TE3_BETA"][0]
    liquid = [cs for cs in pt.stable_composition_sets if cs.phase_record.phase_name == "LIQUID"][0]
    zpf_point = map_utils._generate_point_with_fixed_cs(pt, beta, liquid)

    conds = copy.deepcopy(zpf_point.global_conditions)
    conds[v.T] = 1135
    below_results = update_equilibrium_with_new_conditions(zpf_point, conds, v.X("TE"))
    assert below_results is not None
    assert not _degenerate_zero_width_tieline(below_results[0].stable_composition_sets)

    conds = copy.deepcopy(below_results[0].global_conditions)
    conds[v.T] = 1140
    above_results = update_equilibrium_with_new_conditions(below_results[0], conds, v.X("TE"))
    assert above_results is not None
    assert _degenerate_zero_width_tieline(above_results[0].stable_composition_sets)

def test_unary_pt_mapping_not_flagged_as_degenerate():
    """
    A univariant line in a unary P-T diagram is not ended as a degenerate zero-width tie-line
    """
    dbf = Database("""
    ELEMENT A BLANK 0 0 0 !
    TYPE_DEFINITION % SEQ * !
    PHASE ALPHA % 1 1 !
    CONSTITUENT ALPHA :A: !
    PHASE BETA % 1 1 !
    CONSTITUENT BETA :A: !
    PARAMETER G(ALPHA,A;0) 1 0; 6000 N !
    PARAMETER G(BETA,A;0) 1 1000-T+1E-7*P; 6000 N !
    """)
    strategy = TielineStrategy(dbf, ["A"], ["ALPHA", "BETA"], {v.N: 1, v.T: (900, 1300, 50), v.P: (0, 2e9, 2e8)})

    # ALPHA+BETA coexist along T = 1000 K + 1e-7 K/Pa * P, where both phases trivially have the same composition
    conds = {v.N: 1.0, v.T: 1100.0, v.P: 1e9}
    alpha, beta = CompositionSet(strategy.phase_records["ALPHA"]), CompositionSet(strategy.phase_records["BETA"])
    state_variables = np.array([conds[sv] for sv in strategy.phase_records["ALPHA"].state_variables])
    alpha.update(np.array([1.0]), 0.0, state_variables)
    alpha.fixed = True
    beta.update(np.array([1.0]), 1.0, state_variables)
    zpf_line = ZPFLine(["ALPHA"], ["BETA"], [Point(conds, np.zeros(1), [alpha], [beta])],
                       axis_var=v.T, axis_direction=Direction.POSITIVE, current_delta=strategy.axis_delta[v.T])

    # Stepping in T (freeing P) stays on the univariant line, which must not be ended
    strategy._attempt_to_add_point(zpf_line, strategy._step_zpf_line(zpf_line))
    assert zpf_line.status == ZPFState.NOT_FINISHED
    assert len(zpf_line.points) == 2
    new_conds = zpf_line.points[-1].global_conditions
    np.testing.assert_allclose([new_conds[v.T], new_conds[v.P]], [1150, 1.5e9], rtol=1e-6)

@select_database("cumg.tdb")
def test_no_degenerate_edge_pinned_zpf_lines(load_database):
    """
    A zpf line stepping past the melting point of Mg onto the pure Mg edge is ended there
    """
    import pycalphad.mapping.utils as map_utils

    dbf = load_database()
    comps, phases = ["CU", "MG", "VA"], ["HCP_A3", "LIQUID"]
    conds = {v.P: 101325, v.N: 1, v.T: (850, 1100, 20), v.X("MG"): (0.9, 1, 0.05)}
    strategy = TielineStrategy(dbf, comps, phases, conds)

    # HCP_A3 (pure Mg) at zero amount and LIQUID on the Mg-rich liquidus, below the melting point of Mg (923 K)
    pt = point_from_equilibrium(dbf, comps, phases, {v.P: 101325, v.N: 1, v.T: 900, v.X("MG"): 0.99},
                                models=strategy.models, phase_record_factory=strategy.phase_records)
    assert set(pt.stable_phases) == {"HCP_A3", "LIQUID"}
    hcp = [cs for cs in pt.stable_composition_sets if cs.phase_record.phase_name == "HCP_A3"][0]
    liquid = [cs for cs in pt.stable_composition_sets if cs.phase_record.phase_name == "LIQUID"][0]
    zpf_line = ZPFLine(["HCP_A3"], ["LIQUID"], [map_utils._generate_point_with_fixed_cs(pt, hcp, liquid)],
                       axis_var=v.T, axis_direction=Direction.POSITIVE, current_delta=strategy.axis_delta[v.T])

    def _tieline_width(point):
        return np.amax(np.ptp([np.asarray(cs.X) for cs in point.stable_composition_sets], axis=0))

    # 920 K: still a real boundary with a finite tie-line, so the zpf line continues
    strategy._attempt_to_add_point(zpf_line, strategy._step_zpf_line(zpf_line))
    assert zpf_line.status == ZPFState.NOT_FINISHED
    assert len(zpf_line.points) == 2
    assert _tieline_width(zpf_line.points[-1]) > 1e-3

    # 940 K: the solver converges to a zero-width tie-line pinned at x(MG)=1, which must end the zpf line
    strategy._attempt_to_add_point(zpf_line, strategy._step_zpf_line(zpf_line))
    assert len(zpf_line.points) == 3
    assert _tieline_width(zpf_line.points[-1]) < 1e-7
    assert zpf_line.status == ZPFState.REACHED_LIMIT

def test_circular_loop_check_normalizes_axes():
    """
    A liquidus back at its starting temperature after a congruent minimum (e.g. Ba-Ca) is not ended as a loop
    """
    from pycalphad.mapping.zpf_checks import check_circular_loop

    dbf = Database("""
    ELEMENT A BLANK 0 0 0 !
    ELEMENT B BLANK 0 0 0 !
    PHASE LIQUID % 1 1 !
    CONSTITUENT LIQUID :A,B: !
    """)
    prf = PhaseRecordFactory(dbf, ["A", "B"], {v.N, v.P, v.T}, instantiate_models(dbf, ["A", "B"], ["LIQUID"]))

    def _point(T, x):
        conds = {v.N: 1.0, v.P: 101325.0, v.T: T, v.X("B"): x}
        compset = CompositionSet(prf["LIQUID"])
        compset.update(np.array([1 - x, x]), 1.0, np.array([conds[sv] for sv in prf["LIQUID"].state_variables]))
        return Point(conds, np.zeros(2), [], [compset])

    # Axis data and normalization (by the axis steps) as a T-X map strategy passes them to the check
    axis_delta = {v.T: 20, v.X("B"): 0.05}
    axis_data = {"axis_vars": [v.T, v.X("B")], "axis_delta": axis_delta, "axis_lims": {v.T: (700, 1250), v.X("B"): (0, 1)}}

    # Seeded at the A melting point, down through the minimum and back up to 1 K above the starting temperature.
    # In raw units, the 16 K from the previous point swamps the 0.79 mole fraction from the first point
    zpf_line = ZPFLine([], ["LIQUID"])
    zpf_line.points = [_point(1000, 0.01), _point(894, 0.45), _point(985, 0.78)]
    check_circular_loop(zpf_line, (_point(1001, 0.80), []), axis_data, normalize_factor=axis_delta)
    assert zpf_line.status == ZPFState.NOT_FINISHED

    # Coming back near the first point in both temperature and composition is still a loop
    zpf_line = ZPFLine([], ["LIQUID"])
    zpf_line.points = [_point(1000, 0.30), _point(1060, 0.45), _point(1030, 0.40)]
    check_circular_loop(zpf_line, (_point(1005, 0.32), []), axis_data, normalize_factor=axis_delta)
    assert zpf_line.status == ZPFState.REACHED_LIMIT

def test_automatic_starting_points_keep_forced_starting_points():
    """
    Parentless recovery nodes from the edge step maps are kept, e.g. for the narrow La BCC window (1134-1194 K)

    The coarse step map along the x(LA)~1 edge of the Ga-La map (dT = 185 K) straddles the window and only
    records the LIQUID+BCC_A2 field by the recovery point it force-adds (without a parent) at 1153.8 K
    """
    from pycalphad.mapping.primitives import ExitHint, MIN_COMPOSITION
    # Starting points are built from the composition sets of the step map nodes, so no phase energies are needed
    dbf = Database("""
    ELEMENT GA BLANK 0 0 0 !
    ELEMENT LA BLANK 0 0 0 !
    PHASE LIQUID % 1 1 !
    CONSTITUENT LIQUID :GA,LA: !
    PHASE BCC_A2 % 1 1 !
    CONSTITUENT BCC_A2 :GA,LA: !
    """)
    phases = ["LIQUID", "BCC_A2"]
    conds = {v.P: 101325, v.N: 1, v.T: (300, 4000, 20), v.X("LA"): (0.75, 1, 0.05)}
    strategy = TielineStrategy(dbf, ["GA", "LA"], phases, conds)
    step_conds = {v.P: 101325, v.N: 1, v.T: (300, 4000, 185), v.X("LA"): 1 - MIN_COMPOSITION}
    step = StepStrategy(dbf, ["GA", "LA"], phases, step_conds, models=strategy.models, phase_record_factory=strategy.phase_records)

    # Recovery point at the Ga-La LIQUID+BCC_A2 equilibrium, added like StepStrategy.add_nodes_from_conditions does
    node_conds = {v.P: 101325, v.N: 1, v.T: 1153.8, v.X("LA"): 1 - MIN_COMPOSITION}
    state_variables = np.array([node_conds[sv] for sv in strategy.phase_records["LIQUID"].state_variables], dtype=np.float64)
    liquid, bcc = CompositionSet(strategy.phase_records["LIQUID"]), CompositionSet(strategy.phase_records["BCC_A2"])
    liquid.update(np.array([0.0212, 0.9788]), 4.6e-5, state_variables)
    bcc.update(np.array([2.05e-8, 1 - 2.05e-8]), 1 - 4.6e-5, state_variables)
    point = Point(node_conds, np.array([-244788.1, -85941.1]), [], [liquid, bcc])
    step.node_queue.add_node(step._create_node_from_point(point, None, None, Direction.NEGATIVE, ExitHint.POINT_IS_EXIT), True)

    strategy.add_starting_points_from_step(step)
    starting_points = [node for node in strategy.node_queue.nodes if sorted(node.stable_phases) == ["BCC_A2", "LIQUID"]]
    assert sorted(node.axis_direction.value for node in starting_points) == [-1, 1], (
        "The LIQUID+BCC_A2 recovery point inside the La BCC window was not added as a starting point in both directions"
    )

def test_automatic_starting_points_for_every_merged_field():
    """
    Fields merged into one same-name step zpf line each get a starting point (Au-rich fcc+Au3Cu at 300 K)

    The fcc-based phases of Au-Cu (fcc, Au3Cu, AuCu, ...) share one phase name, so the step map along the
    T = 300 K edge slides the same composition sets from the Au3Cu+AuCu field into the fcc+Au3Cu field
    without a phase-change node, and both fields end up in one step zpf line
    """
    # Starting points are built from the composition sets of the step zpf line, so no phase energies are needed
    dbf = Database("""
    ELEMENT AU BLANK 0 0 0 !
    ELEMENT CU BLANK 0 0 0 !
    PHASE FCC_A1 % 1 1 !
    CONSTITUENT FCC_A1 :AU,CU: !
    """)
    conds = {v.P: 101325, v.N: 1, v.T: (300, 700, 20), v.X("CU"): (0, 1, 0.05)}
    strategy = TielineStrategy(dbf, ["AU", "CU"], ["FCC_A1"], conds)
    step_conds = {v.P: 101325, v.N: 1, v.T: 300, v.X("CU"): (0, 1, 0.05)}
    step = StepStrategy(dbf, ["AU", "CU"], ["FCC_A1"], step_conds, models=strategy.models, phase_record_factory=strategy.phase_records)

    # The Au-Cu step zpf line at 300 K, as each field's tie-line (x(CU) of the two composition sets) and the
    # x(CU) of its points: first the Au3Cu+AuCu field, then the fcc+Au3Cu field
    fields = [
        ((0.2803, 0.4616), [0.4616, 0.4566, 0.4466, 0.4266, 0.3866, 0.3366, 0.2866]),
        ((0.2374, 0.0780), [0.2366, 0.1866, 0.1366, 0.0866, 0.0780]),
    ]
    state_variables = np.array([1, 101325, 300], dtype=np.float64)
    zpf_line = ZPFLine([], ["FCC_A1", "FCC_A1"])
    for tieline, x_cus in fields:
        for x_cu in x_cus:
            comp_sets = [CompositionSet(strategy.phase_records["FCC_A1"]) for _ in tieline]
            phase_frac = (x_cu - tieline[0]) / (tieline[1] - tieline[0])
            comp_sets[0].update(np.array([1 - tieline[0], tieline[0]]), 1 - phase_frac, state_variables)
            comp_sets[1].update(np.array([1 - tieline[1], tieline[1]]), phase_frac, state_variables)
            zpf_line.append(Point({v.P: 101325, v.N: 1, v.T: 300, v.X("CU"): x_cu}, np.zeros(2), [], comp_sets))
    step.zpf_lines.append(zpf_line)

    strategy.add_starting_points_from_step(step)
    # One starting point in each direction per field, labeled by the Au-rich end of its tie-line
    starting_points = sorted((round(min(cs.X[1] for cs in node.stable_composition_sets), 3), node.axis_direction.value)
                             for node in strategy.node_queue.nodes)
    assert starting_points == [(0.078, -1), (0.078, 1), (0.28, -1), (0.28, 1)], (
        "Each field merged into the step zpf line should get one starting point in each direction, "
        "including the Au-rich fcc+Au3Cu field (x(CU) = 0.078-0.237)"
    )

def test_starting_point_dedup_distinguishes_twin_fields():
    """
    A starting point in the Cr-rich BCC_A2+C15_LAVES field of Cr-Ta is not already mapped by the line of the
    Ta-rich BCC_A2+C15_LAVES field on the other side of the C15_LAVES compound
    """
    dbf = Database("""
    ELEMENT CR BLANK 0 0 0 !
    ELEMENT TA BLANK 0 0 0 !
    PHASE BCC_A2 % 1 1 !
    CONSTITUENT BCC_A2 :CR,TA: !
    PHASE C15_LAVES % 2 2 1 !
    CONSTITUENT C15_LAVES :CR:TA: !
    """)
    conds = {v.P: 101325, v.N: 1, v.T: (300, 4000, 20), v.X("TA"): (0, 1, 0.05)}
    strategy = TielineStrategy(dbf, ["CR", "TA"], ["BCC_A2", "C15_LAVES"], conds)

    def bcc_c15_point(T, x_bcc, x, bcc_fixed=False):
        # Tie-line between BCC_A2 with x(TA) = x_bcc and C15_LAVES (Cr2Ta) at overall x(TA) = x
        state_vars = np.array([1, 101325, T], dtype=np.float64)
        bcc = CompositionSet(strategy.phase_records["BCC_A2"])
        c15 = CompositionSet(strategy.phase_records["C15_LAVES"])
        c15_amount = 1.0 if bcc_fixed else (x - x_bcc) / (1/3 - x_bcc)
        bcc.update(np.array([1 - x_bcc, x_bcc]), 1 - c15_amount, state_vars)
        c15.update(np.array([1.0, 1.0]), c15_amount, state_vars)
        point_conds = {v.P: 101325, v.N: 1, v.T: T, v.X("TA"): x}
        if bcc_fixed:
            bcc.fixed = True
            return Point(point_conds, np.zeros(2), [bcc], [c15])
        return Point(point_conds, np.zeros(2), [], [bcc, c15])

    # Mapped line of the Ta-rich field, which lies at the C15_LAVES composition with BCC_A2 fixed
    ta_rich_line = ZPFLine(["BCC_A2"], ["C15_LAVES"])
    for T in [300, 302, 306, 314, 330, 350, 370, 390]:
        ta_rich_line.append(bcc_c15_point(T, 1 - 1e-6, 1/3, bcc_fixed=True))
    strategy.zpf_lines.append(ta_rich_line)

    # Starting point of the Cr-rich field from the 300 K edge, a fraction of a step from the mapped line
    cr_rich_start = bcc_c15_point(300, 1e-6, 0.3325)
    for d in [Direction.POSITIVE, Direction.NEGATIVE]:
        assert not strategy._exit_already_mapped(cr_rich_start, v.T, d), (
            "The Cr-rich BCC_A2+C15_LAVES field should not be covered by the Ta-rich BCC_A2+C15_LAVES line"
        )

    # A point on a mapped tie-line of the Ta-rich field is covered
    ta_rich_point = bcc_c15_point(350, 1 - 1e-6, 0.5)
    for d in [Direction.POSITIVE, Direction.NEGATIVE]:
        assert strategy._exit_already_mapped(ta_rich_point, v.T, d)

@select_database("BaCa-86Alc.tdb")
def test_starting_point_coverage_semantics(load_database):
    """
    A mapped zpf line covers its interior in both directions and its start only in the direction it was mapped
    """
    dbf = load_database()
    conds = {v.P: 101325, v.N: 1, v.T: (700, 1250, 20), v.X("CA"): (0, 1, 0.05)}
    strategy = TielineStrategy(dbf, ["BA", "CA", "VA"], list(dbf.phases.keys()), conds)
    strategy.do_map()

    long_lines = [zl for zl in strategy.zpf_lines if len(zl.points) > 10]
    assert len(long_lines) > 0
    zpf_line = long_lines[0]
    Ts = [np.squeeze(pt.get_property(v.T)) for pt in zpf_line.points]
    mapped_dir = Direction.POSITIVE if Ts[1] > Ts[0] else Direction.NEGATIVE
    opposite_dir = Direction.NEGATIVE if mapped_dir == Direction.POSITIVE else Direction.POSITIVE

    assert strategy._exit_already_mapped(zpf_line.points[0], v.T, mapped_dir), (
        "The start of a line should be covered in the direction it was mapped"
    )
    assert not strategy._exit_already_mapped(zpf_line.points[0], v.T, opposite_dir), (
        "The start of a line must not cover its opposite-direction sibling"
    )
    # an interior point where the line is monotonic in T
    i = next(i for i in range(4, len(Ts) - 4) if (Ts[i+4] - Ts[i]) * (Ts[i] - Ts[i-4]) > 0)
    for d in [Direction.POSITIVE, Direction.NEGATIVE]:
        assert strategy._exit_already_mapped(zpf_line.points[i], v.T, d), (
            "A point on an already mapped tie-line should be covered in both directions"
        )

@select_database("CaMg-06Zho.tdb")
def test_no_duplicate_boundary_remapping(load_database):
    """
    Recovery starting points do not map the CAMG2_C14+FCC_A1 boundary again
    """
    dbf = load_database()
    conds = {v.P: 101325, v.N: 1, v.T: (300, 4000, 20), v.X("MG"): (0, 1, 0.05)}
    strategy = TielineStrategy(dbf, ["CA", "MG", "VA"], list(dbf.phases.keys()), conds)
    strategy.do_map()

    target_lines = [zl for zl in strategy.zpf_lines
                    if sorted(set(zl.stable_phases)) == ["CAMG2_C14", "FCC_A1"] and len(zl.points) > 1]
    assert 1 <= len(target_lines) <= 2, (
        f"CAMG2_C14+FCC_A1 mapped {len(target_lines)} times; expected at most 2 segments"
    )
    # Still mapped from 300 K to the ~710 K invariant
    Ts = [np.squeeze(pt.get_property(v.T)) for zl in target_lines for pt in zl.points]
    assert np.min(Ts) < 310 and np.max(Ts) > 700

@select_database("AuCu-98Sun-LB.tdb")
def test_no_remapping_from_start_of_mapped_line(load_database):
    """
    An exit at the start of a mapped zpf line, heading the same way, does not map it again
    """
    dbf = load_database()
    conds = {v.P: 101325, v.N: 1, v.T: (300, 700, 20), v.X("CU"): (0, 1, 0.05)}
    strategy = TielineStrategy(dbf, ["AU", "CU", "VA"], list(dbf.phases.keys()), conds)
    strategy.do_map()

    def tielines(zpf_line):
        rows = []
        for pt in zpf_line.points:
            xs = sorted(np.squeeze(pt.get_local_property(cs, v.X("CU"))) for cs in pt.stable_composition_sets)
            rows.append([np.squeeze(pt.get_property(v.T))] + xs)
        return np.array(rows)

    lines = [zl for zl in strategy.zpf_lines if len(zl.points) > 1]
    for i, zpf_line in enumerate(lines):
        for other in lines[i+1:]:
            if sorted(zpf_line.stable_phases) != sorted(other.stable_phases):
                continue
            other_tielines = tielines(other)
            num_shared = sum(any(np.allclose(tl, other_tl, atol=1e-3) for other_tl in other_tielines) for tl in tielines(zpf_line))
            assert num_shared < 2, f"{zpf_line.stable_phases} boundary mapped more than once"

@select_database("CrV-92Lee-LB.tdb")
def test_degenerate_check_does_not_veto_narrow_lens(load_database):
    """
    The narrow Cr-V melting lens is mapped
    """
    dbf = load_database()
    conds = {v.P: 101325, v.N: 1, v.T: (300, 4000, 20), v.X("V"): (0, 1, 0.05)}
    strategy = TielineStrategy(dbf, ["CR", "V", "VA"], list(dbf.phases.keys()), conds)
    strategy.do_map()

    for zpf_line in strategy.zpf_lines:
        if sorted(set(zpf_line.stable_phases)) == ["BCC_A2", "LIQUID"]:
            xs = [np.squeeze(pt.get_property(v.X("V"))) for pt in zpf_line.points]
            if np.min(xs) < 0.05 and np.max(xs) > 0.9:
                break
    else:
        assert False, "BCC_A2+LIQUID melting lens was not mapped across the composition range"

@select_database("CrPt-98Spe-LB.tdb")
def test_degenerate_check_does_not_end_line_at_congruent_extremum(load_database):
    """
    The Cr-Pt fcc liquidus is mapped through its congruent maximum (~2058 K, x(PT)~0.78)
    """
    dbf = load_database()
    conds = {v.P: 101325, v.N: 1, v.T: (1700, 2200, 10), v.X("PT"): (0, 1, 0.01)}
    strategy = TielineStrategy(dbf, ["CR", "PT", "VA"], list(dbf.phases.keys()), conds)
    strategy.do_map()

    for zpf_line in strategy.zpf_lines:
        if sorted(set(zpf_line.stable_phases)) == ["FCC_A1", "LIQUID"]:
            xs = [np.squeeze(pt.get_property(v.X("PT"))) for pt in zpf_line.points]
            if np.max(xs) > 0.95:
                break
    else:
        assert False, "FCC_A1+LIQUID liquidus was not mapped past the congruent maximum to the Pt side"

@select_database("Al-Cu-Y.tdb")
def test_issue_662_phase_boundary_loop(load_database):
    T = 2260
    dbf = load_database()
    comps = ['AL', 'CU', 'Y', 'VA']
    phases = list(dbf.phases.keys())
    conds = {v.T: T, v.P:101325, v.X('AL'): (0,1,0.015), v.X('Y'): (0,1,0.015)}
    strat = TielineStrategy(dbf, comps, phases, conds)
    strat.add_nodes_from_conditions({v.T: T, v.P: 101325, v.X('AL'): 0.33, v.X('Y'): 0.33})

    # this is pretty much the loop in strat.do_map()
    # the conditions we set here should be 90 iterations
    # so mapping does not finish by 200 iterations, then something would be wrong
    strat.do_map(200)
    zpf_finished = strat.zpf_lines[-1].status != ZPFState.NOT_FINISHED
    no_more_exits = strat._exit_index >= len(strat._exits)
    no_more_nodes = strat.node_queue.size() == 0
    assert zpf_finished
    assert no_more_exits
    assert no_more_nodes

    # should only have zpf lines for liquid-AlCuY
    # there should be two zpf lines since the node we manually add
    # will map in the positive and negative axis directions
    assert len(strat.zpf_lines) == 2
    for z in strat.zpf_lines:
        assert len(set(z.stable_phases) - {'LIQUID', 'ALCUY'}) == 0

    # this will trigger plotting the zpf line as a point, so just make sure this plots
    # without fail
    plot_ternary(strat, label_nodes=True)

@select_database("Al-Cu-Y.tdb")
def test_starting_points_across_stoichiometric_tielines(load_database):
    """
    Exits between two stoichiometric phases add the tie-triangle across the tie-line as a starting point
    """
    dbf = load_database()
    conds = {v.T: 700, v.P: 101325, v.N: 1, v.X("CU"): (0, 1, 0.02), v.X("Y"): (0, 1, 0.02)}
    strategy = TielineStrategy(dbf, ["AL", "CU", "Y", "VA"], list(dbf.phases.keys()), conds)
    # ALCUY+ALY2+HCP_A3 and AL2Y3+ALCUY+ALY both neighbor AL2Y3+ALCUY+ALY2
    strategy.add_nodes_from_conditions({v.T: 700, v.P: 101325, v.N: 1, v.X("CU"): 0.05, v.X("Y"): 0.65})
    strategy.add_nodes_from_conditions({v.T: 700, v.P: 101325, v.N: 1, v.X("CU"): 0.05, v.X("Y"): 0.5})
    while strategy.node_queue._current_node_index < 2 or strategy._exit_index < len(strategy._exits):
        if strategy.iterate():
            break

    assert len(strategy.zpf_lines) == 0
    new_nodes = {tuple(sorted(n.stable_phases)): n for n in strategy.node_queue.nodes[2:] if len(n.stable_composition_sets) == 3}
    assert sorted(new_nodes) == [("AL2Y", "ALCUY", "ALY"), ("AL2Y3", "ALCUY", "ALY2"), ("ALCUY", "CUY", "HCP_A3")]
    # The tie-line crossed to reach each node is not an exit
    remaining_exits = {phases: sorted(tuple(sorted(e.stable_phases)) for e in strategy._find_exits_from_node(node)[0]) for phases, node in new_nodes.items()}
    assert remaining_exits[("AL2Y", "ALCUY", "ALY")] == [("AL2Y", "ALCUY"), ("AL2Y", "ALY")]
    assert remaining_exits[("AL2Y3", "ALCUY", "ALY2")] == [("AL2Y3", "ALY2")]
    assert remaining_exits[("ALCUY", "CUY", "HCP_A3")] == [("ALCUY", "CUY"), ("CUY", "HCP_A3")]

@select_database("alzn_mey.tdb")
def test_step_strategy_get_data_respects_display_units_for_state_and_model_properties(load_database):
    dbf = load_database()
    strategy = StepStrategy(dbf, ["AL", "VA"], ["FCC_A1", "LIQUID"], conditions={v.T: (920, 940, 5), v.P: 101325})
    strategy.do_map()

    gm_mass = as_property("GM")["J/g"]
    data = strategy.get_data(v.T["degC"], gm_mass, global_y=True, set_nan_to_zero=False)
    assert len(data.data) == 1 and data.data[0].phase == "SYSTEM"

    expected_x = []
    expected_y = []
    for zpf_line in strategy.zpf_lines:
        for point in zpf_line.points:
            expected_x.append(point.get_property(v.T) - 273.15)
            expected_y.append(to_display_units(point.get_property(gm_mass), point.stable_composition_sets, gm_mass))
    expected_x = np.asarray(expected_x)
    expected_y = np.asarray(expected_y)
    argsort = np.argsort(expected_x)

    np.testing.assert_allclose(data.data[0].x, expected_x[argsort])
    np.testing.assert_allclose(data.data[0].y, expected_y[argsort])

@select_database("pbsn.tdb")
def test_step_strategy_get_data_computes_mass_fraction(load_database):
    dbf = load_database()
    strategy = StepStrategy(dbf, ["PB", "SN", "VA"], ["FCC_A1", "LIQUID"], conditions={v.T: (500, 520, 10), v.X("SN"): 0.5, v.P: 101325})
    strategy.do_map()

    data = strategy.get_data(v.X("SN"), v.W("SN"), global_x=True, global_y=True, set_nan_to_zero=False)
    assert len(data.data) == 1 and data.data[0].phase == "SYSTEM"

    x_sn = data.data[0].x
    w_sn = data.data[0].y
    pb_mass = dbf.refstates["PB"]["mass"]
    sn_mass = dbf.refstates["SN"]["mass"]
    expected_w_sn = x_sn*sn_mass / (x_sn*sn_mass + (1 - x_sn)*pb_mass)

    np.testing.assert_allclose(w_sn, expected_w_sn)
    assert np.nanmax(np.abs(w_sn - x_sn)) > 1e-3

@select_database("alzn_mey.tdb")
def test_strategy_plotting_respects_units(load_database):
    """Test that giving state variables with units to ZPFLine.get_var_list converts units appropriately"""

    dbf = load_database()
    strategy = StepStrategy(dbf, ["AL", "VA"], ["FCC_A1", "LIQUID"], conditions={v.T: (920, 940, 1), v.P: 101325})
    strategy.do_map()

    # we should have found a line ending at a node, extract that here
    node_zpf_lines = [zl for zl in strategy.zpf_lines if zl.status == ZPFState.NEW_NODE_FOUND]
    assert len(node_zpf_lines) == 1
    node_zpf_line = node_zpf_lines[0]
    np.testing.assert_allclose(node_zpf_line.get_var_list(_get_phase_specific_variable(None, v.T))[-1], 933.600, atol=1e-3)  # value in Kelvin
    np.testing.assert_allclose(node_zpf_line.get_var_list(_get_phase_specific_variable(None, v.T["celsius"]))[-1], 933.600 - 273.15, atol=1e-3)  # value in Celsius
    np.testing.assert_allclose(node_zpf_line.get_var_list(_get_phase_specific_variable(None, v.T["degC"]))[-1], 933.600 - 273.15, atol=1e-3)  # value in Celsius

def test_mapping_custom_models():
    dbf = Database("""
    ELEMENT A A 1 1 1 !
    ELEMENT B B 1 1 1 !

    TYPE_DEFINITION % SEQ * !
    DEFINE_SYSTEM_DEFAULT VA !

    PHASE S %  1 1.0 !
    CONSTITUENT S :A,B: !

    PHASE M %  1 1.0 !
    CONSTITUENT M :A,B: !
    """)

    def make_custom_model(gibbs_function):
        class CustomModel(Model):
            contributions = [("custom", "custom_gibbs")]

            def build_phase(self, dbe):
                super(CustomModel, self).build_phase(dbe)

            def custom_gibbs(self, dbe):
                tmp = sorted(list(dbe.species)) # guarantee order to be [A, B]
                xs = [v.SiteFraction(self.phase_name, 0, s) for s in tmp]
                return gibbs_function(xs, v.T)

        return CustomModel

    def custom_energy_1(x, T):
        return (x[0]-0.1)**2 + (x[1]-0.9)**2 + T
    def custom_energy_2(x, T):
        return (x[0]-0.9)**2 + (x[1]-0.1)**2 + 0.99*T
    # exact phase boundaries for the above energies
    def exact_sm(T):
        return (0.1-0.003125*T)
    def exact_ms(T):
        return (0.9-0.003125*T)

    ens = [custom_energy_1, custom_energy_2]
    phases = ["S", "M"]
    comps = ["A", "B"]
    models = { k: make_custom_model(v)(dbf, comps, k) for k,v  in zip(phases, ens)}
    conds = {v.N : 1, v.P : 1e5, v.T : (0, 10, 1), v.X("A") : (0, 1, 0.05) }
    binary = TielineStrategy(dbf,  comps, phases, conditions=conds, models=models)
    binary.do_map()
    assert len(binary.zpf_lines) > 0 # without the fix passing in models this will be empty for the dummy DB
    for tl in binary.get_tieline_data(v.X("A"), v.T):
        T = tl["S"].y
        # error is mostly O(1e-11), though some O(1e-10) occur -> set tolerance above this
        np.testing.assert_allclose([tl["S"].x, tl["M"].x], [exact_sm(T), exact_ms(T)], atol=1e-9, rtol=1e-9)
        # diff from machine precision probably via search/solver tolerances
