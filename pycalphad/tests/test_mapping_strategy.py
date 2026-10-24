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
    # Ideal solution with a line compound that is stable below ~605 K
    dbf = Database("""
    ELEMENT A   VACUUM                    0 0 0!
    ELEMENT B   VACUUM                    0 0 0!

    PHASE ALPHA % 1 1 !
    CONSTITUENT ALPHA :A,B: !

    PHASE AB % 2 1 1 !
    CONSTITUENT AB :A:B: !
    PARAMETER G(AB,A:B;0) 1 -10000+5*T; 6000 N !
    """)
    comps, phases = ["A", "B"], ["ALPHA", "AB"]
    tieline = TielineStrategy(dbf, comps, phases, {v.P: 101325, v.T: (445, 455, 10), v.X("B"): (0, 1, 0.05)})
    # A strategy only picks up a stray X(*,B) if it was created before the axis variable was changed
    steps = [StepStrategy(dbf, comps, phases, {v.P: 101325, v.T: 450, v.X("B"): (0, 0.1, 0.05)}) for _ in range(2)]
    assert tieline.axis_vars[1] is steps[0].axis_vars[0] is steps[1].axis_vars[0]
    # Start from a single ALPHA + AB tie-line instead of searching the axis limits for starting points
    tieline.add_nodes_from_conditions({v.P: 101325, v.T: 450, v.X("B"): 0.3})

    tieline.get_tieline_data(v.X("B"), v.T)
    tieline.get_tieline_data(v.T, v.X("B"))
    tieline.get_invariant_data(v.X("B"), v.T)
    tieline.get_invariant_data(v.T, v.X("B"))
    assert len(tieline.zpf_lines) > 0
    # A stray X(*,B) condition used to make this fail with nonzero degrees of freedom
    steps[0].do_map()
    assert len(steps[0].zpf_lines) > 0

    steps[0].get_data(v.X("B"), v.NP("*"))
    steps[0].get_data(v.NP("*"), v.X("B"))
    steps[1].do_map()
    assert len(steps[1].zpf_lines) > 0


@select_database("AlTe-19Shi.tdb")
def test_degenerate_tieline_rejected_above_congruent_melting(load_database):
    """
    The AL2TE3_BETA+LIQUID boundary ends at the congruent melting point (~1138.7 K) instead of following a zero-width tie-line above it
    """
    dbf = load_database()
    # Only the phases of the tie-line; the congruent melting of AL2TE3_BETA does not depend on the others.
    # The coarse composition step keeps the line stepping in temperature up to the congruent point, so the
    # step past it converges to the zero-width tie-line with the liquid collapsed onto the AL2TE3 associate
    conds = {v.P: 101325, v.N: 1, v.T: (1100, 1200, 5), v.X("TE"): (0.5, 0.7, 0.1)}
    strategy = TielineStrategy(dbf, ["AL", "TE", "VA"], ["AL2TE3_BETA", "LIQUID"], conds)
    # Te-rich side of the AL2TE3_BETA+LIQUID region, mapped up in temperature toward the congruent point
    assert strategy.add_nodes_from_conditions({v.P: 101325, v.N: 1, v.T: 1120, v.X("TE"): 0.61}, Direction.POSITIVE)
    strategy.do_map()

    assert len(strategy.zpf_lines) == 1
    zpf_line = strategy.zpf_lines[0]
    widths = [np.ptp([np.asarray(cs.X) for cs in pt.stable_composition_sets], axis=0).max() for pt in zpf_line.points]
    zero_width_Ts = [float(pt.get_property(v.T)) for pt, width in zip(zpf_line.points, widths) if width < 1e-7]
    # The boundary reaches the congruent point, where the tie-line has zero width, and ends there
    # (it used to keep stepping up in temperature along the zero-width tie-line at x(TE)=0.6)
    assert widths[-1] < 1e-7
    assert len(zero_width_Ts) == 1, f"ZPF line continued along a zero-width tie-line at T={zero_width_Ts}"
    assert zpf_line.status == ZPFState.REACHED_LIMIT, f"ZPF line was not ended at the zero-width tie-line ({zpf_line.status})"

def test_unary_pt_mapping_not_flagged_as_degenerate():
    """
    A univariant line in a unary P-T diagram is mapped across the diagram, not ended as a degenerate zero-width tie-line
    """
    # ALPHA and BETA (both pure A) coexist along T = 1000 K + 1e-7 K/Pa * P
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
    comps, phases = ["A"], ["ALPHA", "BETA"]
    strategy = TielineStrategy(dbf, comps, phases, {v.N: 1, v.T: (900, 1300, 50), v.P: (0, 2e9, 2e8)})
    # Starting point: the ALPHA -> BETA transition at 1000 K, found by stepping in T along the P = 0 edge,
    # as the automatic starting point search does
    step = StepStrategy(dbf, comps, phases, {v.N: 1, v.P: 0, v.T: (900, 1300, 50)}, models=strategy.models, phase_record_factory=strategy.phase_records)
    step.do_map()
    strategy.add_starting_points_from_step(step)
    strategy.do_map()

    # Every point on the line has a zero-width tie-line, but the line must still be followed from P = 0 to the P = 2 GPa limit
    zpf_lines = [zpf_line for zpf_line in strategy.zpf_lines if set(zpf_line.stable_phases) == {"ALPHA", "BETA"}]
    assert len(zpf_lines) > 0, "No ALPHA+BETA line was mapped"
    P = np.concatenate([zpf_line.get_var_list(v.P) for zpf_line in zpf_lines])
    T = np.concatenate([zpf_line.get_var_list(v.T) for zpf_line in zpf_lines])
    assert np.max(P) == pytest.approx(2e9), "ALPHA+BETA line was ended before reaching P = 2 GPa"
    np.testing.assert_allclose(T, 1000 + 1e-7*P, atol=1e-3)

@select_database("cumg.tdb")
def test_no_degenerate_edge_pinned_zpf_lines(load_database):
    """
    The Mg-rich HCP_A3+LIQUID boundary ends at the melting point of Mg instead of following the pure Mg edge above it
    """
    dbf = load_database()
    conds = {v.P: 101325, v.N: 1, v.T: (850, 1100, 20), v.X("MG"): (0.9, 1, 0.05)}
    strategy = TielineStrategy(dbf, ["CU", "MG", "VA"], ["HCP_A3", "LIQUID"], conds)
    # The automatic starting point of this boundary, on the 850 K edge
    strategy.add_nodes_from_conditions({v.P: 101325, v.N: 1, v.T: 850, v.X("MG"): 0.95})
    strategy.do_map()

    zpf_lines = [zl for zl in strategy.zpf_lines if set(zl.stable_phases) == {"HCP_A3", "LIQUID"}]
    T = np.concatenate([zl.get_var_list(v.T) for zl in zpf_lines])
    x_liq = np.concatenate([zl.get_var_list(v.X("LIQUID", "MG")) for zl in zpf_lines])
    # The liquidus is mapped from the starting point up to the melting point of Mg (923 K)
    below_melting = T < 923
    assert np.min(x_liq[below_melting]) < 0.93
    assert np.max(x_liq[below_melting]) > 0.99
    # Above it, the tie-line collapses to zero width at x(MG)=1, which ends the line within a step or two
    # instead of following the pure Mg edge up to the T limit (1100 K)
    assert np.max(T) < 923 + 2 * 20

def test_circular_loop_check_normalizes_axes():
    """
    A liquidus back at its starting temperature after a congruent minimum (e.g. Ba-Ca) is not ended as a loop
    """
    # A melts at 1000 K and B at 1200 K, and the BCC interaction gives a congruent minimum near 890 K
    dbf = Database("""
    ELEMENT A BLANK 0 0 0 !
    ELEMENT B BLANK 0 0 0 !
    PHASE LIQUID % 1 1 !
    CONSTITUENT LIQUID :A,B: !
    PARAMETER G(LIQUID,A;0) 1 10000-10*T; 6000 N !
    PARAMETER G(LIQUID,B;0) 1 12000-10*T; 6000 N !
    PHASE BCC % 1 1 !
    CONSTITUENT BCC :A,B: !
    PARAMETER L(BCC,A,B;0) 1 8000; 6000 N !
    """)
    strategy = TielineStrategy(dbf, ["A", "B"], ["BCC", "LIQUID"], {v.P: 101325, v.T: (850, 1250, 20), v.X("B"): (0, 1, 0.05)})
    # Start just below the A melting point, so the liquidus comes back to the starting temperature on the B-rich side
    strategy.add_nodes_from_conditions({v.P: 101325, v.T: 990, v.X("B"): 0.02})
    strategy.do_map()

    T = np.concatenate([zpf_line.get_var_list(v.T) for zpf_line in strategy.zpf_lines])
    x_liq = np.concatenate([zpf_line.get_var_list(v.X("LIQUID", "B")) for zpf_line in strategy.zpf_lines])
    assert np.min(T) < 900
    # In raw units, the 20 K steps swamped the mole fraction distance and the liquidus was ended as a loop near 1000 K
    assert np.max(x_liq) > 0.95

@select_database("GaLa-11Idb.tdb")
def test_automatic_starting_points_keep_forced_starting_points(load_database):
    """
    Parentless recovery nodes from the edge step maps are kept, e.g. for the narrow La BCC window (1134-1194 K)
    """
    from pycalphad.mapping.primitives import MIN_COMPOSITION
    dbf = load_database()
    # La side of Ga-La. FCC_A1 is needed for the metastable node that the edge step map lands on
    phases = ["LIQUID", "BCC_A2", "FCC_A1"]
    conds = {v.P: 101325, v.N: 1, v.T: (1100, 1250, 20), v.X("LA"): (0.9, 1, 0.05)}
    strategy = TielineStrategy(dbf, ["GA", "LA", "VA"], phases, conds)
    # Step map of the automatic starting point search along the x(LA)=1 edge of the full Ga-La map (300-4000 K).
    # Its coarse step (dT = 185 K) jumps over the BCC window to a metastable FCC_A1+LIQUID node, and the step map
    # only records the LIQUID+BCC_A2 field by the recovery point that it force-adds (without a parent) below the node
    step_conds = {v.P: 101325, v.N: 1, v.T: (300, 4000, 185), v.X("LA"): 1 - MIN_COMPOSITION}
    step = StepStrategy(dbf, ["GA", "LA", "VA"], phases, step_conds, models=strategy.models, phase_record_factory=strategy.phase_records)
    step.do_map()
    bcc_liquid_nodes = [node for node in step.node_queue.nodes if sorted(node.stable_phases) == ["BCC_A2", "LIQUID"]]
    assert len(bcc_liquid_nodes) == 1 and bcc_liquid_nodes[0].parent is None

    strategy.add_starting_points_from_step(step)
    strategy.do_map()

    for zpf_line in strategy.zpf_lines:
        if sorted(zpf_line.stable_phases) == ["BCC_A2", "LIQUID"]:
            Ts = zpf_line.get_var_list(v.T)
            if np.min(Ts) < 1160 and np.max(Ts) > 1190:
                break
    else:
        assert False, "No BCC_A2+LIQUID ZPF line mapped inside the La BCC window (1134-1194 K)"

@select_database("AuCu-98Sun-LB.tdb")
def test_automatic_starting_points_for_every_merged_field(load_database):
    """
    Fields merged into one same-name step zpf line each get a starting point (Au-rich fcc+Au3Cu at 300 K)
    """
    dbf = load_database()
    # Au-rich half of Au-Cu. The fcc-based phases (fcc, Au3Cu, AuCu, ...) are all FCC_4SL
    conds = {v.P: 101325, v.N: 1, v.T: (300, 540, 20), v.X("CU"): (0, 0.5, 0.05)}
    strategy = TielineStrategy(dbf, ["AU", "CU", "VA"], ["FCC_4SL"], conds)
    # Step map along the T = 300 K edge. It starts in the middle of its range, in the Au3Cu+AuCu field, and its
    # 0.05 steps jump over the narrow Au3Cu field (x(CU) = 0.237-0.280) into the fcc+Au3Cu field: the same two
    # composition sets slide from one field into the other, so both fields end up in one step zpf line
    step_conds = {v.P: 101325, v.N: 1, v.T: 300, v.X("CU"): (0.2, 0.52, 0.05)}
    step = StepStrategy(dbf, ["AU", "CU", "VA"], ["FCC_4SL"], step_conds, models=strategy.models, phase_record_factory=strategy.phase_records)
    step.do_map()
    assert any(len(zpf_line.stable_phases) == 2 and np.min(zpf_line.get_var_list(v.X("CU"))) < 0.237
               and np.max(zpf_line.get_var_list(v.X("CU"))) > 0.280 for zpf_line in step.zpf_lines)

    strategy.add_starting_points_from_step(step)
    # One starting point in each direction per field, labeled by the Au-rich end of its tie-line
    starting_points = sorted((round(min(cs.X[1] for cs in node.stable_composition_sets), 3), node.axis_direction.value)
                             for node in strategy.node_queue.nodes)
    assert starting_points == [(0.078, -1), (0.078, 1), (0.28, -1), (0.28, 1)], (
        "Each field merged into the step zpf line should get one starting point in each direction, "
        "including the Au-rich fcc+Au3Cu field (x(CU) = 0.078-0.237)"
    )

    strategy.do_map()
    for zpf_line in strategy.zpf_lines:
        xs = zpf_line.get_var_list(v.X("CU"))
        if np.min(xs) < 0.15 and np.max(xs) > 0.20:
            break
    else:
        assert False, "Au-rich fcc+Au3Cu field (x(CU)~0.08-0.24) was never mapped"

@select_database("CrTa-93Dup-LB.tdb")
def test_starting_point_dedup_distinguishes_twin_fields(load_database):
    """
    BCC_A2+C15_LAVES fields on both sides of the C15_LAVES compound (Cr2Ta) are mapped from their 300 K starting points

    The starting point of the Cr-rich field is a fraction of a step from the mapped line of the Ta-rich field at the
    C15_LAVES composition, but it is on another tie-line, so it is not already mapped
    """
    dbf = load_database()
    comps, phases = ["CR", "TA", "VA"], ["BCC_A2", "C15_LAVES"]
    conds = {v.P: 101325, v.N: 1, v.T: (300, 340, 20), v.X("TA"): (0, 1, 0.05)}
    strategy = TielineStrategy(dbf, comps, phases, conds)
    # Step map along the 300 K edge from x(TA) = 0.5, like the automatic starting point search over the whole edge,
    # with the same starting points: first the Ta-rich field at x(TA) = 0.5, then the Cr-rich field at x(TA) = 0.3325
    step_conds = {v.P: 101325, v.N: 1, v.T: 300, v.X("TA"): (0.3, 0.7, 0.05)}
    step = StepStrategy(dbf, comps, phases, step_conds, models=strategy.models, phase_record_factory=strategy.phase_records)
    step.do_map()
    strategy.add_starting_points_from_step(step)
    strategy.do_map()

    # The two fields differ in the composition of BCC_A2: x(TA) ~ 0 in the Cr-rich field and ~ 1 in the Ta-rich field
    mapped_fields = []
    for zpf_line in strategy.zpf_lines:
        Ts = zpf_line.get_var_list(v.T)
        if sorted(zpf_line.stable_phases) == ["BCC_A2", "C15_LAVES"] and np.min(Ts) < 301 and np.max(Ts) > 339:
            mapped_fields.append("Ta-rich" if np.mean(zpf_line.get_var_list(v.X("BCC_A2", "TA"))) > 0.5 else "Cr-rich")
    assert sorted(mapped_fields) == ["Cr-rich", "Ta-rich"], (
        f"Each BCC_A2+C15_LAVES field should be mapped once over 300-340 K, got {mapped_fields}"
    )

def test_starting_point_coverage_semantics():
    """
    A mapped zpf line covers its interior in both directions and its start only in the direction it was mapped

    So both directions of a starting point in the middle of a boundary are mapped, and starting points at the start
    of a mapped line (heading the same way) or inside it do not map it again
    """
    from pycalphad.mapping.primitives import MIN_COMPOSITION
    # BCC_A2 and C15_LAVES of CrTa-93Dup-LB.tdb, with the pure-element references dropped
    # (they are linear in composition, so the phase boundaries are unchanged)
    dbf = Database("""
    ELEMENT VA VACUUM 0 0 0 !
    ELEMENT CR BCC_A2 51.996 4050.0 23.5429 !
    ELEMENT TA BCC_A2 180.9479 5681.872 41.4718 !
    TYPE_DEFINITION % SEQ * !
    TYPE_DEFINITION B GES AMEND_PHASE_DESCRIPTION @ MAGNETIC -1 0.4 !

    PHASE BCC_A2 %B 2 1 3 !
    CONSTITUENT BCC_A2 :CR,TA:VA: !
    PARAMETER G(BCC_A2,CR:VA;0) 298.15 0; 6000 N !
    PARAMETER G(BCC_A2,TA:VA;0) 298.15 0; 6000 N !
    PARAMETER TC(BCC_A2,CR:VA;0) 298.15 -311.5; 6000 N !
    PARAMETER BMAGN(BCC_A2,CR:VA;0) 298.15 -0.008; 6000 N !
    PARAMETER L(BCC_A2,CR,TA:VA;0) 298.15 46800-11.4*T; 6000 N !
    PARAMETER L(BCC_A2,CR,TA:VA;1) 298.15 37200-17.3*T; 6000 N !
    PARAMETER L(BCC_A2,CR,TA:VA;2) 298.15 16200-5.4*T; 6000 N !

    PHASE C15_LAVES % 2 2 1 !
    CONSTITUENT C15_LAVES :CR,TA:CR,TA: !
    PARAMETER G(C15_LAVES,CR:CR;0) 298.15 15000; 6000 N !
    PARAMETER G(C15_LAVES,TA:TA;0) 298.15 15000; 6000 N !
    PARAMETER G(C15_LAVES,CR:TA;0) 298.15 -26850+0.69*T; 6000 N !
    PARAMETER G(C15_LAVES,TA:CR;0) 298.15 56850-0.69*T; 6000 N !
    PARAMETER L(C15_LAVES,CR,TA:CR;0) 298.15 270000-90*T; 6000 N !
    PARAMETER L(C15_LAVES,CR,TA:TA;0) 298.15 270000-90*T; 6000 N !
    PARAMETER L(C15_LAVES,CR:CR,TA;0) 298.15 117000-39*T; 6000 N !
    PARAMETER L(C15_LAVES,TA:CR,TA;0) 298.15 117000-39*T; 6000 N !
    """)
    comps, phases = ["CR", "TA", "VA"], ["BCC_A2", "C15_LAVES"]
    conds = {v.P: 101325, v.N: 1, v.T: (800, 900, 20), v.X("TA"): (0, 1, 0.05)}
    strategy = TielineStrategy(dbf, comps, phases, conds)
    # Step map along the x(TA) = 0 edge, as the automatic starting point search does. Its starting point is where
    # C15_LAVES precipitates from Cr-rich BCC_A2 (849 K), in the middle of the solvus, which goes on to lower x(TA)
    # below 849 K and to higher x(TA) above
    edge_conds = {v.P: 101325, v.N: 1, v.T: (800, 900, 5), v.X("TA"): MIN_COMPOSITION}
    edge_step = StepStrategy(dbf, comps, phases, edge_conds, models=strategy.models, phase_record_factory=strategy.phase_records)
    edge_step.do_map()
    strategy.add_starting_points_from_step(edge_step)
    strategy.do_map()

    # The line mapped up from 849 K has its first points a fraction of a step above the starting point, but it does
    # not cover the other direction, which maps the solvus down to 800 K
    Ts = np.concatenate([zpf_line.get_var_list(v.T) for zpf_line in strategy.zpf_lines])
    assert np.min(Ts) < 810 and np.max(Ts) > 890, (
        f"The solvus should be mapped from 849 K up to 900 K and down to 800 K, got {np.min(Ts):.0f}-{np.max(Ts):.0f} K"
    )
    num_lines = len(strategy.zpf_lines)

    # The same starting points again, each at the start of the line it mapped and heading the same way
    strategy.add_starting_points_from_step(edge_step)
    strategy.do_map()
    assert len(strategy.zpf_lines) == num_lines, "Starting points at the start of mapped lines mapped them again"

    # A starting point inside the line above 849 K, from a step map across the Cr-rich BCC_A2+C15_LAVES field at 870 K
    field_conds = {v.P: 101325, v.N: 1, v.T: 870, v.X("TA"): (0.05, 0.3, 0.05)}
    field_step = StepStrategy(dbf, comps, phases, field_conds, models=strategy.models, phase_record_factory=strategy.phase_records)
    field_step.do_map()
    strategy.add_starting_points_from_step(field_step)
    strategy.do_map()
    assert len(strategy.zpf_lines) == num_lines, "A starting point inside a mapped line mapped it again"

@select_database("CaMg-06Zho.tdb")
def test_no_duplicate_boundary_remapping(load_database):
    """
    A recovery starting point on a tie-line of the mapped CAMG2_C14+FCC_A1 boundary does not map it again
    """
    from pycalphad.mapping.primitives import MIN_COMPOSITION
    dbf = load_database()
    conds = {v.P: 101325, v.N: 1, v.T: (670, 700, 20), v.X("MG"): (0, 1, 0.05)}
    strategy = TielineStrategy(dbf, ["CA", "MG", "VA"], ["CAMG2_C14", "FCC_A1"], conds)
    # Automatic starting points on the boundary: one at the T edge, and a recovery node of the x(MG)=0 edge
    # step map at an off-grid temperature. Mapped line points have FCC_A1 at zero amount, so their global
    # composition is at the CAMG2_C14 end of the tie-line and the recovery node's is at the FCC_A1 end
    strategy.add_nodes_from_conditions({v.P: 101325, v.N: 1, v.T: 670, v.X("MG"): 0.3})
    strategy.add_nodes_from_conditions({v.P: 101325, v.N: 1, v.T: 691.243, v.X("MG"): MIN_COMPOSITION})
    for node in strategy.node_queue.nodes:
        assert sorted(node.stable_phases) == ["CAMG2_C14", "FCC_A1"]
        node.skip_if_mapped = True
    strategy.do_map()

    assert len(strategy.zpf_lines) == 1, f"CAMG2_C14+FCC_A1 mapped {len(strategy.zpf_lines)} times"
    Ts = [np.squeeze(pt.get_property(v.T)) for pt in strategy.zpf_lines[0].points]
    assert np.min(Ts) < 671 and np.max(Ts) > 699

def test_no_remapping_from_start_of_mapped_line():
    """
    An exit at the start of a mapped zpf line, heading the same way, does not map it again
    """
    from pycalphad.mapping.primitives import ExitHint
    # Miscibility gap, so that both composition sets are the same phase as in the FCC_4SL+FCC_4SL boundaries of Au-Cu
    dbf = Database("""
    ELEMENT A BLANK 0 0 0 !
    ELEMENT B BLANK 0 0 0 !
    PHASE ALPHA % 1 1 !
    CONSTITUENT ALPHA :A,B: !
    PARAMETER G(ALPHA,A,B;0) 1 20000; 6000 N !
    """)
    conds = {v.P: 101325, v.N: 1, v.T: (800, 820, 20), v.X("B"): (0, 1, 0.05)}
    strategy = TielineStrategy(dbf, ["A", "B"], ["ALPHA"], conds)
    # The same node found twice by mapping (like an invariant reached from both boundaries ending there), each
    # with an exit on the same tie-line, from which the boundary only rises since 800 K is the axis limit
    for x in [0.3, 0.6]:
        point = point_from_equilibrium(dbf, strategy.components, strategy.phases, {v.P: 101325, v.N: 1, v.T: 800, v.X("B"): x},
                                       models=strategy.models, phase_record_factory=strategy.phase_records)
        strategy.node_queue.add_node(strategy._create_node_from_point(point, None, None, None, ExitHint.NORMAL), True)
    strategy.do_map()
    assert len(strategy.zpf_lines) == 1, f"Boundary mapped {len(strategy.zpf_lines)} times"

@select_database("CrV-92Lee-LB.tdb")
def test_degenerate_check_does_not_veto_narrow_lens(load_database):
    """
    The exit direction test does not veto the narrow Cr-V melting lens at the pure Cr edge
    """
    from pycalphad.mapping.primitives import MIN_COMPOSITION
    from pycalphad.mapping.zpf_equilibrium import update_equilibrium_with_new_conditions
    from pycalphad.mapping.zpf_checks import _degenerate_zero_width_tieline

    dbf = load_database()
    conds = {v.P: 101325, v.N: 1, v.T: (300, 4000, 20), v.X("V"): (0, 1, 0.0002)}
    strategy = TielineStrategy(dbf, ["CR", "V", "VA"], ["BCC_A2", "LIQUID"], conds)

    # Liquidus at the pure Cr edge (~2180 K), with BCC_A2 fixed at zero amount
    prf = strategy.phase_records
    edge_conds = {v.N: 1.0, v.P: 101325.0, v.T: 2180.0, v.X("V"): MIN_COMPOSITION}
    state_variables = np.array([edge_conds[sv] for sv in prf["BCC_A2"].state_variables])
    bcc, liquid = CompositionSet(prf["BCC_A2"]), CompositionSet(prf["LIQUID"])
    bcc.update(np.array([1 - MIN_COMPOSITION, MIN_COMPOSITION, 1.0]), 0.0, state_variables)
    liquid.update(np.array([1 - MIN_COMPOSITION, MIN_COMPOSITION]), 1.0, state_variables)
    bcc.fixed = True
    edge_results = update_equilibrium_with_new_conditions(Point(edge_conds, np.zeros(2), [bcc], [liquid]), copy.deepcopy(edge_conds), v.T)
    assert edge_results is not None
    edge_point = edge_results[0]

    # The minimum delta trial step lands where the lens is genuinely narrower than the degenerate tolerance
    trial_delta = strategy.axis_delta[v.X("V")] * strategy.MIN_DELTA_RATIO
    trial_results = strategy._take_step(edge_point, v.X("V"), trial_delta, strategy.axis_lims[v.X("V")], Direction.POSITIVE)
    assert trial_results is not None
    assert _degenerate_zero_width_tieline(trial_results[0].stable_composition_sets)

    assert strategy._test_direction(edge_point, v.X("V"), Direction.POSITIVE) is not None

@select_database("CrPt-98Spe-LB.tdb")
def test_degenerate_check_does_not_end_line_at_congruent_extremum(load_database):
    """
    A zpf line stepping onto the Cr-Pt congruent maximum (~2058 K, x(PT)~0.7756) is not ended as a degenerate tie-line
    """
    from pycalphad.mapping.zpf_equilibrium import update_equilibrium_with_new_conditions
    from pycalphad.mapping.zpf_checks import check_degenerate_tieline, _degenerate_zero_width_tieline

    dbf = load_database()
    comps = ["CR", "PT", "VA"]
    prf = PhaseRecordFactory(dbf, comps, {v.N, v.P, v.T}, instantiate_models(dbf, comps, ["FCC_A1", "LIQUID"]))

    # FCC_A1 solidus one composition step below the congruent maximum, with LIQUID fixed at zero amount
    conds = {v.N: 1.0, v.P: 101325.0, v.T: 2057.8, v.X("PT"): 0.765557}
    state_variables = np.array([conds[sv] for sv in prf["FCC_A1"].state_variables])
    fcc, liquid = CompositionSet(prf["FCC_A1"]), CompositionSet(prf["LIQUID"])
    fcc.update(np.array([0.234443, 0.765557, 1.0]), 1.0, state_variables)
    liquid.update(np.array([0.24, 0.76]), 0.0, state_variables)
    liquid.fixed = True
    prev_results = update_equilibrium_with_new_conditions(Point(conds, np.zeros(2), [liquid], [fcc]), copy.deepcopy(conds), v.T)
    assert prev_results is not None
    prev_point = prev_results[0]

    zpf_line = ZPFLine(prev_point.fixed_phases, prev_point.free_phases)
    zpf_line.points.append(prev_point)
    zpf_line.axis_var = v.X("PT")
    zpf_line.axis_direction = Direction.POSITIVE
    zpf_line.current_delta = 0.01

    # A full step lands where the tie-line width passes through zero
    new_conds = copy.deepcopy(prev_point.global_conditions)
    new_conds[v.X("PT")] += zpf_line.current_delta
    step_results = update_equilibrium_with_new_conditions(prev_point, new_conds, v.T)
    assert step_results is not None
    assert _degenerate_zero_width_tieline(step_results[0].stable_composition_sets)

    axis_data = {
        "axis_vars": [v.X("PT"), v.T],
        "axis_delta": {v.X("PT"): 0.01, v.T: 10},
        "axis_lims": {v.X("PT"): (0, 1), v.T: (1700, 2200)},
    }
    check_degenerate_tieline(zpf_line, step_results, axis_data, normalize_factor=axis_data["axis_delta"])
    assert zpf_line.status == ZPFState.NOT_FINISHED

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

def test_starting_points_across_stoichiometric_tielines():
    """
    Exits between two stoichiometric phases add the tie-triangle across the tie-line as a starting point,
    once, and without the crossed tie-lines as exits
    """
    # Al-Cu-Y.tdb phases around AL2Y3+ALCUY+ALY2 at 700 K, with the pure-element references dropped
    # (they are linear in composition, so the tie-triangles are unchanged)
    dbf = Database("""
    ELEMENT VA VACUUM 0 0 0 !
    ELEMENT AL FCC_A1 26.982 4577.3 28.322 !
    ELEMENT CU FCC_A1 63.546 5004.1 33.15 !
    ELEMENT Y HCP_A3 88.906 0 0 !
    TYPE_DEFINITION % SEQ * !

    PHASE ALY % 2 .5 .5 !
    CONSTITUENT ALY :AL:Y: !
    PARAMETER G(ALY,AL:Y;0) 298.15 -54911.9+5.48155*T; 6000 N !

    PHASE AL2Y3 % 2 .4 .6 !
    CONSTITUENT AL2Y3 :AL:Y: !
    PARAMETER G(AL2Y3,AL:Y;0) 298.15 -48406.3+4.82944*T; 6000 N !

    PHASE ALY2 % 2 .33333 .66667 !
    CONSTITUENT ALY2 :AL:Y: !
    PARAMETER G(ALY2,AL:Y;0) 298.15 -44886.6+7.39566*T; 6000 N !

    PHASE ALCUY % 3 .333334 .333333 .333333 !
    CONSTITUENT ALCUY :AL:CU:Y: !
    PARAMETER G(ALCUY,AL:CU:Y;0) 298.15 -60989+2*T; 6000 N !

    PHASE HCP_A3 % 2 1 .5 !
    CONSTITUENT HCP_A3 :AL,CU,Y:VA: !
    PARAMETER G(HCP_A3,AL:VA;0) 298.15 5481-1.8*T; 6000 N !
    PARAMETER G(HCP_A3,CU:VA;0) 298.15 600+0.2*T; 6000 N !
    PARAMETER G(HCP_A3,Y:VA;0) 298.15 0; 6000 N !
    PARAMETER G(HCP_A3,CU,Y:VA;0) 298.15 80*T; 6000 N !
    """)
    conds = {v.T: 700, v.P: 101325, v.N: 1, v.X("CU"): (0, 1, 0.02), v.X("Y"): (0, 1, 0.02)}
    strategy = TielineStrategy(dbf, ["AL", "CU", "Y", "VA"], list(dbf.phases.keys()), conds)
    # ALCUY+ALY2+HCP_A3 and AL2Y3+ALCUY+ALY both neighbor AL2Y3+ALCUY+ALY2
    strategy.add_nodes_from_conditions({v.T: 700, v.P: 101325, v.N: 1, v.X("CU"): 0.05, v.X("Y"): 0.65})
    strategy.add_nodes_from_conditions({v.T: 700, v.P: 101325, v.N: 1, v.X("CU"): 0.05, v.X("Y"): 0.5})
    starting_nodes = list(strategy.node_queue.nodes)
    for node, tieline in zip(starting_nodes, [["ALCUY", "ALY2"], ["AL2Y3", "ALCUY"]]):
        exit_point = next(e for e in strategy._find_exits_from_node(node)[0] if sorted(e.stable_phases) == tieline)
        # No zpf line can start along the tie-line, so the tie-triangle across it is added as a starting point instead
        assert strategy._determine_start_direction(node, exit_point, None) is None

    # Reached across both tie-lines, but added once
    new_nodes = strategy.node_queue.nodes[len(starting_nodes):]
    assert [sorted(n.stable_phases) for n in new_nodes] == [["AL2Y3", "ALCUY", "ALY2"]]
    # Neither crossed tie-line is an exit of the new node
    assert [sorted(e.stable_phases) for e in strategy._find_exits_from_node(new_nodes[0])[0]] == [["AL2Y3", "ALY2"]]

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
