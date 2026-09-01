import copy
from importlib.resources import files

import numpy as np
import matplotlib.pyplot as plt

from pycalphad import binplot, ternplot, Database, variables as v
from pycalphad.tests.fixtures import select_database, load_database
from pycalphad.core.utils import instantiate_models, get_state_variables
from pycalphad.codegen.phase_record_factory import PhaseRecordFactory
from pycalphad.core.composition_set import CompositionSet
from pycalphad.property_framework import as_property
from pycalphad.property_framework.units import Q_, unit_conversion_context, to_display_units

from pycalphad.mapping import StepStrategy, IsoplethStrategy, BinaryStrategy, TernaryStrategy, TielineStrategy, plot_step, plot_isopleth, plot_ternary
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
    strategy = TernaryStrategy(dbf, comps, phases, conds)

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
    Tests how TernaryStrategy deals with nodes that are metastable

    This is done by purposely creating a known metastable node, which
    the TernaryStrategy should be able to detect whether a node is metastable
    and perform the following:
        a) if node is metastable, do not add to node queue and end the zpf line
           (the zpf line is kept since its points have passed their own global min checks)
        b) if node is stable, add to node queue
    """
    # Create system
    dbf = load_database()
    comps = ['CR', 'FE', 'NB', 'VA']
    phases = list(dbf.phases.keys())
    map_conds = {v.T: 1323, v.P: 101325, v.N: 1, v.X('CR'): (0, 1, 0.01), v.X('FE'): (0, 1, 0.01)}

    strategy = TernaryStrategy(dbf, comps, phases, map_conds, initialize=False)

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
    # Test that zpf line was kept, but ended for leading to a metastable node
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
    strat = TernaryStrategy(dbf, comps, phases, conds)

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
    strat = BinaryStrategy(dbf, comps, phases, conds)

    strat.add_nodes_from_conditions({v.T: 650, v.P: 101325, v.X('SN'): 0.2})
    initial_nodes = len(strat.node_queue.nodes)
    # stop until there's 4 nodes or if mapping finished
    while strat.node_queue._current_node_index < (initial_nodes+1):
        if strat.iterate():
            break

    nodes = [set(n.stable_phases) for n in strat.node_queue.nodes]
    assert {'LIQUID', 'HCP_A3', 'AUSN_B81'} in nodes

@select_database("cumg.tdb")
def test_mapping_runs_just_in_time_on_data_retrieval(load_database):
    """
    Data retrieval (get_* methods, and therefore plotting) should run mapping just in
    time, so users do not need to call do_map() explicitly. Adding starting points
    after mapping resets the mapping complete flag so the new points are processed on
    the next data retrieval.
    """
    dbf = load_database()
    conds = {v.P: 101325, v.N: 1, v.T: (850, 1000, 20), v.X("MG"): (0.9, 1, 0.05)}
    strategy = BinaryStrategy(dbf, ["CU", "MG", "VA"], ["HCP_A3", "LIQUID"], conds)
    assert not strategy._mapping_complete

    # Retrieving data without calling do_map() runs the mapping
    tieline_data = strategy.get_tieline_data(v.X("MG"), v.T)
    assert strategy._mapping_complete
    assert len(tieline_data) > 0
    assert len(strategy.zpf_lines) > 0
    num_zpf_lines = len(strategy.zpf_lines)

    # Retrieving data again does not re-run the mapping
    strategy.get_tieline_data(v.X("MG"), v.T)
    assert len(strategy.zpf_lines) == num_zpf_lines

    # Adding a starting point (here, inside the HCP_A3+LIQUID two-phase region)
    # resets the flag, and the next retrieval maps the new starting points
    added = strategy.add_nodes_from_conditions({v.T: 900, v.P: 101325, v.X("MG"): 0.99})
    assert added
    assert not strategy._mapping_complete
    strategy.get_tieline_data(v.X("MG"), v.T)
    assert strategy._mapping_complete
    assert len(strategy.zpf_lines) > num_zpf_lines

@select_database("AlTe-19Shi.tdb")
def test_degenerate_tieline_rejected_above_congruent_melting(load_database):
    """
    ZPF line for AL2TE3_BETA+LIQUID should not extent above a congruent melting point

    This constructs the ZPF line configuration (compound fixed at zero amount, liquid
    free) and steps it just past the congruent point, where the solver converges to
    the degenerate zero-width result, and checks that the degenerate tie-line check
    rejects it while accepting the real result below the congruent point.
    """
    from pycalphad.mapping.zpf_equilibrium import update_equilibrium_with_new_conditions
    from pycalphad.mapping.zpf_checks import simple_check_degenerate_tieline
    import pycalphad.mapping.utils as map_utils

    dbf = load_database()
    comps = ["AL", "TE", "VA"]
    phases = list(dbf.phases.keys())

    # Two-phase point on the Te-rich branch of the AL2TE3_BETA+LIQUID region
    # (congruent melting of AL2TE3_BETA is at ~1138.7 K, x(TE)=0.6)
    pt = point_from_equilibrium(dbf, comps, phases, {v.P: 101325, v.N: 1, v.T: 1100, v.X("TE"): 0.61})
    assert set(pt.stable_phases) == {"AL2TE3_BETA", "LIQUID"}
    beta = [cs for cs in pt.stable_composition_sets if cs.phase_record.phase_name == "AL2TE3_BETA"][0]
    liquid = [cs for cs in pt.stable_composition_sets if cs.phase_record.phase_name == "LIQUID"][0]
    zpf_point = map_utils._generate_point_with_fixed_cs(pt, beta, liquid)

    # Below the congruent point, the boundary is real (finite tie-line width)
    conds = copy.deepcopy(zpf_point.global_conditions)
    conds[v.T] = 1135
    below_results = update_equilibrium_with_new_conditions(zpf_point, conds, v.X("TE"))
    assert below_results is not None
    assert simple_check_degenerate_tieline(below_results)

    # Just above the congruent point, the solver converges to the degenerate
    # zero-width solution (liquid collapsed onto the AL2TE3 associate), which
    # must be rejected so mapping does not track it
    conds = copy.deepcopy(below_results[0].global_conditions)
    conds[v.T] = 1140
    above_results = update_equilibrium_with_new_conditions(below_results[0], conds, v.X("TE"))
    if above_results is not None:
        assert not simple_check_degenerate_tieline(above_results)

@select_database("cfe_broshe.tdb")
def test_unary_pt_mapping_not_flagged_as_degenerate(load_database):
    """
    In a unary P-T diagram, every multi-phase equilibrium trivially has all phases at
    the same pure composition, but two-phase coexistence along a univariant line is
    allowed by the Gibbs phase rule. The degenerate tie-line detection (see
    test_no_degenerate_edge_pinned_zpf_lines) must not end these zpf lines.
    """
    dbf = load_database()
    # Window of the Fe P-T diagram containing the bcc-fcc-hcp triple point
    # (~799 K, ~9.7 GPa in this database) and its three univariant lines
    conds = {v.N: 1, v.T: (600, 1300, 50), v.P: (0, 20e9, 2e9)}
    strategy = TielineStrategy(dbf, ["FE", "VA"], ["BCC_A2", "FCC_A1", "HCP_A3", "LIQUID"], conds)
    strategy.do_map()

    mapped_sets = [set(zl.stable_phases) for zl in strategy.zpf_lines if len(zl.points) > 1]
    for pair in [{"BCC_A2", "FCC_A1"}, {"BCC_A2", "HCP_A3"}, {"FCC_A1", "HCP_A3"}]:
        assert pair in mapped_sets, f"No univariant line mapped for {pair}"

@select_database("cumg.tdb")
def test_no_degenerate_edge_pinned_zpf_lines(load_database):
    """
    Mapping over a temperature range extending above a pure-element melting point
    should not produce spurious two-phase ZPF lines pinned to the pure-element edge.

    At a pure-element composition, a "two-phase" equilibrium is degenerate: the second
    phase has vanishing amount and both phases have essentially the pure-element
    composition, so the ZPF conditions are trivially satisfiable at any temperature.
    Without detecting this, the mapper follows these zero-width lines from the melting
    point up to the temperature axis limit (e.g. HCP_A3+LIQUID above the melting point
    of Mg and FCC_A1+LIQUID above the melting point of Cu in Cu-Mg).
    """
    dbf = load_database()
    # A window around the melting point of Mg (923 K) is sufficient to reproduce the
    # bug and keeps the test fast. Without the degenerate equilibrium detection, the
    # HCP_A3+LIQUID zpf line follows the x(MG)=1 edge from 923 K to the T axis limit.
    conds = {v.P: 101325, v.N: 1, v.T: (850, 1100, 20), v.X("MG"): (0.9, 1, 0.05)}
    strategy = TielineStrategy(dbf, ["CU", "MG", "VA"], ["HCP_A3", "LIQUID"], conds)
    strategy.do_map()

    for zpf_line in strategy.zpf_lines:
        num_degenerate_points = 0
        for point in zpf_line.points:
            comp_sets = point.stable_composition_sets
            if len(comp_sets) < 2:
                continue
            comps = np.array([np.asarray(cs.X, dtype=float) for cs in comp_sets])
            all_pure = np.all(np.max(comps, axis=1) > 1 - 1e-5)
            same_component = len(set(np.argmax(comps, axis=1))) == 1
            if all_pure and same_component:
                num_degenerate_points += 1
        # A real boundary may legitimately end at a pure-element melting point, so a
        # single point at the pure composition is fine, but a line tracking a
        # degenerate equilibrium along the pure-element edge is not
        assert num_degenerate_points <= 1, (
            f"ZPF line {zpf_line.stable_phases_with_multiplicity} has "
            f"{num_degenerate_points} points pinned at a pure-element composition"
        )

    # The real phase boundaries should be unaffected, including the ones that
    # terminate at the pure-element melting points
    def _composition_extent(zpf_line):
        comps = [
            point.get_local_property(cs, v.X("MG"))
            for point in zpf_line.points
            for cs in point.stable_composition_sets
        ]
        return np.nanmax(comps) - np.nanmin(comps)

    matching_lines = [zl for zl in strategy.zpf_lines if set(zl.stable_phases) == {"HCP_A3", "LIQUID"}]
    assert any(_composition_extent(zl) > 0.02 for zl in matching_lines), (
        "No non-degenerate HCP_A3+LIQUID ZPF line found"
    )

@select_database("BaCa-86Alc.tdb")
def test_circular_loop_check_normalizes_axes(load_database):
    """
    The liquidus must be traced past a congruent minimum, back up to the
    temperature it started at.

    Ba-Ca is isomorphous (BCC) with a congruent liquidus minimum at ~894 K. The
    BCC_A2+LIQUID ZPF line seeded at one pure-element melting point (Ba: ~1000 K,
    Ca: ~1115 K) traces down through the minimum and back up to the other
    element's melting point, so its temperature necessarily returns to its
    starting temperature partway along.

    check_circular_loop ends a line when it gets closer to its first point than
    to its previous point. With raw axis values, the kelvin scale of the
    temperature axis swamps the mole-fraction axis and the line is silently
    ended the moment its temperature comes back within one step of the starting
    temperature, truncating the liquidus at exactly the lower-melting element's
    melting point. Axis distances must be normalized for the check to only
    catch genuine loops.
    """
    dbf = load_database()
    conds = {v.P: 101325, v.N: 1, v.T: (700, 1250, 20), v.X("CA"): (0, 1, 0.05)}
    strategy = TielineStrategy(dbf, ["BA", "CA", "VA"], list(dbf.phases.keys()), conds)
    strategy.do_map()

    # Zero-extent 2-point stubs at the pure-element melting points (ended by the
    # degenerate tie-line check after a single step) are not the lens; only
    # substantial lines are held to the full-composition-range requirement
    liq_lines = []
    for zl in strategy.zpf_lines:
        if set(zl.stable_phases) == {"BCC_A2", "LIQUID"}:
            xs = [np.squeeze(pt.get_property(v.X("CA"))) for pt in zl.points]
            if np.max(xs) - np.min(xs) > 0.01:
                liq_lines.append((zl, xs))
    assert len(liq_lines) > 0, "No BCC_A2+LIQUID ZPF line mapped"
    for zl, xs in liq_lines:
        assert np.min(xs) < 0.05 and np.max(xs) > 0.95, (
            f"BCC_A2+LIQUID line truncated: x(CA) spans [{np.min(xs):.3f}, {np.max(xs):.3f}] "
            "instead of the full composition range"
        )

@select_database("GaLa-11Idb.tdb")
def test_edge_harvest_keeps_forced_starting_points(load_database):
    """
    Phase fields only recorded by a force-added recovery point in an edge step map
    must still be seeded.

    La has a narrow BCC window (1134-1194 K) between DHCP/FCC and melting. The
    coarse starting-point search along the x(LA)~1 edge (step = T-range/20 = 185 K)
    straddles the window, lands on a metastable FCC+LIQUID node, fails the exit
    direction test, and force-adds a recovery point (a parentless node) at the true
    LIQUID+BCC_A2 equilibrium inside the window. Harvesting only parented step-map
    nodes discards that recovery point, so the whole La-side BCC_A2+LIQUID boundary
    is lost and the La liquidus dead-ends below the window.
    """
    dbf = load_database()
    conds = {v.P: 101325, v.N: 1, v.T: (300, 4000, 20), v.X("LA"): (0.75, 1, 0.05)}
    strategy = TielineStrategy(dbf, ["GA", "LA", "VA"], list(dbf.phases.keys()), conds)
    strategy.do_map()

    for zpf_line in strategy.zpf_lines:
        if sorted(set(zpf_line.stable_phases)) == ["BCC_A2", "LIQUID"]:
            Ts = [np.squeeze(pt.get_property(v.T)) for pt in zpf_line.points]
            xs = [np.squeeze(pt.get_property(v.X("LA"))) for pt in zpf_line.points]
            if np.min(Ts) < 1195 and np.max(Ts) > 1140 and np.max(xs) > 0.9:
                break
    else:
        assert False, "No BCC_A2+LIQUID ZPF line mapped inside the La BCC window (1134-1194 K)"

@select_database("AuCu-98Sun-LB.tdb")
def test_edge_harvest_seeds_every_merged_field(load_database):
    """
    Every two-phase field merged into a single same-name edge step line must get its
    own starting point.

    In Au-Cu all fcc-ordered phases (fcc, AuCu3, AuCu, ...) share the FCC_4SL phase
    name, so the step map along the T=300 K edge merges several distinct two-phase
    fields into single step lines: the warm-started solver slides the same
    composition sets from one field into the next without creating a phase-change
    node. Harvesting one starting point per step line then seeds only one of the
    merged fields; the Au-rich fcc+AuCu3 field (tie-lines spanning x(CU)=[0.078,
    0.237] at 300 K) is fully present in the step results but was never traced.
    Harvesting one starting point per contiguous segment (tie-line endpoints are
    constant within a field at fixed potentials, so a jump marks a new field)
    recovers it.
    """
    dbf = load_database()
    conds = {v.P: 101325, v.N: 1, v.T: (300, 700, 20), v.X("CU"): (0, 1, 0.05)}
    strategy = TielineStrategy(dbf, ["AU", "CU", "VA"], list(dbf.phases.keys()), conds)
    strategy.do_map()

    for zpf_line in strategy.zpf_lines:
        xs = [np.squeeze(pt.get_property(v.X("CU"))) for pt in zpf_line.points]
        if np.min(xs) < 0.15 and np.max(xs) > 0.20:
            break
    else:
        assert False, "Au-rich fcc+AuCu3 field (x(CU)~0.08-0.24) was never traced"

@select_database("CrTa-93Dup-LB.tdb")
def test_starting_point_dedup_distinguishes_twin_fields(load_database):
    """
    Two distinct two-phase fields with the same phase pair flanking a line compound
    must both be traced.

    In Cr-Ta, BCC_A2+C15_LAVES fields exist on both sides of the CR2TA_C15 line
    compound. The x-edge harvested starting point for the Cr-rich field sits (in
    condition space) within a fraction of a step of the already-traced Ta-rich
    field's line at the compound composition, so a coverage check based on
    condition-space position alone discards it and the whole Cr-rich low-T field
    (x < 0.03, 300-849 K) is lost. Coverage must compare tie-lines (potential
    coordinates plus phase compositions), which differ between the two fields by
    a mole fraction of ~0.3.
    """
    dbf = load_database()
    conds = {v.P: 101325, v.N: 1, v.T: (300, 4000, 20), v.X("TA"): (0, 1, 0.05)}
    strategy = TielineStrategy(dbf, ["CR", "TA", "VA"], list(dbf.phases.keys()), conds)
    strategy.do_map()

    for zpf_line in strategy.zpf_lines:
        if sorted(set(zpf_line.stable_phases)) == ["BCC_A2", "C15_LAVES"]:
            Ts = [np.squeeze(pt.get_property(v.T)) for pt in zpf_line.points]
            xs = [np.squeeze(pt.get_property(v.X("TA"))) for pt in zpf_line.points]
            if np.min(xs) < 0.05 and np.min(Ts) < 350:
                break
    else:
        assert False, "Cr-rich BCC_A2+C15_LAVES field (x(TA)<0.05) was not traced down to 300 K"

@select_database("BaCa-86Alc.tdb")
def test_starting_point_coverage_semantics(load_database):
    """
    The coverage test used to skip redundant harvested starting points must:
    - treat a point deep on a traced line as covered (that is its purpose), and
    - NOT treat the start region of a line as covering: the opposite-direction
      sibling of a starting point sits at (or within a refined first step of) the
      line's first point, and skipping it would lose the whole other side of the
      boundary (e.g. the entire lens below a top-edge starting point).
    """
    dbf = load_database()
    conds = {v.P: 101325, v.N: 1, v.T: (700, 1250, 20), v.X("CA"): (0, 1, 0.05)}
    strategy = TielineStrategy(dbf, ["BA", "CA", "VA"], list(dbf.phases.keys()), conds)
    strategy.do_map()

    long_lines = [zl for zl in strategy.zpf_lines if len(zl.points) > 10]
    assert len(long_lines) > 0
    zpf_line = long_lines[0]
    # a point in the start region must not count as covered
    assert not strategy._point_on_existing_zpf_line(zpf_line.points[1]), (
        "Start region of a line must not cover its opposite-direction sibling"
    )
    # a point deep on the line is covered
    assert strategy._point_on_existing_zpf_line(zpf_line.points[8]), (
        "A point on an already-traced tie-line should be reported as covered"
    )

@select_database("CaMg-06Zho.tdb")
def test_no_duplicate_boundary_retracing(load_database):
    """
    Recovery starting points harvested from the edge step maps must not re-trace a
    boundary that is already mapped.

    In Ca-Mg the CAMG2_C14+FCC_A1 boundary gets one seed from the T=300 K edge and
    additional parentless recovery seeds at the x(MG)=0 edge; without tie-line-based
    coverage checking the same boundary is traced up to five times (visible as
    interleaved duplicate tie-lines in the plot, since the re-traces start at
    off-grid temperatures).
    """
    dbf = load_database()
    conds = {v.P: 101325, v.N: 1, v.T: (300, 4000, 20), v.X("MG"): (0, 1, 0.05)}
    strategy = TielineStrategy(dbf, ["CA", "MG", "VA"], list(dbf.phases.keys()), conds)
    strategy.do_map()

    target_lines = [zl for zl in strategy.zpf_lines
                    if sorted(set(zl.stable_phases)) == ["CAMG2_C14", "FCC_A1"] and len(zl.points) > 1]
    assert 1 <= len(target_lines) <= 2, (
        f"CAMG2_C14+FCC_A1 traced {len(target_lines)} times; expected at most 2 segments"
    )
    # deduplication must not cost coverage: the boundary still spans 300 K up to
    # the ~710 K invariant
    Ts = [np.squeeze(pt.get_property(v.T)) for zl in target_lines for pt in zl.points]
    assert np.min(Ts) < 310 and np.max(Ts) > 700

@select_database("CrV-92Lee-LB.tdb")
def test_degenerate_check_does_not_veto_narrow_lens(load_database):
    """
    A genuinely narrow melting lens must still be traced.

    Cr-V is isomorphous with a solidus-liquidus lens only a few kelvin tall; near
    the pure-element edges its tie-line width is genuinely below the degenerate
    zero-width tolerance. Running the degenerate tie-line check inside the exit
    direction test (whose trial step uses the minimum delta, right at the edge)
    vetoes both directions from the edge melting nodes, so the entire lens - and
    with it the whole diagram - is lost. The direction test must not apply the
    degenerate check; line tracing's own check still ends truly degenerate lines
    one step later.
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
        assert False, "BCC_A2+LIQUID melting lens was not traced across the composition range"

@select_database("CrPt-98Spe-LB.tdb")
def test_degenerate_check_does_not_end_line_at_congruent_extremum(load_database):
    """
    A boundary crossing a congruent extremum must be traced through it.

    In Cr-Pt the fcc liquidus passes over a congruent maximum at x(PT)~0.78,
    2058 K, where the tie-line width passes continuously through zero. At a fine
    composition step the degenerate tie-line check landed a point inside the
    (genuinely) sub-tolerance pinch and ended the line - with no node and no
    restart - losing the entire Pt-rich liquidus down to the pure-Pt melting
    point. A line whose composition is still advancing step over step is crossing
    a pinch, not tracking a degenerate boundary (those are pinned at a fixed
    composition), and must not be ended.
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
        assert False, "FCC_A1+LIQUID liquidus was not traced past the congruent maximum to the Pt side"

@select_database("Al-Cu-Y.tdb")
def test_issue_662_phase_boundary_loop(load_database):
    T = 2260
    dbf = load_database()
    comps = ['AL', 'CU', 'Y', 'VA']
    phases = list(dbf.phases.keys())
    conds = {v.T: T, v.P:101325, v.X('AL'): (0,1,0.015), v.X('Y'): (0,1,0.015)}
    strat = TernaryStrategy(dbf, comps, phases, conds)
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
