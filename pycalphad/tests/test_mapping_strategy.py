import copy
from importlib.resources import files

import numpy as np
import matplotlib.pyplot as plt

from pycalphad import binplot, ternplot, Database, variables as v
from pycalphad.tests.fixtures import select_database, load_database
from pycalphad.core.utils import instantiate_models, get_state_variables
from pycalphad.codegen.phase_record_factory import PhaseRecordFactory
from pycalphad.core.composition_set import CompositionSet

from pycalphad.mapping import StepStrategy, IsoplethStrategy, BinaryStrategy, TernaryStrategy, plot_step, plot_isopleth
from pycalphad.mapping.starting_points import point_from_equilibrium
from pycalphad.mapping.zpf_equilibrium import find_global_min_point
from pycalphad.mapping.primitives import Point, Node, Direction, ZPFLine, ZPFState
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

    strategy = IsoplethStrategy(dbf, ["CR", "TI", "V", "VA"], ["BCC_A2", "LIQUID"], conditions={v.T: (1500, 2100, 40), v.X("TI"): (0, 0.2, 0.05), v.X("V"): 0.2, v.P: 101325})
    strategy.do_map()

    # Check that plot_isopleth runs without fail
    plot_isopleth(strategy)
    #plt.show()

    # Two-phase regions intended to show up in the Cr-Ti-V system
    desired_zpf_sets = [{"BCC_A2", "LIQUID"}]

    # All unique phase regions
    mapping_sets = [set(zpf_line.stable_phases_with_multiplicity) for zpf_line in strategy.zpf_lines]

    # Make sure that the phase regions from mapping contains all the desired regions
    # NOTE: this will not test for extra phase regions that mapping may produce
    for dzs in desired_zpf_sets:
        assert dzs in mapping_sets

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
        a) if node is metastable, do not add to node queue and remove zpf line
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

    # In _process_new_node, this will fail the global min check and remove the zpf line
    num_nodes = len(strategy.node_queue.nodes)
    strategy._process_new_node(strategy.zpf_lines[0], metastable_node)
    # Test that zpf line was removed for being a metastable node
    assert len(strategy.zpf_lines) == 0
    assert len(strategy.node_queue.nodes) == num_nodes

    # _process_new_node with correct/stable node
    # This will pass the _check_full_global_equilibrium test and the node will be added
    # to the node queue
    eq_conds = {v.T: 1323, v.P: 101325, v.N: 1, v.X('CR'): 0.1, v.X('FE'): 0.35}
    stable_point = point_from_equilibrium(strategy.dbf, strategy.components, strategy.phases, eq_conds, models=strategy.models, phase_record_factory=strategy.phase_records)
    stable_node = Node(stable_point.global_conditions, stable_point.chemical_potentials, [], stable_point.stable_composition_sets, None)

    strategy.zpf_lines.append(ZPFLine([], point.stable_phases))
    strategy.zpf_lines[0].axis_var = v.X('FE')
    strategy.zpf_lines[0].axis_direction = Direction.POSITIVE

    num_nodes = len(strategy.node_queue.nodes)
    num_zpf_lines = len(strategy.zpf_lines)
    strategy._process_new_node(strategy.zpf_lines[0], stable_node)
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

    # First point in the first zpf line should be a node
    zpf_line = strategy.zpf_lines[0]
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

