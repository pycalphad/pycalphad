import copy
from importlib.resources import files

import numpy as np
import matplotlib.pyplot as plt

from pycalphad import binplot, ternplot, stepplot, isoplethplot, Database, variables as v
from pycalphad.tests.fixtures import select_database, load_database

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
    #dbf = Database(str(files(pycalphad.tests.databases).joinpath('alcocrni.tdb')))

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
    #plt.show()

@select_database("crtiv_ghosh.tdb")
def test_ternary_strategy(load_database):
    dbf = load_database()
    #dbf = Database(str(files(pycalphad.tests.databases).joinpath('crtiv_ghosh.tdb')))

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
    #plt.show()

@select_database("alcocrni.tdb")
def test_step_strategy(load_database):
    dbf = load_database()
    #dbf = Database(str(files(pycalphad.tests.databases).joinpath('alcocrni.tdb')))

    ax, strategy = stepplot(dbf, ["CR", "NI", "VA"], None, conditions={v.T: (1000, 2500, 40), v.X("CR"): 0.8, v.P: 101325}, return_strategy=True)
    
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
    #plt.show()

@select_database("crtiv_ghosh.tdb")
def test_isopleth_strategy(load_database):
    dbf = load_database()
    #dbf = Database(str(files(pycalphad.tests.databases).joinpath('crtiv_ghosh.tdb')))

    ax, strategy = isoplethplot(dbf, ["CR", "TI", "V", "VA"], None, conditions={v.T: (1073, 2073, 20), v.X("TI"): (0, 0.8, 0.01), v.X("V"): 0.2, v.P: 101325}, return_strategy=True)
    
    # Two-phase regions intended to show up in the Cr-Ni system
    desired_zpf_sets = [{"BCC_A2", "LAVES_C15"}, {"BCC_A2", "LIQUID"}]

    # All unique phase regions
    mapping_sets = [set(zpf_line.stable_phases_with_multiplicity) for zpf_line in strategy.zpf_lines]
    #node_sets = [set(node.stable_phases_with_multiplicity) for node in strategy.node_queue.nodes]

    # Make sure that the phase regions from mapping contains all the desired regions
    # NOTE: this will not test for extra phase regions that mapping may produce
    for dzs in desired_zpf_sets:
        assert dzs in mapping_sets
    #plt.show()