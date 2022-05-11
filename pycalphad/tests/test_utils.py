"""
The utils test module contains tests for pycalphad utilities.
"""

import pytest
from importlib_resources import files
from pycalphad import Database, Model
from pycalphad.core.utils import filter_phases, unpack_components, instantiate_models, generate_symmetric_group
import pycalphad.tests.databases
from pycalphad.tests.fixtures import select_database, load_database


@select_database("alcocrni.tdb")
def test_filter_phases_removes_disordered_phases_from_order_disorder(load_database):
    """Databases with order-disorder models should have the disordered phases be filtered if candidate_phases kwarg is not passed to filter_phases.
    If candidate_phases kwarg is passed, disordered phases just are filtered if respective ordered phases are inactive"""
    dbf = load_database()
    ALNIPT_DBF = Database(str(files(pycalphad.tests.databases).joinpath("alnipt.tdb")))
    ALCRNI_DBF = Database(str(files(pycalphad.tests.databases).joinpath("alcrni.tdb")))
    all_phases = set(ALNIPT_DBF.phases.keys())
    filtered_phases = set(filter_phases(ALNIPT_DBF, unpack_components(ALNIPT_DBF, ['AL', 'NI', 'PT', 'VA'])))
    assert all_phases.difference(filtered_phases) == {'FCC_A1'}
    comps = unpack_components(ALCRNI_DBF, ['NI', 'AL', 'CR', 'VA'])
    filtered_phases = set(filter_phases(ALCRNI_DBF, comps, ['FCC_A1', 'L12_FCC', 'LIQUID', 'BCC_A2']))
    assert filtered_phases == {'L12_FCC', 'LIQUID', 'BCC_A2'}
    filtered_phases = set(filter_phases(ALCRNI_DBF, comps, ['FCC_A1', 'LIQUID', 'BCC_A2']))
    assert filtered_phases == {'FCC_A1', 'LIQUID', 'BCC_A2'}
    filtered_phases = set(filter_phases(ALCRNI_DBF, comps, ['FCC_A1']))
    assert filtered_phases == {'FCC_A1'}
    # Test that phases are removed if there are no ordered/disorder model hints on the disordered configuration
    filtered_phases = set(filter_phases(dbf, unpack_components(dbf, ['AL', 'NI', 'VA']), ['BCC_A2', 'BCC_B2']))
    assert filtered_phases == {'BCC_B2'}


@select_database("alnipt.tdb")
def test_filter_phases_removes_phases_with_inactive_sublattices(load_database):
    """Phases that have no active components in any sublattice should be filtered"""
    dbf = load_database()
    all_phases = set(dbf.phases.keys())
    filtered_phases = set(filter_phases(dbf, unpack_components(dbf, ['AL', 'NI', 'VA'])))
    assert all_phases.difference(filtered_phases) == {'FCC_A1', 'PT8AL21', 'PT5AL21', 'PT2AL', 'PT2AL3', 'PT5AL3', 'ALPT2'}


@select_database("alnipt.tdb")
def test_instantiate_models_only_returns_desired_phases(load_database):
    """instantiate_models should only return phases passed"""
    dbf = load_database()
    comps = ['AL', 'NI', 'VA']
    phases = ['FCC_A1', 'LIQUID']

    # models are overspecified w.r.t. phases
    too_many_phases = ['FCC_A1', 'LIQUID', 'AL3NI1']
    too_many_models = {phase: Model(dbf, comps, phase) for phase in too_many_phases}
    inst_mods = instantiate_models(dbf, comps, phases, model=too_many_models)
    assert len(inst_mods) == len(phases)

    # models are underspecified w.r.t. phases
    too_few_phases = ['FCC_A1']
    too_few_models = {phase: Model(dbf, comps, phase) for phase in too_few_phases}
    inst_mods = instantiate_models(dbf, comps, phases, model=too_few_models)
    assert len(inst_mods) == len(phases)


def test_symmetric_group_can_be_generated_for_2_sl_mixing_with_symmetry():
    """A phase with two sublattices that are mixing should generate a cross interaction"""
    symm_groups = generate_symmetric_group((('AL', 'CO'), ('AL', 'CO')), [[0, 1]])
    assert symm_groups == [(('AL', 'CO'), ('AL', 'CO'))]


def test_symmetric_group_can_be_generated_for_2_sl_endmembers_with_symmetry():
    """A phase with symmetric sublattices should find a symmetric endmember """
    symm_groups = generate_symmetric_group(('AL', 'CO'), [[0, 1]])
    assert symm_groups == [('AL', 'CO'), ('CO', 'AL')]


def test_generating_symmetric_group_works_without_symmetry():
    """generate_symmetric_group returns the passed configuration if symmetry=None"""

    config_D03_A3B = ["A", "A", "A", "B"]
    symm_groups = generate_symmetric_group(config_D03_A3B, None)
    assert symm_groups == [("A", "A", "A", "B")]

    symm_groups = generate_symmetric_group((("CR", "FE"), "VA"), None)
    assert symm_groups == [
        (("CR", "FE"), "VA")
    ]


def test_generating_symmetric_group_bcc_4sl():
    """Binary BCC 4SL ordered symmetric configurations can be generated"""
    bcc_4sl_symmetry = [[0, 1], [2, 3]]

    config_D03_A3B = ["A", "A", "A", "B"]
    symm_groups = generate_symmetric_group(config_D03_A3B, bcc_4sl_symmetry)
    assert symm_groups == [
        ("A", "A", "A", "B"),
        ("A", "A", "B", "A"),
        ("A", "B", "A", "A"),
        ("B", "A", "A", "A"),
    ]

    config_B2_A2B2 = ["A", "A", "B", "B"]
    symm_groups = generate_symmetric_group(config_B2_A2B2, bcc_4sl_symmetry)
    assert symm_groups == [
        ("A", "A", "B", "B"),
        ("B", "B", "A", "A"),
    ]

    config_B32_A2B2 = ["A", "B", "A", "B"]
    symm_groups = generate_symmetric_group(config_B32_A2B2, bcc_4sl_symmetry)
    assert symm_groups == [
        ("A", "B", "A", "B"),
        ("A", "B", "B", "A"),
        ("B", "A", "A", "B"),
        ("B", "A", "B", "A"),
    ]


def test_generating_symmetric_group_fcc_4sl():
    """Binary FCC 4SL ordered symmetric configurations can be generated"""
    fcc_4sl_symmetry = [[0, 1, 2, 3]]

    config_L1_2_A3B = ["A", "A", "A", "B"]
    symm_groups = generate_symmetric_group(config_L1_2_A3B, fcc_4sl_symmetry)
    assert symm_groups == [
        ("A", "A", "A", "B"),
        ("A", "A", "B", "A"),
        ("A", "B", "A", "A"),
        ("B", "A", "A", "A"),
    ]

    config_L1_0_A2B2 = ["A", "A", "B", "B"]
    symm_groups = generate_symmetric_group(config_L1_0_A2B2, fcc_4sl_symmetry)
    assert symm_groups == [
        ("A", "A", "B", "B"),
        ("A", "B", "A", "B"),
        ("A", "B", "B", "A"),
        ("B", "A", "A", "B"),
        ("B", "A", "B", "A"),
        ("B", "B", "A", "A"),
    ]


def test_generating_symmetric_group_works_with_interstitial_sublattice():
    """Symmetry groups for phases with an inequivalent vacancy sublattice are correctly generated"""
    bcc_4sl_symmetry = [[0, 1], [2, 3]]
    config_D03_A3B = ["A", "A", "A", "B", "VA"]
    symm_groups = generate_symmetric_group(config_D03_A3B, bcc_4sl_symmetry)
    assert symm_groups == [
        ("A", "A", "A", "B", "VA"),
        ("A", "A", "B", "A", "VA"),
        ("A", "B", "A", "A", "VA"),
        ("B", "A", "A", "A", "VA"),
    ]

    fcc_4sl_symmetry = [[0, 1, 2, 3]]
    config_L1_2_A3B = ["A", "A", "A", "B", "VA"]
    symm_groups = generate_symmetric_group(config_L1_2_A3B, fcc_4sl_symmetry)
    assert symm_groups == [
        ("A", "A", "A", "B", "VA"),
        ("A", "A", "B", "A", "VA"),
        ("A", "B", "A", "A", "VA"),
        ("B", "A", "A", "A", "VA"),
    ]

    # "Unrealistic" cases where the vacancy sublattice is in the middle at index 2
    bcc_4sl_symmetry = [[0, 1], [3, 4]]
    config_D03_A3B = ["A", "A", "VA", "A", "B"]
    symm_groups = generate_symmetric_group(config_D03_A3B, bcc_4sl_symmetry)
    assert symm_groups == [
        ("A", "A", "VA", "A", "B"),
        ("A", "A", "VA", "B", "A"),
        ("A", "B", "VA", "A", "A"),
        ("B", "A", "VA", "A", "A"),
    ]

    fcc_4sl_symmetry = [[0, 1, 3, 4]]
    config_L1_2_A3B = ["A", "A", "VA", "A", "B"]
    symm_groups = generate_symmetric_group(config_L1_2_A3B, fcc_4sl_symmetry)
    assert symm_groups == [
        ("A", "A", "VA", "A", "B"),
        ("A", "A", "VA", "B", "A"),
        ("A", "B", "VA", "A", "A"),
        ("B", "A", "VA", "A", "A"),
    ]
