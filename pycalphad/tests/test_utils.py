"""
The utils test module contains tests for pycalphad utilities.
"""

import pytest
from importlib.resources import files
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


def test_filter_ordered_disordered_phases_with_unmatched_constituents():
    # Ordered phase has more constituents than disordered phase
    TDB = """
    ELEMENT A   PHASE             0.0                0.00            0.00      !
    ELEMENT B   PHASE             0.0                0.00            0.00      !
    ELEMENT C   PHASE             0.0                0.00            0.00      !
    ELEMENT D   PHASE             0.0                0.00            0.00      !
    ELEMENT VA  VACUUM            0.0                0.00            0.00      !
    TYPE_DEFINITION X GES AMEND_PHASE_DESCRIPTION BCC_B2 DIS_PART BCC_A2 !
    PHASE BCC_B2 X  3 0.5 0.5   3 !
    CONSTITUENT BCC_B2  : A,B,C,D : A,B,C,D : VA : !
    PHASE BCC_A2  X  2 1   3 !
    CONSTITUENT BCC_A2  :A,B,C : VA :  !
    """

    # Ordered phase has different (but same length) set of constituents
    TDB2 = """
    ELEMENT A   PHASE             0.0                0.00            0.00      !
    ELEMENT B   PHASE             0.0                0.00            0.00      !
    ELEMENT C   PHASE             0.0                0.00            0.00      !
    ELEMENT D   PHASE             0.0                0.00            0.00      !
    ELEMENT VA  VACUUM            0.0                0.00            0.00      !
    TYPE_DEFINITION X GES AMEND_PHASE_DESCRIPTION BCC_B2 DIS_PART BCC_A2 !
    PHASE BCC_B2 X  3 0.5 0.5   3 !
    CONSTITUENT BCC_B2  : B,C,D : B,C,D : VA : !
    PHASE BCC_A2  X  2 1   3 !
    CONSTITUENT BCC_A2  :A,B,C : VA :  !
    """

    # Ordered phase has less constituents than disordered phase
    TDB3 = """
    ELEMENT A   PHASE             0.0                0.00            0.00      !
    ELEMENT B   PHASE             0.0                0.00            0.00      !
    ELEMENT C   PHASE             0.0                0.00            0.00      !
    ELEMENT D   PHASE             0.0                0.00            0.00      !
    ELEMENT VA  VACUUM            0.0                0.00            0.00      !
    TYPE_DEFINITION X GES AMEND_PHASE_DESCRIPTION BCC_B2 DIS_PART BCC_A2 !
    PHASE BCC_B2 X  3 0.5 0.5   3 !
    CONSTITUENT BCC_B2  : A,B,C : A,B,C : VA : !
    PHASE BCC_A2  X  2 1   3 !
    CONSTITUENT BCC_A2  :A,B,C,D : VA :  !
    """
    from pycalphad import variables as v

    dbf1 = Database(TDB)
    dbf2 = Database(TDB2)
    dbf3 = Database(TDB3)

    # dbf1 gives BCC_B2 with [A, B, C, VA] as components
    # dbf2 gives BCC_A2 with [A, B, C, VA] as components
    # dbf3 gives BCC_B2 with [A, B, C, VA] as components
    comps = ['A', 'B', 'C', 'VA']

    phases1 = filter_phases(dbf1, unpack_components(dbf1, comps))
    model1 = Model(dbf1, comps, phases1[0])
    assert len(phases1) == 1 and 'BCC_B2' in phases1
    assert len(set([v.Species('A'), v.Species('B'), v.Species('C'), v.Species('VA')]).symmetric_difference(model1.components)) == 0

    phases2 = filter_phases(dbf2, unpack_components(dbf2, comps))
    model2 = Model(dbf2, comps, phases2[0])
    assert len(phases2) == 1 and 'BCC_A2' in phases2
    assert len(set([v.Species('A'), v.Species('B'), v.Species('C'), v.Species('VA')]).symmetric_difference(model2.components)) == 0

    phases3 = filter_phases(dbf3, unpack_components(dbf3, comps))
    model3 = Model(dbf3, comps, phases3[0])
    assert len(phases3) == 1 and 'BCC_B2' in phases3
    assert len(set([v.Species('A'), v.Species('B'), v.Species('C'), v.Species('VA')]).symmetric_difference(model3.components)) == 0

    # dbf1 gives BCC_B2 with [A, B, C, VA] as components
    # dbf2 gives BCC_A2 with [A, B, C, VA] as components
    # dbf3 gives BCC_A2 with [A, B, C, D, VA] as components
    comps = ['A', 'B', 'C', 'D', 'VA']

    phases1 = filter_phases(dbf1, unpack_components(dbf1, comps))
    model1 = Model(dbf1, comps, phases1[0])
    assert len(phases1) == 1 and 'BCC_B2' in phases1
    assert len(set([v.Species('A'), v.Species('B'), v.Species('C'), v.Species('VA')]).symmetric_difference(model1.components)) == 0

    phases2 = filter_phases(dbf2, unpack_components(dbf2, comps))
    model2 = Model(dbf2, comps, phases2[0])
    assert len(phases2) == 1 and 'BCC_A2' in phases2
    assert len(set([v.Species('A'), v.Species('B'), v.Species('C'), v.Species('VA')]).symmetric_difference(model2.components)) == 0

    phases3 = filter_phases(dbf3, unpack_components(dbf3, comps))
    model3 = Model(dbf3, comps, phases3[0])
    assert len(phases3) == 1 and 'BCC_A2' in phases3
    assert len(set([v.Species('A'), v.Species('B'), v.Species('C'), v.Species('D'), v.Species('VA')]).symmetric_difference(model3.components)) == 0

    # dbf1 gives BCC_B2 with [B, C, VA] as components
    # dbf2 gives BCC_B2 with [B, C, VA] as components
    # dbf3 gives BCC_A2 with [B, C, D, VA] as components
    comps = ['B', 'C', 'D', 'VA']

    phases1 = filter_phases(dbf1, unpack_components(dbf1, comps))
    model1 = Model(dbf1, comps, phases1[0])
    assert len(phases1) == 1 and 'BCC_B2' in phases1
    assert len(set([v.Species('B'), v.Species('C'), v.Species('VA')]).symmetric_difference(model1.components)) == 0

    phases2 = filter_phases(dbf2, unpack_components(dbf2, comps))
    model2 = Model(dbf2, comps, phases2[0])
    assert len(phases2) == 1 and 'BCC_B2' in phases2
    assert len(set([v.Species('B'), v.Species('C'), v.Species('VA')]).symmetric_difference(model2.components)) == 0

    phases3 = filter_phases(dbf3, unpack_components(dbf3, comps))
    model3 = Model(dbf3, comps, phases3[0])
    assert len(phases3) == 1 and 'BCC_A2' in phases3
    assert len(set([v.Species('B'), v.Species('C'), v.Species('D'), v.Species('VA')]).symmetric_difference(model3.components)) == 0
