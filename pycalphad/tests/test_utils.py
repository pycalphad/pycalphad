"""
The utils test module contains tests for pycalphad utilities.
"""

import pytest
from importlib.resources import files
from pycalphad import Database, Model
from pycalphad.core.utils import filter_phases, unpack_species, instantiate_models, generate_symmetric_group
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
    filtered_phases = set(filter_phases(ALNIPT_DBF, unpack_species(ALNIPT_DBF, ['AL', 'NI', 'PT', 'VA'])))
    assert all_phases.difference(filtered_phases) == {'FCC_A1'}
    comps = unpack_species(ALCRNI_DBF, ['NI', 'AL', 'CR', 'VA'])
    filtered_phases = set(filter_phases(ALCRNI_DBF, comps, ['FCC_A1', 'L12_FCC', 'LIQUID', 'BCC_A2']))
    assert filtered_phases == {'L12_FCC', 'LIQUID', 'BCC_A2'}
    filtered_phases = set(filter_phases(ALCRNI_DBF, comps, ['FCC_A1', 'LIQUID', 'BCC_A2']))
    assert filtered_phases == {'FCC_A1', 'LIQUID', 'BCC_A2'}
    filtered_phases = set(filter_phases(ALCRNI_DBF, comps, ['FCC_A1']))
    assert filtered_phases == {'FCC_A1'}
    # Test that phases are removed if there are no ordered/disorder model hints on the disordered configuration
    filtered_phases = set(filter_phases(dbf, unpack_species(dbf, ['AL', 'NI', 'VA']), ['BCC_A2', 'BCC_B2']))
    assert filtered_phases == {'BCC_B2'}


@select_database("alnipt.tdb")
def test_filter_phases_removes_phases_with_inactive_sublattices(load_database):
    """Phases that have no active components in any sublattice should be filtered"""
    dbf = load_database()
    all_phases = set(dbf.phases.keys())
    filtered_phases = set(filter_phases(dbf, unpack_species(dbf, ['AL', 'NI', 'VA'])))
    assert all_phases.difference(filtered_phases) == {'FCC_A1', 'PT8AL21', 'PT5AL21', 'PT2AL', 'PT2AL3', 'PT5AL3', 'ALPT2'}


def test_filter_phases_removes_pure_vacancy_phases():
    """Phases where only vacancies are active on every sublattice should be filtered.

    Model raises DofError for such phases, so leaving them active breaks
    calculate/equilibrium/Workspace with default phase lists.
    """
    tdb = """
    ELEMENT AL FCC_A1 26.98 4577.3 28.32 !
    ELEMENT VA VACUUM 0.0 0.0 0.0 !
    TYPE_DEFINITION % SEQ * !
    PHASE FCC_A1 % 2 1.0 1.0 !
    CONSTITUENT FCC_A1 :AL:VA: !
    PHASE PURE_VA % 1 1.0 !
    CONSTITUENT PURE_VA :VA: !
    PHASE PURE_VA_2SL % 2 1.0 3.0 !
    CONSTITUENT PURE_VA_2SL :VA:VA: !
    PHASE AL_VA_MIX % 2 1.0 1.0 !
    CONSTITUENT AL_VA_MIX :AL,VA:VA: !
    PARAMETER G(FCC_A1,AL:VA;0) 298.15 -10000; 6000 N !
    PARAMETER G(PURE_VA,VA;0) 298.15 0; 6000 N !
    PARAMETER G(PURE_VA_2SL,VA:VA;0) 298.15 0; 6000 N !
    PARAMETER G(AL_VA_MIX,AL:VA;0) 298.15 -5000; 6000 N !
    PARAMETER G(AL_VA_MIX,VA:VA;0) 298.15 0; 6000 N !
    """
    dbf = Database(tdb)
    # Phases mixing atoms and vacancies stay; phases with only vacancies active are removed
    filtered_phases = set(filter_phases(dbf, unpack_species(dbf, ['AL', 'VA'])))
    assert filtered_phases == {'FCC_A1', 'AL_VA_MIX'}
    # Every phase that passes the filter can build a Model
    instantiate_models(dbf, ['AL', 'VA'], sorted(filtered_phases))
    # With AL inactive, all phases with atoms lose their active sublattices too
    assert filter_phases(dbf, unpack_species(dbf, ['VA'])) == []


@select_database("CoV-20Wan.tdb")
def test_filter_phases_removes_disordered_part_of_never_disorder_models(load_database):
    """Phases that are the disordered part of a never disorder model should be removed"""
    dbf = load_database()
    all_phases = set(dbf.phases.keys())
    filtered_phases = set(filter_phases(dbf, unpack_species(dbf, ['CO', 'V', 'VA'])))

    # A1_FCC and DIS_SIGMA are disordered parts and should be removed
    assert all_phases.difference(filtered_phases) == {"A1_FCC", "DIS_SIGMA"}


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
    # Valid TDB
    TDB_VALID = """
    ELEMENT A   PHASE             0.0                0.00            0.00      !
    ELEMENT B   PHASE             0.0                0.00            0.00      !
    ELEMENT C   PHASE             0.0                0.00            0.00      !
    ELEMENT D   PHASE             0.0                0.00            0.00      !
    ELEMENT VA  VACUUM            0.0                0.00            0.00      !
    TYPE_DEFINITION X GES AMEND_PHASE_DESCRIPTION BCC_B2 DIS_PART BCC_A2 !
    PHASE BCC_B2 X  3 0.5 0.5   3 !
    CONSTITUENT BCC_B2  : A,B,C,D : A,B,C,D : VA : !
    PHASE BCC_A2  X  2 1   3 !
    CONSTITUENT BCC_A2  :A,B,C,D : VA :  !

    TYPE_DEFINITION Y GES AMEND_PHASE_DESCRIPTION FCC_L12 DIS_PART FCC_A1 !
    PHASE FCC_L12 Y  2 0.75 0.25 !
    CONSTITUENT FCC_L12  : A,B,C,D : A,B,C,D : !
    PHASE FCC_A1  Y  1 1 !
    CONSTITUENT FCC_A1  :A,B,C,D :  !
    """

    # Ordered phase is superset of disordered phase
    # if active components is [A, B, C], then this is valid
    # if active components include [D], this should raise
    # ValueError since energy of D in disordered is undefined
    TDB_O_SUPERSET = """
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

    TYPE_DEFINITION Y GES AMEND_PHASE_DESCRIPTION FCC_L12 DIS_PART FCC_A1 !
    PHASE FCC_L12 Y  2 0.75 0.25 !
    CONSTITUENT FCC_L12  : A,B,C,D : A,B,C,D : !
    PHASE FCC_A1  Y  1 1 !
    CONSTITUENT FCC_A1  :A,B,C :  !
    """

    # Ordered phase has different (but same length) set of constituents
    # if active components is [B, C], then this is valid and no changes to ordered
    # if active components is [A, B, C], then this is valid and ordered will add a 0 term for A
    # if active components include [D], then this should raise
    # ValueError, since energy of D in disordered in undefined
    TDB_O_DIFFERENT = """
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

    TYPE_DEFINITION Y GES AMEND_PHASE_DESCRIPTION FCC_L12 DIS_PART FCC_A1 !
    PHASE FCC_L12 Y  2 0.75 0.25 !
    CONSTITUENT FCC_L12  : B,C,D : B,C,D : !
    PHASE FCC_A1  Y  1 1 !
    CONSTITUENT FCC_A1  :A,B,C :  !
    """

    # Disordered phase is superset of ordered phase
    # if active components is [A, B, C], then this is valid and no changes to ordered
    # if active components include [D], then this should extended ordered phase to include D
    TDB_D_SUPERSET = """
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

    TYPE_DEFINITION Y GES AMEND_PHASE_DESCRIPTION FCC_L12 DIS_PART FCC_A1 !
    PHASE FCC_L12 Y  2 0.75 0.25 !
    CONSTITUENT FCC_L12  : A,B,C : A,B,C : !
    PHASE FCC_A1  Y  1 1 !
    CONSTITUENT FCC_A1  :A,B,C,D :  !
    """

    # different interstitial sublattice
    # this will throw a value error during filter phases
    TDB_I_DIFFERENT = """
    ELEMENT A   PHASE             0.0                0.00            0.00      !
    ELEMENT B   PHASE             0.0                0.00            0.00      !
    ELEMENT C   PHASE             0.0                0.00            0.00      !
    ELEMENT D   PHASE             0.0                0.00            0.00      !
    ELEMENT E   PHASE             0.0                0.00            0.00      !
    ELEMENT VA  VACUUM            0.0                0.00            0.00      !
    TYPE_DEFINITION X GES AMEND_PHASE_DESCRIPTION BCC_B2 DIS_PART BCC_A2 !
    PHASE BCC_B2 X  3 0.5 0.5   3 !
    CONSTITUENT BCC_B2  : A,B,C,D : A,B,C,D : VA : !
    PHASE BCC_A2  X  2 1   3 !
    CONSTITUENT BCC_A2  :A,B,C,D : E :  !
    """

    # different interstitial sublattice
    # this will throw a value error during model construction
    TDB_I_D_SUPERSET = """
    ELEMENT A   PHASE             0.0                0.00            0.00      !
    ELEMENT B   PHASE             0.0                0.00            0.00      !
    ELEMENT C   PHASE             0.0                0.00            0.00      !
    ELEMENT D   PHASE             0.0                0.00            0.00      !
    ELEMENT E   PHASE             0.0                0.00            0.00      !
    ELEMENT VA  VACUUM            0.0                0.00            0.00      !
    TYPE_DEFINITION X GES AMEND_PHASE_DESCRIPTION BCC_B2 DIS_PART BCC_A2 !
    PHASE BCC_B2 X  3 0.5 0.5   3 !
    CONSTITUENT BCC_B2  : A,B,C,D : A,B,C,D : VA : !
    PHASE BCC_A2  X  2 1   3 !
    CONSTITUENT BCC_A2  :A,B,C,D : VA, E :  !
    """

    # different interstitial sublattice
    # this will throw a value error from filter phases
    TDB_I_O_SUPERSET = """
    ELEMENT A   PHASE             0.0                0.00            0.00      !
    ELEMENT B   PHASE             0.0                0.00            0.00      !
    ELEMENT C   PHASE             0.0                0.00            0.00      !
    ELEMENT D   PHASE             0.0                0.00            0.00      !
    ELEMENT E   PHASE             0.0                0.00            0.00      !
    ELEMENT VA  VACUUM            0.0                0.00            0.00      !
    TYPE_DEFINITION X GES AMEND_PHASE_DESCRIPTION BCC_B2 DIS_PART BCC_A2 !
    PHASE BCC_B2 X  3 0.5 0.5   3 !
    CONSTITUENT BCC_B2  : A,B,C,D : A,B,C,D : VA, E : !
    PHASE BCC_A2  X  2 1   3 !
    CONSTITUENT BCC_A2  :A,B,C,D : VA :  !
    """
    from pycalphad import variables as v

    dbf_valid = Database(TDB_VALID)
    dbf_o_superset = Database(TDB_O_SUPERSET)
    dbf_o_different = Database(TDB_O_DIFFERENT)
    dbf_d_superset = Database(TDB_D_SUPERSET)
    dbf_i_different = Database(TDB_I_DIFFERENT)
    dbf_i_d_superset = Database(TDB_I_D_SUPERSET)
    dbf_i_o_superset = Database(TDB_I_O_SUPERSET)

    abc_set = set([v.Species(e) for e in ['A', 'B', 'C']])
    abc_va_set = set([v.Species(e) for e in ['A', 'B', 'C', 'VA']])
    bc_set = set([v.Species(e) for e in ['B', 'C']])
    bc_va_set = set([v.Species(e) for e in ['B', 'C', 'VA']])
    abcd_set = set([v.Species(e) for e in ['A', 'B', 'C', 'D']])
    abcd_va_set = set([v.Species(e) for e in ['A', 'B', 'C', 'D', 'VA']])
    abcde_set = set([v.Species(e) for e in ['A', 'B', 'C', 'D', 'E']])
    abcde_va_set = set([v.Species(e) for e in ['A', 'B', 'C', 'D', 'E', 'VA']])

    # valid dbf with [A, B, C] will return ordered phase if it is specified
    comps = ['A', 'B', 'C', 'VA']
    systems = [{'dis': 'BCC_A2', 'ord': 'BCC_B2', 'comp': abc_va_set}, {'dis': 'FCC_A1', 'ord': 'FCC_L12', 'comp': abc_set}]
    for sys in systems:
        phases = filter_phases(dbf_valid, unpack_species(dbf_valid, comps), [sys['dis']])
        assert len(phases) == 1 and phases[0] == sys['dis']
        assert len(sys['comp'].symmetric_difference(Model(dbf_valid, comps, phases[0]).components)) == 0
        phases = filter_phases(dbf_valid, unpack_species(dbf_valid, comps), [sys['ord']])
        assert len(phases) == 1 and phases[0] == sys['ord']
        assert len(sys['comp'].symmetric_difference(Model(dbf_valid, comps, phases[0]).components)) == 0
        phases = filter_phases(dbf_valid, unpack_species(dbf_valid, comps), [sys['dis'], sys['ord']])
        assert len(phases) == 1 and phases[0] == sys['ord']
        assert len(sys['comp'].symmetric_difference(Model(dbf_valid, comps, phases[0]).components)) == 0

    # O is superset of D
    # comps of [A, B, C] will be valid
    # comps of [A, B, C, D] will raise ValueError if order phase is selected
    comps = ['A', 'B', 'C', 'VA']
    systems = [{'dis': 'BCC_A2', 'ord': 'BCC_B2', 'comp': abc_va_set}, {'dis': 'FCC_A1', 'ord': 'FCC_L12', 'comp': abc_set}]
    for sys in systems:
        phases = filter_phases(dbf_o_superset, unpack_species(dbf_o_superset, comps), [sys['dis']])
        assert len(phases) == 1 and phases[0] == sys['dis']
        assert len(sys['comp'].symmetric_difference(Model(dbf_o_superset, comps, phases[0]).components)) == 0
        phases = filter_phases(dbf_o_superset, unpack_species(dbf_o_superset, comps), [sys['ord']])
        assert len(phases) == 1 and phases[0] == sys['ord']
        assert len(sys['comp'].symmetric_difference(Model(dbf_o_superset, comps, phases[0]).components)) == 0
        phases = filter_phases(dbf_o_superset, unpack_species(dbf_o_superset, comps), [sys['dis'], sys['ord']])
        assert len(phases) == 1 and phases[0] == sys['ord']
        assert len(sys['comp'].symmetric_difference(Model(dbf_o_superset, comps, phases[0]).components)) == 0

    comps = ['A', 'B', 'C', 'D', 'VA']
    # for the disordered phase, Model will only include the defined components [A, B, C]
    systems = [{'dis': 'BCC_A2', 'ord': 'BCC_B2', 'comp': abc_va_set}, {'dis': 'FCC_A1', 'ord': 'FCC_L12', 'comp': abc_set}]
    for sys in systems:
        phases = filter_phases(dbf_o_superset, unpack_species(dbf_o_superset, comps), [sys['dis']])
        assert len(phases) == 1 and phases[0] == sys['dis']
        assert len(sys['comp'].symmetric_difference(Model(dbf_o_superset, comps, phases[0]).components)) == 0
        with pytest.raises(ValueError):
            phases = filter_phases(dbf_o_superset, unpack_species(dbf_o_superset, comps), [sys['ord']])
        with pytest.raises(ValueError):
            phases = filter_phases(dbf_o_superset, unpack_species(dbf_o_superset, comps), [sys['dis'], sys['ord']])

    # O has same number, but different constituents of D
    # comps of [B, C] will be valid with no changes in B2
    # comps of [A, B, C] will be valid and A will be added to B2
    # comps of [A, B, C, D] will raise ValueError if order phase is selected
    comps = ['B', 'C', 'VA']
    systems = [{'dis': 'BCC_A2', 'ord': 'BCC_B2', 'comp': bc_va_set}, {'dis': 'FCC_A1', 'ord': 'FCC_L12', 'comp': bc_set}]
    for sys in systems:
        phases = filter_phases(dbf_o_different, unpack_species(dbf_o_different, comps), [sys['dis']])
        assert len(phases) == 1 and phases[0] == sys['dis']
        assert len(sys['comp'].symmetric_difference(Model(dbf_o_different, comps, phases[0]).components)) == 0
        phases = filter_phases(dbf_o_different, unpack_species(dbf_o_different, comps), [sys['ord']])
        assert len(phases) == 1 and phases[0] == sys['ord']
        assert len(sys['comp'].symmetric_difference(Model(dbf_o_different, comps, phases[0]).components)) == 0
        phases = filter_phases(dbf_o_different, unpack_species(dbf_o_different, comps), [sys['dis'], sys['ord']])
        assert len(phases) == 1 and phases[0] == sys['ord']
        assert len(sys['comp'].symmetric_difference(Model(dbf_o_different, comps, phases[0]).components)) == 0

    comps = ['A', 'B', 'C', 'VA']
    systems = [{'dis': 'BCC_A2', 'ord': 'BCC_B2', 'comp': abc_va_set}, {'dis': 'FCC_A1', 'ord': 'FCC_L12', 'comp': abc_set}]
    for sys in systems:
        phases = filter_phases(dbf_o_different, unpack_species(dbf_o_different, comps), [sys['dis']])
        assert len(phases) == 1 and phases[0] == sys['dis']
        assert len(sys['comp'].symmetric_difference(Model(dbf_o_different, comps, phases[0]).components)) == 0
        phases = filter_phases(dbf_o_different, unpack_species(dbf_o_different, comps), [sys['ord']])
        assert len(phases) == 1 and phases[0] == sys['ord']
        assert len(sys['comp'].symmetric_difference(Model(dbf_o_different, comps, phases[0]).components)) == 0
        phases = filter_phases(dbf_o_different, unpack_species(dbf_o_different, comps), [sys['dis'], sys['ord']])
        assert len(phases) == 1 and phases[0] == sys['ord']
        assert len(sys['comp'].symmetric_difference(Model(dbf_o_different, comps, phases[0]).components)) == 0

    comps = ['A', 'B', 'C', 'D', 'VA']
    systems = [{'dis': 'BCC_A2', 'ord': 'BCC_B2', 'comp': abc_va_set}, {'dis': 'FCC_A1', 'ord': 'FCC_L12', 'comp': abc_set}]
    for sys in systems:
        phases = filter_phases(dbf_o_different, unpack_species(dbf_o_different, comps), [sys['dis']])
        assert len(phases) == 1 and phases[0] == sys['dis']
        assert len(sys['comp'].symmetric_difference(Model(dbf_o_different, comps, phases[0]).components)) == 0
        with pytest.raises(ValueError):
            phases = filter_phases(dbf_o_different, unpack_species(dbf_o_different, comps), [sys['ord']])
        with pytest.raises(ValueError):
            phases = filter_phases(dbf_o_different, unpack_species(dbf_o_different, comps), [sys['dis'], sys['ord']])

    # D is superset of O
    # comps of [A, B, C] will be valid with no changes to B2
    # comps of [A, B, C, D] will be valid with D added to B2
    comps = ['A', 'B', 'C', 'VA']
    systems = [{'dis': 'BCC_A2', 'ord': 'BCC_B2', 'comp': abc_va_set}, {'dis': 'FCC_A1', 'ord': 'FCC_L12', 'comp': abc_set}]
    for sys in systems:
        phases = filter_phases(dbf_d_superset, unpack_species(dbf_d_superset, comps), [sys['dis']])
        assert len(phases) == 1 and phases[0] == sys['dis']
        assert len(sys['comp'].symmetric_difference(Model(dbf_d_superset, comps, phases[0]).components)) == 0
        phases = filter_phases(dbf_d_superset, unpack_species(dbf_d_superset, comps), [sys['ord']])
        assert len(phases) == 1 and phases[0] == sys['ord']
        assert len(sys['comp'].symmetric_difference(Model(dbf_d_superset, comps, phases[0]).components)) == 0
        phases = filter_phases(dbf_d_superset, unpack_species(dbf_d_superset, comps), [sys['dis'], sys['ord']])
        assert len(phases) == 1 and phases[0] == sys['ord']
        assert len(sys['comp'].symmetric_difference(Model(dbf_d_superset, comps, phases[0]).components)) == 0

    comps = ['A', 'B', 'C', 'D', 'VA']
    systems = [{'dis': 'BCC_A2', 'ord': 'BCC_B2', 'comp': abcd_va_set}, {'dis': 'FCC_A1', 'ord': 'FCC_L12', 'comp': abcd_set}]
    for sys in systems:
        phases = filter_phases(dbf_d_superset, unpack_species(dbf_d_superset, comps), [sys['dis']])
        assert len(phases) == 1 and phases[0] == sys['dis']
        assert len(sys['comp'].symmetric_difference(Model(dbf_d_superset, comps, phases[0]).components)) == 0
        phases = filter_phases(dbf_d_superset, unpack_species(dbf_d_superset, comps), [sys['ord']])
        assert len(phases) == 1 and phases[0] == sys['ord']
        assert len(sys['comp'].symmetric_difference(Model(dbf_d_superset, comps, phases[0]).components)) == 0
        phases = filter_phases(dbf_d_superset, unpack_species(dbf_d_superset, comps), [sys['dis'], sys['ord']])
        assert len(phases) == 1 and phases[0] == sys['ord']
        assert len(sys['comp'].symmetric_difference(Model(dbf_d_superset, comps, phases[0]).components)) == 0

    # test for different intersitial sublattice
    comps = ['A', 'B', 'C', 'D', 'E', 'VA']
    # same number, but different constituents
    phases = filter_phases(dbf_i_different, unpack_species(dbf_i_different, comps), ['BCC_A2'])
    assert len(phases) == 1 and phases[0] == 'BCC_A2'
    assert len(abcde_set.symmetric_difference(Model(dbf_i_different, comps, phases[0]).components)) == 0
    with pytest.raises(ValueError):
        phases = filter_phases(dbf_i_different, unpack_species(dbf_i_different, comps), ['BCC_B2'])
        model = Model(dbf_i_different, comps, phases[0])

    # interstitials of d is superset of o
    phases = filter_phases(dbf_i_d_superset, unpack_species(dbf_i_d_superset, comps), ['BCC_A2'])
    assert len(phases) == 1 and phases[0] == 'BCC_A2'
    assert len(abcde_va_set.symmetric_difference(Model(dbf_i_d_superset, comps, phases[0]).components)) == 0
    with pytest.raises(ValueError):
        phases = filter_phases(dbf_i_d_superset, unpack_species(dbf_i_d_superset, comps), ['BCC_B2'])
        model = Model(dbf_i_d_superset, comps, phases[0])

    # interstitials of o is superset of d
    phases = filter_phases(dbf_i_o_superset, unpack_species(dbf_i_o_superset, comps), ['BCC_A2'])
    assert len(phases) == 1 and phases[0] == 'BCC_A2'
    assert len(abcd_va_set.symmetric_difference(Model(dbf_i_o_superset, comps, phases[0]).components)) == 0
    with pytest.raises(ValueError):
        phases = filter_phases(dbf_i_o_superset, unpack_species(dbf_i_o_superset, comps), ['BCC_B2'])
        model = Model(dbf_i_o_superset, comps, phases[0])
