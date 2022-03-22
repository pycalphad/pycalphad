"""
The utils test module contains tests for pycalphad utilities.
"""

import pytest
from importlib_resources import files
from pycalphad import Database, Model
from pycalphad.core.utils import filter_phases, unpack_components, instantiate_models
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
