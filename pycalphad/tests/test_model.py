"""
The test_model module contains unit tests for the Model object.
"""
from pycalphad import Database, Model, variables as v, equilibrium
from pycalphad.tests.datasets import ALCRNI_TDB, ALNIPT_TDB, ALFE_TDB, ZRO2_CUBIC_BCC_TDB, TDB_PARAMETER_FILTERS_TEST
from pycalphad.core.errors import DofError
from tinydb import where
import numpy as np
import pytest

ALCRNI_DBF = Database(ALCRNI_TDB)
ALNIPT_DBF = Database(ALNIPT_TDB)
ALFE_DBF = Database(ALFE_TDB)

def test_model_eq():
    "Model equality comparison."
    test_model = Model(ALCRNI_DBF, ['AL', 'CR'], 'L12_FCC')
    assert test_model == test_model
    assert test_model == Model(ALCRNI_DBF, ['AL', 'CR'], 'L12_FCC')
    assert not (test_model == Model(ALCRNI_DBF, ['NI', 'CR'], 'L12_FCC'))
    # literals which don't have __dict__
    assert not (test_model == 42)
    assert not (test_model == None)
    assert not (42 == test_model)
    assert not (None == test_model)

def test_model_ne():
    "Model inequality comparison."
    test_model = Model(ALCRNI_DBF, ['AL', 'CR'], 'L12_FCC')
    assert not (test_model != test_model)
    assert not (test_model != Model(ALCRNI_DBF, ['AL', 'CR'], 'L12_FCC'))
    assert test_model != Model(ALCRNI_DBF, ['NI', 'CR'], 'L12_FCC')
    # literals which don't have __dict__
    assert test_model != 42
    assert test_model != None
    assert 42 != test_model
    assert None != test_model

def test_export_import():
    "Equivalence of Model using re-imported database."
    test_model = Model(Database.from_string(ALNIPT_DBF.to_string(fmt='tdb', if_incompatible='ignore'), fmt='tdb'), ['PT', 'NI', 'VA'], 'FCC_L12')
    ref_model = Model(ALNIPT_DBF, ['NI', 'PT', 'VA'], 'FCC_L12')
    assert test_model == ref_model

def test_custom_model_contributions():
    "Building a custom model using contributions."
    class CustomModel(Model):
        contributions = [('zzz', 'test'), ('xxx', 'test2'), ('yyy', 'test3')]
        def test(self, dbe):
            return 0
        def test2(self, dbe):
            return 0
        def test3(self, dbe):
            return 0
    CustomModel(ALCRNI_DBF, ['AL', 'CR'], 'L12_FCC')


def test_degree_of_ordering():
    "Degree of ordering should be calculated properly."
    my_phases = ['B2_BCC']
    comps = ['AL', 'FE', 'VA']
    conds = {v.T: 300, v.P: 101325, v.X('AL'): 0.25}
    eqx = equilibrium(ALFE_DBF, comps, my_phases, conds, output='degree_of_ordering', verbose=True)
    print('Degree of ordering: {}'.format(eqx.degree_of_ordering.sel(vertex=0).values.flatten()))
    assert np.isclose(eqx.degree_of_ordering.sel(vertex=0).values.flatten(), np.array([0.66663873]))

def test_detect_pure_vacancy_phases():
    "Detecting a pure vacancy phase"
    dbf = Database(ZRO2_CUBIC_BCC_TDB)
    with pytest.raises(DofError):
        Model(dbf,['AL','CU','VA'],'ZRO2_CUBIC')

def test_bad_constituents_not_in_model():
    dbf = Database(TDB_PARAMETER_FILTERS_TEST)
    mod = Model(dbf, ['A', 'B'], 'ALPHA')
    assert v.SiteFraction('ALPHA', 0, 'B') not in mod.ast.free_symbols

def test_bad_constituents_do_not_affect_equilibrium():
    dbf = Database(TDB_PARAMETER_FILTERS_TEST)
    assert np.isclose(equilibrium(dbf, ['A', 'B'], ['ALPHA'], {v.P: 101325, v.T: 300, v.N: 1, v.X('B'): 0.5}).GM.values.squeeze(), -10.0)

def test_interation_test_method():
    dbf = Database(TDB_PARAMETER_FILTERS_TEST)
    interacting_consts = dbf.search(
        where('parameter'))[4]['constituent_array']
    non_interacing_consts = dbf.search(
        where('parameter'))[1]['constituent_array']
    assert Model(dbf,['B','C'],'BETA')._interaction_test(interacting_consts) == True
    assert Model(dbf, ['A','B'],'ALPHA')._interaction_test(non_interacing_consts) == False

def test_params_array_validity():
    dbf = Database(TDB_PARAMETER_FILTERS_TEST)
    bad_comp_param = dbf.search(
        where('parameter'))[5]['constituent_array']
    extra_subl_param = dbf.search(
        where('parameter'))[6]['constituent_array']
    assert Model(dbf,['B','C'],'BETA')._interaction_test(bad_comp_param) == False
    assert Model(dbf,['B','C'],'BETA')._interaction_test(extra_subl_param) == False
