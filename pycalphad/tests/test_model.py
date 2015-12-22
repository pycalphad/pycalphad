"""
The test_model module contains unit tests for the Model object.
"""
from __future__ import print_function
from pycalphad import Database, Model
from pycalphad.tests.datasets import ALCRNI_TDB

ALCRNI_DBF = Database(ALCRNI_TDB)

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
    test_model = Model(Database.from_string(ALCRNI_DBF.to_string(fmt='tdb'), fmt='tdb'), ['CR', 'NI'], 'L12_FCC')
    ref_model = Model(ALCRNI_DBF, ['CR', 'NI'], 'L12_FCC')
    assert test_model == ref_model