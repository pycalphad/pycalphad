"""
The test_database module contains tests for the Database object.
"""
from __future__ import print_function
from pycalphad import Database, Model
from pycalphad.tests.datasets import ALCRNI_TDB, ROSE_TDB
import nose.tools
try:
    # Python 2
    from StringIO import StringIO
except ImportError:
    # Python 3
    from io import StringIO

#
# DATABASE LOADING TESTS
#

# This uses the backwards compatible behavior
# Underneath it's calling many of the same routines, so we can't guarantee
# the Database is correct; that's okay, other tests check correctness.
# We're only checking consistency and exercising error checking here.
REFERENCE_DBF = Database(ALCRNI_TDB)
REFERENCE_MOD = Model(REFERENCE_DBF, ['CR', 'NI'], 'L12_FCC')

def test_database_eq():
    "Database equality comparison."
    test_dbf = Database(ALCRNI_TDB)
    assert test_dbf == test_dbf
    assert test_dbf == REFERENCE_DBF
    assert not (test_dbf == Database(ROSE_TDB))
    # literals which don't have __dict__
    assert not (test_dbf == 42)
    assert not (test_dbf == None)
    assert not (42 == test_dbf)
    assert not (None == test_dbf)

def test_database_ne():
    "Database inequality comparison."
    test_dbf = Database(ALCRNI_TDB)
    assert not (test_dbf != test_dbf)
    assert not (test_dbf != REFERENCE_DBF)
    assert test_dbf != Database(ROSE_TDB)
    # literals which don't have __dict__
    assert test_dbf != 42
    assert test_dbf != None
    assert None != test_dbf
    assert 42 != test_dbf

def test_load_from_string():
    "Test database loading from a string."
    test_model = Model(Database.from_string(ALCRNI_TDB, fmt='tdb'), ['CR', 'NI'], 'L12_FCC')
    assert test_model == REFERENCE_MOD

def test_export_import():
    "Equivalence of re-imported database to original."
    test_dbf = Database.from_string(REFERENCE_DBF.to_string(fmt='tdb'), fmt='tdb')
    assert test_dbf == REFERENCE_DBF

@nose.tools.raises(ValueError)
def test_unspecified_format_from_string():
    "from_string: Unspecified string format raises ValueError."
    Database.from_string(ALCRNI_TDB)

@nose.tools.raises(NotImplementedError)
def test_unknown_format_from_string():
    "from_string: Unknown import string format raises NotImplementedError."
    Database.from_string(ALCRNI_TDB, fmt='_fail_')

@nose.tools.raises(NotImplementedError)
def test_unknown_format_to_string():
    "to_string: Unknown export file format raises NotImplementedError."
    REFERENCE_DBF.to_string(fmt='_fail_')

def test_load_from_stringio():
    "Test database loading from a file-like object."
    test_tdb = Database.from_file(StringIO(ALCRNI_TDB), fmt='tdb')
    assert test_tdb == REFERENCE_DBF

@nose.tools.raises(ValueError)
def test_unspecified_format_from_file():
    "from_file: Unspecified format for file descriptor raises ValueError."
    Database.from_file(StringIO(ALCRNI_TDB))

@nose.tools.raises(ValueError)
def test_unspecified_format_to_file():
    "to_file: Unspecified format for file descriptor raises ValueError."
    REFERENCE_DBF.to_file(StringIO())

@nose.tools.raises(NotImplementedError)
def test_unknown_format_from_file():
    "from_string: Unknown import file format raises NotImplementedError."
    Database.from_string(StringIO(ALCRNI_TDB), fmt='_fail_')

@nose.tools.raises(NotImplementedError)
def test_unknown_format_to_file():
    "to_file: Unknown export file format raises NotImplementedError."
    REFERENCE_DBF.to_file(StringIO(), fmt='_fail_')