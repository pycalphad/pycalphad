"""
The test_database module contains tests for the Database object.
"""
from __future__ import print_function
from pycalphad import Database, Model
from pycalphad.io.tdb import expand_keyword
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
    Database.from_string(ALCRNI_TDB, fmt='_fail_')

@nose.tools.raises(NotImplementedError)
def test_unknown_format_to_file():
    "to_file: Unknown export file format raises NotImplementedError."
    REFERENCE_DBF.to_file(StringIO(), fmt='_fail_')

def test_expand_keyword():
    "expand_keyword expands command abbreviations."
    test_list = [
        'PARAMETER',
        'ELEMENT',
        'CALCULATE_EQUILIBRIUM',
        'CALCULATE_ALL_EQUILIBRIA',
        'LIST_EQUILIBRIUM',
        'LIST_INITIAL_EQUILIBRIUM',
        'LOAD_INITIAL_EQUILIBRIUM',
        'LIST_PHASE_DATA',
        'SET_ALL_START_VALUES',
        'SET_AXIS_VARIABLE',
        'SET_START_CONSTITUENT',
        'SET_START_VALUE',
        'SET_AXIS_PLOT_STATUS',
        'SET_AXIS_TEXT_STATUS',
        'SET_AXIS_TYPE',
        'SET_OPTIMIZING_CONDITION',
        'SET_OPTIMIZING_VARIABLE',
        'SET_OUTPUT_LEVEL'
    ]
    test_input = [
        ('Par', ['PARAMETER']),
        ('Elem', ['ELEMENT']),
        ('PAR', ['PARAMETER']),
        ('C-E', ['CALCULATE_EQUILIBRIUM']),
        ('C-A', ['CALCULATE_ALL_EQUILIBRIA']),
        ('LI-I-E', ['LIST_INITIAL_EQUILIBRIUM']),
        ('LO-I-E', ['LOAD_INITIAL_EQUILIBRIUM']),
        ('L-P-D', ['LIST_PHASE_DATA']),
        ('S-A-S', ['SET_ALL_START_VALUES']),
        ('S-AL', ['SET_ALL_START_VALUES']),
        ('S-A-V', ['SET_AXIS_VARIABLE']),
        ('S-S-C', ['SET_START_CONSTITUENT']),
        ('S-S-V', ['SET_START_VALUE']),
        ('S-A-P', ['SET_AXIS_PLOT_STATUS']),
        ('S-A-T-S', ['SET_AXIS_TEXT_STATUS']),
        ('S-A-TE', ['SET_AXIS_TEXT_STATUS']),
        ('S-A-TY', ['SET_AXIS_TYPE']),
        ('S-O-C', ['SET_OPTIMIZING_CONDITION']),
        ('S-O-V', ['SET_OPTIMIZING_VARIABLE']),
        ('S-O-L', ['SET_OUTPUT_LEVEL']),
        ('S-OU', ['SET_OUTPUT_LEVEL'])
    ]
    assert all([full == expand_keyword(test_list, abbrev) for abbrev, full in test_input])
