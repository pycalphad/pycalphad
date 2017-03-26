"""
The test_database module contains tests for the Database object.
"""
from __future__ import print_function
import warnings
import hashlib
from copy import deepcopy
from pyparsing import ParseException
from sympy import Symbol, Piecewise, And
from pycalphad import Database, Model, variables as v
from pycalphad.io.tdb import expand_keyword
from pycalphad.io.tdb import _apply_new_symbol_names, DatabaseExportError
from pycalphad.tests.datasets import ALCRNI_TDB, ALFE_TDB, ALNIPT_TDB, ROSE_TDB, DIFFUSION_TDB
import nose.tools
try:
    # Python 2
    from StringIO import StringIO
except ImportError:
    # Python 3
    from io import StringIO

warnings.simplefilter("always", UserWarning) # so we can test warnings

#
# DATABASE LOADING TESTS
#

# This uses the backwards compatible behavior
# Underneath it's calling many of the same routines, so we can't guarantee
# the Database is correct; that's okay, other tests check correctness.
# We're only checking consistency and exercising error checking here.
REFERENCE_DBF = Database(ALCRNI_TDB)
REFERENCE_MOD = Model(REFERENCE_DBF, ['CR', 'NI'], 'L12_FCC')

INVALID_TDB_STR="""$ Note: database that invalidates the minimum compatibility subset for TDBs in different softwares
$ functions names must be <=8 characters (Thermo-Calc)
FUNCTION A_VERY_LONG_FUNCTION_NAME  298.15 -42; 6000 N !
FUNCTION COMPAT 298.15 +9001; 6000 N !
"""

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

def test_database_diffusion():
    "Diffusion database support."
    assert Database(DIFFUSION_TDB).phases == \
           Database.from_string(Database(DIFFUSION_TDB).to_string(fmt='tdb'), fmt='tdb').phases
    # Won't work until sympy/sympy#10560 is fixed to prevent precision loss
    #assert Database(DIFFUSION_TDB) == Database.from_string(Database(DIFFUSION_TDB).to_string(fmt='tdb'), fmt='tdb')

def test_load_from_string():
    "Test database loading from a string."
    test_model = Model(Database.from_string(ALCRNI_TDB, fmt='tdb'), ['CR', 'NI'], 'L12_FCC')
    assert test_model == REFERENCE_MOD

def test_export_import():
    "Equivalence of re-imported database to original."
    test_dbf = Database(ALNIPT_TDB)
    assert Database.from_string(test_dbf.to_string(fmt='tdb', if_incompatible='ignore'), fmt='tdb') == test_dbf
    test_dbf = Database(ALFE_TDB)
    assert Database.from_string(test_dbf.to_string(fmt='tdb'), fmt='tdb') == test_dbf

def test_incompatible_db_warns_by_default():
    "Symbol names too long for Thermo-Calc warn and write the database as given by default."
    test_dbf = Database.from_string(INVALID_TDB_STR, fmt='tdb')
    with warnings.catch_warnings(record=True) as w:
        invalid_dbf = test_dbf.to_string(fmt='tdb')
        assert len(w) > 0
    assert test_dbf == Database.from_string(invalid_dbf, fmt='tdb')

@nose.tools.raises(DatabaseExportError)
def test_incompatible_db_raises_error_with_kwarg_raise():
    "Symbol names too long for Thermo-Calc raise error on write with kwarg raise."
    test_dbf = Database.from_string(INVALID_TDB_STR, fmt='tdb')
    test_dbf.to_string(fmt='tdb', if_incompatible='raise')

def test_incompatible_db_warns_with_kwarg_warn():
    "Symbol names too long for Thermo-Calc warn and write the database as given."
    test_dbf = Database.from_string(INVALID_TDB_STR, fmt='tdb')
    with warnings.catch_warnings(record=True) as w:
        invalid_dbf = test_dbf.to_string(fmt='tdb', if_incompatible='warn')
        assert len(w) > 0
    assert test_dbf == Database.from_string(invalid_dbf, fmt='tdb')

def test_incompatible_db_ignores_with_kwarg_ignore():
    "Symbol names too long for Thermo-Calc are ignored the database written as given."
    test_dbf = Database.from_string(INVALID_TDB_STR, fmt='tdb')
    with warnings.catch_warnings(record=True) as w:
        invalid_dbf = test_dbf.to_string(fmt='tdb', if_incompatible='ignore')
        assert len(w) == 0
    assert test_dbf == Database.from_string(invalid_dbf, fmt='tdb')

def test_incompatible_db_mangles_names_with_kwarg_fix():
    "Symbol names too long for Thermo-Calc are mangled and replaced in symbol names, symbol expressions, and parameter expressions."
    test_dbf = Database.from_string(INVALID_TDB_STR, fmt='tdb')
    test_dbf_copy = deepcopy(test_dbf)
    mangled_dbf = Database.from_string(test_dbf.to_string(fmt='tdb', if_incompatible='fix'), fmt='tdb')
    # check that the long function name was hashed correctly
    a_very_long_function_name_hash_symbol = 'F' + str(hashlib.md5('A_VERY_LONG_FUNCTION_NAME'.encode('UTF-8')).hexdigest()).upper()[:7]
    assert a_very_long_function_name_hash_symbol in mangled_dbf.symbols.keys()
    assert 'COMPAT' in mangled_dbf.symbols.keys() # test that compatible keys are not removed
    assert test_dbf_copy == test_dbf # make sure test_dbf has not mutated
    assert test_dbf != mangled_dbf # also make sure test_dbf has not mutated

def test_symbol_names_are_propagated_through_symbols_and_parameters():
    """A map of old symbol names to new symbol names should propagate through symbol and parameter SymPy expressions"""
    tdb_propagate_str = """$ Mangled function names should propagate through other symbols and parameters
    ELEMENT A PH 0 0 0 !
    FUNCTION FN1  298.15 -42; 6000 N !
    FUNCTION FN2 298.15 FN1#; 6000 N !
    PARAMETER G(PH,A;0) 298.15 FN1# + FN2#; 6000 N !
    """
    test_dbf = Database.from_string(tdb_propagate_str, fmt='tdb')
    rename_map = {'FN1': 'RENAMED_FN1', 'FN2': 'RENAMED_FN2'}
    _apply_new_symbol_names(test_dbf, rename_map)
    assert 'RENAMED_FN1' in test_dbf.symbols
    assert 'FN1' not in test_dbf.symbols # check that the old key was removed
    assert test_dbf.symbols['RENAMED_FN2'] == Piecewise((Symbol('RENAMED_FN1'), And(v.T < 6000.0, v.T >= 298.15)), (0, True))
    assert test_dbf._parameters.all()[0]['parameter'] == Piecewise((Symbol('RENAMED_FN1')+Symbol('RENAMED_FN2'), And(v.T < 6000.0, v.T >= 298.15)), (0, True))

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
    test_tdb = Database(StringIO(ALCRNI_TDB))
    assert test_tdb == REFERENCE_DBF

def test_load_from_stringio_from_file():
    "Test database loading from a file-like object with the from_file method."
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


@nose.tools.raises(ParseException)
def test_tdb_missing_terminator_element():
    tdb_str = """$ Note missing '!' in next line
               ELEMENT ZR   BCT_A5
               FUNCTION EMBCCTI    298.15 -39.72; 6000 N !"""
    Database(tdb_str)
