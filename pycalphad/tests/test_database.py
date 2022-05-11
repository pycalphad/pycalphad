"""
The test_database module contains tests for the Database object.
"""
from io import StringIO
import pytest
from importlib_resources import files
import hashlib
import os
import pickle
from copy import deepcopy
import numpy as np
from pyparsing import ParseException
from symengine import Symbol, Piecewise, And, S
from pycalphad import Database, Model, calculate, variables as v
from pycalphad.variables import Species
from pycalphad.io.tdb import expand_keyword, reflow_text, TCPrinter
from pycalphad.io.tdb import _apply_new_symbol_names, DatabaseExportError
import pycalphad.tests.databases
from pycalphad.tests.fixtures import select_database, load_database


#
# DATABASE LOADING TESTS
#

# This uses the backwards compatible behavior
# Underneath it's calling many of the same routines, so we can't guarantee
# the Database is correct; that's okay, other tests check correctness.
# We're only checking consistency and exercising error checking here.
REFERENCE_DBF = Database(str(files(pycalphad.tests.databases).joinpath("alcrni.tdb")))
REFERENCE_MOD = Model(REFERENCE_DBF, ['CR', 'NI'], 'L12_FCC')

INVALID_TDB_STR="""$ Note: database that invalidates the minimum compatibility subset for TDBs in different softwares
$ functions names must be <=8 characters (Thermo-Calc)
FUNCTION A_VERY_LONG_FUNCTION_NAME  298.15 -42; 6000 N !
FUNCTION COMPAT 298.15 +9001; 6000 N !
"""

@select_database("rose.tdb")
def test_database_eq(load_database):
    "Database equality comparison."
    test_dbf = Database(str(files(pycalphad.tests.databases).joinpath("alcrni.tdb")))
    assert test_dbf == test_dbf
    assert test_dbf == REFERENCE_DBF
    assert not (test_dbf == load_database())
    # literals which don't have __dict__
    assert not (test_dbf == 42)
    assert not (test_dbf == None)
    assert not (42 == test_dbf)
    assert not (None == test_dbf)


@select_database("rose.tdb")
def test_database_ne(load_database):
    "Database inequality comparison."
    test_dbf = Database(str(files(pycalphad.tests.databases).joinpath("alcrni.tdb")))
    assert not (test_dbf != test_dbf)
    assert not (test_dbf != REFERENCE_DBF)
    assert test_dbf != load_database()
    # literals which don't have __dict__
    assert test_dbf != 42
    assert test_dbf != None
    assert None != test_dbf
    assert 42 != test_dbf

def test_database_pickle():
    "Database pickle roundtrip."
    test_dbf = Database(str(files(pycalphad.tests.databases).joinpath("alcrni.tdb")))
    new_dbf = pickle.loads(pickle.dumps(test_dbf))
    assert test_dbf == new_dbf


def test_database_diffusion():
    "Diffusion database support."
    DIFFUSION_TDB = open(str(files(pycalphad.tests.databases).joinpath("diffusion.tdb")), "r").read()
    assert Database(DIFFUSION_TDB).phases == \
           Database.from_string(Database(DIFFUSION_TDB).to_string(fmt='tdb'), fmt='tdb').phases
    # Won't work until sympy/sympy#10560 is fixed to prevent precision loss
    #assert Database(DIFFUSION_TDB) == Database.from_string(Database(DIFFUSION_TDB).to_string(fmt='tdb'), fmt='tdb')

def test_load_from_string():
    "Test database loading from a string."
    test_model = Model(Database.from_string(open(files(pycalphad.tests.databases).joinpath("alcrni.tdb"), "r").read()
                                            , fmt='tdb'), ['CR', 'NI'], 'L12_FCC')
    assert test_model == REFERENCE_MOD


@select_database("alfe.tdb")
def test_export_import(load_database):
    "Equivalence of re-imported database to original."
    test_dbf = Database(str(files(pycalphad.tests.databases).joinpath("alnipt.tdb")))
    assert Database.from_string(test_dbf.to_string(fmt='tdb', if_incompatible='ignore'), fmt='tdb') == test_dbf
    test_dbf = load_database()
    assert Database.from_string(test_dbf.to_string(fmt='tdb'), fmt='tdb') == test_dbf

def test_incompatible_db_warns_by_default():
    "Symbol names too long for Thermo-Calc warn and write the database as given by default."
    test_dbf = Database.from_string(INVALID_TDB_STR, fmt='tdb')
    with pytest.warns(UserWarning, match='Ignoring that the following function names are beyond the 8 character TDB limit'):
        invalid_dbf = test_dbf.to_string(fmt='tdb')
    assert test_dbf == Database.from_string(invalid_dbf, fmt='tdb')

def test_incompatible_db_raises_error_with_kwarg_raise():
    "Symbol names too long for Thermo-Calc raise error on write with kwarg raise."
    test_dbf = Database.from_string(INVALID_TDB_STR, fmt='tdb')
    with pytest.raises(DatabaseExportError):
        test_dbf.to_string(fmt='tdb', if_incompatible='raise')

def test_incompatible_db_warns_with_kwarg_warn():
    "Symbol names too long for Thermo-Calc warn and write the database as given."
    test_dbf = Database.from_string(INVALID_TDB_STR, fmt='tdb')
    with pytest.warns(UserWarning, match='Ignoring that the following function names are beyond the 8 character TDB limit'):
        invalid_dbf = test_dbf.to_string(fmt='tdb', if_incompatible='warn')
    assert test_dbf == Database.from_string(invalid_dbf, fmt='tdb')

@pytest.mark.filterwarnings("error")
def test_incompatible_db_ignores_with_kwarg_ignore():
    "Symbol names too long for Thermo-Calc are ignored the database written as given."
    test_dbf = Database.from_string(INVALID_TDB_STR, fmt='tdb')
    invalid_dbf = test_dbf.to_string(fmt='tdb', if_incompatible='ignore')
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

def test_tdb_content_after_line_end_is_neglected():
    """Any characters after the line ending '!' are neglected as in commercial software."""
    tdb_line_ending_str = """$ Characters after line endings should be discarded.
    PARAMETER G(PH,A;0) 298.15 +42; 6000 N ! SHOULD_NOT_RAISE_ERROR
    $ G(PH,C;0) should not parse
    PARAMETER G(PH,B;0) 298.15 +9001; 6000 N ! PARAMETER G(PH,C;0) 298.15 +2; 600 N !
    PARAMETER G(PH,D;0) 298.15 -42; 6000 N !
    """
    test_dbf = Database.from_string(tdb_line_ending_str, fmt='tdb')
    assert len(test_dbf._parameters) == 3

@pytest.fixture
def _testwritetdb():
    fname = 'testwritedb.tdb'
    yield fname  # run the test
    os.remove(fname)


@select_database("alnipt.tdb")
def test_to_file_defaults_to_raise_if_exists(load_database, _testwritetdb):
    "Attempting to use Database.to_file should raise by default if it exists"
    fname = _testwritetdb
    test_dbf = load_database()
    test_dbf.to_file(fname)  # establish the initial file
    with pytest.raises(FileExistsError):
        test_dbf.to_file(fname)  # test if_exists behavior


@select_database("alnipt.tdb")
def test_to_file_raises_with_bad_if_exists_argument(load_database, _testwritetdb):
    "Database.to_file should raise if a bad behavior string is passed to if_exists"
    fname = _testwritetdb
    test_dbf = load_database()
    test_dbf.to_file(fname)  # establish the initial file
    with pytest.raises(FileExistsError):
        test_dbf.to_file(fname, if_exists='TEST_BAD_ARGUMENT')  # test if_exists behavior


@select_database("alnipt.tdb")
def test_to_file_overwrites_with_if_exists_argument(load_database, _testwritetdb):
    "Database.to_file should overwrite if 'overwrite' is passed to if_exists"
    import time
    fname = _testwritetdb
    test_dbf = load_database()
    test_dbf.to_file(fname)  # establish the initial file
    inital_modification_time = os.path.getmtime(fname)
    time.sleep(1)  # this test can fail intermittently without waiting.
    test_dbf.to_file(fname, if_exists='overwrite')  # test if_exists behavior
    overwrite_modification_time = os.path.getmtime(fname)
    assert overwrite_modification_time > inital_modification_time


def test_unspecified_format_from_string():
    "from_string: Unspecified string format raises ValueError."
    with pytest.raises(ValueError):
        Database.from_string(str(files(pycalphad.tests.databases).joinpath("alcrni.tdb")))

def test_unknown_format_from_string():
    "from_string: Unknown import string format raises NotImplementedError."
    with pytest.raises(NotImplementedError):
        Database.from_string(str(files(pycalphad.tests.databases).joinpath("alcrni.tdb")), fmt='_fail_')

def test_unknown_format_to_string():
    "to_string: Unknown export file format raises NotImplementedError."
    with pytest.raises(NotImplementedError):
        REFERENCE_DBF.to_string(fmt='_fail_')

def test_load_from_stringio():
    "Test database loading from a file-like object."
    test_tdb = Database(StringIO(open(str(files(pycalphad.tests.databases).joinpath("alcrni.tdb")), "r").read()))
    assert test_tdb == REFERENCE_DBF

def test_load_from_stringio_from_file():
    "Test database loading from a file-like object with the from_file method."
    test_tdb = Database.from_file(StringIO(open(str(files(pycalphad.tests.databases).joinpath("alcrni.tdb")), "r").read()), fmt='tdb')
    assert test_tdb == REFERENCE_DBF

def test_unspecified_format_from_file():
    "from_file: Unspecified format for file descriptor raises ValueError."
    with pytest.raises(ValueError):
        Database.from_file(StringIO(str(files(pycalphad.tests.databases).joinpath("alcrni.tdb"))))

def test_unspecified_format_to_file():
    "to_file: Unspecified format for file descriptor raises ValueError."
    with pytest.raises(ValueError):
        REFERENCE_DBF.to_file(StringIO())

def test_unknown_format_from_file():
    "from_string: Unknown import file format raises NotImplementedError."
    with pytest.raises(NotImplementedError):
        Database.from_string(str(files(pycalphad.tests.databases).joinpath("alcrni.tdb")), fmt='_fail_')

def test_unknown_format_to_file():
    "to_file: Unknown export file format raises NotImplementedError."
    with pytest.raises(NotImplementedError):
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


def test_tdb_species_are_parsed_correctly():
    """The TDB speciescommand should be properly parsed."""
    tdb_species_str = """
ELEMENT /-   ELECTRON_GAS              0.0000E+00  0.0000E+00  0.0000E+00!
ELEMENT VA   VACUUM                    0.0000E+00  0.0000E+00  0.0000E+00!
ELEMENT AL   FCC_A1                    2.6982E+01  4.5773E+03  2.8321E+01!
ELEMENT O    1/2_MOLE_O2(G)            1.5999E+01  4.3410E+03  1.0252E+02!

SPECIES AL+3                        AL1/+3!
SPECIES O-2                         O1/-2!
SPECIES O1                          O!
SPECIES O2                          O2!
SPECIES O3                          O3!
SPECIES AL1O1                       AL1O1!
SPECIES AL1O2                       AL1O2!
SPECIES AL2                         AL2!
SPECIES AL2O                        AL2O1!
SPECIES AL2O1                       AL2O1!
SPECIES AL2O2                       AL2O2!
SPECIES AL2O3                       AL2O3!
SPECIES ALO                         AL1O1!
SPECIES ALO2                        AL1O2!
SPECIES ALO3/2                      AL1O1.5!
    """
    test_dbf = Database.from_string(tdb_species_str, fmt='tdb')
    assert len(test_dbf.species) == 19
    species_dict = {sp.name: sp for sp in test_dbf.species}
    assert species_dict['AL'].charge == 0
    assert species_dict['O2'].constituents['O'] == 2
    assert species_dict['O1'].constituents['O'] == 1
    assert species_dict['AL1O2'].constituents['AL'] == 1
    assert species_dict['AL1O2'].constituents['O'] == 2
    assert species_dict['ALO3/2'].constituents['O'] == 1.5


def test_tdb_species_with_charge_are_parsed_correctly():
    """The TDB species that have a charge should be properly parsed."""
    tdb_species_str = """
ELEMENT /-   ELECTRON_GAS              0.0000E+00  0.0000E+00  0.0000E+00!
ELEMENT VA   VACUUM                    0.0000E+00  0.0000E+00  0.0000E+00!
ELEMENT AL   FCC_A1                    2.6982E+01  4.5773E+03  2.8321E+01!
ELEMENT O    1/2_MOLE_O2(G)            1.5999E+01  4.3410E+03  1.0252E+02!

SPECIES AL+3                        AL1/+3!
SPECIES O-2                         O1/-2!
SPECIES O2                          O2!
SPECIES AL2                         AL2!
    """
    test_dbf = Database.from_string(tdb_species_str, fmt='tdb')
    assert len(test_dbf.species) == 8
    species_dict = {sp.name: sp for sp in test_dbf.species}
    assert species_dict['AL'].charge == 0
    assert species_dict['AL+3'].charge == 3
    assert species_dict['O-2'].charge == -2


def test_writing_tdb_with_species_gives_same_result():
    """Species defined in the tdb should be written back to the TDB correctly"""
    tdb_species_str = """
ELEMENT /-   ELECTRON_GAS              0.0000E+00  0.0000E+00  0.0000E+00!
ELEMENT VA   VACUUM                    0.0000E+00  0.0000E+00  0.0000E+00!
ELEMENT AL   FCC_A1                    2.6982E+01  4.5773E+03  2.8321E+01!
ELEMENT O    1/2_MOLE_O2(G)            1.5999E+01  4.3410E+03  1.0252E+02!

SPECIES AL+3                        AL1/+3!
SPECIES O-2                         O1/-2!
SPECIES O2                          O2!
SPECIES AL2                         AL2!
    """
    test_dbf = Database.from_string(tdb_species_str, fmt='tdb')
    written_tdb_str = test_dbf.to_string(fmt='tdb')
    test_dbf_reread = Database.from_string(written_tdb_str, fmt='tdb')
    assert len(test_dbf_reread.species) == 8
    species_dict = {sp.name: sp for sp in test_dbf_reread.species}
    assert species_dict['AL'].charge == 0
    assert species_dict['AL+3'].charge == 3
    assert species_dict['O-2'].charge == -2


def test_species_are_parsed_in_tdb_phases_and_parameters():
    """Species defined in the tdb phases and parameters should be parsed."""
    tdb_str = """
ELEMENT /-   ELECTRON_GAS              0.0000E+00  0.0000E+00  0.0000E+00!
ELEMENT VA   VACUUM                    0.0000E+00  0.0000E+00  0.0000E+00!
ELEMENT AL   FCC_A1                    2.6982E+01  4.5773E+03  2.8321E+01!
ELEMENT O    1/2_MOLE_O2(G)            1.5999E+01  4.3410E+03  1.0252E+02!

SPECIES AL+3                        AL1/+3!
SPECIES O-2                         O1/-2!
SPECIES O2                          O2!
SPECIES AL2                         AL2!


 PHASE TEST_PH % 1 1 !
 CONSTITUENT TEST_PH :AL,AL2,O-2: !
 PARA G(TEST_PH,AL;0) 298.15          +10; 6000 N !
 PARA G(TEST_PH,AL2;0) 298.15          +100; 6000 N !
 PARA G(TEST_PH,O-2;0) 298.15          +1000; 6000 N !

 PHASE T2SL % 2 1 1 !
  CONSTITUENT T2SL :AL+3:O-2: !
  PARA L(T2SL,AL+3:O-2;0) 298.15 +2; 6000 N !
    """
    from tinydb import where
    test_dbf = Database.from_string(tdb_str, fmt='tdb')
    written_tdb_str = test_dbf.to_string(fmt='tdb')
    test_dbf_reread = Database.from_string(written_tdb_str, fmt='tdb')
    assert set(test_dbf_reread.phases.keys()) == {'TEST_PH', 'T2SL'}
    assert test_dbf_reread.phases['TEST_PH'].constituents[0] == {Species('AL'), Species('AL2'), Species('O-2')}
    assert len(test_dbf_reread._parameters.search(where('constituent_array') == ((Species('AL'),),))) == 1
    assert len(test_dbf_reread._parameters.search(where('constituent_array') == ((Species('AL2'),),))) == 1
    assert len(test_dbf_reread._parameters.search(where('constituent_array') == ((Species('O-2'),),))) == 1

    assert test_dbf_reread.phases['T2SL'].constituents == ({Species('AL+3')}, {Species('O-2')})
    assert len(test_dbf_reread._parameters.search(where('constituent_array') == ((Species('AL+3'),),(Species('O-2'),)))) == 1

def test_tdb_missing_terminator_element():
    tdb_str = """$ Note missing '!' in next line
               ELEMENT ZR   BCT_A5
               FUNCTION EMBCCTI    298.15 -39.72; 6000 N !"""
    with pytest.raises(ParseException):
        Database(tdb_str)


def test_database_parsing_of_floats_with_no_values_after_decimal():
    """Floats with no values after the decimal should be properly parsed (gh-143)"""
    tdb_string = """$ The element has no values after the decimal in '5004.'
        ELEMENT CU   FCC_A1           63.546           5004.             33.15      !"""
    dbf = Database.from_string(tdb_string, fmt='tdb')
    assert "CU" in dbf.elements


def test_database_parsing_of_floats_with_multiple_leading_zeros():
    """Floats with multiple leading zeros should be properly parsed (gh-143)"""
    tdb_string = """$ The element has multiple leading zeros in '00.546'
        ELEMENT CU   FCC_A1           00.546           5004.0             33.15      !"""
    dbf = Database.from_string(tdb_string, fmt='tdb')
    assert "CU" in dbf.elements


def test_comma_templims():
    """Accept TEMPERATURE_LIMITS and default-limit commas."""
    tdb_string = """
     ELEMENT VA   VACUUM                      0.0          0.0      0.0    !
     ELEMENT AL   FCC_A1                     26.98154   4540.      28.30   !
     ELEMENT C    GRAPHITE                   12.011     1054.0      5.7423 !
     ELEMENT CO   HCP_A3                     58.9332    4765.567   30.0400 !
     ELEMENT CR   BCC_A2                     51.996     4050.0     23.5429 !
     ELEMENT FE   BCC_A2                     55.847     4489.0     27.2797 !
     ELEMENT MN   CBCC_A12                   54.9380    4995.696   32.2206 !
     ELEMENT NI   FCC_A1                     58.69      4787.0     29.7955 !
     TEMP-LIM 298 6000 !
    $ ------------------------------------------------------------------------------
    $
    $ Fcc (cF4, Fm-3m) and MeX (cF8, Fm-3m)
    $
     PHASE FCC_A1 %A 2 1 1 !
     CONST FCC_A1 : AL% CO% CR FE% MN% NI% : C VA% : !
    $
    $ Disordered part of FCC_4SL, identical to FCC_A1
    $
     PHASE A1_FCC %A 2 1 1 !
     CONST A1_FCC : AL CO CR FE MN NI : C VA% : !
    $
    $ Bcc (cI2, Im-3m)
    $
     PHASE BCC_A2 %B 2 1 3 !
     CONST BCC_A2 : AL CO CR% FE% MN% NI : C VA% : !
    $
    $ Disordered part of B2_BCC, identical to BCC_A2 (except Va)
    $
     PHASE A2_BCC %B 2 1 3 !
     CONST A2_BCC : AL CO CR FE MN NI VA : C VA% : !
    $
    $ Prototype CsCl (cP2, Pm-3m)
    $
     PHASE B2_BCC %BO 3 0.5 0.5 3 !
     CONST B2_BCC : AL CO CR FE MN% NI VA : AL CO CR FE MN NI% VA : C VA% : !
    $
    $ Hcp (hP2, P6_3/mmc) and Me2X (NiAs-type, hP4, P6_3/mmc, B8_1)
    $
     PHASE HCP_A3 %A 2 1 0.5 !
     CONST HCP_A3 : AL CO% CR FE MN NI : C VA% : !
    $ ------------------------------------------------------------------------------
    $ Defaults
    $
     DEFINE-SYSTEM-DEFAULT ELEMENT 2 !
     DEFAULT-COM DEFINE_SYSTEM_ELEMENT VA !
     DEFAULT-COM REJECT_PHASE FCC_A1 BCC_A2 !
    $DEFAULT-COM REJECT_PHASE A1_FCC FCC_4SL A2_BCC B2_BCC !
     TYPE-DEF % SEQ * !
     TYPE-DEF A GES AMEND_PHASE_DESCRIPTION @ MAGNETIC -3 0.28 !
     TYPE-DEF B GES AMEND_PHASE_DESCRIPTION @ MAGNETIC -1 0.4 !
     TYPE-DEF O GES AMEND_PHASE_DESCRIPTION B2_BCC DIS_PART A2_BCC !
    $ The following type definition is commented out because the FCC_4SL phase
    $ is not required by this test and are not defined in this database.
    $ TYPE-DEF Y GES AMEND_PHASE_DESCRIPTION FCC_4SL DIS_PART A1_FCC !
     FUNCTION ZERO      298.15  0;                                         6000 N !
     FUNCTION UN_ASS    298.15  0;                                         6000 N !
     FUNCTION R         298.15  +8.31451;                                  6000 N !
    $ ------------------------------------------------------------------------------
    $ Element data
    $ ------------------------------------------------------------------------------
    $ Al
    $
    $ BCT_A5 and DIAMOND_A4 added in unary 3.0
    $
     PAR  G(FCC_A1,AL:VA),,                 +GHSERAL;                ,, N 91Din !
     PAR  G(A1_FCC,AL:VA),,                 +GHSERAL;                , N 91Din !
     PAR  G(BCC_A2,AL:VA),,                 +GHSERAL+10083-4.813*T;  2900 N 91Din !
     PAR  G(A2_BCC,AL:VA),,                 +GHSERAL+10083-4.813*T;  2900 N 91Din !
     PAR  G(HCP_A3,AL:VA),,                 +GHSERAL+5481-1.8*T;     2900 N 91Din !
     PAR  G(CBCC_A12,AL:VA),,               +GHSERAL
                 +10083.4-4.813*T;                                   2900 N 91Din !
     PAR  G(CUB_A13,AL:VA),,                +GHSERAL
                 +10920.44-4.8116*T;                                 2900 N 91Din !
     PAR  G(BCT_A5,AL),,                    +GHSERAL+10083-4.813*T;  2900 N SGCOST !
     PAR  G(DIAMOND_A4,AL),,                +GHSERAL+30*T;           2900 N SGCOST !
     FUNCTION GHSERAL   298.15  -7976.15+137.093038*T-24.3671976*T*LN(T)
           -0.001884662*T**2-8.77664E-07*T**3+74092*T**(-1);
           700.00  Y  -11276.24+223.048446*T-38.5844296*T*LN(T)
           +0.018531982*T**2 -5.764227E-06*T**3+74092*T**(-1);
           933.47  Y  -11278.378+188.684153*T-31.748192*T*LN(T)
           -1.230524E+28*T**(-9);
          2900.00  N !
        """
    dbf = Database.from_string(tdb_string, fmt='tdb')
    assert "AL" in dbf.elements


def test_database_parameter_with_species_that_is_not_a_stoichiometric_formula():
    """Species names used in parameters do not have to be stoichiometric formulas"""

    # names are taken from the Thermo-Calc documentation set, Database Manager Guide, SPECIES

    tdb_string = """
     ELEMENT /-   ELECTRON_GAS              0.0000E+00  0.0000E+00  0.0000E+00!
     ELEMENT VA   VACUUM                    0.0000E+00  0.0000E+00  0.0000E+00!
     ELEMENT O    1/2_MOLE_O2(G)            0.0000E+00  0.0000E+00  0.0000E+00!
     ELEMENT SI   HCP_A3                    0.0000E+00  0.0000E+00  0.0000E+00!
     ELEMENT NA   HCP_A3                    0.0000E+00  0.0000E+00  0.0000E+00!
     ELEMENT SB   RHOMBOHEDRAL_A7           0.0000E+00  0.0000E+00  0.0000E+00!
     ELEMENT H    1/2_MOLE_H2(G)            0.0000E+00  0.0000E+00  0.0000E+00!

     SPECIES SILICA                      SI1O2 !  $ tests for arbitrary names
     SPECIES NASB_6OH                    NA1SB1O6H6 !  $ tests for underscores
     SPECIES SB-3                        SB/-3 !  $ tests for charge
     SPECIES ALCL2OH.3WATER                        AL1O1H1CL2H6O3 !  $ tests for charge


     PHASE LIQUID:L %  1  1.0  !

     CONSTITUENT LIQUID:L : O, SI, NA, SB, H, SILICA, NASB_6OH, SB-3, ALCL2OH.3WATER  :  !
     PARAMETER G(LIQUID,SILICA;0)      298.15  10;      3000 N !
     PARAMETER G(LIQUID,NASB_6OH;0)    298.15  100;      3000 N !
     PARAMETER G(LIQUID,ALCL2OH.3WATER;0)    298.15  1000;      3000 N !
     PARAMETER G(LIQUID,SB-3;0)        298.15  10000;      3000 N !

     """

    dbf = Database.from_string(tdb_string, fmt='tdb')

    species_dict = {sp.name: sp for sp in dbf.species}
    species_names = list(species_dict.keys())

    # check that the species are found
    assert 'SILICA' in species_names
    assert 'NASB_6OH' in species_names
    assert 'ALCL2OH.3WATER' in species_names
    assert 'SB-3' in species_names

    import tinydb
    silica = dbf._parameters.search(tinydb.where('constituent_array') == ((species_dict['SILICA'],),))
    assert len(silica) == 1
    assert silica[0]['parameter'].args[0] == float(10)

    nasb_6oh = dbf._parameters.search(tinydb.where('constituent_array') == ((species_dict['NASB_6OH'],),))
    assert len(nasb_6oh) == 1
    assert nasb_6oh[0]['parameter'].args[0] == float(100)

    alcl2oh_3water = dbf._parameters.search(tinydb.where('constituent_array') == ((species_dict['ALCL2OH.3WATER'],),))
    assert len(alcl2oh_3water) == 1
    assert alcl2oh_3water[0]['parameter'].args[0] == float(1000)

    sbminus3 = dbf._parameters.search(tinydb.where('constituent_array') == ((species_dict['SB-3'],),))
    assert len(sbminus3) == 1
    assert sbminus3[0]['parameter'].args[0] == float(10000)


def test_database_sympy_namespace_clash():
    """Symbols that clash with sympy special objects are replaced (gh-233)"""
    Database.from_string("""FUNCTION TEST 0.01 T*LN(CC)+FF; 6000 N TW !""", fmt='tdb')


def test_tdb_order_disorder_model_hints_applied_correctly():
    """Phases using the order/disorder model should have model_hints added to
    both phases, regardless of the order by which the phases were specified.
    Model hints should also be applied correctly if only one of the phases has
    the order/disorder type defintion applied, since this is allowed by
    commercial software.
    """

    # This test creates a starting template and then tries to add the phase and
    # type definitions in any order. In this case, the BCC_A2 phase does not
    # have the type definition while the BCC_B2 phase does.
    TEMPLATE_TDB = """
     ELEMENT VA   VACUUM                     .0000E+00   .0000E+00   .0000E+00!
     ELEMENT AL   FCC_A1                    2.6982E+01  4.5773E+03  2.8322E+01!
     ELEMENT NI   FCC_A1                    5.8690E+01  4.7870E+03  2.9796E+01!
     TYPE_DEFINITION % SEQ *!
    """

    PHASE_A2 = """
     PHASE BCC_A2  %  2 1   3 !
     CONST BCC_A2  :AL,NI : VA :  !
     """

    PHASE_B2 = """
     PHASE BCC_B2  %C  3 .5 .5    3 !
     CONST BCC_B2  :AL,NI : AL,NI : VA: !
    """

    TYPE_DEF_ORD = """
     TYPE_DEFINITION C GES A_P_D BCC_B2 DIS_PART BCC_A2 !
    """
    import itertools
    for (k1, v1), (k2, v2), (k3, v3) in itertools.permutations([('PHASE_A2 ', PHASE_A2), ('PHASE_B2 ', PHASE_B2), ('TYPE_DEF ', TYPE_DEF_ORD)]):
        print(k1 + k2 + k3)
        dbf = Database(TEMPLATE_TDB + v1 + v2 + v3)
        assert 'disordered_phase' in dbf.phases['BCC_A2'].model_hints
        assert 'ordered_phase' in dbf.phases['BCC_A2'].model_hints
        assert 'disordered_phase' in dbf.phases['BCC_B2'].model_hints
        assert 'ordered_phase' in dbf.phases['BCC_B2'].model_hints
        roundtrip_dbf = Database.from_string(dbf.to_string(fmt='tdb'), fmt='tdb')
        assert roundtrip_dbf == dbf


def test_database_applies_late_type_def():
    """If type definitions are defined after phase defintions in the TDB, the
    model_hints from the type definition should be correctly applied."""
    dbf = Database("""
     ELEMENT AL   FCC_A1                    2.6982E+01  4.5773E+03  2.8322E+01!
     ELEMENT NI   FCC_A1                    5.8690E+01  4.7870E+03  2.9796E+01!
     ELEMENT VA   VACUUM                     .0000E+00   .0000E+00   .0000E+00!
     TYPE_DEFINITION % SEQ *!
     DEFINE_SYSTEM_DEFAULT E 2 !
     DEFAULT_COMMAND DEF_SYS_ELEMENT VA !

     PHASE BCC_A2  %B  2 1   3 !
     CONST BCC_A2  :AL,NI,VA : VA :  !

    $ Note: type definition after PHASE definition
     TYPE_DEFINITION B GES A_P_D @ MAGNETIC  -1.0 .40 !

     FUNCTION GHSERAL    298.15
        -7976.15+137.093038*T-24.3671976*T*LN(T)
        -.001884662*T**2-8.77664E-07*T**3+74092*T**(-1);
                         700.00  Y
        -11276.24+223.048446*T-38.5844296*T*LN(T)
        +.018531982*T**2-5.764227E-06*T**3+74092*T**(-1);
                         933.60  Y
        -11278.378+188.684153*T-31.748192*T*LN(T)
        -1.231E+28*T**(-9);,,  N !
     FUNCTION GHSERNI    298.14
        -5179.159+117.854*T-22.096*T*LN(T)
        -.0048407*T**2;
                         1728.0  Y
        -27840.655+279.135*T-43.1*T*LN(T)+1.12754E+31*T**(-9);,,  N   !

     FUNCTION GBCCAL     298.15  +10083-4.813*T+GHSERAL;,,N !
     FUNCTION GBCCNI     298.15  +8715.084-3.556*T+GHSERNI;,,,   N !

     PARAMETER G(BCC_A2,AL:VA;0)  298.15  +GBCCAL;,,N 91DIN !
       FUNC B2ALVA 295.15 10000-T;,,N !
       FUNC LB2ALVA 298.15 150000;,,N !
     PARAMETER L(BCC_A2,AL,VA:VA;0)  298.15  B2ALVA+LB2ALVA;,,N 99DUP !

     PARAMETER G(BCC_A2,NI:VA;0)  298.15  +GBCCNI;,,N 91DIN !
     PARAMETER TC(BCC_A2,NI:VA;0)  298.15  575;,,N 89DIN !
     PARAMETER BMAGN(BCC_A2,NI:VA;0)  298.15  .85;,,N 89DIN !
       FUNC B2NIVA 295.15 +162397.3-27.40575*T;,,N !
       FUNC LB2NIVA 298.15 -64024.38+26.49419*T;,,N !
     PARAMETER L(BCC_A2,NI,VA:VA;0)  298.15  B2NIVA+LB2NIVA;,,N 99DUP !

       FUNC B2ALNI 295.15 -152397.3+26.40575*T;,,N !
       FUNC LB2ALNI 298.15 -52440.88+11.30117*T;,,N !
     PARAMETER L(BCC_A2,AL,NI:VA;0)  298.15  B2ALNI+LB2ALNI;,,N 99DUP!

    """)

    assert 'ihj_magnetic_afm_factor' in dbf.phases['BCC_A2'].model_hints
    assert dbf.phases['BCC_A2'].model_hints['ihj_magnetic_afm_factor'] == -1.0
    assert 'ihj_magnetic_structure_factor' in dbf.phases['BCC_A2'].model_hints
    assert dbf.phases['BCC_A2'].model_hints['ihj_magnetic_structure_factor'] == 0.4


def test_tdb_parser_raises_unterminated_parameters():
    """A TDB FUNCTION or PARAMETER should give an error if the parsed
    Piecewise expression does not end the line."""
    # The PARAMETER G(BCC,FE:H;0) parameter is not terminated by an `!`.
    # The parser merges all newlines until the `!`, meaning both parameters
    # will be joined on one "line". The parser should raise an error.
    UNTERMINATED_PARAM_STR = """     PARAMETER G(BCC,FE:H;0) 298.15  +GHSERFE+1.5*GHSERHH
        +258000-3170*T+498*T*LN(T)-0.275*T**2; 1811.00  Y
        +232264+82*T+1*GHSERFE+1.5*GHSERHH; 6000.00  N

     PARAMETER G(BCC,FE:VA;0)      298.15 +GHSERFE; 6000 N ZIM !
    """
    with pytest.raises(ParseException):
        Database(UNTERMINATED_PARAM_STR)


@select_database("alfe.tdb")
def test_load_database_when_given_in_lowercase(load_database):
    "Test loading a database coerced to lowercase loads correctly."
    ALFE_TDB = load_database().to_string(fmt='tdb')
    dbf = Database.from_string(ALFE_TDB, fmt='tdb')
    dbf_lower = Database.from_string(ALFE_TDB.lower(), fmt='tdb')

    assert dbf == dbf_lower


def test_reflow_text_raises_on_unbreakable_lines():
    """Long lines without linebreak characters should raise an error."""
    very_long_line = "NOBREAKCHARACTERS" * 50
    with pytest.raises(ValueError):
        reflow_text(very_long_line, 80)


# TODO: when the new database-as-files test fixture is merged, replace with unary 50 proper.
def test_long_constituent_line_writes_correctly():
    TDB = """
    $ From SGTE Unary 50
    ELEMENT /-   ELECTRON_GAS              0.0000E+00  0.0000E+00  0.0000E+00 !
    ELEMENT VA   VACUUM                    0.0000E+00  0.0000E+00  0.0000E+00 !
    ELEMENT AG   FCC_A1                    1.0787E+02  5.7446E+03  4.2551E+01 !
    ELEMENT AL   FCC_A1                    2.6982E+01  4.5773E+03  2.8322E+01 !
    ELEMENT AM   DHCP                      2.4306E+02  0.0000E+00  0.0000E+00 !
    ELEMENT AS   RHOMBOHEDRAL_A7           7.4922E+01  0.0000E+00  0.0000E+00 !
    ELEMENT AU   FCC_A1                    1.9697E+02  6.0166E+03  4.7488E+01 !
    ELEMENT B    BETA_RHOMBO_B             1.0811E+01  1.2220E+03  5.9000E+00 !
    ELEMENT BA   BCC_A2                    1.3733E+02  0.0000E+00  0.0000E+00 !
    ELEMENT BE   HCP_A3                    9.0122E+00  0.0000E+00  0.0000E+00 !
    ELEMENT BI   RHOMBOHEDRAL_A7           2.0898E+02  6.4266E+03  5.6735E+01 !
    ELEMENT C    GRAPHITE                  1.2011E+01  1.0540E+03  5.7400E+00 !
    ELEMENT CA   FCC_A1                    4.0078E+01  6.1965E+03  4.1589E+01 !
    ELEMENT CD   HCP_A3                    1.1241E+02  6.2509E+03  5.1798E+01 !
    ELEMENT CE   FCC_A1                    1.4011E+02  0.0000E+00  0.0000E+00 !
    ELEMENT CO   HCP_A3                    5.8933E+01  0.0000E+00  0.0000E+00 !
    ELEMENT CR   BCC_A2                    5.1996E+01  4.0500E+03  2.3560E+01 !
    ELEMENT CS   BCC_A2                    1.3291E+02  7.7153E+03  8.5149E+01 !
    ELEMENT CU   FCC_A1                    6.3546E+01  5.0041E+03  3.3150E+01 !
    ELEMENT DY   HCP_A3                    1.6250E+02  0.0000E+00  0.0000E+00 !
    ELEMENT ER   HCP_A3                    1.6726E+02  7.3923E+03  7.3178E+01 !
    ELEMENT EU   BCC_A2                    1.5197E+02  0.0000E+00  8.0793E+01 !
    ELEMENT FE   BCC_A2                    5.5847E+01  4.4890E+03  2.7280E+01 !
    ELEMENT GA   ORTHORHOMBIC_GA           6.9723E+01  5.5731E+03  4.0828E+01 !
    ELEMENT GD   HCP_A3                    1.5725E+02  0.0000E+00  0.0000E+00 !
    ELEMENT GE   DIAMOND_A4                7.2610E+01  4.6275E+03  3.1087E+01 !
    ELEMENT HF   HCP_A3                    1.7849E+02  0.0000E+00  0.0000E+00 !
    ELEMENT HG   LIQUID                    2.0059E+02  0.0000E+00  0.0000E+00 !
    ELEMENT HO   HCP_A3                    1.6493E+02  0.0000E+00  0.0000E+00 !
    ELEMENT IN   TETRAGONAL_A6             1.1482E+02  6.6100E+03  5.7650E+01 !
    ELEMENT IR   FCC_A1                    1.9222E+02  5.2677E+03  3.5505E+01 !
    ELEMENT K    BCC_A2                    3.9098E+01  7.0835E+03  6.4672E+01 !
    ELEMENT LA   DHCP                      1.3891E+02  0.0000E+00  0.0000E+00 !
    ELEMENT LI   BCC_A2                    6.9410E+00  4.6233E+03  2.9095E+01 !
    ELEMENT LU   HCP_A3                    1.7497E+02  0.0000E+00  0.0000E+00 !
    ELEMENT MG   HCP_A3                    2.4305E+01  4.9980E+03  3.2671E+01 !
    ELEMENT MN   CBCC_A12                  5.4938E+01  4.9960E+03  3.2008E+01 !
    ELEMENT MO   BCC_A2                    9.5940E+01  4.5890E+03  2.8560E+01 !
    ELEMENT N    1/2_MOLE_N2(G)            1.4007E+01  4.3350E+03  9.5751E+01 !
    ELEMENT NA   BCC_A2                    2.2990E+01  6.4475E+03  5.1447E+01 !
    ELEMENT NB   BCC_A2                    9.2906E+01  5.2200E+03  3.6270E+01 !
    ELEMENT ND   DHCP                      1.4424E+02  0.0000E+00  0.0000E+00 !
    ELEMENT NI   FCC_A1                    5.8690E+01  4.7870E+03  2.9796E+01 !
    ELEMENT NP   ORTHORHOMBIC_AC           2.3705E+02  0.0000E+00  0.0000E+00 !
    ELEMENT O    1/2_MOLE_O2(G)            1.5999E+01  4.3410E+03  1.0252E+02 !
    ELEMENT OS   HCP_A3                    1.9020E+02  0.0000E+00  3.2635E+01 !
    ELEMENT P    WHITE_P                   3.0974E+01  0.0000E+00  0.0000E+00 !
    ELEMENT PA   BCT_AA                    2.3104E+02  0.0000E+00  0.0000E+00 !
    ELEMENT PB   FCC_A1                    2.0720E+02  6.8785E+03  6.4785E+01 !
    ELEMENT PD   FCC_A1                    1.0642E+02  5.4685E+03  3.7823E+01 !
    ELEMENT PR   DHCP                      1.4091E+02  0.0000E+00  0.0000E+00 !
    ELEMENT PT   FCC_A1                    1.9508E+02  5.7237E+03  4.1631E+01 !
    ELEMENT PU   ALPHA_PU                  2.4406E+02  0.0000E+00  0.0000E+00 !
    ELEMENT RB   BCC_A2                    8.5468E+01  7.4894E+03  7.6776E+01 !
    ELEMENT RE   HCP_A3                    1.8621E+02  5.3555E+03  3.6526E+01 !
    ELEMENT RH   FCC_A1                    1.0291E+02  4.9204E+03  3.1505E+01 !
    ELEMENT RU   HCP_A3                    1.0107E+02  4.6024E+03  2.8535E+01 !
    ELEMENT S    ORTHORHOMBIC_S            3.2066E+01  0.0000E+00  0.0000E+00 !
    ELEMENT SB   RHOMBOHEDRAL_A7           1.2175E+02  5.8702E+03  4.5522E+01 !
    ELEMENT SC   HCP_A3                    4.4956E+01  0.0000E+00  0.0000E+00 !
    ELEMENT SE   HEXAGONAL_A8              7.8960E+01  5.5145E+03  4.1966E+01 !
    ELEMENT SI   DIAMOND_A4                2.8085E+01  3.2175E+03  1.8820E+01 !
    ELEMENT SM   RHOMBOHEDRAL_SM           1.5036E+02  0.0000E+00  0.0000E+00 !
    ELEMENT SN   BCT_A5                    1.1871E+02  6.3220E+03  5.1195E+01 !
    ELEMENT SR   FCC_A1                    8.7620E+01  0.0000E+00  0.0000E+00 !
    ELEMENT TA   BCC_A2                    1.8095E+02  5.6819E+03  4.1472E+01 !
    ELEMENT TB   HCP_A3                    1.5893E+02  0.0000E+00  0.0000E+00 !
    ELEMENT TC   HCP_A3                    9.7907E+01  0.0000E+00  0.0000E+00 !
    ELEMENT TE   HEXAGONAL_A8              1.2760E+02  6.1212E+03  4.9497E+01 !
    ELEMENT TH   FCC_A1                    2.3204E+02  0.0000E+00  0.0000E+00 !
    ELEMENT TI   HCP_A3                    4.7880E+01  4.8100E+03  3.0648E+01 !
    ELEMENT TL   HCP_A3                    2.0438E+02  6.8283E+03  6.4183E+01 !
    ELEMENT TM   HCP_A3                    1.6893E+02  7.3973E+03  7.4015E+01 !
    ELEMENT U    ORTHORHOMBIC_A20          2.3803E+02  0.0000E+00  0.0000E+00 !
    ELEMENT V    BCC_A2                    5.0941E+01  4.5070E+03  3.0890E+01 !
    ELEMENT W    BCC_A2                    1.8385E+02  4.9700E+03  3.2620E+01 !
    ELEMENT Y    HCP_A3                    8.8906E+01  0.0000E+00  0.0000E+00 !
    ELEMENT YB   FCC_A1                    1.7304E+02  0.0000E+00  0.0000E+00 !
    ELEMENT ZN   HCP_A3                    6.5390E+01  5.6568E+03  4.1631E+01 !
    ELEMENT ZR   HCP_A3                    9.1224E+01  5.5663E+03  3.9181E+01 !

    PHASE LIQUID % 1 1 !
    CONSTITUENT LIQUID : AG,AL,AM,AS,AU,B,BA,BE,BI,C,CA,CD,CE,CO,CR,CS,CU,DY,ER,
    EU,FE,GA,GD,GE,HF,HG,HO,IN,IR,K,LA,LI,LU,MG,MN,MO,N,NA,NB,ND,NI,NP,O,OS,P,PA,
    PB,PD,PR,PT,PU,RB,RE,RH,RU,S,SB,SC,SE,SI,SM,SN,SR,TA,TB,TC,TE,TH,TI,TL,TM,U,V,
    W,Y,YB,ZN,ZR : !
    """
    dbf = Database(TDB)
    assert len(dbf.elements) == 80
    assert len(dbf.phases['LIQUID'].constituents[0]) == 78  # No VA or /-
    reloaded_dbf = Database(dbf.to_string(fmt='tdb'))
    dbf == reloaded_dbf
    assert len(dbf.elements) == len(reloaded_dbf.elements)
    assert len(dbf.phases['LIQUID'].constituents[0]) == len(reloaded_dbf.phases['LIQUID'].constituents[0])

def test_database_passes_with_diffusion_commands():
    "DIFFUSION and ZEROVOLUME_SPECIES commands do not raise errors while parsing (doesn't test implementation)."
    tdb_string = """
     ELEMENT /-   ELECTRON_GAS              0.0000E+00  0.0000E+00  0.0000E+00 !
     ELEMENT VA   VACUUM                    0.0000E+00  0.0000E+00  0.0000E+00 !

     ZEROVOLUME_SPECIES VA !
     DIFFUSION MAGNETIC BCC_A2 ALPHA=0.3 !
    """

    dbf = Database.from_string(tdb_string, fmt='tdb')

def test_tc_printer_no_division_symbols():
    "TCPrinter does not produce division symbols in string output of symbolic expressions."
    test_expr = Piecewise((S('VV0000/T + T*VV0004 + T**2*VV0001 + T**3*VV0002 + T*LOG(T)*VV0003'), v.T>0))
    result = TCPrinter().doprint(test_expr)
    assert '/' not in result

def test_tc_printer_exp():
    "TCPrinter prints the exponential function when the argument is not an integer."
    test_expr = S('exp(-300T**(-1))')
    result = TCPrinter()._stringify_expr(test_expr)
    assert result == 'exp(-300 * T**(-1))'


@select_database("Al-Fe_sundman2009.tdb")
def test_database_symmetry_options_are_generated(load_database):
    """Loading a database with option B and F generates new ordering parameters."""
    # Tests for the correctness of which configurations are generated is done
    # by testing generate_symmetric_group
    # This test aims to see that the parameters are loaded (resulting in a
    # particular number of parameters) and that removing those parameters is
    # possible.
    dbf = load_database()
    assert len(dbf._parameters) == 375  # number after adding automatically

    # The parameters should be filtered out when when writing such that a
    # read/write is a no-op
    read_dbf = Database.from_string(dbf.to_string(fmt="tdb"), fmt="tdb")
    assert len(read_dbf._parameters) == 375