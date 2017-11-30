"""
The test_database module contains tests for the Database object.
"""
from __future__ import print_function
import warnings
import hashlib
import os
from copy import deepcopy
from pyparsing import ParseException
from sympy import Symbol, Piecewise, And
from pycalphad import Database, Model, variables as v
from pycalphad.io.database import FileExistsError
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
        assert len(w) >= 1
        expected_string_fragment = 'Ignoring that the following function names are beyond the 8 character TDB limit'
        assert any([expected_string_fragment in str(warning.message) for warning in w])
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
        assert len(w) >= 1
        expected_string_fragment = 'Ignoring that the following function names are beyond the 8 character TDB limit'
        assert any([expected_string_fragment in str(warning.message) for warning in w])
    assert test_dbf == Database.from_string(invalid_dbf, fmt='tdb')

def test_incompatible_db_ignores_with_kwarg_ignore():
    "Symbol names too long for Thermo-Calc are ignored the database written as given."
    test_dbf = Database.from_string(INVALID_TDB_STR, fmt='tdb')
    with warnings.catch_warnings(record=True) as w:
        invalid_dbf = test_dbf.to_string(fmt='tdb', if_incompatible='ignore')
        not_expected_string_fragment = 'Ignoring that the following function names are beyond the 8 character TDB limit'
        assert all([not_expected_string_fragment not in str(warning.message) for warning in w])
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

def _remove_file_with_name_testwritedb():
    fname = 'testwritedb.tdb'
    os.remove(fname)

@nose.tools.with_setup(None, _remove_file_with_name_testwritedb)
@nose.tools.raises(FileExistsError)
def test_to_file_defaults_to_raise_if_exists():
    "Attempting to use Database.to_file should raise by default if it exists"
    fname = 'testwritedb.tdb'
    test_dbf = Database(ALNIPT_TDB)
    test_dbf.to_file(fname)  # establish the initial file
    test_dbf.to_file(fname)  # test if_exists behavior

@nose.tools.with_setup(None, _remove_file_with_name_testwritedb)
@nose.tools.raises(FileExistsError)
def test_to_file_raises_with_bad_if_exists_argument():
    "Database.to_file should raise if a bad behavior string is passed to if_exists"
    fname = 'testwritedb.tdb'
    test_dbf = Database(ALNIPT_TDB)
    test_dbf.to_file(fname)  # establish the initial file
    test_dbf.to_file(fname, if_exists='TEST_BAD_ARGUMENT')  # test if_exists behavior

@nose.tools.with_setup(None, _remove_file_with_name_testwritedb)
def test_to_file_overwrites_with_if_exists_argument():
    "Database.to_file should overwrite if 'overwrite' is passed to if_exists"
    fname = 'testwritedb.tdb'
    test_dbf = Database(ALNIPT_TDB)
    test_dbf.to_file(fname)  # establish the initial file
    inital_modification_time = os.path.getmtime(fname)
    test_dbf.to_file(fname, if_exists='overwrite')  # test if_exists behavior
    overwrite_modification_time = os.path.getmtime(fname)
    assert overwrite_modification_time > inital_modification_time

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
    assert test_dbf_reread.phases['TEST_PH'].constituents[0] == {'AL', 'AL2', 'O-2'}
    assert len(test_dbf_reread._parameters.search(where('constituent_array') == (('AL',),))) == 1
    assert len(test_dbf_reread._parameters.search(where('constituent_array') == (('AL2',),))) == 1
    assert len(test_dbf_reread._parameters.search(where('constituent_array') == (('O-2',),))) == 1

    assert test_dbf_reread.phases['T2SL'].constituents == ({'AL+3'}, {'O-2'})
    assert len(test_dbf_reread._parameters.search(where('constituent_array') == (('AL+3',),('O-2',)))) == 1

@nose.tools.raises(ParseException)
def test_tdb_missing_terminator_element():
    tdb_str = """$ Note missing '!' in next line
               ELEMENT ZR   BCT_A5
               FUNCTION EMBCCTI    298.15 -39.72; 6000 N !"""
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
    $ ------------------------------------------------------------------------------
    $ Phase definitions
    $
     PHASE LIQUID:L % 1 1 !
     CONST LIQUID:L : AL C CO CR FE MN NI : !
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
    $ Prototype AuCu3 (cP4, Pm-3m, L1_2) and AuCu (tP4, P4/mmm, L1_0)
    $
     PHASE FCC_4SL:F %AY 5 0.25 0.25 0.25 0.25 1 !
     CONST FCC_4SL:F : AL CO CR FE MN NI : AL CO CR FE MN NI
                     : AL CO CR FE MN NI : AL CO CR FE MN NI : C VA% :  !
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
    $
    $ Prototype C (cF8, Fd-3m)
    $
     PHASE DIAMOND_A4 % 1 1 !
     CONST DIAMOND_A4 : C : !
    $
    $ Prototype C (hP4, P6_3/mmc)
    $
     PHASE GRAPHITE_A9 % 1 1 !
     CONST GRAPHITE_A9 : C : !
    $
    $ Prototype alpha-Mn (cI58, I-43m)
    $
     PHASE CBCC_A12 %A 2 1 1 !
     CONST CBCC_A12 : AL CO CR FE MN% NI : C VA% : !
    $
    $ Prototype beta-Mn (cP20, P4_132)
    $
     PHASE CUB_A13 % 2 1 1 !
     CONST CUB_A13 : AL CO CR FE MN% NI : C VA% : !
    $
    $ Prototype Al4C3 (hR7, R-3m)
    $
     PHASE AL4C3_D71 % 2 4 3 !
     CONST AL4C3_D71 : AL : C : !
    $
    $ Prototype Al5Co2 (hP28, P6_3/mmc), also Al10Fe3Ni
    $
     PHASE AL5CO2_D811 % 2 5 2 !
     CONST AL5CO2_D811 : AL : CO FE NI : !
    $
    $ Unknown structure
    $
     PHASE AL3CO % 2 3 1 !
     CONST AL3CO : AL : CO : !
    $
    $ Prototype Al19Co6 (mC100, C2/m)
    $
     PHASE AL13CO4 % 2 13 4 !
     CONST AL13CO4 : AL : CO : !
    $
    $ Prototype Al9Co2 (mP22, P2_1/c), also Al9FeNi
    $
     PHASE AL9CO2 % 2 9 2 !
     CONST AL9CO2 : AL : CO FE NI : !
    $
    $ Prototype Al45V7 (mC104, C2/m)
    $
     PHASE AL13CR2 % 2 13 2 !
     CONST AL13CR2 : AL : CR : !
    $
    $ Prototype Al5Cr (mC732, C2/c)
    $
     PHASE AL11CR2 % 3 10 1 2 !
     CONST AL11CR2 : AL : AL : CR :  !
    $
    $ Prototype Al4Mn-mu (hP574, P6_3/mmc)
    $
     PHASE AL4CR % 2 4 1 !
     CONST AL4CR : AL : CR : !
    $
    $ Unknown structure
    $
     PHASE AL9CR4_H % 2 9 4 !
     CONST AL9CR4_H : AL : CR : !
    $
    $ Prototype Al9Cr4 (cI52, I-43m ?)
    $
     PHASE AL9CR4_L % 2 9 4 !
     CONST AL9CR4_L : AL : CR : !
    $
    $ Prototype Cu5Zn8 (cI52, I-43m)
    $
     PHASE AL8CR5_D82 % 2 8 5 !
     CONST AL8CR5_D82 : AL : CR : !
    $
    $ Prototype Al8Cr5 (hR26, R3m)
    $
     PHASE AL8CR5_D810 % 2 8 5 !
     CONST AL8CR5_D810 : AL : CR : !
    $
    $ Prototype MoSi2 (tI6, I4/mmm)
    $
     PHASE ALCR2_C11B % 2 1 2 !
     CONST ALCR2_C11B : AL : CR : !
    $
    $ Prototype Al13Fe4 (mC102, C2/m)
    $
     PHASE AL13FE4 % 3 0.6275 0.235 0.1375 !
     CONST AL13FE4 : AL : FE% MN NI : AL VA : !
    $
    $ Prototype Al5Fe2 (oC24, Cmcm)
    $ Al5Fe2 does not have the Al5Co2 (D8_11) structure
    $
     PHASE AL5FE2 % 2 5 2 !
     CONST AL5FE2 : AL : FE% NI : !
    $
    $ Prototype Al2Fe (aP18, P1)
    $
     PHASE AL2FE % 2 2 1 !
     CONST AL2FE : AL : FE% NI : !
    $
    $ Prototype Cu8Zn5 (cI52, I-43m)
    $
     PHASE AL8FE5_D82 % 2 8 5 !
     CONST AL8FE5_D82 : AL FE : AL FE : !
    $
    $ Decagonal (quasicrystal)
    $
     PHASE AL71FE5NI24 % 3 0.71 0.05 0.24 !
     CONST AL71FE5NI24 : AL : FE : NI : !
    $
    $ Prototype Al12W (cI26, Im-3)
    $
     PHASE AL12MN % 2 12 1 !
     CONST AL12MN : AL : MN : !
    $
    $ Prototype Al6Mn (oC28, Cmcm)
    $
     PHASE AL6MN_D2H % 2 6 1 !
     CONST AL6MN_D2H : AL : FE MN% : !
    $
    $ Prototype lambda-Al4Mn (hP586, P6_3/m)
    $
     PHASE AL4MN_LAMBDA % 2 0.81162 0.18838 !
     CONST AL4MN_LAMBDA : AL : MN : !
    $
    $ Prototype mu-Al4Mn (hP574, P6_3/mmc)
    $
     PHASE AL4MN_MU % 2 4 1 !
     CONST AL4MN_MU : AL : FE MN% : !
    $
    $ Prototype Al11Mn4 (aP15, P-1)
    $
     PHASE AL11MN4_LT % 2 11 4 !
     CONST AL11MN4_LT : AL : MN : !
    $
    $ Prototype Al3Mn (oP156, Pnma)
    $
     PHASE AL11MN4_HT % 3 9 2 2 !
     CONST AL11MN4_HT : AL : AL MN : MN : !
    $
    $ Prototype Al8Cr5 (hR26, R3m)
    $
     PHASE AL8MN5_D810 % 3 12 5 9 !
     CONST AL8MN5_D810 : AL : MN : AL FE MN : !
    $
    $ Prototype Fe3C (oP16, Pnma)
    $
     PHASE AL3NI_D011 % 2 0.75 0.25 !
     CONST AL3NI_D011 : AL : FE NI% : !
    $
    $ Prototype Al3Ni2 (hP5, P-3m1)
    $
     PHASE AL3NI2_D513 % 3 3 2 1 !
     CONST AL3NI2_D513 : AL : AL FE NI% : NI VA% : !
    $
    $ Prototype Ga4Ni3 (cI112, Ia-3d)
    $
     PHASE AL4NI3 % 2 4 3 !
     CONST AL4NI3 : AL : NI : !
    $
    $ Prototype Ga3Pt5 (oC16, Cmmm)
    $
     PHASE AL3NI5 % 2 0.375 0.625 !
     CONST AL3NI5 : AL : NI : !
    $
    $ Prototype Cr3C2 (oP20, Pnma)
    $
     PHASE CR3C2_D510 % 2 3 2 !
     CONST CR3C2_D510 : CO CR% : C : !
    $
    $ Similar to alpha-Mn (cI58)
    $
     PHASE CR3MN5 % 2 3 5 !
     CONST CR3MN5 : CR : MN : !
    $
    $ Prototype MoPt2 (oP6, Immm)
    $
     PHASE CRNI2 % 2 1 2 !
     CONST CRNI2 : CR : NI : !
    $
    $ Prototype Cr23C6 (cF116, Fm-3m)
    $
     PHASE M23C6_D84 % 3 20 3 6 !
     CONST M23C6_D84 : CO CR FE MN NI : CO CR FE MN NI : C : !
    $
    $ Prototype Fe3C (oP16, Pnma)
    $
     PHASE CEMENTITE_D011 % 2 3 1 !
     CONST CEMENTITE_D011 : CO CR FE MN NI : C : !
    $
    $ Prototype Mn5C2 (mC28, C2/c), Haegg carbide, chi
    $
     PHASE M5C2 % 2 5 2 !
     CONST M5C2 : FE MN : C : !
    $
    $ Prototype Cr7C3 (oP40, Pnma)
    $
     PHASE M7C3_D101 % 2 7 3 !
     CONST M7C3_D101 : CO CR FE MN NI : C : !
    $
    $ Prototype CaTiO3 (cP5, Pm-3m)
    $
     PHASE KAPPA_E21 % 3 1 3 1 !
     CONST KAPPA_E21 : AL : CO FE MN NI : C% VA : !
    $
    $ Prototype Cr2AlC (hP8, P6_3/mmc)
    $
     PHASE MAX_PHASE % 3 2 1 1 !
     CONST MAX_PHASE : AL CR% : AL : C : !
    $
    $ Prototype CrFe (tP30, P4_2/mnm)
    $
     PHASE SIGMA_D8B % 3 10 4 16 !
     CONST SIGMA_D8B : AL CO FE MN NI : CR : AL CO CR FE MN NI : !
    $
    $ Prototype CrFe (tP30, P4_2/mnm, D8b)
    $
     PHASE HIGH_SIGMA % 3 10 4 16 !
     CONST HIGH_SIGMA : AL CO FE MN NI : CR : AL CO FE CR MN NI : !
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
     TYPE-DEF Y GES AMEND_PHASE_DESCRIPTION FCC_4SL DIS_PART A1_FCC !
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
     PAR  G(FCC_A1,AL:VA),,                 +GHSERAL;                2900 N 91Din !
     PAR  G(A1_FCC,AL:VA),,                 +GHSERAL;                2900 N 91Din !
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