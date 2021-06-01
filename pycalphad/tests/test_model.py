"""
The test_model module contains unit tests for the Model object.
"""
from pycalphad import Database, Model, variables as v, equilibrium
from pycalphad.tests.datasets import ALCRNI_TDB, ALNIPT_TDB, ALFE_TDB, ZRO2_CUBIC_BCC_TDB, TDB_PARAMETER_FILTERS_TEST
from pycalphad.core.errors import DofError
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
    assert np.isclose(eqx.degree_of_ordering.sel(vertex=0).values.flatten(), np.array([0.6666]), atol=1e-4)

def test_detect_pure_vacancy_phases():
    "Detecting a pure vacancy phase"
    dbf = Database(ZRO2_CUBIC_BCC_TDB)
    with pytest.raises(DofError):
        Model(dbf,['AL','CU','VA'],'ZRO2_CUBIC')


def test_constituents_not_in_model():
    """Test that parameters with constituent arrays not matching the phase model are filtered out correctly"""
    dbf = Database(TDB_PARAMETER_FILTERS_TEST)
    modA = Model(dbf, ['A', 'B'], 'ALPHA')
    modB = Model(dbf, ['B', 'C'], 'BETA')
    assert v.SiteFraction('ALPHA', 0, 'B') not in modA.ast.free_symbols
    assert v.SiteFraction('BETA', 1, 'D') not in modB.ast.free_symbols
    assert v.SiteFraction('BETA', 2, 'C') not in modB.ast.free_symbols


def test_order_disorder_interstital_sublattice_validation():
    # Check that substitutional/interstitial sublattices that break our
    # assumptions raise errors
    DBF_OrderDisorder_broken = Database("""
    ELEMENT VA   VACUUM   0.0000E+00  0.0000E+00  0.0000E+00 !
    ELEMENT A    DISORD     0.0000E+00  0.0000E+00  0.0000E+00 !
    ELEMENT B    DISORD     0.0000E+00  0.0000E+00  0.0000E+00 !
    ELEMENT C    DISORD     0.0000E+00  0.0000E+00  0.0000E+00 !

    DEFINE_SYSTEM_DEFAULT ELEMENT 2 !
    DEFAULT_COMMAND DEF_SYS_ELEMENT VA !

    TYPE_DEFINITION % SEQ *!
    TYPE_DEFINITION ' GES A_P_D ORD_MORE_INSTL DIS_PART DISORD ,,,!
    TYPE_DEFINITION & GES A_P_D ORD_LESS_INSTL DIS_PART DISORD ,,,!
    TYPE_DEFINITION ) GES A_P_D ORD_SUBS_INSTL DIS_PART DISORD ,,,!

    PHASE DISORD  %  2 1   3 !
    CONSTITUENT DISORD  : A,B,VA : VA :  !

    $ Has one more interstitial sublattice than disordered:
    PHASE ORD_MORE_INSTL %'  4 0.5  0.5  3 1 !
    CONSTITUENT ORD_MORE_INSTL  : A,B,VA : A,B,VA : VA : A : !

    $ Has one less interstitial sublattice than disordered:
    PHASE ORD_LESS_INSTL %&  2 0.5  0.5 !
    CONSTITUENT ORD_LESS_INSTL  : A,B,VA : A,B,VA : !

    $ The interstitial sublattice has the same species as the substitutional
    $ and cannot be distinguished:
    PHASE ORD_SUBS_INSTL %)  3 0.5  0.5 3 !
    CONSTITUENT ORD_SUBS_INSTL  : A,B,VA : A,B,VA : A,B,VA :  !

    """)

    # Case 1: Ordered phase has one more interstitial sublattice than disordered
    with pytest.raises(ValueError):
        Model(DBF_OrderDisorder_broken, ["A", "B", "VA"], "ORD_MORE_INSTL")

    # Case 2: Ordered phase has one more interstitial sublattice than disordered
    with pytest.raises(ValueError):
        Model(DBF_OrderDisorder_broken, ["A", "B", "VA"], "ORD_LESS_INSTL")

    # Case 3: The ordered phase has interstitial sublattice has the same species
    # as the substitutional and cannot be distinguished
    with pytest.raises(ValueError):
        Model(DBF_OrderDisorder_broken, ["A", "B", "VA"], "ORD_SUBS_INSTL")
