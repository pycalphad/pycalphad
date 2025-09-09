"""
The test_model module contains unit tests for the Model object.
"""
from pycalphad import Database, Model, variables as v, equilibrium
from pycalphad.tests.fixtures import select_database, load_database
from pycalphad.core.errors import DofError
from symengine import Piecewise
import numpy as np
import pickle
import pytest


@select_database("alcrni.tdb")
def test_model_eq(load_database):
    "Model equality comparison."
    dbf = load_database()
    test_model = Model(dbf, ['AL', 'CR'], 'L12_FCC')
    assert test_model == test_model
    assert test_model == Model(dbf, ['AL', 'CR'], 'L12_FCC')
    assert not (test_model == Model(dbf, ['NI', 'CR'], 'L12_FCC'))
    # literals which don't have __dict__
    assert not (test_model == 42)
    assert not (test_model == None)
    assert not (42 == test_model)
    assert not (None == test_model)


@select_database("alcrni.tdb")
def test_model_ne(load_database):
    "Model inequality comparison."
    dbf = load_database()
    test_model = Model(dbf, ['AL', 'CR'], 'L12_FCC')
    assert not (test_model != test_model)
    assert not (test_model != Model(dbf, ['AL', 'CR'], 'L12_FCC'))
    assert test_model != Model(dbf, ['NI', 'CR'], 'L12_FCC')
    # literals which don't have __dict__
    assert test_model != 42
    assert test_model != None
    assert 42 != test_model
    assert None != test_model


@select_database("alnipt.tdb")
def test_export_import(load_database):
    "Equivalence of Model using re-imported database."
    dbf = load_database()
    test_model = Model(Database.from_string(dbf.to_string(fmt='tdb', if_incompatible='ignore'), fmt='tdb'), ['PT', 'NI', 'VA'], 'FCC_L12')
    ref_model = Model(dbf, ['NI', 'PT', 'VA'], 'FCC_L12')
    assert test_model == ref_model


@select_database("alnipt.tdb")
def test_model_pickle(load_database):
    "Model pickle roundtrip."
    dbf = load_database()
    test_model = Model(dbf, ['NI', 'PT', 'VA'], 'FCC_L12')
    new_model = pickle.loads(pickle.dumps(test_model))
    assert test_model == new_model


@select_database("alcrni.tdb")
def test_custom_model_contributions(load_database):
    "Building a custom model using contributions."
    dbf = load_database()
    class CustomModel(Model):
        contributions = [('zzz', 'test'), ('xxx', 'test2'), ('yyy', 'test3')]
        def test(self, dbe):
            return 0
        def test2(self, dbe):
            return 0
        def test3(self, dbe):
            return 0
    CustomModel(dbf, ['AL', 'CR'], 'L12_FCC')


@select_database("alfe.tdb")
def test_degree_of_ordering(load_database):
    "Degree of ordering should be calculated properly."
    dbf = load_database()
    my_phases = ['B2_BCC']
    comps = ['AL', 'FE', 'VA']
    conds = {v.T: 300, v.P: 101325, v.X('AL'): 0.25}
    eqx = equilibrium(dbf, comps, my_phases, conds, output='degree_of_ordering', verbose=True)
    print('Degree of ordering: {}'.format(eqx.degree_of_ordering.values.flatten()))
    assert np.isclose(eqx.degree_of_ordering.values.flatten(), np.array([0.6666]), atol=1e-4)

def test_detect_pure_vacancy_phases():
    "Detecting a pure vacancy phase"
    ZRO2_CUBIC_BCC_TDB = """
        ELEMENT /-   ELECTRON_GAS              0.0000E+00  0.0000E+00  0.0000E+00!
        ELEMENT VA   VACUUM                    0.0000E+00  0.0000E+00  0.0000E+00!
        ELEMENT AL   FCC_A1                    2.6982E+01  4.5773E+03  2.8321E+01!
        ELEMENT CU   FCC_A1                    6.3546E+01  5.0041E+03  3.3150E+01!
        ELEMENT O    1/2_MOLE_O2(G)            1.5999E+01  4.3410E+03  1.0252E+02!
        ELEMENT ZR   HCP_A3                    9.1224E+01  5.5663E+03  3.9181E+01!

        PHASE BCC_A2  %  2 1   3 !
            CONSTITUENT BCC_A2  :AL,CU,ZR : O,VA% :  !

        PHASE ZRO2_CUBIC  %  2 1   2 !
            CONSTITUENT ZRO2_CUBIC  :VA,ZR% : O%,VA :  !
    """
    dbf = Database(ZRO2_CUBIC_BCC_TDB)
    with pytest.raises(DofError):
        Model(dbf,['AL','CU','VA'],'ZRO2_CUBIC')


@select_database("parameter_filter_test.tdb")
def test_constituents_not_in_model(load_database):
    """Test that parameters with constituent arrays not matching the phase model are filtered out correctly"""
    dbf = load_database()
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

@pytest.mark.filterwarnings("ignore:The order-disorder model for \"ORD_FCC\" has a contribution from the physical property model*:UserWarning")
@select_database("FeNi_deep_branching.tdb")
def test_model_deep_branching(load_database):
    "Models with very deep piecewise branching are optimized at construction time"
    dbf = load_database()
    mod = Model(dbf, ['FE', 'NI', 'VA'], 'ORD_FCC')
    # All we really care about is that the energy Hessian will calculate without hanging
    # However, that is a relatively long test. This just checks that the deep branches were cleaned up.
    # Without optimization/unwrapping, this would be about 57
    assert len(mod.GM.atoms(Piecewise)) < 30

def test_model_extrapolate_temperature():
    "Models extrapolate temperature bounds outside upper/lower limits"
    TDB_extrapolate = """
    ELEMENT VA   VACUUM   0.0000E+00  0.0000E+00  0.0000E+00 !
    ELEMENT A    DISORD     0.0000E+00  0.0000E+00  0.0000E+00 !
    ELEMENT B    DISORD     0.0000E+00  0.0000E+00  0.0000E+00 !

    FUNCTION GSYM1  500 +100;  5000.00 Y 40000; 10000 N !

    DEFINE_SYSTEM_DEFAULT ELEMENT 2 !
    DEFAULT_COMMAND DEF_SYS_ELEMENT VA !

    TYPE_DEFINITION % SEQ *!
    TYPE_DEFINITION ' GES A_P_D ORDERED DIS_PART DISORD ,,,!

    PHASE DISORD  %  2 1   3 !
    PHASE ORDERED %'  3 0.5  0.5  3  !

    CONSTITUENT DISORD  : A,B : VA :  !
    CONSTITUENT ORDERED  : A,B: A,B : VA :  !

    PARAMETER G(DISORD,A:VA;0)  298.15  -10000+GSYM1#; 1000 Y -3000-GSYM1#; 6000 N !
    PARAMETER G(DISORD,B:VA;0)  298.15  -10000+GSYM1#; 1000 Y -3000-GSYM1#; 6000 N !

    """
    dbf = Database(TDB_extrapolate)
    # First, confirm that turning the extrapolation off reproduces the correct behavior
    class modtype(Model):
        extrapolate_temperature_bounds = False
    mod = modtype(dbf, ['A', 'B', 'VA'], 'ORDERED')
    # Remove ideal mixing effects so we can test extrapolation easily
    mod.models['idmix'] = 0
    dof = {v.Y('ORDERED', 0, 'A'): 0.5, v.Y('ORDERED', 0, 'B'): 0.5,
               v.Y('ORDERED', 1, 'A'): 0.5, v.Y('ORDERED', 1, 'B'): 0.5,
               v.Y('ORDERED', 2, 'VA'): 1.0}
    assert mod.GM.subs(dof).subs({v.T: 200}).n(real=True) == 0.
    assert mod.GM.subs(dof).subs({v.T: 50000}).n(real=True) == 0.

    # Next, test that the default behavior (extrapolation) works as intended
    mod = Model(dbf, ['A', 'B', 'VA'], 'ORDERED')
    # Remove ideal mixing effects so we can test extrapolation easily
    mod.models['idmix'] = 0
    dof = {v.Y('ORDERED', 0, 'A'): 0.5, v.Y('ORDERED', 0, 'B'): 0.5,
               v.Y('ORDERED', 1, 'A'): 0.5, v.Y('ORDERED', 1, 'B'): 0.5,
               v.Y('ORDERED', 2, 'VA'): 1.0}
    assert mod.GM.subs(dof).subs({v.T: 200}).n(real=True) == -9900.
    assert mod.GM.subs(dof).subs({v.T: 50000}).n(real=True) == -43000.

def test_error_raised_for_higher_order_reciprocal_parameter():
    """
    The composition dependent reciprocal parameter on more than two sublattices
    should not be supported and should raise an error if a database has these parameters
    """
    test_tdb = """
    ELEMENT VA   VACUUM                    0.0000E+00  0.0000E+00  0.0000E+00!
    ELEMENT C    GRAPHITE                  1.2011E+01  1.0540E+03  5.7423E+00!
    ELEMENT MO   BCC_A2                    9.5940E+01  4.5890E+03  2.8560E+01!
    ELEMENT NB   BCC_A2                    9.2906E+01  5.2200E+03  3.6270E+01!
    ELEMENT AL   BCC_A2                    9.2906E+01  5.2200E+03  3.6270E+01!

    PHASE PHASE_CORRECT % 3 1 1 1 !
    CONSTITUENT PHASE_CORRECT : MO,NB : NB,AL : C,VA : !

    PARAMETER G(PHASE_CORRECT,MO,NB:NB,AL:C,VA;0) 298.15 -300000; 6000 N !

    PHASE PHASE_SUBLATTICE % 3 1 1 1 !
    CONSTITUENT PHASE_SUBLATTICE : MO,NB : NB,AL : C,VA : !

    PARAMETER G(PHASE_SUBLATTICE,MO,NB:NB,AL:C,VA;1) 298.15 -300000; 6000 N !

    PHASE PHASE_HIGH_ORDER % 3 1 1 1 !
    CONSTITUENT PHASE_HIGH_ORDER : MO,NB : C,VA : !

    PARAMETER G(PHASE_HIGH_ORDER,MO,NB:C,VA;3) 298.15 -300000; 6000 N !
    """
    dbf = Database(test_tdb)
    # Check that a model with 0th order reciprocal parameter loads correctly
    mod = Model(dbf, ["AL", "MO", "NB", "C", "VA"], "PHASE_CORRECT")

    # Check that a model with a higher order reciprocal parameter on three sublattices throws an error
    with pytest.raises(ValueError):
        mod = Model(dbf, ["AL", "MO", "NB", "C", "VA"], "PHASE_SUBLATTICE")

    # Check that a model with a higher order reciprocal parameter > 2 throws an error
    with pytest.raises(ValueError):
        mod = Model(dbf, ["MO", "NB", "C", "VA"], "PHASE_HIGH_ORDER")
