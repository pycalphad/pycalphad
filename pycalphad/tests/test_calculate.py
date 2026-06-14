"""
The calculate test module verifies that calculate() calculates
Model quantities correctly.
"""

import pytest
from pycalphad import Database, calculate, Model, variables as v
from itertools import chain
import numpy as np
from numpy.testing import assert_allclose
from pycalphad.codegen.phase_record_factory import PhaseRecordFactory
from pycalphad.core.calculate import _jacobian_from_constraints
from pycalphad.core.constants import MIN_SITE_FRACTION
from pycalphad.core.polytope import sample
from pycalphad.core.utils import instantiate_models
from pycalphad import ConditionError
from pycalphad.tests.fixtures import select_database, load_database


@select_database("alcrni.tdb")
def test_surface(load_database):
    "Bare minimum: calculation produces a result."
    dbf = load_database()
    calculate(dbf, ['AL', 'CR', 'NI'], 'L12_FCC',
                T=1273., mode='numpy')


@select_database("alcrni.tdb")
def test_unknown_model_attribute(load_database):
    "Sampling an unknown model attribute raises exception."
    dbf = load_database()
    with pytest.raises(AttributeError):
        calculate(dbf, ['AL', 'CR', 'NI'], 'L12_FCC', T=1400.0, output='_fail_')


@select_database("alcrni.tdb")
def test_statevar_upcast(load_database):
    "Integer state variable values are cast to float."
    dbf = load_database()
    calculate(dbf, ['AL', 'CR', 'NI'], 'L12_FCC',
                T=1273, mode='numpy')


@select_database("alcrni.tdb")
def test_points_kwarg_multi_phase(load_database):
    "Multi-phase calculation works when internal dof differ (gh-41)."
    dbf = load_database()
    calculate(dbf, ['AL', 'CR', 'NI'], ['L12_FCC', 'LIQUID'],
                T=1273, points={'L12_FCC': [0.20, 0.05, 0.75, 0.05, 0.20, 0.75]}, mode='numpy')


@select_database("alcrni.tdb")
def test_issue116(load_database):
    "Calculate gives correct result when a state variable is left as default (gh-116)."
    dbf = load_database()
    result_one = calculate(dbf, ['AL', 'CR', 'NI'], 'LIQUID', T=400)
    result_one_values = result_one.GM.values
    result_two = calculate(dbf, ['AL', 'CR', 'NI'], 'LIQUID', T=400, P=101325)
    result_two_values = result_two.GM.values
    result_three = calculate(dbf, ['AL', 'CR', 'NI'], 'LIQUID', T=400, P=101325, N=1)
    result_three_values = result_three.GM.values
    np.testing.assert_array_equal(np.squeeze(result_one_values), np.squeeze(result_two_values))
    np.testing.assert_array_equal(np.squeeze(result_one_values), np.squeeze(result_three_values))
    # N is not a coordinate of calculate(); only P and T are.
    # An explicit N kwarg is ignored (filtered out) and does not add a dimension.
    assert len(result_one_values.shape) == 2  # T, points
    assert result_one_values.shape[0] == 1
    assert len(result_two_values.shape) == 3  # P, T, points
    assert result_two_values.shape[:2] == (1, 1)
    assert len(result_three_values.shape) == 3  # P, T, points (N ignored)
    assert result_three_values.shape[:2] == (1, 1)
    # N is not a coordinate of the calculate() output
    assert 'N' not in result_one.coords
    assert 'N' not in result_three.coords

    # NOT passing temperature when temperature is required by the Model raises
    with pytest.raises(ConditionError):
        calculate(dbf, ['AL', 'CR', 'NI'], 'LIQUID', P=101325, pdens=10)  # no T


@select_database("alfe.tdb")
def test_calculate_some_phases_filtered(load_database):
    """
    Phases are filtered out from calculate() when some cannot be built.
    """
    dbf = load_database()
    # should not raise; AL13FE4 should be filtered out
    calculate(dbf, ['AL', 'VA'], ['FCC_A1', 'AL13FE4'], T=1200, P=101325)


@select_database("alfe.tdb")
def test_calculate_raises_with_no_active_phases_passed(load_database):
    """Passing inactive phases to calculate() raises a ConditionError."""
    dbf = load_database()
    # Phase cannot be built without FE
    with pytest.raises(ConditionError):
        calculate(dbf, ['AL', 'VA'], ['AL13FE4'], T=1200, P=101325)


@select_database("cumg_parameters.tdb")
def test_calculate_with_parameters_vectorized(load_database):
    # Second set of parameter values are directly copied from the TDB
    dbf = load_database()
    parameters = {'VV0000': [-33134.699474175846, -32539.5], 'VV0001': [7734.114029426941, 8236.3],
                  'VV0002': [-13498.542175596054, -14675.0], 'VV0003': [-26555.048975092268, -24441.2],
                  'VV0004': [20777.637577083482, 20149.6], 'VV0005': [41915.70425630003, 46500.0],
                  'VV0006': [-34525.21964215504, -39591.3], 'VV0007': [95457.14639216446, 104160.0],
                  'VV0008': [21139.578967453144, 21000.0], 'VV0009': [19047.833726419598, 17772.0],
                  'VV0010': [20468.91829601273, 21240.0], 'VV0011': [19601.617855958328, 14321.1],
                  'VV0012': [-4546.9325861738, -4923.18], 'VV0013': [-1640.6354331231278, -1962.8],
                  'VV0014': [-35682.950005357634, -31626.6]}
    res = calculate(dbf, ['CU', 'MG'], ['HCP_A3'], parameters=parameters, T=743.15, P=1e5)
    res_noparams = calculate(dbf, ['CU', 'MG'], ['HCP_A3'], parameters=None, T=743.15, P=1e5)
    param_values = []
    for symbol in sorted(parameters.keys()):
        param_values.append(parameters[symbol])
    param_values = np.array(param_values).T
    assert all(res['param_symbols'] == sorted([str(x) for x in parameters.keys()]))
    assert_allclose(np.squeeze(res['param_values'].values), param_values)
    assert_allclose(res.GM.isel(samples=1).values, res_noparams.GM.values)


@select_database("alcrni.tdb")
def test_incompatible_model_instance_raises(load_database):
    "Calculate raises when an incompatible Model instance built with a different phase is passed."
    dbf = load_database()
    comps = ['AL', 'CR', 'NI']
    phase_name = 'L12_FCC'
    mod = Model(dbf, comps, 'LIQUID')  # Model instance does not match the phase
    with pytest.raises(ValueError):
        calculate(dbf, comps, phase_name, T=1400.0, output='_fail_', model=mod)


@select_database("alcrni.tdb")
def test_single_model_instance_raises(load_database):
    "Calculate raises when a single Model instance is passed with multiple phases."
    dbf = load_database()
    comps = ['AL', 'CR', 'NI']
    mod = Model(dbf, comps, 'L12_FCC')  # Model instance does not match the phase
    with pytest.raises(ValueError):
        calculate(dbf, comps, ['LIQUID', 'L12_FCC'], T=1400.0, output='_fail_', model=mod)


@select_database("alfe.tdb")
def test_missing_phase_records_passed_to_calculate_raises(load_database):
    "calculate should raise an error if all the active phases are not included in the phase_records"
    dbf = load_database()
    my_phases = ['LIQUID', 'FCC_A1']
    subset_phases = ['FCC_A1']
    comps = ['AL', 'FE', 'VA']

    models = instantiate_models(dbf, comps, subset_phases)
    # Dummy conditions are just needed to get all the state variables into the PhaseRecord objects
    phase_records = PhaseRecordFactory(dbf, comps, [v.T, v.P, v.N], models)

    with pytest.raises(ValueError):
        calculate(dbf, comps, my_phases, T=1200, P=101325, N=1, phase_records=phase_records)


def test_no_neutral_endmembers_single():
    "calculate returns the feasible configuration in a charge-constrained phase, when no endmembers are neutral"
    tdb = """
    ELEMENT Al FCC_A1 0 0 0 !
    ELEMENT CL GAS 0 0 0 !
    ELEMENT VA VACUUM 0 0 0 !

    SPECIES AL+3 AL1/+3 !
    SPECIES CL-1 CL1/-1 !

    PHASE ALCL3 % 2 1 1 !
    CONSTITUENT ALCL3 : AL+3, VA : CL-1 : !
    PARAMETER G(ALCL3,AL+3:CL-1;0) 1 -100000; 10000 N !
    PARAMETER G(ALCL3,VA:CL-1;0) 1 0; 10000 N !
    """
    dbf = Database(tdb)
    calc_res = calculate(dbf, ['AL', 'CL', 'VA'], ['ALCL3'], N=1, P=101325, T=300)
    np.testing.assert_allclose(np.squeeze(calc_res.Y.values), np.array([1/3, 2/3, 1]))


@pytest.mark.filterwarnings("ignore:No valid points found for phase PYROCHLORE*:UserWarning")  # Filter out an expected warning so we don't fail the test
@select_database("zrlayalo.tdb")
def test_pyrochlore_infeasible(load_database):
    "calculate raises an error when it is impossible to satisfy a phase's constraints"
    dbf = load_database()
    with pytest.raises(ConditionError):
        calculate(dbf, ['LA', 'Y', 'O'], 'PYROCHLORE', T=600, P=1e5, pdens=10)


@select_database("zrlayalo.tdb")
def test_pyrochlore_complex(load_database):
    "calculate generates feasible points for complex charged phase"
    dbf = load_database()
    # PYROCHLORE
    # 2   2   6   1   1
    # LA+3,Y+3,ZR+4 : LA+3,Y+3,ZR+4 : O-2,VA : O-2 :  O-2,VA :  !
    constraint_jac = np.array([[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                               [2*3, 2*3, 2*4, 2*3, 2*3, 2*4, 6*-2, 6*0, 1*-2, 1*-2, 1*0]])
    constraint_rhs = np.array([1, 1, 1, 1, 1, 0])
    res = calculate(dbf, ['LA', 'Y', 'ZR', 'O', 'VA'], 'PYROCHLORE', T=600, P=1e5, pdens=10)
    output = np.squeeze(res.Y.values)
    assert output.shape[0] > 0
    assert np.all(output > 0)
    cons_infeasibility = np.max(np.abs(constraint_jac.dot(output.T).T - constraint_rhs))
    assert cons_infeasibility < 1e-10


@select_database("zrlayalo.tdb")
def test_pyrochlore_no_freedom(load_database):
    "calculate generates at least one feasible point for a phase with no degrees of freedom"
    dbf = load_database()
    # PYROCHLORE
    # 2   2   6   1   1
    # LA+3,Y+3,ZR+4 : LA+3,Y+3,ZR+4 : O-2 : O-2 :  O-2 :  !
    # Note that we are not including VA on purpose here
    constraint_jac = np.array([[1, 1, 1, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 1, 1, 1, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 1, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 1, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 1],
                               [2*3, 2*3, 2*4, 2*3, 2*3, 2*4, 6*-2, 1*-2, 1*-2]])
    constraint_rhs = np.array([1, 1, 1, 1, 1, 0])
    res = calculate(dbf, ['LA', 'Y', 'ZR', 'O'], 'PYROCHLORE', T=600, P=1e5, pdens=10)
    output = np.squeeze(res.Y.values)
    assert output.shape[0] > 0
    assert np.all(output > 0)
    cons_infeasibility = np.max(np.abs(constraint_jac.dot(output.T).T - constraint_rhs))
    assert cons_infeasibility < 1e-10


def test_charged_infeasible_minimum_norm():
    "calculate generates a uniform sample when the minimum norm constraint solution is infeasible"
    tdb = """
 ELEMENT /-   ELECTRON_GAS              0.0000E+00  0.0000E+00  0.0000E+00!
 ELEMENT VA   VACUUM                    0.0000E+00  0.0000E+00  0.0000E+00!
 ELEMENT ND   DOUBLE_HCP(ABAC)          1.4424E+02  0.0000E+00  0.0000E+00!
 ELEMENT O    1/2_MOLE_O2(G)            1.5999E+01  4.3410E+03  1.0252E+02!
 ELEMENT Y    HCP_A3                    8.8906E+01  5.9664E+03  4.4434E+01!
 SPECIES ND+3                        ND1/+3!
 SPECIES Y+3                         Y1/+3!
 SPECIES O-2                         O1/-2!
 PHASE M2O3B:I %  3 2   3   1 !
 CONSTITUENT M2O3B:I :ND+3,Y+3 : O-2 : O-2,VA :  !
    """
    dbf = Database(tdb)
    res = calculate(dbf, ['ND', 'Y', 'O', 'VA'], 'M2O3B', T=600, P=1e5, pdens=10)
    output = np.squeeze(res.Y.values)
    assert output.shape[0] > 10
    assert np.all(output > 0)
    # Check that the point sample didn't get 'stuck' in part of the space
    assert np.any(np.logical_and(output[:, 1] > 0.05, output[:, 1] < 0.15))
    assert np.any(np.logical_and(output[:, 1] > 0.25, output[:, 1] < 0.35))


@pytest.mark.filterwarnings("ignore:The order-disorder model for \"BCC_4SL\" has a contribution from the physical property model*:UserWarning")
@pytest.mark.filterwarnings("ignore:The order-disorder model for \"BCC_NOB\" has a contribution from the physical property model*:UserWarning")
@select_database("Al-Fe_sundman2009.tdb")
def test_BCC_phase_with_symmetry_option_B(load_database):
    """Loading a database with option B and F generates new ordering parameters."""
    # This database has a BCC_4SL:B phase and a BCC_NOB phase that
    # does _not_ use option B and gives all parameters manually.
    # After applying the symmetry relationships, energies should be equal
    dbf = load_database()
    bcc_4sl_calc_res = calculate(dbf, ["AL", "FE", "VA"], "BCC_4SL", T=300, N=1, P=101325, pdens=10)
    bcc_no_B_calc_res = calculate(dbf, ["AL", "FE", "VA"], "BCC_NOB", T=300, N=1, P=101325, pdens=10)
    assert np.allclose(bcc_4sl_calc_res.GM.values.squeeze(), bcc_no_B_calc_res.GM.values.squeeze())

def test_complex_multisublattice():
    "calculate returns a result without running out of memory for a complex multi-sublattice phase"
    tdb = """
ELEMENT CR BCC_A2 51.996 4050.0 23.543 !
ELEMENT FE BCC_A2 55.847 4489.0 27.28 !
ELEMENT MO BCC_A2 95.94 4589.0 28.56 !
ELEMENT NB BCC_A2 92.906 5220.0 36.27 !
ELEMENT NI FCC_A1 58.69 4787.0 29.796 !

PHASE MU_PHASE %  5 2.0 2.0 2.0 6.0 1.0 !
CONSTITUENT MU_PHASE
   :CR,FE,MO,NB,NI:CR,FE,MO,NB,NI:CR,FE,MO,NB,NI:
   CR,FE,MO,NB,NI:CR,FE,MO,NB,NI: !
    """
    dbf = Database(tdb)
    calc_res = calculate(dbf, ['CR', 'FE', 'MO', 'NB', 'NI'], ['MU_PHASE'], N=1, P=101325, T=300, pdens=60)
    assert calc_res.GM.size < 100000

@pytest.mark.solver
@select_database("alni_dupin_2001.tdb")
def test_calculation_symengine_evalf_energy_difference(load_database):
    "Platform-dependent numerical differences stemming from optimizations of the Model object representation (gh-431)"
    dbf = load_database()
    comps = ['AL', 'NI', 'VA']
    res = calculate(dbf, comps, ['FCC_L12'], P=101325, T=1600, N=1, pdens=60)
    points = res.Y.values
    mod = Model(dbf, comps, 'FCC_L12')
    for point_idx in range(points.shape[-2]):
        dof = dict(zip(mod.variables, chain([1600., *points[..., point_idx, :].flat])))
        np.testing.assert_allclose(res.GM.values.flat[point_idx], float(mod.GM.subs(dof).evalf(real=True)))

def test_molar_volume_isothermal():
    TDB = """
        ELEMENT FE   BCC_A2                     55.847     4489.0     27.2797 !
        TYPE_DEFINITION % SEQ * !
        PHASE BCC_A2 % 1 1 !
        CONSTITUENT BCC_A2 :FE: !
        PARAMETER V0(BCC_A2,FE;0) 0.01 7.00790E-6; 6000 N !
        PARAMETER VA(BCC_A2,FE;0) 0.01 3.42756E-5*T +8.14005E-9*T**2 +0.291672*T**-1; 6000 N !
        """

    db = Database(TDB)
    mv_300K = calculate(db, ["FE"], "BCC_A2", T=300, N=1, P=101325, output="molar_volume", pdens=10)
    mv_1200K = calculate(db, ["FE"], "BCC_A2", T=1200, N=1, P=101325, output="molar_volume", pdens=10)

    assert np.allclose(mv_300K['molar_volume'], 7.092e-6)
    assert np.allclose(mv_1200K['molar_volume'], 7.39e-6)

def test_volume_energy():
    TDB = """
    ELEMENT FE   BCC_A2                     55.847     4489.0     27.2797 !
    TYPE_DEFINITION % SEQ * !
    PHASE BCC_A2 % 1 1 !
    CONSTITUENT BCC_A2 :FE: !
    PARAMETER G(BCC_A2,FE;0) 0.01 T;  6000.00 N !

    PARAMETER V0(BCC_A2,FE;0) 0.01 1E-6; 6000 N !
    PARAMETER VA(BCC_A2,FE;0) 0.01 1E-6*T; 6000 N !
    """

    db = Database(TDB)
    energy_hp = 1e-6*np.exp(1e-6*300)*(10e9-101325)

    result_atm = calculate(db, ['FE'], 'BCC_A2', T=300, P=101325, N=1)
    result_hp = calculate(db, ['FE'], 'BCC_A2', T=300, P=10E9, N=1)
    vol_energy_contrib = np.squeeze(result_hp['GM']).values - np.squeeze(result_atm['GM']).values

    assert np.allclose(vol_energy_contrib, energy_hp)

def test_magnetic_volume_contribution():
    """
    Toy problem where pressure required to reduce TC to 500 K is 5.43 GPa.
    Pressures below this value will increase the molar volume due to the magnetic transition.
    Pressures far above this value should approach the isothermal molar volume.
    """

    TDB = """
    ELEMENT FE   BCC_A2                     55.847     4489.0     27.2797 !
    TYPE_DEFINITION % SEQ * !
    TYPE_DEFINITION A GES AMEND_PHASE_DESCRIPTION @ MAGNETIC -1 0.4 !
    PHASE BCC_A2 %A 1 1 !
    CONSTITUENT BCC_A2 :FE: !
    PARAMETER G(BCC_A2,FE;0) 0.01 T;  6000.00 N !
    PARAMETER TC(BCC_A2,FE;0)  298.15 +1043 -1e-7*P;  6000 N !
    PARAMETER BMAGN(BCC_A2,FE;0)  298.15 +2.22;  6000 N !

    PARAMETER V0(BCC_A2,FE;0) 0.01 1E-6; 6000 N !
    PARAMETER VA(BCC_A2,FE;0) 0.01 1E-6*T; 6000 N !
    """

    db = Database(TDB)
    res1 = calculate(db, ['FE'], 'BCC_A2', T=500, P=101325, N=1, output='molar_volume')
    res2 = calculate(db, ['FE'], 'BCC_A2', T=500, P=10e9, N=1, output='molar_volume')

    assert np.allclose(np.squeeze(res1['molar_volume']).values, 1.857e-6)
    assert np.allclose(np.squeeze(res2['molar_volume']).values, 1.0e-6)


@pytest.mark.filterwarnings("ignore:No valid points found for phase SPINEL*:UserWarning")  # Filter out an expected warning so we don't fail the test
def test_calculate_raises_if_no_feasible_points_exist():
    "calculate should raise if there are no feasible points produced"

    TDB = """
    ELEMENT /-   ELECTRON_GAS              0.0000E+00  0.0000E+00  0.0000E+00!
    ELEMENT VA   VACUUM                    0.0000E+00  0.0000E+00  0.0000E+00!
    ELEMENT O    1/2_MOLE_O2(G)            1.5999E+01  4.3410E+03  1.0252E+02!
    ELEMENT ZR   BLANK                     0.0000E+00  0.0000E+00  0.0000E+00!
    SPECIES O-2                         O1/-2!
    SPECIES ZR+4                        ZR1/+4!
    PHASE SPINEL %  4 1 2 2 4 !
    CONSTITUENT SPINEL : ZR+4 : VA : VA : O-2 :  !
    PHASE GAS:G %  1  1.0  !
    CONSTITUENT GAS:G :O,ZR :  !
    """

    dbf = Database(TDB)
    # SPINEL phase cannot charge balance, so even though it contains ZR, O, and VA, it must be suspended
    # as it is the only active phase, there will be no points and should raise
    with pytest.raises(ConditionError):
        grid = calculate(dbf, ["O", "ZR", "VA"], ["SPINEL"], P=1e5, T=1000, fake_points=False, to_xarray=False)
    # SPINEL still suspended, but doesn't raise because GAS phase provides points
    with pytest.warns(match="No valid points found for phase SPINEL"):
        grid = calculate(dbf, ["O", "ZR", "VA"], ["GAS", "SPINEL"], P=1e5, T=1000, fake_points=False, to_xarray=False)
    # fake_points provides points and therefore calculate should not raise for having no points
    # fake_points also prevents the warning here
    grid = calculate(dbf, ["O", "ZR", "VA"], ["SPINEL"], P=1e5, T=1000, fake_points=True, to_xarray=False)


@pytest.mark.filterwarnings("ignore:No valid points found for phase SPINEL*:UserWarning")  # Filter out an expected warning so we don't fail the test
def test_calculate_raises_correctly_when_charged_phases_cannot_charge_balance():
    """calculate should _not_ raise if there are phases that are infeasible, but overall there are still points

    This test tests against a special case where a phase is charged and has internal DOF, but cannot charge balance.
    We should see the same failure modes as above.
    """

    TDB = """
    ELEMENT /-   ELECTRON_GAS              0.0000E+00  0.0000E+00  0.0000E+00!
    ELEMENT VA   VACUUM                    0.0000E+00  0.0000E+00  0.0000E+00!
    ELEMENT AL   BLANK                     0.0000E+00  0.0000E+00  0.0000E+00!
    ELEMENT ZR   BLANK                     0.0000E+00  0.0000E+00  0.0000E+00!
    SPECIES AL+3                        AL1/+3!
    SPECIES ZR+4                        ZR1/+4!
    PHASE SPINEL %  4 1 2 2 4 !
    CONSTITUENT SPINEL : AL+3,ZR+4,VA : VA : VA :  !
    PHASE GAS:G %  1  1.0  !
    CONSTITUENT GAS:G :AL,VA,ZR :  !
    """
    dbf = Database(TDB)

    # Gas can always charge balance, all is well
    grid = calculate(dbf, ["AL", "ZR", "VA"], ["GAS"], P=1e5, T=1000, fake_points=False, to_xarray=False)
    # SPINEL cannot charge balance without a negative ion.
    # As it is the only active phase, there will be no points and should raise.
    with pytest.raises(ConditionError):
        grid = calculate(dbf, ["AL", "ZR", "VA"], ["SPINEL"], P=1e5, T=1000, fake_points=False, to_xarray=False)
    # SPINEL still suspended, but doesn't raise because GAS phase provides points
    with pytest.warns(match="No valid points found for phase SPINEL"):
        grid = calculate(dbf, ["AL", "ZR", "VA"], ["GAS", "SPINEL"], P=1e5, T=1000, fake_points=False, to_xarray=False)
    # fake_points provides points and therefore calculate should not raise for having no points
    # fake_points also prevents the warning here
    grid = calculate(dbf, ["AL", "ZR", "VA"], ["SPINEL"], P=1e5, T=1000, fake_points=True, to_xarray=False)


def test_calculate_produces_no_infeasible_points_for_charged_phase():
    """Test that MgO-Al2O3 SPINEL phase can have points sampled via calculate and not raise due to points volating the constraints"""

    _tdb_str = """
    ELEMENT /- ELECTRON_GAS 0.0 0.0 0.0 !
    ELEMENT AL BLANK 26.982 0.0 0.0 !
    ELEMENT MG BLANK 24.305 0.0 0.0 !
    ELEMENT O BLANK 15.999 0.0 0.0 !
    ELEMENT VA VACUUM 0.0 0.0 0.0 !

    SPECIES AL+3 AL1.0/+3 !
    SPECIES MG+2 MG1.0/+2 !
    SPECIES O-2 O1.0/-2 !

    TYPE_DEFINITION % SEQ * !
    DEFINE_SYSTEM_DEFAULT ELEMENT 2 !
    DEFAULT_COMMAND DEFINE_SYSTEM_ELEMENT /- VA !

    PHASE SPINEL:I %  3 1.0 2.0 4.0 !
    CONSTITUENT SPINEL:I :AL+3,MG+2 : AL+3,MG+2,VA : O-2: !
    """

    dbf = Database(_tdb_str)
    comps = ["MG", "AL", "O", "VA"]
    phase_name = "SPINEL"
    model = Model(dbf, comps, phase_name)

    # When failing, this tends to produce constraint violations where the second sublattice (Al+3, Mg+2, Va) does not mass balance
    num_points = 2
    model_constraints = model.get_internal_constraints()
    constraint_jac, constraint_rhs = _jacobian_from_constraints(model_constraints, model.site_fractions)
    extra_points = sample(num_points, np.full(constraint_jac.shape[1], MIN_SITE_FRACTION), np.ones(constraint_jac.shape[1]), A2=constraint_jac, b2=constraint_rhs)
    print(f"extra_points: {extra_points}")
    constraints_for_points = np.abs(constraint_jac.dot(extra_points.T).T)
    np.testing.assert_allclose(constraints_for_points, np.broadcast_to(constraint_rhs, constraints_for_points.shape), atol=1e-6, err_msg="`extra_points` produced by polytope sampler should produce feasible points")

    # success if the phase does not raise
    calc_res = calculate(dbf, comps, [phase_name], T=1000, pdens=60)


@select_database("alcrni.tdb")
def test_calculate_dof_state_variable_width_matches_phase_record(load_database):
    """The number of state-variable dimensions calculate() produces must equal the phase
    record's num_statevars (the compiled function's expected state-variable input count).

    This is the shape invariant whose violation causes the out-of-bounds read: the dof's
    state-variable width must match what the compiled property function expects."""
    dbf = load_database()
    comps = ["AL", "CR", "NI"]
    models = instantiate_models(dbf, comps, ["LIQUID"])
    prf = PhaseRecordFactory(dbf, comps, {v.N: 1, v.T: 400}, models)
    res = calculate(dbf, comps, ["LIQUID"], T=400, model=models["LIQUID"])
    # Leading dims are the supplied state variables (N, T); trailing dim is 'points'.
    n_statevar_dims = res.GM.values.ndim - 1
    assert n_statevar_dims == prf["LIQUID"].num_statevars == len(prf.state_variables)
