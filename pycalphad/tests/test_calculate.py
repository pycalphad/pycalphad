"""
The calculate test module verifies that calculate() calculates
Model quantities correctly.
"""

import pytest
from pycalphad import Database, calculate, Model, variables as v
import numpy as np
from numpy.testing import assert_allclose
from pycalphad.codegen.callables import build_phase_records
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
    # N is added automatically
    assert len(result_one_values.shape) == 3  # N, T, points
    assert result_one_values.shape[0] == 1
    assert len(result_two_values.shape) == 4  # N, P, T, points
    assert result_two_values.shape[:3] == (1, 1, 1)
    assert len(result_three_values.shape) == 4  # N, P, T, points
    assert result_three_values.shape[:3] == (1, 1, 1)


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
    phase_records = build_phase_records(dbf, comps, subset_phases, [v.T, v.P, v.N], models)

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


@select_database("zrlayalo.tdb")
def test_pyrochlore_infeasible(load_database):
    "calculate raises an error when it is impossible to satisfy a phase's constraints"
    dbf = load_database()
    with pytest.raises(ValueError):
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
