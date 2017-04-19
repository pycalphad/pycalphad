"""
The equilibrium test module verifies that the Equilibrium class produces the
correct solution for thermodynamic equilibrium.
"""

import warnings
from nose.tools import raises
from numpy.testing import assert_allclose
import numpy as np
from pycalphad import Database, calculate, equilibrium, EquilibriumError, ConditionError
import pycalphad.variables as v
from pycalphad.tests.datasets import *

warnings.simplefilter("always", UserWarning) # so we can test warnings

ROSE_DBF = Database(ROSE_TDB)
ALFE_DBF = Database(ALFE_TDB)
ALNIFCC4SL_DBF = Database(ALNIFCC4SL_TDB)
ALCOCRNI_DBF = Database(ALCOCRNI_TDB)
ISSUE43_DBF = Database(ISSUE43_TDB)
TOUGH_CHEMPOT_DBF = Database(ALNI_TOUGH_CHEMPOT_TDB)

# ROSE DIAGRAM TEST
def test_rose_nine():
    "Nine-component rose diagram point equilibrium calculation."
    my_phases_rose = ['TEST']
    comps = ['H', 'HE', 'LI', 'BE', 'B', 'C', 'N', 'O', 'F']
    conds = dict({v.T: 1000, v.P: 101325})
    for comp in comps[:-1]:
        conds[v.X(comp)] = 1.0/float(len(comps))
    eqx = equilibrium(ROSE_DBF, comps, my_phases_rose, conds, pbar=False, verbose=True)
    assert_allclose(eqx.GM.values.flat[0], -5.8351e3, atol=0.1)

# OTHER TESTS
def test_eq_binary():
    "Binary phase diagram point equilibrium calculation with magnetism."
    my_phases = ['LIQUID', 'FCC_A1', 'HCP_A3', 'AL5FE2',
                 'AL2FE', 'AL13FE4', 'AL5FE4']
    comps = ['AL', 'FE', 'VA']
    conds = {v.T: 1400, v.P: 101325, v.X('AL'): 0.55}
    eqx = equilibrium(ALFE_DBF, comps, my_phases, conds, pbar=False)
    assert_allclose(eqx.GM.values.flat[0], -9.608807e4)

def test_eq_single_phase():
    "Equilibrium energy should be the same as for a single phase with no miscibility gaps."
    res = calculate(ALFE_DBF, ['AL', 'FE'], 'LIQUID', T=[1400, 2500], P=101325,
                    points={'LIQUID': [[0.1, 0.9], [0.2, 0.8], [0.3, 0.7],
                                       [0.7, 0.3], [0.8, 0.2]]})
    eq = equilibrium(ALFE_DBF, ['AL', 'FE'], 'LIQUID',
                     {v.T: [1400, 2500], v.P: 101325,
                      v.X('AL'): [0.1, 0.2, 0.3, 0.7, 0.8]}, verbose=True, pbar=False)
    assert_allclose(eq.GM, res.GM, atol=0.1)


def test_eq_b2_without_all_comps():
    """
    All-vacancy endmembers are correctly excluded from the computation when fewer than
    all components in a Database are selected for the calculation.
    """
    equilibrium(Database(ALNIPT_TDB), ['AL', 'NI', 'VA'], 'BCC_B2', {v.X('NI'): 0.4, v.P: 101325, v.T: 1200},
                verbose=True, pbar=False)


@raises(ValueError)
def test_eq_underdetermined_comps():
    """
    The number of composition conditions should yield exactly one dependent component.
    This is the underdetermined case.
    """
    equilibrium(ALFE_DBF, ['AL', 'FE'], 'LIQUID', {v.T: 2000, v.P: 101325}, pbar=False)


@raises(ValueError)
def test_eq_overdetermined_comps():
    """
    The number of composition conditions should yield exactly one dependent component.
    This is the overdetermined case.
    """
    equilibrium(ALFE_DBF, ['AL', 'FE'], 'LIQUID', {v.T: 2000, v.P: 101325,
                                                   v.X('FE'): 0.2, v.X('AL'): 0.8}, pbar=False)

def test_dilute_condition():
    """
    'Zero' and dilute composition conditions are correctly handled.
    """
    eq = equilibrium(ALFE_DBF, ['AL', 'FE', 'VA'], 'FCC_A1', {v.T: 1300, v.P: 101325, v.X('AL'): 0}, pbar=False)
    assert_allclose(np.squeeze(eq.GM.values), -64415.84, atol=0.1)
    eq = equilibrium(ALFE_DBF, ['AL', 'FE', 'VA'], 'FCC_A1', {v.T: 1300, v.P: 101325, v.X('AL'): 1e-8}, pbar=False)
    assert_allclose(np.squeeze(eq.GM.values), -64415.84069827)
    assert_allclose(eq.MU.values, [[[[-335723.04320981,  -64415.8379852]]]], atol=0.1)

def test_eq_illcond_hessian():
    """
    Check equilibrium of a system with an ill-conditioned Hessian.
    This is difficult to reproduce so we only include some known examples here (gh-23).
    """
    # This set of conditions is known to trigger the issue
    eq = equilibrium(ALFE_DBF, ['AL', 'FE', 'VA'], 'LIQUID',
                     {v.X('FE'): 0.73999999999999999, v.T: 401.5625, v.P: 1e5}, pbar=False)
    assert_allclose(eq.GM.values, [[[-16507.22325998]]])
    # chemical potentials were checked in TC and accurate to 1 J/mol
    # pycalphad values used for more significant figures
    # once again, py33 converges to a slightly different value versus every other python
    assert_allclose(eq.MU.values, [[[[-55611.954141,  -2767.72322]]]], atol=0.1)

def test_eq_illcond_magnetic_hessian():
    """
    Check equilibrium of a system with an ill-conditioned Hessian due to magnetism (Tc->0).
    This is difficult to reproduce so we only include some known examples here.
    """
    # This set of conditions is known to trigger the issue
    eq = equilibrium(ALFE_DBF, ['AL', 'FE', 'VA'], ['FCC_A1', 'AL13FE4'],
                     {v.X('AL'): 0.8, v.T: 300, v.P: 1e5}, pbar=False, verbose=True)
    assert_allclose(eq.GM.values, [[[-31414.46677]]])
    assert_allclose(eq.MU.values, [[[[-8490.140, -123111.773]]]], atol=0.1)


def test_eq_composition_cond_sorting():
    """
    Composition conditions are correctly constructed when the dependent component does not
    come last in alphabetical order (gh-21).
    """
    eq = equilibrium(ALFE_DBF, ['AL', 'FE'], 'LIQUID',
                     {v.T: 2000, v.P: 101325, v.X('FE'): 0.2}, pbar=False)
    # Values computed by Thermo-Calc
    tc_energy = -143913.3
    tc_mu_fe = -184306.01
    tc_mu_al = -133815.12
    assert_allclose(eq.GM.values, tc_energy)
    assert_allclose(eq.MU.values, [[[[tc_mu_al, tc_mu_fe]]]], rtol=1e-6)

def test_eq_output_property():
    """
    Extra properties can be specified to `equilibrium`.
    """
    equilibrium(ALFE_DBF, ['AL', 'FE', 'VA'], ['LIQUID', 'B2_BCC'],
                {v.X('AL'): 0.25, v.T: (300, 2000, 500), v.P: 101325},
                output=['heat_capacity', 'degree_of_ordering'], pbar=False)

def test_eq_on_endmember():
    """
    When the composition condition is right on top of an end-member
    the convex hull is still correctly constructed (gh-28).
    """
    equilibrium(ALFE_DBF, ['AL', 'FE', 'VA'], ['LIQUID', 'B2_BCC'],
                {v.X('AL'): [0.4, 0.5, 0.6], v.T: [300, 600], v.P: 101325}, pbar=False)

def test_eq_four_sublattice():
    """
    Balancing mass in a multi-sublattice phase in a single-phase configuration.
    """
    eq = equilibrium(ALNIFCC4SL_DBF, ['AL', 'NI', 'VA'], 'FCC_L12',
                     {v.T: 1073, v.X('NI'): 0.7601, v.P: 101325}, pbar=False)
    assert_allclose(np.squeeze(eq.X.sel(vertex=0).values), [1-.7601, .7601])
    # Not a strict equality here because we can't yet reach TC's value of -87260.6
    assert eq.GM.values < -87256.3

@raises(EquilibriumError)
def test_eq_missing_component():
    """
    Specifying a non-existent component raises an error.
    """
    # No Co or Cr in this database ; Co component specification should cause failure
    equilibrium(ALNIFCC4SL_DBF, ['AL', 'CO', 'CR', 'VA'], ['LIQUID'],
                {v.T: 1523, v.X('AL'): 0.88811111111111107,
                 v.X('CO'): 0.11188888888888888, v.P: 101325}, pbar=False)

@raises(ConditionError)
def test_eq_missing_component():
    """
    Specifying a condition involving a non-existent component raises an error.
    """
    # No Co or Cr in this database ; Co condition specification should cause failure
    equilibrium(ALNIFCC4SL_DBF, ['AL', 'NI', 'VA'], ['LIQUID'],
                {v.T: 1523, v.X('AL'): 0.88811111111111107,
                 v.X('CO'): 0.11188888888888888, v.P: 101325}, pbar=False)

def test_eq_ternary_edge_case_mass():
    """
    Equilibrium along an edge of composition space will still balance mass.
    """
    eq = equilibrium(ALCOCRNI_DBF, ['AL', 'CO', 'CR', 'VA'], ['L12_FCC', 'BCC_B2', 'LIQUID'],
                     {v.T: 1523, v.X('AL'): 0.88811111111111107,
                      v.X('CO'): 0.11188888888888888, v.P: 101325}, pbar=False, verbose=True)
    mass_error = np.nansum(np.squeeze(eq.NP * eq.X), axis=-2) - \
                 [0.88811111111111107, 0.11188888888888888, 0]
    assert np.all(np.abs(mass_error) < 0.01)

def test_eq_ternary_inside_mass():
    """
    Equilibrium in interior of composition space will still balance mass.
    """
    eq = equilibrium(ALCOCRNI_DBF, ['AL', 'CO', 'CR', 'VA'], ['L12_FCC', 'BCC_B2', 'LIQUID'],
                     {v.T: 1523, v.X('AL'): 0.44455555555555554,
                      v.X('CO'): 0.22277777777777777, v.P: 101325}, pbar=False, verbose=True)
    mass_error = np.nansum(np.squeeze(eq.NP * eq.X), axis=-2) - \
                 [0.44455555555555554, 0.22277777777777777, 0.333]
    assert np.all(np.abs(mass_error) < 0.01)

def test_eq_ternary_edge_misc_gap():
    """
    Equilibrium at edge of miscibility gap will still balance mass.
    """
    eq = equilibrium(ALCOCRNI_DBF, ['AL', 'CO', 'CR', 'VA'], ['L12_FCC', 'BCC_B2', 'LIQUID'],
                     {v.T: 1523, v.X('AL'): 0.33366666666666667,
                      v.X('CO'): 0.44455555555555554, v.P: 101325}, pbar=False, verbose=True)
    mass_error = np.nansum(np.squeeze(eq.NP * eq.X), axis=-2) - \
                 [0.33366666666666667, 0.44455555555555554, 0.22177777777777785]
    assert np.all(np.abs(mass_error) < 0.001)

def test_eq_issue43_chempots_misc_gap():
    """
    Equilibrium for complex ternary miscibility gap (gh-43).
    """
    eq = equilibrium(ISSUE43_DBF, ['AL', 'NI', 'CR', 'VA'], 'GAMMA_PRIME',
                     {v.X('AL'): .1246, v.X('CR'): 1e-9, v.T: 1273, v.P: 101325},
                     verbose=True, pbar=False)
    chempots = 8.31451 * np.squeeze(eq['T'].values) * np.array([[[[[-19.47631644, -25.71249032,  -6.0706158]]]]])
    assert_allclose(eq.GM.values, -81933.259)
    assert_allclose(eq.MU.values, chempots, atol=1)

def test_eq_issue43_chempots_tricky_potentials():
    """
    Ternary equilibrium with difficult convergence for chemical potentials (gh-43).
    """
    eq = equilibrium(ISSUE43_DBF, ['AL', 'NI', 'CR', 'VA'], ['FCC_A1', 'GAMMA_PRIME'],
                     {v.X('AL'): .1246, v.X('CR'): 0.6, v.T: 1273, v.P: 101325},
                     verbose=True, pbar=False)
    chempots = np.array([-135620.9960449, -47269.29002414, -92304.23688281])
    assert_allclose(eq.GM.values, -70680.53695)
    assert_allclose(np.squeeze(eq.MU.values), chempots)

def test_eq_stepsize_reduction():
    """
    Step size reduction required for convergence.
    """
    dbf = TOUGH_CHEMPOT_DBF
    eq = equilibrium(dbf, ['AL', 'NI', 'VA'], list(dbf.phases.keys()),
                     {v.P: 101325, v.T: 780, v.X('NI'): 0.625}, verbose=True)
    assert not np.isnan(np.squeeze(eq.GM.values))

def test_eq_issue62_last_component_not_va():
    """
    VA is not last when components are sorted alphabetically.
    """
    test_tdb = """
    ELEMENT VA   VACUUM                    0.0000E+00  0.0000E+00  0.0000E+00!
    ELEMENT AL   FCC_A1                    2.6982E+01  4.5773E+03  2.8322E+01!
    ELEMENT CO   HCP_A3                    5.8933E+01  4.7656E+03  3.0040E+00!
    ELEMENT CR   BCC_A2                    5.1996E+01  4.0500E+03  2.3560E+01!
    ELEMENT W    BCC_A2                    1.8385E+02  4.9700E+03  3.2620E+01!
    PHASE FCC_A1  %  2 1   1 !
    CONSTITUENT FCC_A1  :AL,CO,CR,W : VA% :  !
    """
    equilibrium(Database(test_tdb), ['AL', 'CO', 'CR', 'W', 'VA'], ['FCC_A1'],
                {"T": 1248, "P": 101325, v.X("AL"): 0.081, v.X("CR"): 0.020, v.X("W"): 0.094})

def test_unused_equilibrium_kwarg_warns():
    "Check that an unused keyword argument raises a warning"
    with warnings.catch_warnings(record=True) as w:
        equilibrium(ALFE_DBF, ['AL', 'FE', 'VA'], 'FCC_A1', {v.T: 1300, v.P: 101325, v.X('AL'): 0}, unused_kwarg='should raise a warning')
        categories = [warning.__dict__['_category_name'] for warning in w]
        assert 'UserWarning' in categories
        assert len(w) == 1 # make sure we don't raise other warnings later that make this test falsely pass


