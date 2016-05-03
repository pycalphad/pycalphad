"""
The equilibrium test module verifies that the Equilibrium class produces the
correct solution for thermodynamic equilibrium.
"""

from unittest.case import SkipTest
from nose.tools import raises
from numpy.testing import assert_allclose
import numpy as np
from pycalphad import Database, calculate, equilibrium, EquilibriumError, ConditionError
import pycalphad.variables as v
from pycalphad.tests.datasets import ALNIPT_TDB, ROSE_TDB, ALFE_TDB, ALNIFCC4SL_TDB

ROSE_DBF = Database(ROSE_TDB)
ALFE_DBF = Database(ALFE_TDB)
ALNIFCC4SL_DBF = Database(ALNIFCC4SL_TDB)

# ROSE DIAGRAM TESTS
# This will fail until the equilibrium engine is switched from Newton-Raphson
@SkipTest
def test_rose_nine():
    "Nine-component rose diagram point equilibrium calculation."
    my_phases_rose = ['TEST']
    comps = ['H', 'HE', 'LI', 'BE', 'B', 'C', 'N', 'O', 'F']
    conds = dict({v.T: 1000, v.P: 101325})
    for comp in comps[:-1]:
        conds[v.X(comp)] = 1.0/float(len(comps))
    eqx = equilibrium(ROSE_DBF, comps, my_phases_rose, conds, pbar=False)
    assert_allclose(eqx.GM.values.flat[0], -5.8351e3)

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
                     {v.X('AL'): 0.8, v.T: 300, v.P: 1e5}, pbar=False)
    assert_allclose(eq.GM.values, [[[-31414.46677]]])


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