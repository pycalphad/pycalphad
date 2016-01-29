"""
The equilibrium test module verifies that the Equilibrium class produces the
correct solution for thermodynamic equilibrium.
"""

from unittest.case import SkipTest
from nose.tools import raises
from numpy.testing import assert_allclose
from pycalphad import Database, calculate, equilibrium
import pycalphad.variables as v
from pycalphad.tests.datasets import ALNIPT_TDB, ROSE_TDB

ROSE_DBF = Database(ROSE_TDB)
ALFE_DBF = Database('examples/alfe_sei.TDB')

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
    eqx = equilibrium(ROSE_DBF, comps, my_phases_rose, conds)
    assert_allclose(eqx.GM.values.flat[0], -5.8351e3)

# OTHER TESTS
def test_eq_binary():
    "Binary phase diagram point equilibrium calculation with magnetism."
    my_phases = ['LIQUID', 'FCC_A1', 'HCP_A3', 'AL5FE2',
                 'AL2FE', 'AL13FE4', 'AL5FE4']
    comps = ['AL', 'FE', 'VA']
    conds = {v.T: 1400, v.P: 101325, v.X('AL'): 0.55}
    eqx = equilibrium(ALFE_DBF, comps, my_phases, conds)
    assert_allclose(eqx.GM.values.flat[0], -9.608807e4)

def test_eq_single_phase():
    "Equilibrium energy should be the same as for a single phase with no miscibility gaps."
    res = calculate(ALFE_DBF, ['AL', 'FE'], 'LIQUID', T=[1400, 2500], P=101325,
                    points={'LIQUID': [[0.1, 0.9], [0.2, 0.8], [0.3, 0.7],
                                       [0.4, 0.6], [0.5, 0.5], [0.6, 0.4],
                                       [0.7, 0.3], [0.8, 0.2]]})
    eq = equilibrium(ALFE_DBF, ['AL', 'FE'], 'LIQUID',
                     {v.T: [1400, 2500], v.P: 101325,
                      v.X('AL'): [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]}, verbose=False)
    assert_allclose(eq.GM, res.GM, atol=0.1)


def test_eq_b2_without_all_comps():
    """
    All-vacancy endmembers are correctly excluded from the computation when fewer than
    all components in a Database are selected for the calculation.
    """
    equilibrium(Database(ALNIPT_TDB), ['AL', 'NI', 'VA'], 'BCC_B2', {v.X('NI'): 0.4, v.P: 101325, v.T: 1200}, verbose=False)


@raises(ValueError)
def test_eq_underdetermined_comps():
    """
    The number of composition conditions should yield exactly one dependent component.
    This is the underdetermined case.
    """
    equilibrium(ALFE_DBF, ['AL', 'FE'], 'LIQUID', {v.T: 2000, v.P: 101325})


@raises(ValueError)
def test_eq_overdetermined_comps():
    """
    The number of composition conditions should yield exactly one dependent component.
    This is the overdetermined case.
    """
    equilibrium(ALFE_DBF, ['AL', 'FE'], 'LIQUID', {v.T: 2000, v.P: 101325,
                                                   v.X('FE'): 0.2, v.X('AL'): 0.8})
def test_eq_illcond_hessian():
    """
    Check equilibrium of a system with an ill-conditioned Hessian.
    This is difficult to reproduce so we only include some known examples here (gh-23).
    """
    # This set of conditions is known to trigger the issue
    eq = equilibrium(ALFE_DBF, ['AL', 'FE', 'VA'], 'LIQUID',
                     {v.X('FE'): 0.73999999999999999, v.T: 401.5625, v.P: 1e5})
    assert_allclose(eq.GM.values, [[[-16507.22325998]]])
    # chemical potentials were checked in TC and accurate to 1 J/mol
    # pycalphad values used for more significant figures
    # once again, py33 converges to a slightly different value versus every other python
    assert_allclose(eq.MU.values, [[[[-55611.954141,  -2767.72322]]]], atol=0.1)


def test_eq_composition_cond_sorting():
    """
    Composition conditions are correctly constructed when the dependent component does not
    come last in alphabetical order (gh-21).
    """
    eq = equilibrium(ALFE_DBF, ['AL', 'FE'], 'LIQUID',
                     {v.T: 2000, v.P: 101325, v.X('FE'): 0.2})
    # Values computed by Thermo-Calc
    tc_energy = -143913.3
    tc_mu_fe = -184306.01
    tc_mu_al = -133815.12
    assert_allclose(eq.GM.values, tc_energy)
    assert_allclose(eq.MU.values, [[[[tc_mu_al, tc_mu_fe]]]], rtol=1e-6)

