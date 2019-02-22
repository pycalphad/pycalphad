"""
The equilibrium test module verifies that the Equilibrium class produces the
correct solution for thermodynamic equilibrium.
"""

import warnings
import os
from nose.tools import raises
from sympy import Symbol
from numpy.testing import assert_allclose
import numpy as np
from pycalphad import Database, Model, calculate, equilibrium, EquilibriumError, ConditionError
from pycalphad.codegen.callables import build_callables
from pycalphad.core.solver import SolverBase, InteriorPointSolver
import pycalphad.variables as v
from pycalphad.tests.datasets import *

warnings.simplefilter("always", UserWarning) # so we can test warnings

ROSE_DBF = Database(ROSE_TDB)
ALFE_DBF = Database(ALFE_TDB)
ALNIFCC4SL_DBF = Database(ALNIFCC4SL_TDB)
ALCOCRNI_DBF = Database(ALCOCRNI_TDB)
ISSUE43_DBF = Database(ISSUE43_TDB)
TOUGH_CHEMPOT_DBF = Database(ALNI_TOUGH_CHEMPOT_TDB)
CUO_DBF = Database(CUO_TDB)
PBSN_DBF = Database(PBSN_TDB)
AL_PARAMETER_DBF = Database(AL_PARAMETER_TDB)


# ROSE DIAGRAM TEST
def test_rose_nine():
    "Nine-component rose diagram point equilibrium calculation."
    my_phases_rose = ['TEST']
    comps = ['H', 'HE', 'LI', 'BE', 'B', 'C', 'N', 'O', 'F']
    conds = dict({v.T: 1000, v.P: 101325})
    for comp in comps[:-1]:
        conds[v.X(comp)] = 1.0/float(len(comps))
    eqx = equilibrium(ROSE_DBF, comps, my_phases_rose, conds, verbose=True)
    assert_allclose(eqx.GM.values.flat[0], -5.8351e3, atol=0.1)

# OTHER TESTS
def test_eq_binary():
    "Binary phase diagram point equilibrium calculation with magnetism."
    my_phases = ['LIQUID', 'FCC_A1', 'HCP_A3', 'AL5FE2',
                 'AL2FE', 'AL13FE4', 'AL5FE4']
    comps = ['AL', 'FE', 'VA']
    conds = {v.T: 1400, v.P: 101325, v.X('AL'): 0.55}
    eqx = equilibrium(ALFE_DBF, comps, my_phases, conds, verbose=True)
    assert_allclose(eqx.GM.values.flat[0], -9.608807e4)

def test_eq_single_phase():
    "Equilibrium energy should be the same as for a single phase with no miscibility gaps."
    res = calculate(ALFE_DBF, ['AL', 'FE'], 'LIQUID', T=[1400, 2500], P=101325,
                    points={'LIQUID': [[0.1, 0.9], [0.2, 0.8], [0.3, 0.7],
                                       [0.7, 0.3], [0.8, 0.2]]})
    eq = equilibrium(ALFE_DBF, ['AL', 'FE'], 'LIQUID',
                     {v.T: [1400, 2500], v.P: 101325,
                      v.X('AL'): [0.1, 0.2, 0.3, 0.7, 0.8]}, verbose=True)
    assert_allclose(np.squeeze(eq.GM), np.squeeze(res.GM), atol=0.1)


def test_eq_b2_without_all_comps():
    """
    All-vacancy endmembers are correctly excluded from the computation when fewer than
    all components in a Database are selected for the calculation.
    """
    equilibrium(Database(ALNIPT_TDB), ['AL', 'NI', 'VA'], 'BCC_B2', {v.X('NI'): 0.4, v.P: 101325, v.T: 1200},
                verbose=True)


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

def test_dilute_condition():
    """
    'Zero' and dilute composition conditions are correctly handled.
    """
    eq = equilibrium(ALFE_DBF, ['AL', 'FE', 'VA'], 'FCC_A1', {v.T: 1300, v.P: 101325, v.X('AL'): 0}, verbose=True)
    assert_allclose(np.squeeze(eq.GM.values), -64415.84, atol=0.1)
    eq = equilibrium(ALFE_DBF, ['AL', 'FE', 'VA'], 'FCC_A1', {v.T: 1300, v.P: 101325, v.X('AL'): 1e-8}, verbose=True)
    # Checked in TC
    assert_allclose(np.squeeze(eq.GM.values), -64415.841)
    # We loosen the tolerance a bit here because our convergence tolerance is too low for the last digit
    assert_allclose(np.squeeze(eq.MU.values), [-335723.28,  -64415.838], atol=1.0)

def test_eq_illcond_hessian():
    """
    Check equilibrium of a system with an ill-conditioned Hessian.
    This is difficult to reproduce so we only include some known examples here (gh-23).
    """
    # This set of conditions is known to trigger the issue
    eq = equilibrium(ALFE_DBF, ['AL', 'FE', 'VA'], 'LIQUID',
                     {v.X('FE'): 0.73999999999999999, v.T: 401.5625, v.P: 1e5})
    assert_allclose(np.squeeze(eq.GM.values), -16507.22325998)
    # chemical potentials were checked in TC and accurate to 1 J/mol
    # pycalphad values used for more significant figures
    # once again, py33 converges to a slightly different value versus every other python
    assert_allclose(np.squeeze(eq.MU.values), [-55611.954141,  -2767.72322], atol=0.1)

def test_eq_illcond_magnetic_hessian():
    """
    Check equilibrium of a system with an ill-conditioned Hessian due to magnetism (Tc->0).
    This is difficult to reproduce so we only include some known examples here.
    """
    # This set of conditions is known to trigger the issue
    eq = equilibrium(ALFE_DBF, ['AL', 'FE', 'VA'], ['FCC_A1', 'AL13FE4'],
                     {v.X('AL'): 0.8, v.T: 300, v.P: 1e5}, verbose=True)
    assert_allclose(np.squeeze(eq.GM.values), -31414.46677)
    # These chemical potentials have a strong dependence on MIN_SITE_FRACTION
    # Smaller values tend to shift the potentials +- 1 J/mol
    # Numbers below based on MIN_SITE_FRACTION=1e-12 (TC's default setting)
    assert_allclose(np.squeeze(eq.MU.values), [-8490.140, -123111.773], rtol=1e-4)


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
    assert_allclose(np.squeeze(eq.GM.values), tc_energy)
    assert_allclose(np.squeeze(eq.MU.values), [tc_mu_al, tc_mu_fe], rtol=1e-6)

def test_eq_output_property():
    """
    Extra properties can be specified to `equilibrium`.
    """
    equilibrium(ALFE_DBF, ['AL', 'FE', 'VA'], ['LIQUID', 'B2_BCC'],
                {v.X('AL'): 0.25, v.T: (300, 2000, 500), v.P: 101325},
                output=['heat_capacity', 'degree_of_ordering'])

def test_eq_on_endmember():
    """
    When the composition condition is right on top of an end-member
    the convex hull is still correctly constructed (gh-28).
    """
    equilibrium(ALFE_DBF, ['AL', 'FE', 'VA'], ['LIQUID', 'B2_BCC'],
                {v.X('AL'): [0.4, 0.5, 0.6], v.T: [300, 600], v.P: 101325}, verbose=True)

def test_eq_four_sublattice():
    """
    Balancing mass in a multi-sublattice phase in a single-phase configuration.
    """
    eq = equilibrium(ALNIFCC4SL_DBF, ['AL', 'NI', 'VA'], 'FCC_L12',
                     {v.T: 1073, v.X('NI'): 0.7601, v.P: 101325})
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
                 v.X('CO'): 0.11188888888888888, v.P: 101325})

@raises(ConditionError)
def test_eq_missing_component():
    """
    Specifying a condition involving a non-existent component raises an error.
    """
    # No Co or Cr in this database ; Co condition specification should cause failure
    equilibrium(ALNIFCC4SL_DBF, ['AL', 'NI', 'VA'], ['LIQUID'],
                {v.T: 1523, v.X('AL'): 0.88811111111111107,
                 v.X('CO'): 0.11188888888888888, v.P: 101325})

def test_eq_ternary_edge_case_mass():
    """
    Equilibrium along an edge of composition space will still balance mass.
    """
    eq = equilibrium(ALCOCRNI_DBF, ['AL', 'CO', 'CR', 'VA'], ['L12_FCC', 'BCC_B2', 'LIQUID'],
                     {v.T: 1523, v.X('AL'): 0.88811111111111107,
                      v.X('CO'): 0.11188888888888888, v.P: 101325}, verbose=True)
    mass_error = np.nansum(np.squeeze(eq.NP * eq.X), axis=-2) - \
                 [0.88811111111111107, 0.11188888888888888, 0]
    assert np.all(np.abs(mass_error) < 0.01)

def test_eq_ternary_inside_mass():
    """
    Equilibrium in interior of composition space will still balance mass.
    """
    # This test cannot be checked in TC due to a lack of significant figures in the composition
    eq = equilibrium(ALCOCRNI_DBF, ['AL', 'CO', 'CR', 'VA'], ['L12_FCC', 'BCC_B2', 'LIQUID'],
                     {v.T: 1523, v.X('AL'): 0.44455555555555554,
                      v.X('CO'): 0.22277777777777777, v.P: 101325}, verbose=True)
    assert_allclose(eq.GM.values, -105871.20, atol=0.1)
    assert_allclose(eq.MU.values.flatten(), [-104655.532294, -142591.644379,  -82905.085459], atol=0.1)


def test_eq_ternary_edge_misc_gap():
    """
    Equilibrium at edge of miscibility gap will still balance mass.
    """
    eq = equilibrium(ALCOCRNI_DBF, ['AL', 'CO', 'CR', 'VA'], ['L12_FCC', 'BCC_B2', 'LIQUID'],
                     {v.T: 1523, v.X('AL'): 0.33366666666666667,
                      v.X('CO'): 0.44455555555555554, v.P: 101325}, verbose=True)
    mass_error = np.nansum(np.squeeze(eq.NP * eq.X), axis=-2) - \
                 [0.33366666666666667, 0.44455555555555554, 0.22177777777777785]
    assert np.all(np.abs(mass_error) < 0.001)

def test_eq_issue43_chempots_misc_gap():
    """
    Equilibrium for complex ternary miscibility gap (gh-43).
    """
    eq = equilibrium(ISSUE43_DBF, ['AL', 'NI', 'CR', 'VA'], 'GAMMA_PRIME',
                     {v.X('AL'): .1246, v.X('CR'): 1e-9, v.T: 1273, v.P: 101325},
                     verbose=True)
    chempots = 8.31451 * np.squeeze(eq['T'].values) * np.array([-19.47631644, -25.71249032,  -6.0706158])
    mass_error = np.nansum(np.squeeze(eq.NP * eq.X), axis=-2) - \
                 [0.1246, 1e-9, 1-(.1246+1e-9)]
    assert np.max(np.fabs(mass_error)) < 1e-9
    assert_allclose(np.squeeze(eq.GM.values), -81933.259)
    assert_allclose(np.squeeze(eq.MU.values), chempots, atol=1)

def test_eq_issue43_chempots_tricky_potentials():
    """
    Ternary equilibrium with difficult convergence for chemical potentials (gh-43).
    """
    eq = equilibrium(ISSUE43_DBF, ['AL', 'NI', 'CR', 'VA'], ['FCC_A1', 'GAMMA_PRIME'],
                     {v.X('AL'): .1246, v.X('CR'): 0.6, v.T: 1273, v.P: 101325},
                     verbose=True)
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

def test_eq_avoid_phase_cycling():
    """
    Converge without getting stuck in an add/remove phase cycle.
    """
    # This set of conditions is known to trigger the issue
    my_phases_alfe = ['LIQUID', 'B2_BCC', 'FCC_A1', 'HCP_A3', 'AL5FE2', 'AL2FE', 'AL13FE4', 'AL5FE4']
    equilibrium(ALFE_DBF, ['AL', 'FE', 'VA'], my_phases_alfe, {v.X('AL'): 0.44,
                                                               v.T: 1600, v.P: 101325}, verbose=True)

def test_eq_issue76_dilute_potentials():
    """
    Convergence for two-phase mixtures at dilute composition (gh-76).
    """
    my_phases = ['LIQUID', 'FCC_A1']
    Tvector = np.arange(900.0, 950.0, 1.0)
    eq = equilibrium(ALFE_DBF, ['AL', 'FE', 'VA'], my_phases,
                     {v.X('FE'): 1.5e-3, v.T: Tvector, v.P: 101325}, verbose=True)
    # Spot check at one temperature, plus monotonic decrease of chemical potential
    np.testing.assert_allclose(eq.GM.sel(T=930, P=101325).values, -37799.510894)
    assert np.all(np.diff(eq.MU.sel(component='FE').values <= 0))

def test_eq_model_phase_name():
    """
    Phase name is set in PhaseRecord when using Model-based JIT compilation.
    """
    eq = equilibrium(ALFE_DBF, ['AL', 'FE', 'VA'], 'LIQUID',
                     {v.X('FE'): 0.3, v.T: 1000, v.P: 101325}, model=Model)
    assert eq.Phase.sel(vertex=0).isel(T=0, P=0, X_FE=0) == 'LIQUID'

def test_unused_equilibrium_kwarg_warns():
    "Check that an unused keyword argument raises a warning"
    with warnings.catch_warnings(record=True) as w:
        equilibrium(ALFE_DBF, ['AL', 'FE', 'VA'], 'FCC_A1', {v.T: 1300, v.P: 101325, v.X('AL'): 0}, unused_kwarg='should raise a warning')
        assert len(w) >= 1
        categories = [warning.__dict__['_category_name'] for warning in w]
        assert 'UserWarning' in categories
        expected_string_fragment = 'keyword arguments were passed, but unused'
        assert any([expected_string_fragment in str(warning.message) for warning in w])

def test_eq_unary_issue78():
    "Unary equilibrium calculations work with property calculations."
    eq = equilibrium(ALFE_DBF, ['AL', 'VA'], 'FCC_A1', {v.T: 1200, v.P: 101325}, output='SM')
    np.testing.assert_allclose(eq.SM, 68.143273)
    eq = equilibrium(ALFE_DBF, ['AL', 'VA'], 'FCC_A1', {v.T: 1200, v.P: 101325}, output='SM', parameters={'GHSERAL': 1000})
    np.testing.assert_allclose(eq.GM, 1000)
    np.testing.assert_allclose(eq.SM, 0)

def test_eq_gas_phase():
    eq = equilibrium(CUO_DBF, ['O'], 'GAS', {v.T: 1000, v.P: 1e5}, verbose=True)
    np.testing.assert_allclose(eq.GM, -110380.61071, atol=0.1)
    eq = equilibrium(CUO_DBF, ['O'], 'GAS', {v.T: 1000, v.P: 1e9}, verbose=True)
    np.testing.assert_allclose(eq.GM, -7.20909E+04, atol=0.1)

def test_eq_ionic_liquid():
    eq = equilibrium(CUO_DBF, ['CU', 'O', 'VA'], 'IONIC_LIQ', {v.T: 1000, v.P: 1e5, v.X('CU'): 0.6618}, verbose=True)
    np.testing.assert_allclose(eq.GM, -9.25057E+04, atol=0.1)


def test_eq_parameter_override():
    """
    Check that overriding parameters works in equilibrium().
    """
    comps = ["AL"]
    dbf = AL_PARAMETER_DBF
    phases = ['FCC_A1']
    conds = {v.P: 101325, v.T: 500}

    # Check that current database should work as expected
    eq_res = equilibrium(dbf, comps, phases, conds)
    np.testing.assert_allclose(eq_res.GM.values.squeeze(), 5000.0)

    # Check that overriding parameters works
    eq_res = equilibrium(dbf, comps, phases, conds, parameters={'VV0000': 10000})
    np.testing.assert_allclose(eq_res.GM.values.squeeze(), 10000.0)


def test_eq_build_callables_with_parameters():
    """
    Check build_callables() compatibility with the parameters kwarg.
    """
    comps = ["AL"]
    dbf = AL_PARAMETER_DBF
    phases = ['FCC_A1']
    conds = {v.P: 101325, v.T: 500, v.N: 1}
    # build callables with a parameter of 20000.0
    callables = build_callables(dbf, comps, phases, conds=conds, parameters={'VV0000': 20000})

    # Check that passing callables should skip the build phase, but use the values from 'VV0000' saved in callables
    eq_res = equilibrium(dbf, comps, phases, conds, callables=callables)
    np.testing.assert_allclose(eq_res.GM.values.squeeze(), 20000.0)

    # Check that passing callables should skip the build phase, but use the values from 'VV0000' as passed in parameters
    eq_res = equilibrium(dbf, comps, phases, conds, callables=callables, parameters={'VV0000': 10000})
    np.testing.assert_allclose(eq_res.GM.values.squeeze(), 10000.0)

    # Check that passing callables should skip the build phase,
    # but use the values from Symbol('VV0000') as passed in parameters
    eq_res = equilibrium(dbf, comps, phases, conds, callables=callables, parameters={Symbol('VV0000'): 10000})
    np.testing.assert_allclose(eq_res.GM.values.squeeze(), 10000.0)


def test_eq_some_phases_filtered():
    """
    Phases are filtered out from equilibrium() when some cannot be built.
    """
    # should not raise; AL13FE4 should be filtered out
    equilibrium(ALFE_DBF, ['AL', 'VA'], ['FCC_A1', 'AL13FE4'], {v.T: 1200, v.P: 101325})


def test_equilibrium_result_dataset_can_serialize_to_netcdf():
    """
    The xarray Dataset returned by equilibrium should serializable to a netcdf file.
    """
    fname = 'eq_result_netcdf_test.nc'
    eq = equilibrium(ALFE_DBF, ['AL', 'VA'], 'FCC_A1', {v.T: 1200, v.P: 101325})
    eq.to_netcdf(fname)
    os.remove(fname)  # cleanup


@raises(ConditionError)
def test_equilibrium_raises_with_no_active_phases_passed():
    """Passing inactive phases to equilibrium raises a ConditionError."""
    # the only phases passed are the disordered phases, which are inactive
    equilibrium(ALNIFCC4SL_DBF, ['AL', 'NI', 'VA'], ['FCC_A1', 'BCC_A2'], {v.T: 300, v.P: 101325})


@raises(ConditionError)
def test_equilibrium_raises_when_no_phases_can_be_active():
    """Equliibrium raises when the components passed cannot give any active phases"""
    # all phases contain AL and/or FE in a sublattice, so no phases can be active
    equilibrium(ALFE_DBF, ['VA'], list(ALFE_DBF.phases.keys()), {v.T: 300, v.P: 101325})


def test_dataset_can_hold_maximum_phases_allowed_by_gibbs_phase_rule():
    """Creating datasets from equilibrium results should work when there are the maximum number of phases that can exist by Gibbs phase rule."""
    comps = ['PB', 'SN', 'VA']
    phases = list(PBSN_DBF.phases.keys())
    eq_res = equilibrium(PBSN_DBF, comps, phases, {v.P: 101325, v.T: 454.562, v.X('SN'): 0.738})
    assert eq_res.vertex.size == 3  # C+1
    assert np.sum(~np.isnan(eq_res.NP.values)) == 3
    assert np.sum(eq_res.Phase.values != '') == 3


@raises(NotImplementedError)
def test_equilibrium_raises_with_invalid_solver():
    """
    SolverBase instances passed to equilibrium should raise an error.
    """
    equilibrium(CUO_DBF, ['O'], 'GAS', {v.T: 1000, v.P: 1e5}, solver=SolverBase())


def test_equlibrium_no_opt_solver():
    """Passing in a solver with `ignore_convergence = True` gives a result."""

    class NoOptSolver(InteriorPointSolver):
        ignore_convergence = True

    comps = ['PB', 'SN', 'VA']
    phases = list(PBSN_DBF.phases.keys())
    conds = {v.T: 300, v.P: 101325, v.X('SN'): 0.50}
    ipopt_solver_eq_res = equilibrium(PBSN_DBF, comps, phases, conds, solver=InteriorPointSolver(), verbose=True)
    no_opt_eq_res = equilibrium(PBSN_DBF, comps, phases, conds, solver=NoOptSolver(), verbose=True)

    ipopt_GM = ipopt_solver_eq_res.GM.values.squeeze()
    no_opt_GM = no_opt_eq_res.GM.values.squeeze()
    no_opt_MU = no_opt_eq_res.MU.values.squeeze()
    assert ipopt_GM != no_opt_GM  # global min energy is different from lower convex hull
    assert np.allclose([-17452.5115967], no_opt_GM)  # energy from lower convex hull
    assert np.allclose([-19540.6522632, -15364.3709302], no_opt_MU)  # chempots from lower convex hull


def test_eq_ideal_chempot_cond():
    TDB = """
     ELEMENT A    GRAPHITE                   12.011     1054.0      5.7423 !
     ELEMENT B   BCC_A2                     55.847     4489.0     27.2797 !
     ELEMENT C   BCC_A2                     55.847     4489.0     27.2797 !
     TYPE_DEFINITION % SEQ * !
     PHASE TEST % 1 1 !
     CONSTITUENT TEST : A,B,C: !
    """
    my_phases = ['TEST']
    comps = ['A', 'B', 'C']
    comps = sorted(comps)
    conds = dict({v.T: 1000, v.P: 101325, v.N: 1})
    conds[v.MU('C')] = -1000
    conds[v.X('A')] = 0.01
    eq = equilibrium(Database(TDB), comps, my_phases, conds, verbose=True)
    np.testing.assert_allclose(eq.GM.values.squeeze(), -3219.570565)
    np.testing.assert_allclose(eq.MU.values.squeeze(), [-38289.687511, -18873.23674,  -1000.])
    np.testing.assert_allclose(eq.X.isel(vertex=0).values.squeeze(), [0.01,  0.103321,  0.886679], atol=1e-4)


def test_eq_tricky_chempot_cond():
    """
    Chemical potential condition with difficult convergence for chemical potentials.
    """
    eq = equilibrium(ISSUE43_DBF, ['AL', 'NI', 'CR', 'VA'], ['FCC_A1', 'GAMMA_PRIME'],
                     {v.MU('AL'): -135620.9960449, v.X('CR'): 0.6, v.T: 1273, v.P: 101325},
                     verbose=True)
    chempots = np.array([-135620.9960449, -47269.29002414, -92304.23688281])
    assert_allclose(eq.GM.values, -70680.53695)
    assert_allclose(np.nansum(np.squeeze(eq.NP * eq.X), axis=-2), [0.1246, 0.6, (1-0.1246-0.6)])
    assert_allclose(np.squeeze(eq.MU.values), chempots)

def test_eq_magnetic_chempot_cond():
    """
    Chemical potential condition with an ill-conditioned Hessian due to magnetism (Tc->0).
    This is difficult to reproduce so we only include some known examples here.
    """
    # This set of conditions is known to trigger the issue
    eq = equilibrium(ALFE_DBF, ['AL', 'FE', 'VA'], ['FCC_A1', 'AL13FE4'],
                     {v.MU('FE'): -123110, v.T: 300, v.P: 1e5}, verbose=True)
    assert_allclose(np.squeeze(eq.GM.values), -35427.1, atol=0.1)
    assert_allclose(np.squeeze(eq.MU.values), [-8490.7, -123110], atol=0.1)
