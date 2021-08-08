"""
The equilibrium test module verifies that the Equilibrium class produces the
correct solution for thermodynamic equilibrium.
"""

import warnings
import os
import pytest
from sympy import Symbol
from numpy.testing import assert_allclose
import numpy as np
from pycalphad import Database, Model, calculate, equilibrium, EquilibriumError, ConditionError
from pycalphad.codegen.callables import build_callables, build_phase_records
from pycalphad.core.solver import SolverBase, Solver
from pycalphad.core.utils import get_state_variables, instantiate_models
import pycalphad.variables as v
from pycalphad.tests.datasets import *

warnings.simplefilter("always", UserWarning) # so we can test warnings

ROSE_DBF = Database(ROSE_TDB)
ALFE_DBF = Database(ALFE_TDB)
ALNIFCC4SL_DBF = Database(ALNIFCC4SL_TDB)
ALCOCRNI_DBF = Database(ALCOCRNI_TDB)
ISSUE43_DBF = Database(ISSUE43_TDB)
TOUGH_CHEMPOT_DBF = Database(ALNI_TOUGH_CHEMPOT_TDB)
NI_AL_DUPIN_2001_DBF = Database(NI_AL_DUPIN_2001_TDB)
CUO_DBF = Database(CUO_TDB)
PBSN_DBF = Database(PBSN_TDB)
AL_PARAMETER_DBF = Database(AL_PARAMETER_TDB)
CUMG_PARAMETERS_DBF = Database(CUMG_PARAMETERS_TDB)


@pytest.mark.solver
def test_rose_nine():
    "Nine-component rose diagram point equilibrium calculation."
    my_phases_rose = ['TEST']
    comps = ['H', 'HE', 'LI', 'BE', 'B', 'C', 'N', 'O', 'F']
    conds = dict({v.T: 1000, v.P: 101325})
    for comp in comps[:-1]:
        conds[v.X(comp)] = 1.0/float(len(comps))
    eqx = equilibrium(ROSE_DBF, comps, my_phases_rose, conds, verbose=True)
    assert_allclose(eqx.GM.values.flat[0], -5.8351e3, atol=0.1)


@pytest.mark.solver
def test_eq_binary():
    "Binary phase diagram point equilibrium calculation with magnetism."
    my_phases = ['LIQUID', 'FCC_A1', 'HCP_A3', 'AL5FE2',
                 'AL2FE', 'AL13FE4', 'AL5FE4']
    comps = ['AL', 'FE', 'VA']
    conds = {v.T: 1400, v.P: 101325, v.X('AL'): 0.55}
    eqx = equilibrium(ALFE_DBF, comps, my_phases, conds, verbose=True)
    assert_allclose(eqx.GM.values.flat[0], -9.608807e4)


def test_phase_records_passed_to_equilibrium():
    "Pre-built phase records can be passed to equilibrium."
    my_phases = ['LIQUID', 'FCC_A1', 'HCP_A3', 'AL5FE2', 'AL2FE', 'AL13FE4', 'AL5FE4']
    comps = ['AL', 'FE', 'VA']
    conds = {v.T: 1400, v.P: 101325, v.N: 1.0, v.X('AL'): 0.55}

    models = instantiate_models(ALFE_DBF, comps, my_phases)
    phase_records = build_phase_records(ALFE_DBF, comps, my_phases, conds, models)

    # With models passed
    eqx = equilibrium(ALFE_DBF, comps, my_phases, conds, verbose=True, model=models, phase_records=phase_records)
    assert_allclose(eqx.GM.values.flat[0], -9.608807e4)


def test_missing_models_with_phase_records_passed_to_equilibrium_raises():
    "equilibrium should raise an error if all the active phases are not included in the phase_records"
    my_phases = ['LIQUID', 'FCC_A1', 'HCP_A3', 'AL5FE2', 'AL2FE', 'AL13FE4', 'AL5FE4']
    comps = ['AL', 'FE', 'VA']
    conds = {v.T: 1400, v.P: 101325, v.N: 1.0, v.X('AL'): 0.55}

    models = instantiate_models(ALFE_DBF, comps, my_phases)
    phase_records = build_phase_records(ALFE_DBF, comps, my_phases, conds, models)

    with pytest.raises(ValueError):
        # model=models NOT passed
        equilibrium(ALFE_DBF, comps, my_phases, conds, verbose=True, phase_records=phase_records)


def test_missing_phase_records_passed_to_equilibrium_raises():
    "equilibrium should raise an error if all the active phases are not included in the phase_records"
    my_phases = ['LIQUID', 'FCC_A1']
    subset_phases = ['FCC_A1']
    comps = ['AL', 'FE', 'VA']
    conds = {v.T: 1400, v.P: 101325, v.N: 1.0, v.X('AL'): 0.55}

    models = instantiate_models(ALFE_DBF, comps, my_phases)
    phase_records = build_phase_records(ALFE_DBF, comps, my_phases, conds, models)

    models_subset = instantiate_models(ALFE_DBF, comps, subset_phases)
    phase_records_subset = build_phase_records(ALFE_DBF, comps, subset_phases, conds, models_subset)

    # Under-specified models
    with pytest.raises(ValueError):
        equilibrium(ALFE_DBF, comps, my_phases, conds, verbose=True, model=models_subset, phase_records=phase_records)

    # Under-specified phase_records
    with pytest.raises(ValueError):
        equilibrium(ALFE_DBF, comps, my_phases, conds, verbose=True, model=models, phase_records=phase_records_subset)


@pytest.mark.solver
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


def test_eq_underdetermined_comps():
    """
    The number of composition conditions should yield exactly one dependent component.
    This is the underdetermined case.
    """
    with pytest.raises(ValueError):
        equilibrium(ALFE_DBF, ['AL', 'FE'], 'LIQUID', {v.T: 2000, v.P: 101325})


def test_eq_overdetermined_comps():
    """
    The number of composition conditions should yield exactly one dependent component.
    This is the overdetermined case.
    """
    with pytest.raises(ValueError):
        equilibrium(ALFE_DBF, ['AL', 'FE'], 'LIQUID', {v.T: 2000, v.P: 101325,
                                                   v.X('FE'): 0.2, v.X('AL'): 0.8})

@pytest.mark.solver
def test_dilute_condition():
    """
    'Zero' and dilute composition conditions are correctly handled.
    """
    eq = equilibrium(ALFE_DBF, ['AL', 'FE', 'VA'], 'FCC_A1', {v.T: 1300, v.P: 101325, v.X('AL'): 0}, verbose=True)
    assert_allclose(np.squeeze(eq.GM.values), -64415.84, atol=0.1)
    eq = equilibrium(ALFE_DBF, ['AL', 'FE', 'VA'], 'FCC_A1', {v.T: 1300, v.P: 101325, v.X('AL'): 1e-12}, verbose=True)
    assert_allclose(np.squeeze(eq.GM.values), -64415.841)
    assert_allclose(np.squeeze(eq.MU.values), [-385499.682936,  -64415.837878], atol=1.0)

@pytest.mark.solver
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

@pytest.mark.solver
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

@pytest.mark.solver
def test_eq_four_sublattice():
    """
    Balancing mass in a multi-sublattice phase in a single-phase configuration.
    """
    eq = equilibrium(ALNIFCC4SL_DBF, ['AL', 'NI', 'VA'], 'FCC_L12',
                     {v.T: 1073, v.X('NI'): 0.7601, v.P: 101325})
    assert_allclose(np.squeeze(eq.X.sel(vertex=0).values), [1-.7601, .7601])
    # Not a strict equality here because we can't yet reach TC's value of -87260.6
    assert eq.GM.values < -87256.3

def test_eq_missing_component():
    """
    Specifying a non-existent component raises an error.
    """
    # No Co or Cr in this database ; Co component specification should cause failure
    with pytest.raises(EquilibriumError):
        equilibrium(ALNIFCC4SL_DBF, ['AL', 'CO', 'CR', 'VA'], ['LIQUID'],
                    {v.T: 1523, v.X('AL'): 0.88811111111111107,
                     v.X('CO'): 0.11188888888888888, v.P: 101325})

def test_eq_missing_component():
    """
    Specifying a condition involving a non-existent component raises an error.
    """
    # No Co or Cr in this database ; Co condition specification should cause failure
    with pytest.raises(ConditionError):
        equilibrium(ALNIFCC4SL_DBF, ['AL', 'NI', 'VA'], ['LIQUID'],
                    {v.T: 1523, v.X('AL'): 0.88811111111111107,
                     v.X('CO'): 0.11188888888888888, v.P: 101325})

@pytest.mark.solver
def test_eq_ternary_edge_case_mass():
    """
    Equilibrium along an edge of composition space will still balance mass.
    """
    eq = equilibrium(ALCOCRNI_DBF, ['AL', 'CO', 'CR', 'VA'], ['L12_FCC', 'BCC_B2', 'LIQUID'],
                     {v.T: 1523, v.X('AL'): 0.8881111111,
                      v.X('CO'): 0.1118888888, v.P: 101325}, verbose=True)
    mass_error = np.nansum(np.squeeze(eq.NP * eq.X), axis=-2) - \
                 [0.8881111111, 0.1118888888, 1e-10]
    assert_allclose(eq.GM.values, -97913.542)  # from Thermo-Calc 2017b
    result_chempots = eq.MU.values.flatten()
    assert_allclose(result_chempots[:2], [-86994.575, -184582.17], atol=0.1)  # from Thermo-Calc 2017b
    assert result_chempots[2] < -300000  # Estimated
    assert np.all(np.abs(mass_error) < 1e-10)

@pytest.mark.solver
def test_eq_ternary_inside_mass():
    """
    Equilibrium in interior of composition space will still balance mass.
    """
    eq = equilibrium(ALCOCRNI_DBF, ['AL', 'CO', 'CR', 'VA'], ['L12_FCC', 'BCC_B2', 'LIQUID'],
                     {v.T: 1523, v.X('AL'): 0.44455555555555554,
                      v.X('CO'): 0.22277777777777777, v.P: 101325}, verbose=True)
    assert_allclose(eq.GM.values, -105871.54, atol=0.1)  # Thermo-Calc: -105871.54
    # Thermo-Calc: [-104653.83, -142595.49, -82905.784]
    assert_allclose(eq.MU.values.flatten(), [-104653.83, -142595.49, -82905.784], atol=0.1)


@pytest.mark.solver
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

@pytest.mark.solver
def test_eq_issue43_chempots_misc_gap():
    """
    Equilibrium for complex ternary miscibility gap (gh-43).
    """
    eq = equilibrium(ISSUE43_DBF, ['AL', 'NI', 'CR', 'VA'], 'GAMMA_PRIME',
                     {v.X('AL'): .1246, v.X('CR'): 1e-9, v.T: 1273, v.P: 101325},
                     verbose=True)
    chempots = np.array([-206144.57, -272150.79, -64253.652])
    assert_allclose(np.nansum(np.squeeze(eq.NP * eq.X), axis=-2), [0.1246, 1e-9, 1-(.1246+1e-9)], rtol=2e-5)
    assert_allclose(np.squeeze(eq.MU.values), chempots, rtol=1e-5)
    assert_allclose(np.squeeze(eq.GM.values), -81933.259)

@pytest.mark.solver
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

@pytest.mark.solver
def test_eq_large_vacancy_hessian():
    """
    Vacancy contribution to phase matrix must be included to get the correct answer.
    """
    dbf = NI_AL_DUPIN_2001_DBF
    comps = ['AL', 'NI', 'VA']
    phases = ['BCC_B2']
    eq = equilibrium(dbf, comps, phases, {v.P: 101325, v.T: 1804, v.N: 1, v.X('AL'): 0.4798})
    assert_allclose(eq.GM.values, -154338.129)
    assert_allclose(eq.MU.values.flatten(), [-167636.23822714, -142072.78317111])
    assert_allclose(eq.X.sel(vertex=0).values.flatten(), [0.4798, 0.5202])

@pytest.mark.solver
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

@pytest.mark.solver
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
    np.testing.assert_allclose(eq.SM, 0, atol=1e-14)

@pytest.mark.solver
def test_eq_gas_phase():
    eq = equilibrium(CUO_DBF, ['O'], 'GAS', {v.T: 1000, v.P: 1e5}, verbose=True)
    np.testing.assert_allclose(eq.GM, -110380.61071, atol=0.1)
    eq = equilibrium(CUO_DBF, ['O'], 'GAS', {v.T: 1000, v.P: 1e9}, verbose=True)
    np.testing.assert_allclose(eq.GM, -7.20909E+04, atol=0.1)

@pytest.mark.solver
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
    conds_statevars = get_state_variables(conds=conds)
    models = {'FCC_A1': Model(dbf, comps, 'FCC_A1', parameters=['VV0000'])}
    # build callables with a parameter of 20000.0
    callables = build_callables(dbf, comps, phases,
                                models=models, parameter_symbols=['VV0000'], additional_statevars=conds_statevars,
                                build_gradients=True, build_hessians=True)

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


def test_equilibrium_raises_with_no_active_phases_passed():
    """Passing inactive phases to equilibrium raises a ConditionError."""
    # the only phases passed are the disordered phases, which are inactive
    with pytest.raises(ConditionError):
        equilibrium(ALNIFCC4SL_DBF, ['AL', 'NI', 'VA'], ['FCC_A1', 'BCC_A2'], {v.T: 300, v.P: 101325})


def test_equilibrium_raises_when_no_phases_can_be_active():
    """Equliibrium raises when the components passed cannot give any active phases"""
    # all phases contain AL and/or FE in a sublattice, so no phases can be active
    with pytest.raises(ConditionError):
        equilibrium(ALFE_DBF, ['VA'], list(ALFE_DBF.phases.keys()), {v.T: 300, v.P: 101325})


# Defer test until inclusion of NP conditions, so test can be rewritten properly
# As is, the "correct" test temperature is very sensitive to platform-specific numerical settings
@pytest.mark.skip("Skip until NP conditions are complete.")
def test_dataset_can_hold_maximum_phases_allowed_by_gibbs_phase_rule():
    """Creating datasets from equilibrium results should work when there are the maximum number of phases that can exist by Gibbs phase rule."""
    comps = ['PB', 'SN', 'VA']
    phases = list(PBSN_DBF.phases.keys())
    # "Exact" invariant temperature is very sensitive to solver convergence criteria
    eq_res = equilibrium(PBSN_DBF, comps, phases, {v.P: 101325, v.T: 454.56201, v.X('SN'): 0.738})
    assert eq_res.vertex.size == 3  # C+1
    assert np.sum(~np.isnan(eq_res.NP.values)) == 3
    assert np.sum(eq_res.Phase.values != '') == 3


def test_equilibrium_raises_with_invalid_solver():
    """
    SolverBase instances passed to equilibrium should raise an error.
    """
    with pytest.raises(NotImplementedError):
        equilibrium(CUO_DBF, ['O'], 'GAS', {v.T: 1000, v.P: 1e5}, solver=SolverBase())


def test_equilibrium_no_opt_solver():
    """Passing in a solver with `ignore_convergence = True` gives a result."""

    class NoOptSolver(Solver):
        ignore_convergence = True

    comps = ['PB', 'SN', 'VA']
    phases = list(PBSN_DBF.phases.keys())
    conds = {v.T: 300, v.P: 101325, v.X('SN'): 0.50}
    ipopt_solver_eq_res = equilibrium(PBSN_DBF, comps, phases, conds, solver=Solver(), verbose=True)
    # NoOptSolver's results are pdens-dependent
    no_opt_eq_res = equilibrium(PBSN_DBF, comps, phases, conds,
                                solver=NoOptSolver(), calc_opts={'pdens': 50}, verbose=True)

    ipopt_GM = ipopt_solver_eq_res.GM.values.squeeze()
    no_opt_GM = no_opt_eq_res.GM.values.squeeze()
    no_opt_MU = no_opt_eq_res.MU.values.squeeze()
    assert ipopt_GM != no_opt_GM  # global min energy is different from lower convex hull
    assert np.allclose([-17449.81365585], no_opt_GM)  # energy from lower convex hull
    assert np.allclose([-19540.85816392, -15358.76914778], no_opt_MU)  # chempots from lower convex hull


@pytest.mark.solver
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


@pytest.mark.solver
def test_eq_tricky_chempot_cond():
    """
    Chemical potential condition with difficult convergence for chemical potentials.
    """
    eq = equilibrium(ISSUE43_DBF, ['AL', 'NI', 'CR', 'VA'], ['FCC_A1', 'GAMMA_PRIME'],
                     {v.MU('AL'): -135620.9960449, v.MU('CR'): -47269.29002414, v.T: 1273, v.P: 101325},
                     verbose=True)
    chempots = np.array([-135620.9960449, -47269.29002414, -92304.23688281])
    print(np.nansum(np.squeeze(eq.NP * eq.X), axis=-2))
    assert_allclose(eq.GM.values, -83242.872102)
    assert_allclose(np.nansum(np.squeeze(eq.NP * eq.X), axis=-2), [0.19624727,  0.38996739,  0.41378534])
    assert_allclose(np.squeeze(eq.MU.values), chempots)

@pytest.mark.solver
def test_eq_magnetic_chempot_cond():
    """
    Chemical potential condition with an ill-conditioned Hessian due to magnetism (Tc->0).
    This is difficult to reproduce so we only include some known examples here.
    """
    # This set of conditions is known to trigger the issue
    eq = equilibrium(ALFE_DBF, ['AL', 'FE', 'VA'], ['FCC_A1', 'AL13FE4'],
                     {v.MU('FE'): -123110, v.T: 300, v.P: 1e5}, verbose=True)
    # Checked in Thermo-Calc 2017b
    assert_allclose(np.squeeze(eq.GM.values), -35427.064, atol=0.1)
    assert_allclose(np.squeeze(eq.MU.values), [-8490.6849, -123110], atol=0.1)

def test_eq_calculation_with_parameters():
    parameters = {'VV0000': -33134.699474175846, 'VV0001': 7734.114029426941, 'VV0002': -13498.542175596054,
                  'VV0003': -26555.048975092268, 'VV0004': 20777.637577083482, 'VV0005': 41915.70425630003,
                  'VV0006': -34525.21964215504, 'VV0007': 95457.14639216446, 'VV0008': 21139.578967453144,
                  'VV0009': 19047.833726419598, 'VV0010': 20468.91829601273, 'VV0011': 19601.617855958328,
                  'VV0012': -4546.9325861738, 'VV0013': -1640.6354331231278, 'VV0014': -35682.950005357634}
    eq = equilibrium(CUMG_PARAMETERS_DBF, ['CU', 'MG'], ['HCP_A3'],
                     {v.X('CU'): 0.0001052, v.P: 101325.0, v.T: 743.15, v.N: 1},
                     parameters=parameters, verbose=True)
    assert_allclose(eq.GM.values, -30374.196034, atol=0.1)


@pytest.mark.solver
def test_eq_alni_low_temp():
    """
    Low temperature Al-Ni keeps correct stable set at equilibrium.
    """
    dbf = NI_AL_DUPIN_2001_DBF
    comps = ['AL', 'NI', 'VA']
    phases = sorted(dbf.phases.keys())
    eq = equilibrium(dbf, comps, phases, {v.P: 101325, v.T: 300, v.N: 1, v.X('AL'): 0.4})
    # Verified in TC: https://github.com/pycalphad/pycalphad/pull/329#discussion_r637241358
    assert_allclose(eq.GM.values, -63736.3048)
    assert_allclose(eq.MU.values.flatten(), [-116098.937755,  -28827.882809])
    assert set(np.squeeze(eq.Phase.values)) == {'BCC_B2', 'AL3NI5', ''}
    bcc_idx = np.nonzero(np.squeeze(eq.Phase.values) == 'BCC_B2')[0][0]
    al3ni5_idx = np.nonzero(np.squeeze(eq.Phase.values) == 'AL3NI5')[0][0]
    assert_allclose(np.squeeze(eq.X.sel(vertex=bcc_idx).values), [0.488104, 0.511896], atol=1e-6)
    assert_allclose(np.squeeze(eq.X.sel(vertex=al3ni5_idx).values), [0.375, 0.625], atol=1e-6)


@pytest.mark.solver
def test_eq_alni_high_temp():
    """
    Avoid 'jitter' in high-temperature phase equilibria with dilute site fractions.
    """
    dbf = NI_AL_DUPIN_2001_DBF
    comps = ['AL', 'NI', 'VA']
    phases = sorted(dbf.phases.keys())
    eq = equilibrium(dbf, comps, phases, {v.P: 101325, v.T: 1600, v.N: 1, v.X('AL'): 0.65})
    # if MIN_SITE_FRACTION is set to 1e-16: -131048.695
    assert_allclose(eq.GM.values, -131081.998)
    # if MIN_SITE_FRACTION is set to 1e-16: [-106515.007322, -176611.259853]
    assert_allclose(eq.MU.values.flatten(), [-106284.8589, -177133.82819])
    assert set(np.squeeze(eq.Phase.values)) == {'BCC_B2', 'LIQUID', ''}
    bcc_idx = np.nonzero(np.squeeze(eq.Phase.values) == 'BCC_B2')[0][0]
    liq_idx = np.nonzero(np.squeeze(eq.Phase.values) == 'LIQUID')[0][0]
    assert_allclose(np.squeeze(eq.X.sel(vertex=bcc_idx).values), [0.563528, 0.436472], atol=1e-6)
    assert_allclose(np.squeeze(eq.X.sel(vertex=liq_idx).values), [0.695314, 0.304686], atol=1e-6)


@pytest.mark.solver
def test_eq_issue259():
    """
    Chemical potential condition for phase with internal degrees of freedom.
    """
    dbf = ALFE_DBF
    comps = ['AL', 'FE', 'VA']
    eq = equilibrium(dbf, comps, ['B2_BCC'], {v.P: 101325, v.N: 1, v.T: 1013, v.MU('AL'): -95906})
    assert_allclose(eq.GM.values, -65786.260)
    assert_allclose(eq.MU.values.flatten(), [-95906., -52877.592122])


@pytest.mark.solver
def test_eq_needs_metastable_starting():
    """
    Complex multi-component system with many phases near the starting hyperplane.
    """
    dbf = Database(MC_FECOCRNBTI_TDB)
    phases = list(set(dbf.phases.keys()) - {'GP_MAT'})
    mass_fracs = {v.W('CR'): 28./100, v.W('FE'): 21./100, v.W('NB'): 1./100, v.W('TI'): 1.3/100}
    conds = v.get_mole_fractions(mass_fracs, 'CO', dbf)
    conds[v.T] = 960
    conds[v.P] = 1e5
    conds[v.N] = 1
    eq = equilibrium(dbf, ['CO', 'CR', 'FE', 'NB', 'TI', 'VA'], phases, conds)
    assert set(np.squeeze(eq.Phase.values)) == {'SIGMA', 'BCC_A2', 'MU_PHASE', 'FCC_A1', 'LAVES_PHASE', ''}
    assert_allclose(eq.GM.values, -46868.31620088)
