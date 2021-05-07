"""
The energy test module verifies that the Model class produces the
correct abstract syntax tree for the energy.
"""

import pytest
from sympy import S
from pycalphad import Database, Model, ReferenceState
from pycalphad.core.utils import make_callable
from pycalphad.tests.datasets import ALCRNI_TDB, FEMN_TDB, FE_MN_S_TDB, ALFE_TDB, \
    CRFE_BCC_MAGNETIC_TDB, VA_INTERACTION_TDB, CUMG_TDB, AL_C_FE_B2_TDB
from pycalphad.core.errors import DofError
import pycalphad.variables as v
import numpy as np

DBF = Database(ALCRNI_TDB)
ALFE_DBF = Database(ALFE_TDB)
FEMN_DBF = Database(FEMN_TDB)
CRFE_DBF = Database(CRFE_BCC_MAGNETIC_TDB)
CUMG_DBF = Database(CUMG_TDB)
FE_MN_S_DBF = Database(FE_MN_S_TDB)
VA_INTERACTION_DBF = Database(VA_INTERACTION_TDB)
AL_C_FE_B2_DBF = Database(AL_C_FE_B2_TDB)


def test_sympify_safety():
    "Parsing malformed strings throws exceptions instead of executing code."
    from pycalphad.io.tdb import _sympify_string
    teststr = "().__class__.__base__.__subclasses__()[216]('ls')"
    with pytest.raises(ValueError):
        _sympify_string(teststr)


def calculate_output(model, variables, output, mode='sympy'):
    """
    Calculate the value of the energy at a point.

    Parameters
    ----------
    model, Model
        Energy model for a phase.

    variables, dict
        Dictionary of all input variables.

    output : str
        String of the property to calculate, e.g. 'ast'

    mode, ['numpy', 'sympy'], optional
        Optimization method for the abstract syntax tree.
    """
    # Generate a callable function
    # Normally we would use model.subs(variables) here, but we want to ensure
    # our optimization functions are working.
    prop = make_callable(getattr(model, output), list(variables.keys()), mode=mode)
    # Unpack all the values in the dict and use them to call the function
    return prop(*(list(variables.values())))


def check_output(model, variables, output, known_value, mode='sympy'):
    "Check that our calculated quantity matches the known value."
    desired = calculate_output(model, variables, output, mode)
    known_value = np.array(known_value, dtype=np.complex_)
    desired = np.array(desired, dtype=np.complex_)
    # atol defaults to zero here, but it cannot be zero if desired is zero
    # we set it to a reasonably small number for energies and derivatives (in Joules)
    # An example where expected = 0, but known != 0 is for ideal mix xlogx terms
    # This 1e-8 value is also used in hyperplane, motivating the use here.
    np.testing.assert_allclose(known_value, desired, rtol=1e-5, atol=1e-8)


def check_energy(model, variables, known_value, mode='sympy'):
    "Check that our calculated energy matches the known value."
    check_output(model, variables, 'ast', known_value, mode=mode)

# PURE COMPONENT TESTS
def test_pure_sympy():
    "Pure component end-members in sympy mode."
    check_energy(Model(DBF, ['AL'], 'LIQUID'), \
            {v.T: 2000, v.SiteFraction('LIQUID', 0, 'AL'): 1}, \
        -1.28565e5, mode='sympy')
    check_energy(Model(DBF, ['AL'], 'B2'), \
            {v.T: 1400, v.SiteFraction('B2', 0, 'AL'): 1,
             v.SiteFraction('B2', 1, 'AL'): 1}, \
        -6.57639e4, mode='sympy')
    check_energy(Model(DBF, ['AL'], 'L12_FCC'), \
            {v.T: 800, v.SiteFraction('L12_FCC', 0, 'AL'): 1,
             v.SiteFraction('L12_FCC', 1, 'AL'): 1}, \
        -3.01732e4, mode='sympy')

def test_degenerate_ordered():
    "Degenerate sublattice configuration has same energy as disordered phase."
    mod_l12 = Model(DBF, ['CR', 'NI'], 'L12_FCC')
    mod_a1 = Model(DBF, ['CR', 'NI'], 'FCC_A1')
    l12_subs = {v.T: 500, v.SiteFraction('L12_FCC', 0, 'CR'): 0.33,
                v.SiteFraction('L12_FCC', 0, 'NI'): 0.67,
                v.SiteFraction('L12_FCC', 1, 'CR'): 0.33,
                v.SiteFraction('L12_FCC', 1, 'NI'): 0.67}
    a1_subs = {v.T: 500, v.SiteFraction('FCC_A1', 0, 'CR'): 0.33,
               v.SiteFraction('FCC_A1', 0, 'NI'): 0.67}
    l12_energy = mod_l12.energy.xreplace(l12_subs)
    a1_energy = mod_a1.energy.xreplace(a1_subs)
    np.testing.assert_almost_equal(l12_energy, a1_energy)

def test_degenerate_zero_ordering():
    "Degenerate sublattice configuration has zero ordering energy."
    mod = Model(DBF, ['CR', 'NI'], 'L12_FCC')
    sub_dict = {v.T: 500, v.SiteFraction('L12_FCC', 0, 'CR'): 0.33,
                v.SiteFraction('L12_FCC', 0, 'NI'): 0.67,
                v.SiteFraction('L12_FCC', 1, 'CR'): 0.33,
                v.SiteFraction('L12_FCC', 1, 'NI'): 0.67}
    #print({x: mod.models[x].subs(sub_dict) for x in mod.models})
    desired = mod.models['ord'].xreplace(sub_dict).evalf()
    assert abs(desired - 0) < 1e-5, "%r != %r" % (desired, 0)

# BINARY TESTS
def test_binary_magnetic():
    "Two-component phase with IHJ magnetic model."
    # disordered case
    check_energy(Model(DBF, ['CR', 'NI'], 'L12_FCC'), \
            {v.T: 500, v.SiteFraction('L12_FCC', 0, 'CR'): 0.33,
             v.SiteFraction('L12_FCC', 0, 'NI'): 0.67,
             v.SiteFraction('L12_FCC', 1, 'CR'): 0.33,
             v.SiteFraction('L12_FCC', 1, 'NI'): 0.67}, \
        -1.68840e4, mode='sympy')

def test_binary_magnetic_reimported():
    "Export and re-import a TDB before the calculation."
    dbf_imported = Database.from_string(DBF.to_string(fmt='tdb'), fmt='tdb')
    check_energy(Model(dbf_imported, ['CR', 'NI'], 'L12_FCC'),
                {v.T: 500, v.SiteFraction('L12_FCC', 0, 'CR'): 0.33,
                v.SiteFraction('L12_FCC', 0, 'NI'): 0.67,
                v.SiteFraction('L12_FCC', 1, 'CR'): 0.33,
                v.SiteFraction('L12_FCC', 1, 'NI'): 0.67},
                -1.68840e4, mode='sympy')

def test_binary_magnetic_ordering():
    "Two-component phase with IHJ magnetic model and ordering."
    # ordered case
    check_energy(Model(DBF, ['CR', 'NI'], 'L12_FCC'), \
            {v.T: 300, v.SiteFraction('L12_FCC', 0, 'CR'): 4.86783e-2,
             v.SiteFraction('L12_FCC', 0, 'NI'): 9.51322e-1,
             v.SiteFraction('L12_FCC', 1, 'CR'): 9.33965e-1,
             v.SiteFraction('L12_FCC', 1, 'NI'): 6.60348e-2}, \
        -9.23953e3, mode='sympy')

def test_binary_dilute():
    "Dilute binary solution phase."
    check_energy(Model(DBF, ['CR', 'NI'], 'LIQUID'), \
            {v.T: 300, v.SiteFraction('LIQUID', 0, 'CR'): 1e-12,
             v.SiteFraction('LIQUID', 0, 'NI'): 1.0-1e-12}, \
        5.52773e3, mode='sympy')

def test_binary_xiong_twostate_einstein():
    "Phase with Xiong magnetic, two-state and Einstein energy contributions."
    femn_dbf = Database(FEMN_TDB)
    mod = Model(femn_dbf, ['FE', 'MN', 'VA'], 'LIQUID')
    check_energy(mod, {v.T: 10, v.SiteFraction('LIQUID', 0, 'FE'): 1,
                                v.SiteFraction('LIQUID', 0, 'MN'): 0,
                                v.SiteFraction('LIQUID', 1, 'VA'): 1},
                 10158.591, mode='sympy')
    check_energy(mod, {v.T: 300, v.SiteFraction('LIQUID', 0, 'FE'): 0.3,
                       v.SiteFraction('LIQUID', 0, 'MN'): 0.7,
                       v.SiteFraction('LIQUID', 1, 'VA'): 1},
                 4200.8435, mode='sympy')
    check_energy(mod, {v.T: 1500, v.SiteFraction('LIQUID', 0, 'FE'): 0.8,
                       v.SiteFraction('LIQUID', 0, 'MN'): 0.2,
                       v.SiteFraction('LIQUID', 1, 'VA'): 1},
                 -86332.217, mode='sympy')

# TERNARY TESTS
def test_ternary_rkm_solution():
    "Solution phase with ternary interaction parameters."
    check_energy(Model(DBF, ['AL', 'CR', 'NI'], 'LIQUID'), \
            {v.T: 1500, v.SiteFraction('LIQUID', 0, 'AL'): 0.44,
             v.SiteFraction('LIQUID', 0, 'CR'): 0.20,
             v.SiteFraction('LIQUID', 0, 'NI'): 0.36}, \
        -1.16529e5, mode='sympy')

def test_ternary_symmetric_param():
    "Generate the other two ternary parameters if only the zeroth is specified."
    check_energy(Model(DBF, ['AL', 'CR', 'NI'], 'FCC_A1'), \
            {v.T: 300, v.SiteFraction('FCC_A1', 0, 'AL'): 1.97135e-1,
             v.SiteFraction('FCC_A1', 0, 'CR'): 1.43243e-2,
             v.SiteFraction('FCC_A1', 0, 'NI'): 7.88541e-1},
                 -37433.794, mode='sympy')

def test_ternary_ordered_magnetic():
    "Ternary ordered solution phase with IHJ magnetic model."
    # ordered case
    check_energy(Model(DBF, ['AL', 'CR', 'NI'], 'L12_FCC'), \
            {v.T: 300, v.SiteFraction('L12_FCC', 0, 'AL'): 5.42883e-8,
             v.SiteFraction('L12_FCC', 0, 'CR'): 2.07934e-6,
             v.SiteFraction('L12_FCC', 0, 'NI'): 9.99998e-1,
             v.SiteFraction('L12_FCC', 1, 'AL'): 7.49998e-1,
             v.SiteFraction('L12_FCC', 1, 'CR'): 2.50002e-1,
             v.SiteFraction('L12_FCC', 1, 'NI'): 4.55313e-10}, \
        -40717.204, mode='sympy')

# QUATERNARY TESTS
def test_quaternary():
    "Quaternary ordered solution phase."
    check_energy(Model(DBF, ['AL', 'CR', 'NI', 'VA'], 'B2'), \
            {v.T: 500, v.SiteFraction('B2', 0, 'AL'): 4.03399e-9,
             v.SiteFraction('B2', 0, 'CR'): 2.65798e-4,
             v.SiteFraction('B2', 0, 'NI'): 9.99734e-1,
             v.SiteFraction('B2', 0, 'VA'): 2.68374e-9,
             v.SiteFraction('B2', 1, 'AL'): 3.75801e-1,
             v.SiteFraction('B2', 1, 'CR'): 1.20732e-1,
             v.SiteFraction('B2', 1, 'NI'): 5.03467e-1,
             v.SiteFraction('B2', 1, 'VA'): 1e-12}, \
        -42368.27, mode='sympy')

# SPECIAL CASES
def test_case_sensitivity():
    "Case sensitivity of component and phase names."
    check_energy(Model(DBF, ['Cr', 'nI'], 'Liquid'), \
            {v.T: 300, v.SiteFraction('LIQUID', 0, 'CR'): 1e-12,
             v.SiteFraction('liquid', 0, 'ni'): 1}, \
        5.52773e3, mode='sympy')

def test_zero_site_fraction():
    "Energy of a binary solution phase where one site fraction is zero."
    check_energy(Model(DBF, ['CR', 'NI'], 'LIQUID'), \
            {v.T: 300, v.SiteFraction('LIQUID', 0, 'CR'): 0,
             v.SiteFraction('LIQUID', 0, 'NI'): 1}, \
        5.52773e3, mode='sympy')


def test_reference_energy_of_unary_twostate_einstein_magnetic_is_zero():
    """The referenced energy for the pure elements in a unary Model with twostate and Einstein contributions referenced to that phase is zero."""
    m = Model(FEMN_DBF, ['FE', 'VA'], 'LIQUID')
    statevars = {v.T: 298.15, v.SiteFraction('LIQUID', 0, 'FE'): 1, v.SiteFraction('LIQUID', 1, 'VA'): 1}
    refstates = [ReferenceState(v.Species('FE'), 'LIQUID')]
    m.shift_reference_state(refstates, FEMN_DBF)
    check_output(m, statevars, 'GMR', 0.0)


def test_underspecified_refstate_raises():
    """A Model cannot be shifted to a new reference state unless references for all pure elements are specified."""
    m = Model(FEMN_DBF, ['FE', 'MN', 'VA'], 'LIQUID')
    refstates = [ReferenceState(v.Species('FE'), 'LIQUID')]
    with pytest.raises(DofError):
        m.shift_reference_state(refstates, FEMN_DBF)


def test_reference_energy_of_binary_twostate_einstein_is_zero():
    """The referenced energy for the pure elements in a binary Model with twostate and Einstein contributions referenced to that phase is zero."""
    m = Model(FEMN_DBF, ['FE', 'MN', 'VA'], 'LIQUID')
    refstates = [ReferenceState(v.Species('FE'), 'LIQUID'), ReferenceState(v.Species('MN'), 'LIQUID')]
    m.shift_reference_state(refstates, FEMN_DBF)

    statevars_FE = {v.T: 298.15,
             v.SiteFraction('LIQUID', 0, 'FE'): 1, v.SiteFraction('LIQUID', 0, 'MN'): 0,
             v.SiteFraction('LIQUID', 1, 'VA'): 1}
    check_output(m, statevars_FE, 'GMR', 0.0)

    statevars_CR = {v.T: 298.15,
             v.SiteFraction('LIQUID', 0, 'FE'): 0, v.SiteFraction('LIQUID', 0, 'MN'): 1,
             v.SiteFraction('LIQUID', 1, 'VA'): 1}
    check_output(m, statevars_CR, 'GMR', 0.0)


def test_magnetic_reference_energy_is_zero():
    """The referenced energy binary magnetic Model is zero."""
    m = Model(CRFE_DBF, ['CR', 'FE', 'VA'], 'BCC_A2')
    refstates = [ReferenceState('CR', 'BCC_A2'), ReferenceState('FE', 'BCC_A2')]
    m.shift_reference_state(refstates, CRFE_DBF)

    statevars_FE = {v.T: 300,
             v.SiteFraction('BCC_A2', 0, 'CR'): 0, v.SiteFraction('BCC_A2', 0, 'FE'): 1,
             v.SiteFraction('BCC_A2', 1, 'VA'): 1}
    check_output(m, statevars_FE, 'GMR', 0.0)

    statevars_CR = {v.T: 300,
             v.SiteFraction('BCC_A2', 0, 'CR'): 1, v.SiteFraction('BCC_A2', 0, 'FE'): 0,
             v.SiteFraction('BCC_A2', 1, 'VA'): 1}
    check_output(m, statevars_CR, 'GMR', 0.0)


def test_non_zero_reference_mixing_enthalpy_for_va_interaction():
    """The referenced mixing enthalpy for a Model with a VA interaction parameter is non-zero."""
    m = Model(VA_INTERACTION_DBF, ['AL', 'VA'], 'FCC_A1')
    refstates = [ReferenceState('AL', 'FCC_A1')]
    m.shift_reference_state(refstates, VA_INTERACTION_DBF)

    statevars_pure = {v.T: 300,
         v.SiteFraction('FCC_A1', 0, 'AL'): 1, v.SiteFraction('FCC_A1', 0, 'VA'): 0,
         v.SiteFraction('FCC_A1', 1, 'VA'): 1}
    check_output(m, statevars_pure, 'GMR', 0.0)

    statevars_mix = {v.T: 300,
        v.SiteFraction('FCC_A1', 0, 'AL'): 0.5, v.SiteFraction('FCC_A1', 0, 'VA'): 0.5,
        v.SiteFraction('FCC_A1', 1, 'VA'): 1}
    # 4000.0 * 0.5=2000 +500 # (Y0VA doesn't contribute), but the VA endmember does (not referenced)
    check_output(m, statevars_mix, 'HMR', 2500.0)

    statevars_mix = {v.T: 300,
        v.SiteFraction('FCC_A1', 0, 'AL'): 0.5, v.SiteFraction('FCC_A1', 0, 'VA'): 0.5,
        v.SiteFraction('FCC_A1', 1, 'VA'): 1}
    # 4000.0 * 0.5 (Y0VA doesn't contribute)
    check_output(m, statevars_mix, 'HM_MIX', 2000.0)


def test_reference_energy_for_different_phase():
    """The referenced energy a different phase should be correct."""
    m = Model(ALFE_DBF, ['AL', 'FE', 'VA'], 'AL2FE')
    # formation reference states
    refstates = [ReferenceState('AL', 'FCC_A1'), ReferenceState('FE', 'BCC_A2')]
    m.shift_reference_state(refstates, ALFE_DBF)

    statevars = {v.T: 300, v.SiteFraction('AL2FE', 0, 'AL'): 1, v.SiteFraction('AL2FE', 1, 'FE'): 1}
    check_output(m, statevars, 'GMR', -28732.525)  # Checked in Thermo-Calc


def test_endmember_mixing_energy_is_zero():
    """The mixing energy for an endmember in a multi-sublattice model should be zero."""
    m = Model(CUMG_DBF, ['CU', 'MG', 'VA'], 'CU2MG')
    statevars = {
                    v.T: 300,
                    v.SiteFraction('CU2MG', 0, 'CU'): 1, v.SiteFraction('CU2MG', 0, 'MG'): 0,
                    v.SiteFraction('CU2MG', 1, 'CU'): 0, v.SiteFraction('CU2MG', 1, 'MG'): 1,
                }
    check_output(m, statevars, 'GM_MIX', 0.0)


def test_magnetic_endmember_mixing_energy_is_zero():
    """The mixing energy for an endmember with a magnetic contribution should be zero."""
    m = Model(CRFE_DBF, ['CR', 'FE', 'VA'], 'BCC_A2')
    statevars = {
                    v.T: 300,
                    v.SiteFraction('BCC_A2', 0, 'CR'): 0, v.SiteFraction('BCC_A2', 0, 'FE'): 1,
                    v.SiteFraction('BCC_A2', 1, 'VA'): 1}
    check_output(m, statevars, 'GM_MIX', 0.0)


def test_order_disorder_mixing_energy_is_nan():
    """The endmember-referenced mixing energy is undefined and the energy should be NaN."""
    m = Model(ALFE_DBF, ['AL', 'FE', 'VA'], 'B2_BCC')
    statevars = {
                    v.T: 300,
                    v.SiteFraction('B2_BCC', 0, 'AL'): 1, v.SiteFraction('B2_BCC', 0, 'FE'): 0,
                    v.SiteFraction('B2_BCC', 1, 'AL'): 0, v.SiteFraction('B2_BCC', 1, 'FE'): 1,
                    v.SiteFraction('B2_BCC', 2, 'VA'): 1}
    check_output(m, statevars, 'GM_MIX', np.nan)


def test_changing_model_ast_also_changes_mixing_energy():
    """If a models contribution is modified, the mixing energy should update accordingly."""
    m = Model(CUMG_DBF, ['CU', 'MG', 'VA'], 'CU2MG')
    m.models['mag'] = 1000
    statevars = {
                    v.T: 300,
                    v.SiteFraction('CU2MG', 0, 'CU'): 1, v.SiteFraction('CU2MG', 0, 'MG'): 0,
                    v.SiteFraction('CU2MG', 1, 'CU'): 0, v.SiteFraction('CU2MG', 1, 'MG'): 1,
                }
    check_output(m, statevars, 'GM_MIX', 1000)

    m.endmember_reference_model.models['mag'] = 1000
    check_output(m, statevars, 'GM_MIX', 0)


def test_shift_reference_state_model_contribs_take_effect():
    """Shift reference state with contrib_mods set adds contributions to the pure elements."""
    TDB = """
     ELEMENT A    GRAPHITE                   12.011     1054.0      5.7423 !
     ELEMENT B   BCC_A2                     55.847     4489.0     27.2797 !
     TYPE_DEFINITION % SEQ * !
     PHASE TEST % 1 1 !
     CONSTITUENT TEST : A,B: !
    """
    dbf = Database(TDB)
    comps = ['A', 'B']
    phase = 'TEST'
    m = Model(dbf, comps, phase)
    refstates = [ReferenceState('A', phase), ReferenceState('B', phase)]
    m.shift_reference_state(refstates, dbf)

    statevars =  {
        v.T: 298.15, v.P: 101325,
        v.SiteFraction(phase, 0, 'A'): 0.5, v.SiteFraction(phase, 0, 'B'): 0.5,
        }

    # ideal mixing should be present for GMR
    idmix_val = 2*0.5*np.log(0.5)*v.R*298.15
    check_output(m, statevars, 'GMR', idmix_val)

    # shifting the reference state, adding an excess contribution
    # should see that addition in the output
    m.shift_reference_state(refstates, dbf, contrib_mods={'xsmix': S(1000.0)})
    # each pure element contribution is has xsmix changed from 0 to 1
    # At x=0.5, the reference xsmix energy is added to by 0.5*1000.0, which is
    # then subtracted out of the GM energy
    check_output(m, statevars, 'GMR', idmix_val-1000.0)


def test_ionic_liquid_energy_anion_sublattice():
    """Test that having anions, vacancies, and neutral species in the anion sublattice of a two sublattice ionic liquid produces the correct Gibbs energy"""

    # Uses the sublattice model (FE+2)(S-2, VA, S)
    mod = Model(FE_MN_S_DBF, ['FE', 'S', 'VA'], 'IONIC_LIQ')

    # Same potentials for all test cases here
    potentials = {v.P: 101325, v.T: 1600}

    # All values checked by Thermo-Calc using set-start-constitution and show gm(ionic_liq)

    # Test the three endmembers produce the corect energy
    em_FE_Sneg2 = {
        v.Y('IONIC_LIQ', 0, v.Species('FE+2', {'FE': 1.0}, charge=2)): 1.0,
        v.Y('IONIC_LIQ', 1, v.Species('S-2', {'S': 1.0}, charge=-2)): 1.0,
        v.Y('IONIC_LIQ', 1, v.Species('VA')): 1e-12,
        v.Y('IONIC_LIQ', 1, v.Species('S', {'S': 1.0})): 1e-12,
    }
    out = np.array(mod.ast.subs({**potentials, **em_FE_Sneg2}), dtype=np.complex_)
    assert np.isclose(out, -148395.0, atol=0.1)

    em_FE_VA = {
        v.Y('IONIC_LIQ', 0, v.Species('FE+2', {'FE': 1.0}, charge=2)): 1.0,
        v.Y('IONIC_LIQ', 1, v.Species('S-2', {'S': 1.0}, charge=-2)): 1e-12,
        v.Y('IONIC_LIQ', 1, v.Species('VA')): 1.0,
        v.Y('IONIC_LIQ', 1, v.Species('S', {'S': 1.0})): 1e-12,
    }
    out = np.array(mod.ast.subs({**potentials, **em_FE_VA}), dtype=np.complex_)
    assert np.isclose(out, -87735.077, atol=0.1)

    em_FE_S = {
        v.Y('IONIC_LIQ', 0, v.Species('FE+2', {'FE': 1.0}, charge=2)): 1.0,
        v.Y('IONIC_LIQ', 1, v.Species('S-2', {'S': 1.0}, charge=-2)): 1e-12,
        v.Y('IONIC_LIQ', 1, v.Species('VA')): 1e-12,
        v.Y('IONIC_LIQ', 1, v.Species('S', {'S': 1.0})): 1.0,
    }
    out = np.array(mod.ast.subs({**potentials, **em_FE_S}), dtype=np.complex_)
    assert np.isclose(out, -102463.52, atol=0.1)

    # Test some ficticious "nice" mixing cases
    mix_equal = {
        v.Y('IONIC_LIQ', 0, v.Species('FE+2', {'FE': 1.0}, charge=2)): 1.0,
        v.Y('IONIC_LIQ', 1, v.Species('S-2', {'S': 1.0}, charge=-2)): 0.33333333,
        v.Y('IONIC_LIQ', 1, v.Species('VA')): 0.33333333,
        v.Y('IONIC_LIQ', 1, v.Species('S', {'S': 1.0})): 0.33333333,
    }
    out = np.array(mod.ast.subs({**potentials, **mix_equal}), dtype=np.complex_)
    assert np.isclose(out, -130358.2, atol=0.1)

    mix_unequal = {
        v.Y('IONIC_LIQ', 0, v.Species('FE+2', {'FE': 1.0}, charge=2)): 1.0,
        v.Y('IONIC_LIQ', 1, v.Species('S-2', {'S': 1.0}, charge=-2)): 0.5,
        v.Y('IONIC_LIQ', 1, v.Species('VA')): 0.25,
        v.Y('IONIC_LIQ', 1, v.Species('S', {'S': 1.0})): 0.25,
    }
    out = np.array(mod.ast.subs({**potentials, **mix_unequal}), dtype=np.complex_)
    assert np.isclose(out, -138484.11, atol=0.1)

    # Test the energies for the two equilibrium internal DOF for the conditions
    eq_sf_1 = {
        v.Y('IONIC_LIQ', 0, v.Species('FE+2', {'FE': 1.0}, charge=2)): 1.0,
        v.Y('IONIC_LIQ', 1, v.Species('S', {'S': 1.0})): 3.98906E-01,
        v.Y('IONIC_LIQ', 1, v.Species('VA')): 1.00545E-04,
        v.Y('IONIC_LIQ', 1, v.Species('S-2', {'S': 1.0}, charge=-2)): 6.00994E-01,
    }
    out = np.array(mod.ast.subs({**potentials, **eq_sf_1}), dtype=np.complex_)
    assert np.isclose(out, -141545.37, atol=0.1)

    eq_sf_2 = {
        v.Y('IONIC_LIQ', 0, v.Species('FE+2', charge=2)): 1.0,
        v.Y('IONIC_LIQ', 1, v.Species('S-2', charge=-2)): 1.53788E-02,
        v.Y('IONIC_LIQ', 1, v.Species('VA')): 1.45273E-04,
        v.Y('IONIC_LIQ', 1, v.Species('S')): 9.84476E-01,
    }
    out = np.array(mod.ast.subs({**potentials, **eq_sf_2}), dtype=np.complex_)
    assert np.isclose(out, -104229.18, atol=0.1)


def test_order_disorder_interstitial_sublattice():
    """Test that non-vacancy elements are supported on interstitial sublattices"""

    TDB_OrderDisorder_VA_VA = """
    ELEMENT VA   VACUUM   0.0000E+00  0.0000E+00  0.0000E+00 !
    ELEMENT A    DISORD     0.0000E+00  0.0000E+00  0.0000E+00 !
    ELEMENT B    DISORD     0.0000E+00  0.0000E+00  0.0000E+00 !
    ELEMENT C    DISORD     0.0000E+00  0.0000E+00  0.0000E+00 !

    DEFINE_SYSTEM_DEFAULT ELEMENT 2 !
    DEFAULT_COMMAND DEF_SYS_ELEMENT VA !

    TYPE_DEFINITION % SEQ *!
    TYPE_DEFINITION & GES A_P_D DISORD MAGNETIC  -1.0    4.00000E-01 !
    TYPE_DEFINITION ( GES A_P_D ORDERED MAGNETIC  -1.0    4.00000E-01 !
    TYPE_DEFINITION ' GES A_P_D ORDERED DIS_PART DISORD ,,,!

    PHASE DISORD  %&  2 1   3 !
    PHASE ORDERED %('  3 0.5  0.5  3  !

    CONSTITUENT DISORD  : A,B,VA : VA :  !
    CONSTITUENT ORDERED  : A,B,VA : A,B,VA : VA :  !

    PARAMETER G(DISORD,A:VA;0)  298.15  -10000; 6000 N !
    PARAMETER G(DISORD,B:VA;0)  298.15  -10000; 6000 N !

    """

    TDB_OrderDisorder_VA_VA_C = """
    $ Compared to TDB_OrderDisorder_VA_VA, this database only
    $ differs in the fact that the VA sublattice contains C

    ELEMENT VA   VACUUM   0.0000E+00  0.0000E+00  0.0000E+00 !
    ELEMENT A    DISORD     0.0000E+00  0.0000E+00  0.0000E+00 !
    ELEMENT B    DISORD     0.0000E+00  0.0000E+00  0.0000E+00 !
    ELEMENT C    DISORD     0.0000E+00  0.0000E+00  0.0000E+00 !

    DEFINE_SYSTEM_DEFAULT ELEMENT 2 !
    DEFAULT_COMMAND DEF_SYS_ELEMENT VA !

    TYPE_DEFINITION % SEQ *!
    TYPE_DEFINITION & GES A_P_D DISORD MAGNETIC  -1.0    4.00000E-01 !
    TYPE_DEFINITION ( GES A_P_D ORDERED MAGNETIC  -1.0    4.00000E-01 !
    TYPE_DEFINITION ' GES A_P_D ORDERED DIS_PART DISORD ,,,!

    PHASE DISORD  %&  2 1   3 !
    PHASE ORDERED %('  3 0.5  0.5  3  !

    CONSTITUENT DISORD  : A,B,VA : C,VA :  !
    CONSTITUENT ORDERED  : A,B,VA : A,B,VA : C,VA :  !

    PARAMETER G(DISORD,A:VA;0)  298.15  -10000; 6000 N !
    PARAMETER G(DISORD,B:VA;0)  298.15  -10000; 6000 N !

    """

    TDB_OrderDisorder_VA_C = """
    $ Compared to TDB_OrderDisorder_VA_VA_C, this database only
    $ differs in the fact that the disorderd sublattices do not contain VA

    ELEMENT VA   VACUUM   0.0000E+00  0.0000E+00  0.0000E+00 !
    ELEMENT A    DISORD     0.0000E+00  0.0000E+00  0.0000E+00 !
    ELEMENT B    DISORD     0.0000E+00  0.0000E+00  0.0000E+00 !
    ELEMENT C    DISORD     0.0000E+00  0.0000E+00  0.0000E+00 !

    DEFINE_SYSTEM_DEFAULT ELEMENT 2 !
    DEFAULT_COMMAND DEF_SYS_ELEMENT VA !

    TYPE_DEFINITION % SEQ *!
    TYPE_DEFINITION & GES A_P_D DISORD MAGNETIC  -1.0    4.00000E-01 !
    TYPE_DEFINITION ( GES A_P_D ORDERED MAGNETIC  -1.0    4.00000E-01 !
    TYPE_DEFINITION ' GES A_P_D ORDERED DIS_PART DISORD ,,,!

    PHASE DISORD  %&  2 1   3 !
    PHASE ORDERED %('  3 0.5  0.5  3  !

    CONSTITUENT DISORD  : A,B : C,VA :  !
    CONSTITUENT ORDERED  : A,B : A,B : C,VA :  !

    PARAMETER G(DISORD,A:VA;0)  298.15  -10000; 6000 N !
    PARAMETER G(DISORD,B:VA;0)  298.15  -10000; 6000 N !

    """

    TDB_OrderDisorder_VA_B = """
    $ This database contains B in both substitutional and interstitial sublattices

    ELEMENT VA   VACUUM   0.0000E+00  0.0000E+00  0.0000E+00 !
    ELEMENT A    DISORD     0.0000E+00  0.0000E+00  0.0000E+00 !
    ELEMENT B    DISORD     0.0000E+00  0.0000E+00  0.0000E+00 !
    ELEMENT C    DISORD     0.0000E+00  0.0000E+00  0.0000E+00 !

    DEFINE_SYSTEM_DEFAULT ELEMENT 2 !
    DEFAULT_COMMAND DEF_SYS_ELEMENT VA !

    TYPE_DEFINITION % SEQ *!
    TYPE_DEFINITION & GES A_P_D DISORD MAGNETIC  -1.0    4.00000E-01 !
    TYPE_DEFINITION ( GES A_P_D ORDERED MAGNETIC  -1.0    4.00000E-01 !
    TYPE_DEFINITION ' GES A_P_D ORDERED DIS_PART DISORD ,,,!

    PHASE DISORD  %&  2 1   3 !
    PHASE ORDERED %('  3 0.5  0.5  3  !

    CONSTITUENT DISORD  : A,B : B,VA :  !
    CONSTITUENT ORDERED  : A,B : A,B : B,VA :  !

    PARAMETER G(DISORD,A:VA;0)  298.15  -10000; 6000 N !
    PARAMETER G(DISORD,B:VA;0)  298.15  -10000; 6000 N !
    PARAMETER G(DISORD,A:B;0)  298.15  -20000; 6000 N !
    PARAMETER G(DISORD,B:B;0)  298.15  -20000; 6000 N !

    PARAMETER G(ORDERED,A:B:B;0)  298.15  -1000; 6000 N !
    PARAMETER G(ORDERED,A:B:VA;0)  298.15  -2000; 6000 N !

    """

    db_VA_VA = Database(TDB_OrderDisorder_VA_VA)
    db_VA_C = Database(TDB_OrderDisorder_VA_C)
    db_VA_VA_C = Database(TDB_OrderDisorder_VA_VA_C)

    mod_VA_VA = Model(db_VA_VA, ["A", "B", "VA"], "ORDERED")
    mod_VA_C = Model(db_VA_C, ["A", "B", "VA"], "ORDERED")
    mod_VA_VA_C = Model(db_VA_VA_C, ["A", "B", "VA"], "ORDERED")

    # Site fractions for pure A
    subs_dict = {
        v.Y('ORDERED', 0, v.Species('A')): 1.0,
        v.Y('ORDERED', 0, v.Species('B')): 0.0,
        v.Y('ORDERED', 0, v.Species('VA')): 0.0,
        v.Y('ORDERED', 1, v.Species('A')): 1.0,
        v.Y('ORDERED', 1, v.Species('B')): 0.0,
        v.Y('ORDERED', 1, v.Species('VA')): 0.0,
        v.Y('ORDERED', 2, v.Species('VA')): 1.0,
        v.T: 300.0,
    }

    check_energy(mod_VA_VA, subs_dict, -10000, mode='sympy')
    check_energy(mod_VA_C, subs_dict, -10000, mode='sympy')
    check_energy(mod_VA_VA_C, subs_dict, -10000, mode='sympy')

    db_VA_B = Database(TDB_OrderDisorder_VA_B)
    mod_VA_B = Model(db_VA_B, ["A", "B", "VA"], "ORDERED")

    # A-B disordered substitutional
    disord_subs_dict = {
        v.Y('ORDERED', 0, v.Species('A')): 0.5,
        v.Y('ORDERED', 0, v.Species('B')): 0.5,
        v.Y('ORDERED', 1, v.Species('A')): 0.5,
        v.Y('ORDERED', 1, v.Species('B')): 0.5,
        v.Y('ORDERED', 2, v.Species('B')): 0.25,
        v.Y('ORDERED', 2, v.Species('VA')): 0.75,
        v.T: 300.0,
    }
    # Thermo-Calc energy via set-start-constitution
    check_energy(mod_VA_B, disord_subs_dict, -10535.395, mode='sympy')

    # A-B ordered substitutional
    ord_subs_dict = {
        v.Y('ORDERED', 0, v.Species('A')): 1.0,
        v.Y('ORDERED', 0, v.Species('B')): 0.0,
        v.Y('ORDERED', 1, v.Species('A')): 0.0,
        v.Y('ORDERED', 1, v.Species('B')): 1.0,
        v.Y('ORDERED', 2, v.Species('B')): 0.25,
        v.Y('ORDERED', 2, v.Species('VA')): 0.75,
        v.T: 300.0,
    }
    # Thermo-Calc energy via set-start-constitution
    check_energy(mod_VA_B, ord_subs_dict, -10297.421, mode='sympy')


@pytest.mark.skip("Skip until partitioned physical properties are supported "
                  "in the disordered energy contribution.")
def test_order_disorder_magnetic_ordering():
    """Test partitioned order-disorder models with magnetic ordering contributions"""
    mod = Model(AL_C_FE_B2_DBF, ['AL', 'C', 'FE', 'VA'], 'B2_BCC')
    subs_dict = {
        v.Y('B2_BCC', 0, v.Species('AL')): 0.23632422,
        v.Y('B2_BCC', 0, v.Species('FE')): 0.09387751,
        v.Y('B2_BCC', 0, v.Species('VA')): 0.66979827,
        v.Y('B2_BCC', 1, v.Species('AL')): 0.40269437,
        v.Y('B2_BCC', 1, v.Species('FE')): 0.55906662,
        v.Y('B2_BCC', 1, v.Species('VA')): 0.03823901,
        v.Y('B2_BCC', 2, v.Species('C')): 0.12888967,
        v.Y('B2_BCC', 2, v.Species('VA')): 0.87111033,
        v.T: 300.0,
    }
    check_output(mod, subs_dict, 'TC', 318.65374, mode='sympy')
    check_output(mod, subs_dict, 'BMAG', 0.81435207, mode='sympy')
    check_energy(mod, subs_dict, 34659.484, mode='sympy')
