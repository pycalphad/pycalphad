"""
The energy test module verifies that the Model class produces the
correct abstract syntax tree for the energy.
"""

import nose.tools
from pycalphad import Database, Model, calculate, ReferenceState
from pycalphad.core.utils import make_callable
from pycalphad.tests.datasets import ALCRNI_TDB, FEMN_TDB, ALFE_TDB, \
    CRFE_BCC_MAGNETIC_TDB, VA_INTERACTION_TDB, CUMG_TDB
from pycalphad.core.errors import DofError
import pycalphad.variables as v
import numpy as np
import warnings

DBF = Database(ALCRNI_TDB)
ALFE_DBF = Database(ALFE_TDB)
FEMN_DBF = Database(FEMN_TDB)
CRFE_DBF = Database(CRFE_BCC_MAGNETIC_TDB)
CUMG_DBF = Database(CUMG_TDB)
VA_INTERACTION_DBF = Database(VA_INTERACTION_TDB)

@nose.tools.raises(ValueError)
def test_sympify_safety():
    "Parsing malformed strings throws exceptions instead of executing code."
    from pycalphad.io.tdb import _sympify_string
    teststr = "().__class__.__base__.__subclasses__()[216]('ls')"
    _sympify_string(teststr) # should throw ParseException


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
    known_value = np.array(known_value, dtype=np.complex)
    desired = np.array(desired, dtype=np.complex)
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


@nose.tools.raises(DofError)
def test_underspecified_refstate_raises():
    """A Model cannot be shifted to a new reference state unless references for all pure elements are specified."""
    m = Model(FEMN_DBF, ['FE', 'MN', 'VA'], 'LIQUID')
    refstates = [ReferenceState(v.Species('FE'), 'LIQUID')]
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


# TODO: This currently fails because 'mag' is not properly accounted for in GM_MIX
# After adding the proper mixing fix to pycalphad,
# This should fail because the mixing models are built by `build_mixing_attrs`,
# which is called during `build_phase`. Changing a contribution changes the
# AST (changes GM), however the `_MIX` part is statically built.
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

    m.reference.models['mag'] = 1000
    check_output(m, statevars, 'GM_MIX', 0)



