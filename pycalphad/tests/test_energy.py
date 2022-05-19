"""
The energy test module verifies that the Model class produces the
correct abstract syntax tree for the energy.
"""

import pytest
from symengine import S
from pycalphad import Database, Model, ReferenceState
from pycalphad.tests.fixtures import select_database, load_database
from pycalphad.core.errors import DofError
import pycalphad.variables as v
import numpy as np
from pycalphad.models.model_mqmqa import ModelMQMQA


def make_callable(model, variables, mode=None):
    energy = lambda *vs: model.subs(dict(zip(variables, vs))).n(real=True)
    return energy


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
    return prop(*[float(x) for x in variables.values()])


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
    check_output(model, variables, 'GM', known_value, mode=mode)

# PURE COMPONENT TESTS
@select_database("alcrni.tdb")
def test_pure_sympy(load_database):
    "Pure component end-members in sympy mode."
    dbf = load_database()
    check_energy(Model(dbf, ['AL'], 'LIQUID'), \
            {v.T: 2000, v.SiteFraction('LIQUID', 0, 'AL'): 1}, \
        -1.28565e5, mode='sympy')
    check_energy(Model(dbf, ['AL'], 'B2'), \
            {v.T: 1400, v.SiteFraction('B2', 0, 'AL'): 1,
             v.SiteFraction('B2', 1, 'AL'): 1}, \
        -6.57639e4, mode='sympy')
    check_energy(Model(dbf, ['AL'], 'L12_FCC'), \
            {v.T: 800, v.SiteFraction('L12_FCC', 0, 'AL'): 1,
             v.SiteFraction('L12_FCC', 1, 'AL'): 1}, \
        -3.01732e4, mode='sympy')


@select_database("alcrni.tdb")
def test_degenerate_ordered(load_database):
    "Degenerate sublattice configuration has same energy as disordered phase."
    dbf = load_database()
    mod_l12 = Model(dbf, ['CR', 'NI'], 'L12_FCC')
    mod_a1 = Model(dbf, ['CR', 'NI'], 'FCC_A1')
    l12_subs = {v.T: 500, v.SiteFraction('L12_FCC', 0, 'CR'): 0.33,
                v.SiteFraction('L12_FCC', 0, 'NI'): 0.67,
                v.SiteFraction('L12_FCC', 1, 'CR'): 0.33,
                v.SiteFraction('L12_FCC', 1, 'NI'): 0.67}
    a1_subs = {v.T: 500, v.SiteFraction('FCC_A1', 0, 'CR'): 0.33,
               v.SiteFraction('FCC_A1', 0, 'NI'): 0.67}
    l12_energy = mod_l12.energy.xreplace(l12_subs).n(real=True)
    a1_energy = mod_a1.energy.xreplace(a1_subs).n(real=True)
    np.testing.assert_almost_equal(l12_energy, a1_energy)


@select_database("alcrni.tdb")
def test_degenerate_zero_ordering(load_database):
    "Degenerate sublattice configuration has zero ordering energy."
    dbf = load_database()
    mod = Model(dbf, ['CR', 'NI'], 'L12_FCC')
    sub_dict = {v.T: 500, v.SiteFraction('L12_FCC', 0, 'CR'): 0.33,
                v.SiteFraction('L12_FCC', 0, 'NI'): 0.67,
                v.SiteFraction('L12_FCC', 1, 'CR'): 0.33,
                v.SiteFraction('L12_FCC', 1, 'NI'): 0.67}
    #print({x: mod.models[x].subs(sub_dict) for x in mod.models})
    desired = mod.models['ord'].xreplace(sub_dict).n(real=True)
    assert abs(desired - 0) < 1e-5, "%r != %r" % (desired, 0)


# BINARY TESTS
@select_database("alcrni.tdb")
def test_binary_magnetic(load_database):
    "Two-component phase with IHJ magnetic model."
    dbf = load_database()
    # disordered case
    check_energy(Model(dbf, ['CR', 'NI'], 'L12_FCC'), \
            {v.T: 500, v.SiteFraction('L12_FCC', 0, 'CR'): 0.33,
             v.SiteFraction('L12_FCC', 0, 'NI'): 0.67,
             v.SiteFraction('L12_FCC', 1, 'CR'): 0.33,
             v.SiteFraction('L12_FCC', 1, 'NI'): 0.67}, \
        -1.68840e4, mode='sympy')


@select_database("alcrni.tdb")
def test_binary_magnetic_reimported(load_database):
    "Export and re-import a TDB before the calculation."
    dbf = load_database()
    dbf_imported = Database.from_string(dbf.to_string(fmt='tdb'), fmt='tdb')
    check_energy(Model(dbf_imported, ['CR', 'NI'], 'L12_FCC'),
                {v.T: 500, v.SiteFraction('L12_FCC', 0, 'CR'): 0.33,
                v.SiteFraction('L12_FCC', 0, 'NI'): 0.67,
                v.SiteFraction('L12_FCC', 1, 'CR'): 0.33,
                v.SiteFraction('L12_FCC', 1, 'NI'): 0.67},
                -1.68840e4, mode='sympy')


@select_database("alcrni.tdb")
def test_binary_magnetic_ordering(load_database):
    "Two-component phase with IHJ magnetic model and ordering."
    dbf = load_database()
    # ordered case
    check_energy(Model(dbf, ['CR', 'NI'], 'L12_FCC'), \
            {v.T: 300, v.SiteFraction('L12_FCC', 0, 'CR'): 4.86783e-2,
             v.SiteFraction('L12_FCC', 0, 'NI'): 9.51322e-1,
             v.SiteFraction('L12_FCC', 1, 'CR'): 9.33965e-1,
             v.SiteFraction('L12_FCC', 1, 'NI'): 6.60348e-2}, \
        -9.23953e3, mode='sympy')


@select_database("alcrni.tdb")
def test_binary_dilute(load_database):
    "Dilute binary solution phase."
    dbf = load_database()
    check_energy(Model(dbf, ['CR', 'NI'], 'LIQUID'), \
            {v.T: 300, v.SiteFraction('LIQUID', 0, 'CR'): 1e-12,
             v.SiteFraction('LIQUID', 0, 'NI'): 1.0-1e-12}, \
        5.52773e3, mode='sympy')


@select_database("femn.tdb")
def test_binary_xiong_twostate_einstein(load_database):
    "Phase with Xiong magnetic, two-state and Einstein energy contributions."
    femn_dbf = load_database()
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
@select_database("alcrni.tdb")
def test_ternary_rkm_solution(load_database):
    "Solution phase with ternary interaction parameters."
    dbf = load_database()
    check_energy(Model(dbf, ['AL', 'CR', 'NI'], 'LIQUID'), \
            {v.T: 1500, v.SiteFraction('LIQUID', 0, 'AL'): 0.44,
             v.SiteFraction('LIQUID', 0, 'CR'): 0.20,
             v.SiteFraction('LIQUID', 0, 'NI'): 0.36}, \
        -1.16529e5, mode='sympy')


@select_database("alcrni.tdb")
def test_ternary_symmetric_param(load_database):
    "Generate the other two ternary parameters if only the zeroth is specified."
    dbf = load_database()
    check_energy(Model(dbf, ['AL', 'CR', 'NI'], 'FCC_A1'), \
            {v.T: 300, v.SiteFraction('FCC_A1', 0, 'AL'): 1.97135e-1,
             v.SiteFraction('FCC_A1', 0, 'CR'): 1.43243e-2,
             v.SiteFraction('FCC_A1', 0, 'NI'): 7.88541e-1},
                 -37433.794, mode='sympy')


@select_database("alcrni.tdb")
def test_ternary_ordered_magnetic(load_database):
    "Ternary ordered solution phase with IHJ magnetic model."
    dbf = load_database()
    # ordered case
    check_energy(Model(dbf, ['AL', 'CR', 'NI'], 'L12_FCC'), \
            {v.T: 300, v.SiteFraction('L12_FCC', 0, 'AL'): 5.42883e-8,
             v.SiteFraction('L12_FCC', 0, 'CR'): 2.07934e-6,
             v.SiteFraction('L12_FCC', 0, 'NI'): 9.99998e-1,
             v.SiteFraction('L12_FCC', 1, 'AL'): 7.49998e-1,
             v.SiteFraction('L12_FCC', 1, 'CR'): 2.50002e-1,
             v.SiteFraction('L12_FCC', 1, 'NI'): 4.55313e-10}, \
        -40717.204, mode='sympy')


# QUATERNARY TESTS
@select_database("alcrni.tdb")
def test_quaternary(load_database):
    "Quaternary ordered solution phase."
    dbf = load_database()
    check_energy(Model(dbf, ['AL', 'CR', 'NI', 'VA'], 'B2'), \
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
@select_database("alcrni.tdb")
def test_case_sensitivity(load_database):
    "Case sensitivity of component and phase names."
    dbf = load_database()
    check_energy(Model(dbf, ['Cr', 'nI'], 'Liquid'), \
            {v.T: 300, v.SiteFraction('LIQUID', 0, 'CR'): 1e-12,
             v.SiteFraction('liquid', 0, 'ni'): 1}, \
        5.52773e3, mode='sympy')


@select_database("alcrni.tdb")
def test_zero_site_fraction(load_database):
    "Energy of a binary solution phase where one site fraction is zero."
    dbf = load_database()
    check_energy(Model(dbf, ['CR', 'NI'], 'LIQUID'), \
            {v.T: 300, v.SiteFraction('LIQUID', 0, 'CR'): 0,
             v.SiteFraction('LIQUID', 0, 'NI'): 1}, \
        5.52773e3, mode='sympy')


@select_database("femn.tdb")
def test_reference_energy_of_unary_twostate_einstein_magnetic_is_zero(load_database):
    """The referenced energy for the pure elements in a unary Model with twostate and Einstein contributions referenced to that phase is zero."""
    dbf = load_database()
    m = Model(dbf, ['FE', 'VA'], 'LIQUID')
    statevars = {v.T: 298.15, v.SiteFraction('LIQUID', 0, 'FE'): 1, v.SiteFraction('LIQUID', 1, 'VA'): 1}
    refstates = [ReferenceState(v.Species('FE'), 'LIQUID')]
    m.shift_reference_state(refstates, dbf)
    check_output(m, statevars, 'GMR', 0.0)


@select_database("femn.tdb")
def test_underspecified_refstate_raises(load_database):
    """A Model cannot be shifted to a new reference state unless references for all pure elements are specified."""
    dbf = load_database()
    m = Model(dbf, ['FE', 'MN', 'VA'], 'LIQUID')
    refstates = [ReferenceState(v.Species('FE'), 'LIQUID')]
    with pytest.raises(DofError):
        m.shift_reference_state(refstates, dbf)


@select_database("femn.tdb")
def test_reference_energy_of_binary_twostate_einstein_is_zero(load_database):
    """The referenced energy for the pure elements in a binary Model with twostate and Einstein contributions referenced to that phase is zero."""
    dbf = load_database()
    m = Model(dbf, ['FE', 'MN', 'VA'], 'LIQUID')
    refstates = [ReferenceState(v.Species('FE'), 'LIQUID'), ReferenceState(v.Species('MN'), 'LIQUID')]
    m.shift_reference_state(refstates, dbf)

    statevars_FE = {v.T: 298.15,
             v.SiteFraction('LIQUID', 0, 'FE'): 1, v.SiteFraction('LIQUID', 0, 'MN'): 0,
             v.SiteFraction('LIQUID', 1, 'VA'): 1}
    check_output(m, statevars_FE, 'GMR', 0.0)

    statevars_CR = {v.T: 298.15,
             v.SiteFraction('LIQUID', 0, 'FE'): 0, v.SiteFraction('LIQUID', 0, 'MN'): 1,
             v.SiteFraction('LIQUID', 1, 'VA'): 1}
    check_output(m, statevars_CR, 'GMR', 0.0)


@select_database("crfe_bcc_magnetic.tdb")
def test_magnetic_reference_energy_is_zero(load_database):
    """The referenced energy binary magnetic Model is zero."""
    dbf = load_database()
    m = Model(dbf, ['CR', 'FE', 'VA'], 'BCC_A2')
    refstates = [ReferenceState('CR', 'BCC_A2'), ReferenceState('FE', 'BCC_A2')]
    m.shift_reference_state(refstates, dbf)

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
    
    VA_INTERACTION_TDB = """
    ELEMENT AL   FCC_A1                    26.981539   4577.296    28.3215!
    ELEMENT VA   BLANK                     0.0 0.0 0.0 !

    PHASE FCC_A1  %  2 1 1 !
    CONSTITUENT FCC_A1  :AL,VA:VA:  !
    PARAMETER G(FCC_A1,AL:VA;0)      0.01   100;       6000 N !
    PARAMETER G(FCC_A1,VA:VA;0)      0.01   500;       6000 N !
    PARAMETER G(FCC_A1,AL,VA:VA;0)      0.01   4000;       6000 N !

    """
    VA_INTERACTION_DBF = Database(VA_INTERACTION_TDB)
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


@select_database("alfe.tdb")
def test_reference_energy_for_different_phase(load_database):
    """The referenced energy a different phase should be correct."""
    dbf = load_database()
    m = Model(dbf, ['AL', 'FE', 'VA'], 'AL2FE')
    # formation reference states
    refstates = [ReferenceState('AL', 'FCC_A1'), ReferenceState('FE', 'BCC_A2')]
    m.shift_reference_state(refstates, dbf)

    statevars = {v.T: 300, v.SiteFraction('AL2FE', 0, 'AL'): 1, v.SiteFraction('AL2FE', 1, 'FE'): 1}
    check_output(m, statevars, 'GMR', -28732.525)  # Checked in Thermo-Calc


@select_database("cumg.tdb")
def test_endmember_mixing_energy_is_zero(load_database):
    """The mixing energy for an endmember in a multi-sublattice model should be zero."""
    dbf = load_database()
    m = Model(dbf, ['CU', 'MG', 'VA'], 'CU2MG')
    statevars = {
                    v.T: 300,
                    v.SiteFraction('CU2MG', 0, 'CU'): 1, v.SiteFraction('CU2MG', 0, 'MG'): 0,
                    v.SiteFraction('CU2MG', 1, 'CU'): 0, v.SiteFraction('CU2MG', 1, 'MG'): 1,
                }
    check_output(m, statevars, 'GM_MIX', 0.0)


@select_database("crfe_bcc_magnetic.tdb")
def test_magnetic_endmember_mixing_energy_is_zero(load_database):
    """The mixing energy for an endmember with a magnetic contribution should be zero."""
    dbf = load_database()
    m = Model(dbf, ['CR', 'FE', 'VA'], 'BCC_A2')
    statevars = {
                    v.T: 300,
                    v.SiteFraction('BCC_A2', 0, 'CR'): 0, v.SiteFraction('BCC_A2', 0, 'FE'): 1,
                    v.SiteFraction('BCC_A2', 1, 'VA'): 1}
    check_output(m, statevars, 'GM_MIX', 0.0)


@select_database("alfe.tdb")
def test_order_disorder_mixing_energy_is_nan(load_database):
    """The endmember-referenced mixing energy is undefined and the energy should be NaN."""
    dbf = load_database()
    m = Model(dbf, ['AL', 'FE', 'VA'], 'B2_BCC')
    statevars = {
                    v.T: 300,
                    v.SiteFraction('B2_BCC', 0, 'AL'): 1, v.SiteFraction('B2_BCC', 0, 'FE'): 0,
                    v.SiteFraction('B2_BCC', 1, 'AL'): 0, v.SiteFraction('B2_BCC', 1, 'FE'): 1,
                    v.SiteFraction('B2_BCC', 2, 'VA'): 1}
    check_output(m, statevars, 'GM_MIX', np.nan)


@select_database("cumg.tdb")
def test_changing_model_ast_also_changes_mixing_energy(load_database):
    """If a models contribution is modified, the mixing energy should update accordingly."""
    dbf = load_database()
    m = Model(dbf, ['CU', 'MG', 'VA'], 'CU2MG')
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


@select_database("femns.tdb")
def test_ionic_liquid_energy_anion_sublattice(load_database):
    """Test that having anions, vacancies, and neutral species in the anion sublattice of a two sublattice ionic liquid produces the correct Gibbs energy"""
    dbf = load_database()
    # Uses the sublattice model (FE+2)(S-2, VA, S)
    mod = Model(dbf, ['FE', 'S', 'VA'], 'IONIC_LIQ')

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
    out = np.array(mod.ast.subs({**potentials, **em_FE_Sneg2}).n(real=True), dtype=np.complex_)
    assert np.isclose(out, -148395.0, atol=0.1)

    em_FE_VA = {
        v.Y('IONIC_LIQ', 0, v.Species('FE+2', {'FE': 1.0}, charge=2)): 1.0,
        v.Y('IONIC_LIQ', 1, v.Species('S-2', {'S': 1.0}, charge=-2)): 1e-12,
        v.Y('IONIC_LIQ', 1, v.Species('VA')): 1.0,
        v.Y('IONIC_LIQ', 1, v.Species('S', {'S': 1.0})): 1e-12,
    }
    out = np.array(mod.ast.subs({**potentials, **em_FE_VA}).n(real=True), dtype=np.complex_)
    assert np.isclose(out, -87735.077, atol=0.1)

    em_FE_S = {
        v.Y('IONIC_LIQ', 0, v.Species('FE+2', {'FE': 1.0}, charge=2)): 1.0,
        v.Y('IONIC_LIQ', 1, v.Species('S-2', {'S': 1.0}, charge=-2)): 1e-12,
        v.Y('IONIC_LIQ', 1, v.Species('VA')): 1e-12,
        v.Y('IONIC_LIQ', 1, v.Species('S', {'S': 1.0})): 1.0,
    }
    out = np.array(mod.ast.subs({**potentials, **em_FE_S}).n(real=True), dtype=np.complex_)
    assert np.isclose(out, -102463.52, atol=0.1)

    # Test some ficticious "nice" mixing cases
    mix_equal = {
        v.Y('IONIC_LIQ', 0, v.Species('FE+2', {'FE': 1.0}, charge=2)): 1.0,
        v.Y('IONIC_LIQ', 1, v.Species('S-2', {'S': 1.0}, charge=-2)): 0.33333333,
        v.Y('IONIC_LIQ', 1, v.Species('VA')): 0.33333333,
        v.Y('IONIC_LIQ', 1, v.Species('S', {'S': 1.0})): 0.33333333,
    }
    out = np.array(mod.ast.subs({**potentials, **mix_equal}).n(real=True), dtype=np.complex_)
    assert np.isclose(out, -130358.2, atol=0.1)

    mix_unequal = {
        v.Y('IONIC_LIQ', 0, v.Species('FE+2', {'FE': 1.0}, charge=2)): 1.0,
        v.Y('IONIC_LIQ', 1, v.Species('S-2', {'S': 1.0}, charge=-2)): 0.5,
        v.Y('IONIC_LIQ', 1, v.Species('VA')): 0.25,
        v.Y('IONIC_LIQ', 1, v.Species('S', {'S': 1.0})): 0.25,
    }
    out = np.array(mod.ast.subs({**potentials, **mix_unequal}).n(real=True), dtype=np.complex_)
    assert np.isclose(out, -138484.11, atol=0.1)

    # Test the energies for the two equilibrium internal DOF for the conditions
    eq_sf_1 = {
        v.Y('IONIC_LIQ', 0, v.Species('FE+2', {'FE': 1.0}, charge=2)): 1.0,
        v.Y('IONIC_LIQ', 1, v.Species('S', {'S': 1.0})): 3.98906E-01,
        v.Y('IONIC_LIQ', 1, v.Species('VA')): 1.00545E-04,
        v.Y('IONIC_LIQ', 1, v.Species('S-2', {'S': 1.0}, charge=-2)): 6.00994E-01,
    }
    out = np.array(mod.ast.subs({**potentials, **eq_sf_1}).n(real=True), dtype=np.complex_)
    assert np.isclose(out, -141545.37, atol=0.1)

    eq_sf_2 = {
        v.Y('IONIC_LIQ', 0, v.Species('FE+2', charge=2)): 1.0,
        v.Y('IONIC_LIQ', 1, v.Species('S-2', charge=-2)): 1.53788E-02,
        v.Y('IONIC_LIQ', 1, v.Species('VA')): 1.45273E-04,
        v.Y('IONIC_LIQ', 1, v.Species('S')): 9.84476E-01,
    }
    out = np.array(mod.ast.subs({**potentials, **eq_sf_2}).n(real=True), dtype=np.complex_)
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
@select_database("alcfe_b2.tdb")
def test_order_disorder_magnetic_ordering(load_database):
    """Test partitioned order-disorder models with magnetic ordering contributions"""
    dbf = load_database()
    mod = Model(dbf, ['AL', 'C', 'FE', 'VA'], 'B2_BCC')
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


@select_database("Viitala.dat")
def test_MQMQA_site_fraction_energy(load_database):
    dbf = load_database()
    ZN =  v.Species("ZN+2.0", constituents={"ZN": 1.0}, charge=2)
    FE2 = v.Species("FE+2.0", constituents={"FE": 1.0}, charge=2)
    FE3 = v.Species("FE+3.0", constituents={"FE": 1.0}, charge=3)
    CU1 = v.Species("CU+1.0", constituents={"CU": 1.0}, charge=1)
    CU2 = v.Species("CU+2.0", constituents={"CU": 1.0}, charge=2)
    CL =  v.Species("CL-1.0", constituents={"CL": 1.0}, charge=-1)

    mod = ModelMQMQA(dbf, ["CU", "ZN", "FE", "CL"], "LIQUIDSOLN")

    subs_dict ={mod._X_ijkl(CU1,CU1,CL,CL): 3.6411159329213960E-002,
                mod._X_ijkl(FE3,FE3,CL,CL): 0.19187702069719115,
                mod._X_ijkl(FE2,FE2,CL,CL): 6.6706457325108374E-004,
                mod._X_ijkl(CU2,CU2,CL,CL): 7.4480630453876051E-004,
                mod._X_ijkl(ZN,ZN,CL,CL): 6.3597725840616029E-002,
                mod._X_ijkl(CU1,FE3,CL,CL): 0.26054793342102595,
                mod._X_ijkl(CU1,CU2,CL,CL): 1.1687135533100841E-002,
                mod._X_ijkl(CU2,FE3,CL,CL): 2.3762278972894308E-002,
                mod._X_ijkl(CU1,FE2,CL,CL): 1.1060387601365204E-002,
                mod._X_ijkl(FE2,FE3,CL,CL): 2.5177769772622496E-002,
                mod._X_ijkl(CU1,ZN,CL,CL): 9.3472468895210881E-002,
                mod._X_ijkl(CU2,FE2,CL,CL): 1.6621696354116697E-003,
                mod._X_ijkl(FE3,ZN,CL,CL): 0.25634822067819923,
                mod._X_ijkl(CU2,ZN,CL,CL): 1.1808559140386612E-002,
                mod._X_ijkl(FE2,ZN,CL,CL): 1.1175299604972171E-002,
                v.T: 800
                }

    check_energy(mod, subs_dict, -1.47867E+05, mode="sympy")
    assert np.isclose(float(mod.moles("CU").subs(subs_dict)), 0.07692307692,1e-5)
    assert np.isclose(float(mod.moles("CL").subs(subs_dict)), 0.6923076923,1e-5)
    assert np.isclose(float(mod.moles("ZN").subs(subs_dict)), 0.07692307692,1e-5)
    assert np.isclose(float(mod.moles("FE").subs(subs_dict)), 0.15384615384,1e-5)


@select_database("Shishin_Fe-Sb-O-S_slag.dat")
def test_MQMQA_SUBQ_Q_mixing_1000K(load_database):
    dbf = load_database()

    FE2 = v.Species("FE2++2.0", constituents={"FE": 2.0}, charge=2)
    FE3 = v.Species("FE3++3.0", constituents={"FE": 3.0}, charge=3)
    SB3 = v.Species("SB3++3.0", constituents={"SB": 3.0}, charge=3)
    O = v.Species("O-2.0", constituents={"O": 1.0}, charge=-2)
    S = v.Species("S-2.0", constituents={"S": 1.0}, charge=-2)
    mod = ModelMQMQA(dbf, ["FE", "SB", "O", "S"], "SLAG-LIQ")

    assert FE2 in mod.cations
    assert FE3 in mod.cations
    assert SB3 in mod.cations
    assert O in mod.anions
    assert S in mod.anions

    subs_dict = {  # Thermochimica site fractions
        mod._X_ijkl(FE2,FE2,O,O): 1.1862E-32,
        mod._X_ijkl(FE3,FE3,O,O): 5.5017E-03,
        mod._X_ijkl(SB3,SB3,O,O): 0.26528   ,
        mod._X_ijkl(FE2,FE3,O,O): 6.6681E-17,
        mod._X_ijkl(FE2,SB3,O,O): 1.3130E-16,
        mod._X_ijkl(FE3,SB3,O,O): 7.6407E-02,
        mod._X_ijkl(FE2,FE2,S,S): 3.1247E-29,
        mod._X_ijkl(FE3,FE3,S,S): 0.26528   ,
        mod._X_ijkl(SB3,SB3,S,S): 5.5017E-03,
        mod._X_ijkl(FE2,FE3,S,S): 6.7388E-15,
        mod._X_ijkl(FE2,SB3,S,S): 9.7047E-16,
        mod._X_ijkl(FE3,SB3,S,S): 7.6407E-02,
        mod._X_ijkl(FE2,FE2,O,S): 8.8748E-31,
        mod._X_ijkl(FE3,FE3,O,S): 7.6407E-02,
        mod._X_ijkl(SB3,SB3,O,S): 7.6407E-02,
        mod._X_ijkl(FE2,FE3,O,S): 1.1446E-15,
        mod._X_ijkl(FE2,SB3,O,S): 6.0950E-16,
        mod._X_ijkl(FE3,SB3,O,S): 0.15281   ,
        v.T: 1000.0,
    }

    check_energy(mod, subs_dict, -1.31831E+05, mode="sympy")  # Thermochimica energy
    assert np.isclose(float(mod.moles("FE").subs(subs_dict)), 0.2, 1e-5)
    assert np.isclose(float(mod.moles("SB").subs(subs_dict)), 0.2, 1e-5)
    assert np.isclose(float(mod.moles("O").subs(subs_dict)), 0.3, 1e-5)
    assert np.isclose(float(mod.moles("S").subs(subs_dict)), 0.3, 1e-5)

@select_database("Shishin_Fe-Sb-O-S_slag.dat")
def test_MQMQA_SUBQ_Q_mixing_1000K_FACTSAGE(load_database):
    dbf = load_database()

    FE2 = v.Species("FE2++2.0", constituents={"FE": 2.0}, charge=2)
    FE3 = v.Species("FE3++3.0", constituents={"FE": 3.0}, charge=3)
    SB3 = v.Species("SB3++3.0", constituents={"SB": 3.0}, charge=3)
    O = v.Species("O-2.0", constituents={"O": 1.0}, charge=-2)
    S = v.Species("S-2.0", constituents={"S": 1.0}, charge=-2)
    mod = ModelMQMQA(dbf, ["FE", "SB", "O", "S"], "SLAG-LIQ")

    assert FE2 in mod.cations
    assert FE3 in mod.cations
    assert SB3 in mod.cations
    assert O in mod.anions
    assert S in mod.anions

    subs_dict = {  # FactSage site fractions (Fe2 quadruplet fractions not printed, I assumed 1e-30)
        mod._X_ijkl(FE2,FE2,O,O): 1e-30,
        mod._X_ijkl(FE3,FE3,O,O): 5.5018E-03,
        mod._X_ijkl(SB3,SB3,O,O): 0.26528,
        mod._X_ijkl(FE2,FE3,O,O): 1e-30,
        mod._X_ijkl(FE2,SB3,O,O): 1e-30,
        mod._X_ijkl(FE3,SB3,O,O): 7.6407E-02,
        mod._X_ijkl(FE2,FE2,S,S): 1e-30,
        mod._X_ijkl(FE3,FE3,S,S): 0.26528,
        mod._X_ijkl(SB3,SB3,S,S): 5.5018E-03,
        mod._X_ijkl(FE2,FE3,S,S): 1e-30,
        mod._X_ijkl(FE2,SB3,S,S): 1e-30,
        mod._X_ijkl(FE3,SB3,S,S): 7.6407E-02,
        mod._X_ijkl(FE2,FE2,O,S): 1e-30,
        mod._X_ijkl(FE3,FE3,O,S): 7.6407E-02,
        mod._X_ijkl(SB3,SB3,O,S): 7.6407E-02,
        mod._X_ijkl(FE2,FE3,O,S): 1e-30,
        mod._X_ijkl(FE2,SB3,O,S): 1e-30,
        mod._X_ijkl(FE3,SB3,O,S): 0.15281,
        v.T: 1000.0,
    }
    print(mod.GM.subs(subs_dict))
    check_energy(mod, subs_dict, -131831.0, mode="sympy")  # FactSage energy, from Max
    assert np.isclose(float(mod.moles("FE").subs(subs_dict)), 0.2, 1e-5)
    assert np.isclose(float(mod.moles("SB").subs(subs_dict)), 0.2, 1e-5)
    assert np.isclose(float(mod.moles("O").subs(subs_dict)), 0.3, 1e-5)
    assert np.isclose(float(mod.moles("S").subs(subs_dict)), 0.3, 1e-5)


@select_database("Shishin_Fe-Sb-O-S_slag.dat")
def test_MQMQA_SUBQ_Q_mixing_400K(load_database):
    dbf = load_database()

    FE2 = v.Species("FE2++2.0", constituents={"FE": 2.0}, charge=2)
    FE3 = v.Species("FE3++3.0", constituents={"FE": 3.0}, charge=3)
    SB3 = v.Species("SB3++3.0", constituents={"SB": 3.0}, charge=3)
    O = v.Species("O-2.0", constituents={"O": 1.0}, charge=-2)
    S = v.Species("S-2.0", constituents={"S": 1.0}, charge=-2)
    mod = ModelMQMQA(dbf, ["FE", "SB", "O", "S"], "SLAG-LIQ")

    assert FE2 in mod.cations
    assert FE3 in mod.cations
    assert SB3 in mod.cations
    assert O in mod.anions
    assert S in mod.anions

    subs_dict = {  # Thermochimica site fractions
        mod._X_ijkl(FE2,FE2,O,O): 1.1225E-39,
        mod._X_ijkl(FE3,FE3,O,O): 2.9334E-05,
        mod._X_ijkl(SB3,SB3,O,O): 0.47751   ,
        mod._X_ijkl(FE2,FE3,O,O): 1.2682E-20,
        mod._X_ijkl(FE2,SB3,O,O): 6.9281E-20,
        mod._X_ijkl(FE3,SB3,O,O): 7.4853E-03,
        mod._X_ijkl(FE2,FE2,S,S): 4.2377E-28,
        mod._X_ijkl(FE3,FE3,S,S): 0.47751   ,
        mod._X_ijkl(SB3,SB3,S,S): 2.9334E-05,
        mod._X_ijkl(FE2,FE3,S,S): 4.2568E-14,
        mod._X_ijkl(FE2,SB3,S,S): 3.3364E-16,
        mod._X_ijkl(FE3,SB3,S,S): 7.4853E-03,
        mod._X_ijkl(FE2,FE2,O,S): 2.0947E-35,
        mod._X_ijkl(FE3,FE3,O,S): 7.4853E-03,
        mod._X_ijkl(SB3,SB3,O,S): 7.4853E-03,
        mod._X_ijkl(FE2,FE3,O,S): 5.7263E-18,
        mod._X_ijkl(FE2,SB3,O,S): 1.1849E-18,
        mod._X_ijkl(FE3,SB3,O,S): 1.4971E-02,
        v.T: 400.0,
    }

    check_energy(mod, subs_dict, -9.60740E+04, mode="sympy")  # Thermochimica energy
    assert np.isclose(float(mod.moles("FE").subs(subs_dict)), 0.2, 1e-5)
    assert np.isclose(float(mod.moles("SB").subs(subs_dict)), 0.2, 1e-5)
    assert np.isclose(float(mod.moles("O").subs(subs_dict)), 0.3, 1e-5)
    assert np.isclose(float(mod.moles("S").subs(subs_dict)), 0.3, 1e-5)


@select_database("Shishin_Fe-Sb-O-S_slag.dat")
def test_MQMQA_SUBQ_Q_mixing_Sb_O_S_400K(load_database):
    dbf = load_database()

    FE2 = v.Species("FE2++2.0", constituents={"FE": 2.0}, charge=2)
    FE3 = v.Species("FE3++3.0", constituents={"FE": 3.0}, charge=3)
    SB3 = v.Species("SB3++3.0", constituents={"SB": 3.0}, charge=3)
    O = v.Species("O-2.0", constituents={"O": 1.0}, charge=-2)
    S = v.Species("S-2.0", constituents={"S": 1.0}, charge=-2)
    mod = ModelMQMQA(dbf, ["SB", "O", "S"], "SLAG-LIQ")

    assert FE2 not in mod.cations
    assert FE3 not in mod.cations
    assert SB3 in mod.cations
    assert O in mod.anions
    assert S in mod.anions

    subs_dict = {  # Thermochimica site fractions
        mod._X_ijkl(SB3,SB3,O,O): 0.25,
        mod._X_ijkl(SB3,SB3,S,S): 0.25,
        mod._X_ijkl(SB3,SB3,O,S): 0.5,
        v.T: 400.0,
    }

    check_energy(mod, subs_dict, -8.04978E+04, mode="sympy")  # Thermochimica energy
    assert np.isclose(float(mod.moles("SB").subs(subs_dict)), 0.4, 1e-5)
    assert np.isclose(float(mod.moles("O").subs(subs_dict)), 0.3, 1e-5)
    assert np.isclose(float(mod.moles("S").subs(subs_dict)), 0.3, 1e-5)

@select_database("KF-NIF2_switched.dat")
def test_DAT_coordination_numbers_are_order_invariant(load_database):
    """Coordination number parameters should have the coordinations sorted in the correct order.
    
    This test confirms that if database cation ordering is not alphabetical in
    the source database (in particular, for coordination numbers), the energy
    will be correctly computed.
    """
    dbf = load_database()

    F = v.Species('F-1.0',constituents={'F':1.0}, charge=-1)
    K = v.Species('K+1.0',constituents={'K':1.0}, charge=1)
    NI = v.Species('NI+2.0',constituents={'NI':1.0}, charge=2)
    mod = ModelMQMQA(dbf, ["K", "NI", "F"], "LIQUID2")

    assert K in mod.cations
    assert NI  in mod.cations
    assert F in mod.anions

    subs_dict={mod._X_ijkl(NI,NI,F,F):0.36820754040431064,
              mod._X_ijkl(K,NI,F,F):0.52716983838275622,
              mod._X_ijkl(K,K,F,F):0.10462262121293306,
              v.T: 1600}

    check_energy(mod, subs_dict, -3.35720E+05, mode="sympy")  # Thermochimica energy
    assert np.isclose(float(mod.moles("K").subs(subs_dict)), 0.2, 1e-5)
    assert np.isclose(float(mod.moles("NI").subs(subs_dict)), 0.2, 1e-5)
    assert np.isclose(float(mod.moles("F").subs(subs_dict)), 0.6, 1e-5)

@select_database("Shishin_Fe-Sb-O-S_slag.dat")
def test_MQMQA_SUBQ_Q_mixing_Sb_O_S_1000K(load_database):
    dbf = load_database()

    FE2 = v.Species("FE2++2.0", constituents={"FE": 2.0}, charge=2)
    FE3 = v.Species("FE3++3.0", constituents={"FE": 3.0}, charge=3)
    SB3 = v.Species("SB3++3.0", constituents={"SB": 3.0}, charge=3)
    O = v.Species("O-2.0", constituents={"O": 1.0}, charge=-2)
    S = v.Species("S-2.0", constituents={"S": 1.0}, charge=-2)
    mod = ModelMQMQA(dbf, ["SB", "O", "S"], "SLAG-LIQ")

    assert FE2 not in mod.cations
    assert FE3 not in mod.cations
    assert SB3 in mod.cations
    assert O in mod.anions
    assert S in mod.anions

    subs_dict = {  # Thermochimica site fractions
        mod._X_ijkl(SB3,SB3,O,O): 0.25,
        mod._X_ijkl(SB3,SB3,S,S): 0.25,
        mod._X_ijkl(SB3,SB3,O,S): 0.5,
        v.T: 1000.0,
    }

    check_energy(mod, subs_dict, -1.18391E+05, mode="sympy")  # Thermochimica energy
    assert np.isclose(float(mod.moles("SB").subs(subs_dict)), 0.4, 1e-5)
    assert np.isclose(float(mod.moles("O").subs(subs_dict)), 0.3, 1e-5)
    assert np.isclose(float(mod.moles("S").subs(subs_dict)), 0.3, 1e-5)


@select_database("Shishin_Fe-Sb-O-S_slag.dat")
def test_MQMQA_SUBQ_Q_mixing_Fe_O_S(load_database):
    dbf = load_database()

    FE2 = v.Species("FE2++2.0", constituents={"FE": 2.0}, charge=2)
    FE3 = v.Species("FE3++3.0", constituents={"FE": 3.0}, charge=3)
    SB3 = v.Species("SB3++3.0", constituents={"SB": 3.0}, charge=3)
    O = v.Species("O-2.0", constituents={"O": 1.0}, charge=-2)
    S = v.Species("S-2.0", constituents={"S": 1.0}, charge=-2)
    mod = ModelMQMQA(dbf, ["FE", "O", "S"], "SLAG-LIQ")

    assert FE2 in mod.cations
    assert FE3 in mod.cations
    assert SB3 not in mod.cations
    assert O in mod.anions
    assert S in mod.anions

    subs_dict = {  # Thermochimica site fractions
        mod._X_ijkl(FE2,FE2,O,O): 1.2662E-30,
        mod._X_ijkl(FE3,FE3,O,O): 0.25000   ,
        mod._X_ijkl(FE2,FE3,O,O): 4.6228E-15,
        mod._X_ijkl(FE2,FE2,S,S): 1.1655E-28,
        mod._X_ijkl(FE3,FE3,S,S): 0.25000   ,
        mod._X_ijkl(FE2,FE3,S,S): 1.2576E-14,
        mod._X_ijkl(FE2,FE2,O,S): 1.7709E-29,
        mod._X_ijkl(FE3,FE3,O,S): 0.50000   ,
        mod._X_ijkl(FE2,FE3,O,S): 1.3019E-14,
        v.T: 1000.0,
    }

    check_energy(mod, subs_dict, -1.37905E+05, mode="sympy")  # Thermochimica energy
    assert np.isclose(float(mod.moles("FE").subs(subs_dict)), 0.4, 1e-5)
    assert np.isclose(float(mod.moles("O").subs(subs_dict)), 0.3, 1e-5)
    assert np.isclose(float(mod.moles("S").subs(subs_dict)), 0.3, 1e-5)

@select_database("Shishin_Fe-Sb-O-S_slag.dat")
def test_MQMQA_SUBQ_Q_mixing_Fe_O_S_2(load_database):
    dbf = load_database()

    FE2 = v.Species("FE2++2.0", constituents={"FE": 2.0}, charge=2)
    FE3 = v.Species("FE3++3.0", constituents={"FE": 3.0}, charge=3)
    SB3 = v.Species("SB3++3.0", constituents={"SB": 3.0}, charge=3)
    O = v.Species("O-2.0", constituents={"O": 1.0}, charge=-2)
    S = v.Species("S-2.0", constituents={"S": 1.0}, charge=-2)
    mod = ModelMQMQA(dbf, ["FE", "O", "S"], "SLAG-LIQ")

    assert FE2 in mod.cations
    assert FE3 in mod.cations
    assert SB3 not in mod.cations
    assert O in mod.anions
    assert S in mod.anions

    subs_dict = {  # Thermochimica site fractions
        mod._X_ijkl(FE2,FE2,O,O): 5.0999E-03,
        mod._X_ijkl(FE3,FE3,O,O): 0.16282   ,
        mod._X_ijkl(FE2,FE3,O,O): 0.14021   ,
        mod._X_ijkl(FE2,FE2,S,S): 0.16575   ,
        mod._X_ijkl(FE3,FE3,S,S): 2.1897E-02,
        mod._X_ijkl(FE2,FE3,S,S): 0.12049   ,
        mod._X_ijkl(FE2,FE2,O,S): 4.2382E-02,
        mod._X_ijkl(FE3,FE3,O,S): 0.11942   ,
        mod._X_ijkl(FE2,FE3,O,S): 0.22193   ,
        v.T: 1000.0,
    }

    check_energy(mod, subs_dict, -1.39362E+05, mode="sympy")  # Thermochimica energy
    assert np.isclose(float(mod.moles("FE").subs(subs_dict)), 0.45, 1e-5)
    assert np.isclose(float(mod.moles("O").subs(subs_dict)), 0.275, 1e-5)
    assert np.isclose(float(mod.moles("S").subs(subs_dict)), 0.275, 1e-5)


@select_database("Shishin_Fe-Sb-O-S_slag.dat")
def test_MQMQA_SUBQ_Q_mixing_Fe3_Sb_S(load_database):
    dbf = load_database()

    FE2 = v.Species("FE2++2.0", constituents={"FE": 2.0}, charge=2)
    FE3 = v.Species("FE3++3.0", constituents={"FE": 3.0}, charge=3)
    SB3 = v.Species("SB3++3.0", constituents={"SB": 3.0}, charge=3)
    O = v.Species("O-2.0", constituents={"O": 1.0}, charge=-2)
    S = v.Species("S-2.0", constituents={"S": 1.0}, charge=-2)
    mod = ModelMQMQA(dbf, ["FE", "SB", "S"], "SLAG-LIQ")

    assert FE2 in mod.cations
    assert FE3 in mod.cations
    assert SB3 in mod.cations
    assert O not in mod.anions
    assert S in mod.anions

    subs_dict = {  # Thermochimica site fractions
        mod._X_ijkl(FE2,FE2,S,S): 3.8675E-29,
        mod._X_ijkl(FE3,FE3,S,S): 0.25000   ,
        mod._X_ijkl(SB3,SB3,S,S): 0.25000   ,
        mod._X_ijkl(FE2,FE3,S,S): 7.2682E-15,
        mod._X_ijkl(FE2,SB3,S,S): 7.2682E-15,
        mod._X_ijkl(FE3,SB3,S,S): 0.50000   ,
        v.T: 1000.0,
    }

    check_energy(mod, subs_dict, -6.76687E+04, mode="sympy")  # Thermochimica energy
    assert np.isclose(float(mod.moles("FE").subs(subs_dict)), 0.2, 1e-5)
    assert np.isclose(float(mod.moles("SB").subs(subs_dict)), 0.2, 1e-5)
    assert np.isclose(float(mod.moles("S").subs(subs_dict)), 0.6, 1e-5)


@select_database("Shishin_Fe-Sb-O-S_slag.dat")
def test_MQMQA_SUBQ_Q_mixing_Fe3_Sb_O(load_database):
    dbf = load_database()

    FE2 = v.Species("FE2++2.0", constituents={"FE": 2.0}, charge=2)
    FE3 = v.Species("FE3++3.0", constituents={"FE": 3.0}, charge=3)
    SB3 = v.Species("SB3++3.0", constituents={"SB": 3.0}, charge=3)
    O = v.Species("O-2.0", constituents={"O": 1.0}, charge=-2)
    S = v.Species("S-2.0", constituents={"S": 1.0}, charge=-2)
    mod = ModelMQMQA(dbf, ["FE", "SB", "O"], "SLAG-LIQ")

    assert FE2 in mod.cations
    assert FE3 in mod.cations
    assert SB3 in mod.cations
    assert O in mod.anions
    assert S not in mod.anions

    subs_dict = {  # Thermochimica site fractions
        mod._X_ijkl(FE2,FE2,O,O): 3.6575E-29,
        mod._X_ijkl(FE3,FE3,O,O): 0.25000   ,
        mod._X_ijkl(SB3,SB3,O,O): 0.25000   ,
        mod._X_ijkl(FE2,FE3,O,O): 3.2565E-14,
        mod._X_ijkl(FE2,SB3,O,O): 9.2343E-15,
        mod._X_ijkl(FE3,SB3,O,O): 0.50000   ,
        v.T: 1000.00,
    }

    check_energy(mod, subs_dict, -1.86322E+05, mode="sympy")  # Thermochimica energy
    assert np.isclose(float(mod.moles("FE").subs(subs_dict)), 0.2, 1e-5)
    assert np.isclose(float(mod.moles("SB").subs(subs_dict)), 0.2, 1e-5)
    assert np.isclose(float(mod.moles("O").subs(subs_dict)), 0.6, 1e-5)


@select_database("Shishin_Fe-Sb-O-S_slag.dat")
def test_MQMQA_SUBQ_Q_mixing_Fe2_Fe3_Sb_S(load_database):
    dbf = load_database()

    FE2 = v.Species("FE2++2.0", constituents={"FE": 2.0}, charge=2)
    FE3 = v.Species("FE3++3.0", constituents={"FE": 3.0}, charge=3)
    SB3 = v.Species("SB3++3.0", constituents={"SB": 3.0}, charge=3)
    O = v.Species("O-2.0", constituents={"O": 1.0}, charge=-2)
    S = v.Species("S-2.0", constituents={"S": 1.0}, charge=-2)
    mod = ModelMQMQA(dbf, ["FE", "SB", "S"], "SLAG-LIQ")

    assert FE2 in mod.cations
    assert FE3 in mod.cations
    assert SB3 in mod.cations
    assert O not in mod.anions
    assert S in mod.anions

    subs_dict = {  # Thermochimica site fractions
        mod._X_ijkl(FE2,FE2,S,S): 8.6505E-02,
        mod._X_ijkl(FE3,FE3,S,S): 0.12457   ,
        mod._X_ijkl(SB3,SB3,S,S): 0.12457   ,
        mod._X_ijkl(FE2,FE3,S,S): 0.20761   ,
        mod._X_ijkl(FE2,SB3,S,S): 0.20761   ,
        mod._X_ijkl(FE3,SB3,S,S): 0.24913   ,
        v.T: 1000.0,
    }

    check_energy(mod, subs_dict, -7.82909E+04, mode="sympy")  # Thermochimica energy
    assert np.isclose(float(mod.moles("FE").subs(subs_dict)), 3.0000E-01, 1e-5)
    assert np.isclose(float(mod.moles("SB").subs(subs_dict)), 0.13333333, 1e-5)
    assert np.isclose(float(mod.moles("S").subs(subs_dict)), 0.56666667, 1e-5)

@select_database("Shishin_Fe-Sb-O-S_slag.dat")
def test_MQMQA_SUBQ_Q_mixing_Fe2_Fe3_Sb_O(load_database):
    dbf = load_database()

    FE2 = v.Species("FE2++2.0", constituents={"FE": 2.0}, charge=2)
    FE3 = v.Species("FE3++3.0", constituents={"FE": 3.0}, charge=3)
    SB3 = v.Species("SB3++3.0", constituents={"SB": 3.0}, charge=3)
    O = v.Species("O-2.0", constituents={"O": 1.0}, charge=-2)
    S = v.Species("S-2.0", constituents={"S": 1.0}, charge=-2)
    mod = ModelMQMQA(dbf, ["FE", "SB", "O"], "SLAG-LIQ")

    assert FE2 in mod.cations
    assert FE3 in mod.cations
    assert SB3 in mod.cations
    assert O in mod.anions
    assert S not in mod.anions

    subs_dict = {  # Thermochimica site fractions
        mod._X_ijkl(FE2,FE2,O,O): 5.6596E-02,
        mod._X_ijkl(FE3,FE3,O,O): 9.1011E-02,
        mod._X_ijkl(SB3,SB3,O,O): 0.14645   ,
        mod._X_ijkl(FE2,FE3,O,O): 0.29296   ,
        mod._X_ijkl(FE2,SB3,O,O): 0.18208   ,
        mod._X_ijkl(FE3,SB3,O,O): 0.23090   ,
        v.T: 1000.00,
    }

    check_energy(mod, subs_dict, -1.83603E+05, mode="sympy")  # Thermochimica energy
    assert np.isclose(float(mod.moles("FE").subs(subs_dict)), 3.0000E-01, 1e-5)
    assert np.isclose(float(mod.moles("SB").subs(subs_dict)), 0.13333333, 1e-5)
    assert np.isclose(float(mod.moles("O").subs(subs_dict)), 0.56666667, 1e-5)


@select_database("Kaye_Pd-Ru-Tc-Mo.dat")
def test_QKTO_binary_mixing(load_database):
    dbf = load_database()
    RU = v.Species("RU")
    PD = v.Species("PD")

    mod = Model(dbf, [ "RU", "PD"], "LIQN")

    assert RU in mod.constituents[0]
    assert PD in mod.constituents[0]

    subs_dict = {  # Thermochimica site fractions
        v.Y("LIQN", 0, RU): 0.40,
        v.Y("LIQN", 0, PD): 0.60,
        v.T: 2500.00,
    }

    assert np.isclose(float(mod.moles("PD").subs(subs_dict)), 0.60, 1e-5)
    assert np.isclose(float(mod.moles("RU").subs(subs_dict)), 0.40, 1e-5)
    check_energy(mod, subs_dict, -1.89736E+05, mode="sympy")  # Thermochimica energy


@select_database("Kaye_Pd-Ru-Tc-Mo.dat")
def test_QKTO_multicomponent_extrapolation_binary_mixing(load_database):
    """Test extrapolation into multi-component from only binary excess parameters"""
    dbf = load_database()
    PD = v.Species("PD")
    RU = v.Species("RU")
    TC = v.Species("TC")
    MO = v.Species("MO")

    mod = Model(dbf, ["PD", "RU", "TC", "MO"], "LIQN")

    assert PD in mod.constituents[0]
    assert RU in mod.constituents[0]
    assert TC in mod.constituents[0]
    assert MO in mod.constituents[0]

    subs_dict = {  # Thermochimica site fractions
        v.Y("LIQN", 0, MO): 0.025,
        v.Y("LIQN", 0, TC): 0.5,
        v.Y("LIQN", 0, RU): 0.4,
        v.Y("LIQN", 0, PD): 0.075,
        v.T: 2500.00,
    }

    check_energy(mod, subs_dict, -1.85308E+05, mode="sympy")  # Thermochimica energy
    assert np.isclose(float(mod.moles("MO").subs(subs_dict)), 0.025, 1e-5)
    assert np.isclose(float(mod.moles("TC").subs(subs_dict)), 0.50, 1e-5)
    assert np.isclose(float(mod.moles("RU").subs(subs_dict)), 0.40, 1e-5)
    assert np.isclose(float(mod.moles("PD").subs(subs_dict)), 0.075, 1e-5)


@select_database("Kaye_Pd-Ru-Tc-Mo.dat")
def test_QKTO_multicomponent_extrapolation(load_database):
    """Test extrapolation into multi-component from binary and ternary excess parameters"""
    dbf = load_database()
    PD = v.Species("PD")
    RU = v.Species("RU")
    TC = v.Species("TC")
    MO = v.Species("MO")

    mod = Model(dbf, ["PD", "RU", "TC", "MO"], "HCPN")

    assert PD in mod.constituents[0]
    assert RU in mod.constituents[0]
    assert TC in mod.constituents[0]
    assert MO in mod.constituents[0]

    subs_dict = {  # Thermochimica site fractions
        v.Y("HCPN", 0, MO): 0.025,
        v.Y("HCPN", 0, TC): 0.500,
        v.Y("HCPN", 0, RU): 0.400,
        v.Y("HCPN", 0, PD): 0.075,
        v.T: 1500.00,
    }

    check_energy(mod, subs_dict, -90169.957, mode="sympy")  # Thermochimica energy
    assert np.isclose(float(mod.moles("MO").subs(subs_dict)), 0.025, 1e-5)
    assert np.isclose(float(mod.moles("TC").subs(subs_dict)), 0.500, 1e-5)
    assert np.isclose(float(mod.moles("RU").subs(subs_dict)), 0.400, 1e-5)
    assert np.isclose(float(mod.moles("PD").subs(subs_dict)), 0.075, 1e-5)
