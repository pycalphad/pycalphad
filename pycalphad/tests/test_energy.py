"""
The energy test module verifies that the Model class produces the
correct abstract syntax tree for the energy.
"""

import nose.tools
from pycalphad import Database, Model
from pycalphad.core.utils import make_callable
from pycalphad.tests.datasets import ALCRNI_TDB, FEMN_TDB
import pycalphad.variables as v
import numpy as np

DBF = Database(ALCRNI_TDB)

@nose.tools.raises(ValueError)
def test_sympify_safety():
    "Parsing malformed strings throws exceptions instead of executing code."
    from pycalphad.io.tdb import _sympify_string
    teststr = "().__class__.__base__.__subclasses__()[216]('ls')"
    _sympify_string(teststr) # should throw ParseException


def calculate_energy(model, variables, mode='numpy'):
    """
    Calculate the value of the energy at a point.

    Parameters
    ----------
    model, Model
        Energy model for a phase.

    variables, dict
        Dictionary of all input variables.

    mode, ['numpy', 'sympy'], optional
        Optimization method for the abstract syntax tree.
    """
    # Generate a callable energy function
    # Normally we would use model.subs(variables) here, but we want to ensure
    # our optimization functions are working.
    energy = make_callable(model.ast, list(variables.keys()), mode=mode)
    # Unpack all the values in the dict and use them to call the function
    return energy(*(list(variables.values())))

def check_energy(model, variables, known_value, mode='numpy'):
    "Check that our calculated energy matches the known value."
    desired = calculate_energy(model, variables, mode)
    known_value = np.array(known_value, dtype=np.complex)
    desired = np.array(desired, dtype=np.complex)
    np.testing.assert_allclose(known_value, desired, rtol=1e-5)

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

def test_pure_numpy():
    "Pure component end-members in numpy mode."
    check_energy(Model(DBF, ['AL'], 'LIQUID'), \
            {v.T: 2000, v.SiteFraction('LIQUID', 0, 'AL'): 1}, \
        -1.28565e5, mode='numpy')
    check_energy(Model(DBF, ['AL'], 'B2'), \
            {v.T: 1400, v.SiteFraction('B2', 0, 'AL'): 1,
             v.SiteFraction('B2', 1, 'AL'): 1}, \
        -6.57639e4, mode='numpy')
    check_energy(Model(DBF, ['AL'], 'L12_FCC'), \
            {v.T: 800, v.SiteFraction('L12_FCC', 0, 'AL'): 1,
             v.SiteFraction('L12_FCC', 1, 'AL'): 1}, \
        -3.01732e4, mode='numpy')

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
    check_energy(Model(DBF, ['CR', 'NI'], 'L12_FCC'), \
            {v.T: 500, v.SiteFraction('L12_FCC', 0, 'CR'): 0.33,
             v.SiteFraction('L12_FCC', 0, 'NI'): 0.67,
             v.SiteFraction('L12_FCC', 1, 'CR'): 0.33,
             v.SiteFraction('L12_FCC', 1, 'NI'): 0.67}, \
        -1.68840e4, mode='numpy')

def test_binary_magnetic_reimported():
    "Export and re-import a TDB before the calculation."
    dbf_imported = Database.from_string(DBF.to_string(fmt='tdb'), fmt='tdb')
    check_energy(Model(dbf_imported, ['CR', 'NI'], 'L12_FCC'),
                {v.T: 500, v.SiteFraction('L12_FCC', 0, 'CR'): 0.33,
                v.SiteFraction('L12_FCC', 0, 'NI'): 0.67,
                v.SiteFraction('L12_FCC', 1, 'CR'): 0.33,
                v.SiteFraction('L12_FCC', 1, 'NI'): 0.67},
                -1.68840e4, mode='numpy')

def test_binary_magnetic_ordering():
    "Two-component phase with IHJ magnetic model and ordering."
    # ordered case
    check_energy(Model(DBF, ['CR', 'NI'], 'L12_FCC'), \
            {v.T: 300, v.SiteFraction('L12_FCC', 0, 'CR'): 4.86783e-2,
             v.SiteFraction('L12_FCC', 0, 'NI'): 9.51322e-1,
             v.SiteFraction('L12_FCC', 1, 'CR'): 9.33965e-1,
             v.SiteFraction('L12_FCC', 1, 'NI'): 6.60348e-2}, \
        -9.23953e3, mode='sympy')
    check_energy(Model(DBF, ['CR', 'NI'], 'L12_FCC'), \
            {v.T: 300, v.SiteFraction('L12_FCC', 0, 'CR'): 4.86783e-2,
             v.SiteFraction('L12_FCC', 0, 'NI'): 9.51322e-1,
             v.SiteFraction('L12_FCC', 1, 'CR'): 9.33965e-1,
             v.SiteFraction('L12_FCC', 1, 'NI'): 6.60348e-2}, \
        -9.23953e3, mode='numpy')

def test_binary_dilute():
    "Dilute binary solution phase."
    check_energy(Model(DBF, ['CR', 'NI'], 'LIQUID'), \
            {v.T: 300, v.SiteFraction('LIQUID', 0, 'CR'): 1e-12,
             v.SiteFraction('LIQUID', 0, 'NI'): 1.0-1e-12}, \
        5.52773e3, mode='sympy')
    check_energy(Model(DBF, ['CR', 'NI'], 'LIQUID'), \
            {v.T: 300, v.SiteFraction('LIQUID', 0, 'CR'): 1e-12,
             v.SiteFraction('LIQUID', 0, 'NI'): 1.0-1e-12}, \
        5.52773e3, mode='numpy')

def test_binary_xiong_twostate_einstein():
    "Phase with Xiong magnetic, two-state and Einstein energy contributions."
    femn_dbf = Database(FEMN_TDB)
    mod = Model(femn_dbf, ['FE', 'MN', 'VA'], 'LIQUID')
    check_energy(mod, {v.T: 10, v.SiteFraction('LIQUID', 0, 'FE'): 1,
                                v.SiteFraction('LIQUID', 0, 'MN'): 0,
                                v.SiteFraction('LIQUID', 1, 'VA'): 1},
                 10158.591, mode='numpy')
    check_energy(mod, {v.T: 300, v.SiteFraction('LIQUID', 0, 'FE'): 0.3,
                       v.SiteFraction('LIQUID', 0, 'MN'): 0.7,
                       v.SiteFraction('LIQUID', 1, 'VA'): 1},
                 4200.8435, mode='numpy')
    check_energy(mod, {v.T: 1500, v.SiteFraction('LIQUID', 0, 'FE'): 0.8,
                       v.SiteFraction('LIQUID', 0, 'MN'): 0.2,
                       v.SiteFraction('LIQUID', 1, 'VA'): 1},
                 -86332.217, mode='numpy')

# TERNARY TESTS
def test_ternary_rkm_solution():
    "Solution phase with ternary interaction parameters."
    check_energy(Model(DBF, ['AL', 'CR', 'NI'], 'LIQUID'), \
            {v.T: 1500, v.SiteFraction('LIQUID', 0, 'AL'): 0.44,
             v.SiteFraction('LIQUID', 0, 'CR'): 0.20,
             v.SiteFraction('LIQUID', 0, 'NI'): 0.36}, \
        -1.16529e5, mode='sympy')
    check_energy(Model(DBF, ['AL', 'CR', 'NI'], 'LIQUID'), \
            {v.T: 1500, v.SiteFraction('LIQUID', 0, 'AL'): 0.44,
             v.SiteFraction('LIQUID', 0, 'CR'): 0.20,
             v.SiteFraction('LIQUID', 0, 'NI'): 0.36}, \
        -1.16529e5, mode='numpy')

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
    check_energy(Model(DBF, ['AL', 'CR', 'NI'], 'L12_FCC'), \
            {v.T: 300, v.SiteFraction('L12_FCC', 0, 'AL'): 5.42883e-8,
             v.SiteFraction('L12_FCC', 0, 'CR'): 2.07934e-6,
             v.SiteFraction('L12_FCC', 0, 'NI'): 9.99998e-1,
             v.SiteFraction('L12_FCC', 1, 'AL'): 7.49998e-1,
             v.SiteFraction('L12_FCC', 1, 'CR'): 2.50002e-1,
             v.SiteFraction('L12_FCC', 1, 'NI'): 4.55313e-10}, \
        -40717.204, mode='numpy')

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
    check_energy(Model(DBF, ['AL', 'CR', 'NI', 'VA'], 'B2'), \
            {v.T: 500, v.SiteFraction('B2', 0, 'AL'): 4.03399e-9,
             v.SiteFraction('B2', 0, 'CR'): 2.65798e-4,
             v.SiteFraction('B2', 0, 'NI'): 9.99734e-1,
             v.SiteFraction('B2', 0, 'VA'): 2.68374e-9,
             v.SiteFraction('B2', 1, 'AL'): 3.75801e-1,
             v.SiteFraction('B2', 1, 'CR'): 1.20732e-1,
             v.SiteFraction('B2', 1, 'NI'): 5.03467e-1,
             v.SiteFraction('B2', 1, 'VA'): 1e-12}, \
        -42368.27, mode='numpy')

# SPECIAL CASES
def test_case_sensitivity():
    "Case sensitivity of component and phase names."
    check_energy(Model(DBF, ['Cr', 'nI'], 'Liquid'), \
            {v.T: 300, v.SiteFraction('LIQUID', 0, 'CR'): 1e-12,
             v.SiteFraction('liquid', 0, 'ni'): 1}, \
        5.52773e3, mode='sympy')
    check_energy(Model(DBF, ['Cr', 'nI'], 'Liquid'), \
            {v.T: 300, v.SiteFraction('LIQUID', 0, 'CR'): 1e-12,
             v.SiteFraction('liquid', 0, 'ni'): 1}, \
        5.52773e3, mode='numpy')

def test_zero_site_fraction():
    "Energy of a binary solution phase where one site fraction is zero."
    check_energy(Model(DBF, ['CR', 'NI'], 'LIQUID'), \
            {v.T: 300, v.SiteFraction('LIQUID', 0, 'CR'): 0,
             v.SiteFraction('LIQUID', 0, 'NI'): 1}, \
        5.52773e3, mode='sympy')
    check_energy(Model(DBF, ['CR', 'NI'], 'LIQUID'), \
            {v.T: 300, v.SiteFraction('LIQUID', 0, 'CR'): 0,
             v.SiteFraction('LIQUID', 0, 'NI'): 1}, \
        5.52773e3, mode='numpy')
