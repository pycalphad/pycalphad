from pycalphad.core.workspace import Workspace
from pycalphad.property_framework import as_property, DotDerivativeComputedProperty, \
    ModelComputedProperty, T0, IsolatedPhase, DormantPhase, ReferenceState
import pycalphad.variables as v
from pycalphad.tests.fixtures import select_database, load_database
import numpy as np
import numpy.testing


def test_as_property_creation():
    assert as_property('T') == v.T
    assert as_property('X(ZN)') == v.X('ZN')
    assert as_property('X(FCC_A1#1,ZN)') == v.X('FCC_A1#1', 'ZN')

def test_as_property_dot_derivative_creation():
    assert as_property('HM.T') == DotDerivativeComputedProperty(ModelComputedProperty('HM'), v.T)
    assert as_property('HM.T') == DotDerivativeComputedProperty('HM', 'T')
    assert as_property('MU(AL).X(ZN)') == DotDerivativeComputedProperty(v.MU('AL'), v.X('ZN'))
    assert as_property('NP(LIQUID).T') == DotDerivativeComputedProperty(v.NP('LIQUID'), v.T)
    assert ModelComputedProperty('SM') != ModelComputedProperty('HM')

def test_property_units():
    model_prop = ModelComputedProperty('test')
    model_prop.implementation_units = 'J/mol'
    model_prop.display_units = 'kJ/mol'
    assert model_prop['J/mol'].display_units == 'J/mol'
    assert v.T['degC'].display_units == 'degC'
    assert v.T.display_units != 'degC'
    assert v.T['degC'] == v.T

@select_database("alzn_mey.tdb")
def test_cpf_phase_energy_curves(load_database):
    wks2 = Workspace(load_database(), ['AL', 'ZN'],
                    ['FCC_A1', 'HCP_A3', 'LIQUID'],
                    {v.X('ZN'):(0,1,0.02), v.T: 600, v.P:101325, v.N: 1})

    props = []
    for phase_name in wks2.phases:
        # Workaround for poor starting point selection in IsolatedPhase
        metastable_wks = wks2.copy()
        metastable_wks.phases = [phase_name]
        prop = IsolatedPhase(phase_name, metastable_wks)(f'GM({phase_name})')
        prop.display_name = phase_name
        props.append(prop)
    result = {}
    for prop, value in wks2.get(*props, values_only=False).items():
        result[prop.display_name] = value
    np.testing.assert_almost_equal(np.nanmax(result['FCC_A1'].magnitude), -20002.975665, decimal=5)
    np.testing.assert_almost_equal(np.nanmin(result['FCC_A1'].magnitude), -26718.58552, decimal=5)
    np.testing.assert_almost_equal(np.nanmax(result['HCP_A3'].magnitude), -15601.975666, decimal=5)
    np.testing.assert_almost_equal(np.nanmin(result['HCP_A3'].magnitude), -28027.206646, decimal=5)
    np.testing.assert_almost_equal(np.nanmax(result['LIQUID'].magnitude), -16099.679946, decimal=5)
    np.testing.assert_almost_equal(np.nanmin(result['LIQUID'].magnitude), -27195.787525, decimal=5)

@select_database("alzn_mey.tdb")
def test_cpf_driving_force(load_database):
    wks3 = Workspace(load_database(), ['AL', 'ZN'],
                ['FCC_A1', 'HCP_A3', 'LIQUID'],
                {v.X('ZN'):(0,1,0.02), v.T: 600, v.P:101325, v.N: 1})
    metastable_liq_wks = wks3.copy()
    metastable_liq_wks.phases = ['LIQUID']
    liq_driving_force = DormantPhase('LIQUID', metastable_liq_wks).driving_force
    liq_driving_force.display_name = 'Liquid Driving Force'
    result, = wks3.get(liq_driving_force)
    np.testing.assert_almost_equal(np.nanmax(result.magnitude), -610.932599, decimal=5)
    np.testing.assert_almost_equal(np.nanmin(result.magnitude), -3903.295718, decimal=5)

@select_database("alzn_mey.tdb")
def test_cpf_tzero(load_database):
    wks4 = Workspace(load_database(), ['AL', 'ZN'],
                    ['FCC_A1', 'HCP_A3', 'LIQUID'],
                    {v.X('ZN'):(0,1,0.02), v.T: 300, v.P:101325, v.N: 1})
    tzero = T0('FCC_A1', 'HCP_A3', wks4)
    tzero.maximum_value = 1700 # ZN reference state in this database is not valid beyond this temperature
    result, = wks4.get(tzero)
    np.testing.assert_almost_equal(np.nanmax(result.magnitude), 1673.2643290, decimal=5)
    np.testing.assert_almost_equal(np.nanmin(result.magnitude), 621.72616, decimal=5)

@select_database("nbre_liu.tdb")
def test_cpf_reference_state(load_database):
    wks = Workspace(load_database(), ["NB", "RE", "VA"], ["LIQUID_RENB"],
                    {v.P: 101325, v.T: 2800, v.X("RE"): (0, 1, 0.005)})

    ref = ReferenceState([("LIQUID_RENB", {v.X("RE"): 0}),
                        ("LIQUID_RENB", {v.X("RE"): 1})
                        ], wks)

    result, = wks.get(ref('HM'))
    np.testing.assert_almost_equal(np.nanmax(result.magnitude), 0, decimal=5)
    np.testing.assert_almost_equal(np.nanmin(result.magnitude), -2034.554367, decimal=5)

@select_database("alzn_mey.tdb")
def test_cpf_calculation(load_database):
    wks4 = Workspace(load_database(), ['AL', 'ZN'],
                    ['LIQUID'],
                    {v.X('ZN'): 0.3, v.T: 700, v.P:101325, v.N: 1})

    results = wks4.get('HM.T', 'MU(AL).X(ZN)')
    np.testing.assert_array_almost_equal(np.squeeze(results), [29.63807, -3460.0878], decimal=5)
    wks4.phases = ['FCC_A1', 'LIQUID', 'HCP_A3']
    wks4.conditions[v.X('ZN')] = 0.7
    results = wks4.get('X(LIQUID, AL).T')
    np.testing.assert_array_almost_equal(np.squeeze(results), [0.00249], decimal=5)
    results = wks4.get('NP(*).T')
    np.testing.assert_array_almost_equal(np.squeeze(results), [-0.01147, float('nan'), 0.01147], decimal=5)
