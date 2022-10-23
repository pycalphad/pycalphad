from pycalphad.property_framework import as_property, DotDerivativeComputedProperty, \
    ModelComputedProperty, T0, IsolatedPhase, DormantPhase
import pycalphad.variables as v


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

