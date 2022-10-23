from numpy.testing import assert_allclose
import numpy as np
from pycalphad import Workspace, variables as v
from pycalphad.property_framework import as_property, ComputableProperty, T0, IsolatedPhase, DormantPhase
from pycalphad.property_framework.units import Q_
from pycalphad.tests.fixtures import load_database, select_database
import pytest

@pytest.mark.solver
@select_database("alzn_mey.tdb")
def test_workspace_creation(load_database):
    dbf = load_database()
    wks = Workspace(dbf=dbf, comps=['AL', 'ZN', 'VA'], phases=['FCC_A1', 'HCP_A3', 'LIQUID'],
                    conditions={v.N: 1, v.P: 1e5, v.T: (300, 1000, 10), v.X('ZN'): 0.3})
    wks2 = Workspace(dbf, ['AL', 'ZN', 'VA'], ['FCC_A1', 'HCP_A3', 'LIQUID'],
                     {v.N: 1, v.P: 1e5, v.T: (300, 1000, 10), v.X('ZN'): 0.3})
    assert_allclose(wks.eq.GM, wks2.eq.GM)

@select_database("alzn_mey.tdb")
def test_meta_property_creation(load_database):
    dbf = load_database()
    wks = Workspace(dbf=dbf, comps=['AL', 'ZN', 'VA'], phases=['FCC_A1', 'HCP_A3', 'LIQUID'],
                    conditions={v.N: 1, v.P: 1e5, v.T: (300, 1000, 10), v.X('ZN'): 0.3})
    my_tzero = T0('FCC_A1', 'HCP_A3', wks=wks)
    assert isinstance(my_tzero, ComputableProperty)

@select_database("alzn_mey.tdb")
def test_tzero_property(load_database):
    dbf = load_database()
    wks = Workspace(dbf=dbf, comps=['AL', 'ZN', 'VA'], phases=['FCC_A1', 'HCP_A3', 'LIQUID'],
                    conditions={v.N: 1, v.P: 1e5, v.T: 600, v.X('ZN'): (0.01,1-0.01,0.01)})
    my_tzero = T0('FCC_A1', 'HCP_A3', wks=wks)
    assert isinstance(my_tzero, ComputableProperty)
    assert my_tzero.property_to_optimize == v.T
    t0_values, = wks.get(my_tzero)
    assert_allclose(np.nanmax(t0_values.magnitude), 2952.90443)
    wks.conditions[v.X('ZN')] = 0.3
    my_tzero.property_to_optimize = v.X('ZN')
    my_tzero.minimum_value = 0.0
    my_tzero.maximum_value = 1.0
    t0_composition, = wks.get(my_tzero)
    assert_allclose(t0_composition[0].magnitude, Q_(0.86119, 'fraction').magnitude, atol=my_tzero.residual_tol)
