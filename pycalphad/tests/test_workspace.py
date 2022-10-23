from numpy.testing import assert_allclose
from pycalphad import Workspace, variables as v
from pycalphad.property_framework import as_property, ComputableProperty, T0, IsolatedPhase, DormantPhase
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

def test_meta_property_creation():
    pass

@select_database("alzn_mey.tdb")
def test_tzero_property_creation(load_database):
    dbf = load_database()
    wks = Workspace(dbf=dbf, comps=['AL', 'ZN', 'VA'], phases=['FCC_A1', 'HCP_A3', 'LIQUID'],
                    conditions={v.N: 1, v.P: 1e5, v.T: (300, 1000, 1), v.X('ZN'): 0.3})
    assert isinstance(T0('FCC_A1', 'HCP_A3', wks=wks), ComputableProperty)
