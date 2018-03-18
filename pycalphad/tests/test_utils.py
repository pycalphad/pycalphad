"""
The utils test module contains tests for pycalphad utilities.
"""

from pycalphad import Database
from pycalphad.core.utils import filter_phases, unpack_components

from pycalphad.tests.datasets import ALNIPT_TDB

ALNIPT_DBF = Database(ALNIPT_TDB)

def test_filter_phases_removes_disordered_phases_from_order_disorder():
    """Databases with order-disorder models should have the disordered phases be filtered."""
    all_phases = set(ALNIPT_DBF.phases.keys())
    filtered_phases = set(filter_phases(ALNIPT_DBF, unpack_components(ALNIPT_DBF, ['AL', 'NI', 'PT', 'VA'])))
    assert all_phases.difference(filtered_phases) == {'FCC_A1'}

def test_filter_phases_removes_phases_with_inactive_sublattices():
    """Phases that have no active components in any sublattice should be filtered"""
    all_phases = set(ALNIPT_DBF.phases.keys())
    filtered_phases = set(filter_phases(ALNIPT_DBF, unpack_components(ALNIPT_DBF, ['AL', 'NI', 'VA'])))
    assert all_phases.difference(filtered_phases) == {'FCC_A1', 'PT8AL21', 'PT5AL21', 'PT2AL', 'PT2AL3', 'PT5AL3', 'ALPT2'}

