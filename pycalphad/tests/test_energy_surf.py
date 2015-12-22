"""
The energy_surf test module verifies that energy_surf() calculates
the Gibbs energy.
"""

import nose.tools
from pycalphad import Database, energy_surf

from pycalphad.tests.datasets import ALCRNI_TDB as TDB_TEST_STRING

DBF = Database(TDB_TEST_STRING)

def test_surface():
    "Bare minimum: energy_surf produces a result."
    energy_surf(DBF, ['AL', 'CR', 'NI'], ['L12_FCC'],
                T=1273., mode='numpy')

@nose.tools.raises(AttributeError)
def test_unknown_model_attribute():
    "Sampling an unknown model attribute raises exception."
    energy_surf(DBF, ['AL', 'CR', 'NI'], ['L12_FCC'],
                T=1400.0, output='_fail_')

