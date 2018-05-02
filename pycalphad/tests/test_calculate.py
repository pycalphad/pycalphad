"""
The calculate test module verifies that calculate() calculates
Model quantities correctly.
"""

import nose.tools
from pycalphad import Database, calculate
import numpy as np
try:
    # Python 2
    from StringIO import StringIO
except ImportError:
    # Python 3
    from io import StringIO

from pycalphad.tests.datasets import ALCRNI_TDB as TDB_TEST_STRING


DBF = Database(TDB_TEST_STRING)

def test_surface():
    "Bare minimum: calculation produces a result."
    calculate(DBF, ['AL', 'CR', 'NI'], 'L12_FCC',
                T=1273., mode='numpy')

@nose.tools.raises(AttributeError)
def test_unknown_model_attribute():
    "Sampling an unknown model attribute raises exception."
    calculate(DBF, ['AL', 'CR', 'NI'], 'L12_FCC',
                T=1400.0, output='_fail_')

def test_statevar_upcast():
    "Integer state variable values are cast to float."
    calculate(DBF, ['AL', 'CR', 'NI'], 'L12_FCC',
                T=1273, mode='numpy')

def test_points_kwarg_multi_phase():
    "Multi-phase calculation works when internal dof differ (gh-41)."
    calculate(DBF, ['AL', 'CR', 'NI'], ['L12_FCC', 'LIQUID'],
                T=1273, points={'L12_FCC': [0.20, 0.05, 0.75, 0.05, 0.20, 0.75]}, mode='numpy')

def test_issue116():
    "Calculate gives correct result when a state variable is left as default (gh-116)."
    result_one = calculate(DBF, ['AL', 'CR', 'NI'], 'LIQUID', T=400)
    result_two = calculate(DBF, ['AL', 'CR', 'NI'], 'LIQUID', T=400, P=101325)
    np.testing.assert_array_equal(result_one.GM.values, result_two.GM.values)

if __name__ == '__main__':
    import nose
    nose.run()