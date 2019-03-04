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
from pycalphad import ConditionError
from pycalphad.tests.datasets import ALCRNI_TDB as TDB_TEST_STRING, ALFE_TDB


DBF = Database(TDB_TEST_STRING)
ALFE_DBF = Database(ALFE_TDB)

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
    result_one_values = result_one.GM.values
    result_two = calculate(DBF, ['AL', 'CR', 'NI'], 'LIQUID', T=400, P=101325)
    result_two_values = result_two.GM.values
    np.testing.assert_array_equal(np.squeeze(result_one_values), np.squeeze(result_two_values))
    assert len(result_one_values.shape) == 2
    assert result_one_values.shape[0] == 1
    assert len(result_two_values.shape) == 3
    assert result_two_values.shape[:2] == (1, 1)


def test_calculate_some_phases_filtered():
    """
    Phases are filtered out from calculate() when some cannot be built.
    """
    # should not raise; AL13FE4 should be filtered out
    calculate(ALFE_DBF, ['AL', 'VA'], ['FCC_A1', 'AL13FE4'], T=1200, P=101325)


@nose.tools.raises(ConditionError)
def test_calculate_raises_with_no_active_phases_passed():
    """Passing inactive phases to calculate() raises a ConditionError."""
    # Phase cannot be built without FE
    calculate(ALFE_DBF, ['AL', 'VA'], ['AL13FE4'], T=1200, P=101325)


if __name__ == '__main__':
    import nose
    nose.run()
