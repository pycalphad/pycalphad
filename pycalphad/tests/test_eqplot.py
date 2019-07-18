"""
The plot test module verifies that the eqplot produces plots without error.
"""

import matplotlib.pyplot as plt
plt.switch_backend('Agg')
from pycalphad import Database, eqplot, equilibrium
import pycalphad.variables as v
from pycalphad.tests.datasets import *
from matplotlib.axes import Axes

ALFE_DBF = Database(ALFE_TDB)
ALCOCRNI_DBF = Database(ALCOCRNI_TDB)


def test_eqplot_binary():
    """
    eqplot should return an axes object when one independent component and one
    independent potential are passed.
    """
    my_phases = ['LIQUID', 'FCC_A1', 'HCP_A3', 'AL5FE2',
                 'AL2FE', 'AL13FE4', 'AL5FE4']
    comps = ['AL', 'FE', 'VA']
    conds = {v.T: (1400, 1500, 50), v.P: 101325, v.X('AL'): (0, 1, 0.5)}
    eq = equilibrium(ALFE_DBF, comps, my_phases, conds)
    ax = eqplot(eq)
    assert isinstance(ax, Axes)


def test_eqplot_ternary():
    """
    eqplot should return an axes object that has a traingular projection when
    two independent components and one independent potential are passed.
    """
    eq = equilibrium(ALCOCRNI_DBF, ['AL', 'CO', 'CR', 'VA'], ['LIQUID'],
                     {v.T: 2500, v.X('AL'): (0,0.5,0.33), v.X('CO'): (0,0.5,0.3), v.P: 101325})
    ax = eqplot(eq)
    assert isinstance(ax, Axes)
    assert ax.name == 'triangular'
