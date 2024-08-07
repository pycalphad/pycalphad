"""
The plot test module verifies that the plotting module produces plots without error.
"""

import matplotlib.pyplot as plt
plt.switch_backend('Agg')
from pycalphad import binplot, eqplot, equilibrium
from pycalphad.tests.fixtures import select_database, load_database
import pycalphad.variables as v
from matplotlib.axes import Axes


@select_database("Al-Mg_Zhong.tdb")
def test_binplot(load_database):
    dbf = load_database()
    comps = ['AL', 'MG', 'VA']
    phases = dbf.phases.keys()
    ax = binplot(dbf, comps, phases,
                 {v.N: 1, v.P:101325, v.T: (300, 1000, 10), v.X('MG'):(0, 1, 0.02)})
    assert isinstance(ax, Axes)


@select_database("alfe.tdb")
def test_eqplot_binary(load_database):
    """
    eqplot should return an axes object when one independent component and one
    independent potential are passed.
    """
    dbf = load_database()
    my_phases = ['LIQUID', 'FCC_A1', 'HCP_A3', 'AL5FE2',
                 'AL2FE', 'AL13FE4', 'AL5FE4']
    comps = ['AL', 'FE', 'VA']
    conds = {v.T: (1400, 1500, 50), v.P: 101325, v.X('AL'): (0, 1, 0.5)}
    eq = equilibrium(dbf, comps, my_phases, conds)
    ax = eqplot(eq)
    assert isinstance(ax, Axes)


@select_database("alcocrni.tdb")
def test_eqplot_ternary(load_database):
    """
    eqplot should return an axes object that has a traingular projection when
    two independent components and one independent potential are passed.
    """
    dbf = load_database()
    eq = equilibrium(dbf, ['AL', 'CO', 'CR', 'VA'], ['LIQUID'],
                     {v.T: 2500, v.X('AL'): (0,0.5,0.33), v.X('CO'): (0,0.5,0.3), v.P: 101325})
    ax = eqplot(eq)
    assert isinstance(ax, Axes)
    assert ax.name == 'triangular'
