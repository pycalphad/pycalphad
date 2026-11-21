"""
The solver test module verifies the solver from explicit composition sets, independent of the starting point.
"""

import numpy as np
from numpy.testing import assert_allclose
import pytest
from pycalphad import Database, Workspace, as_property, variables as v
from pycalphad.codegen.phase_record_factory import PhaseRecordFactory
from pycalphad.core.composition_set import CompositionSet
from pycalphad.core.solver import Solver
from pycalphad.core.utils import instantiate_models
from pycalphad.property_framework.metaproperties import DormantPhase

# Ni-Ti BCC_B2 of the Ni-Ti-Cr database of Lu et al.
NITI_BCC_B2_TDB = """
ELEMENT NI FCC_A1 58.69 4787.0 29.796 !
ELEMENT TI HCP_A3 47.88 4810.0 30.648 !
ELEMENT VA VACUUM 0.0 0.0 0.0 !
FUNCTION GHSERNI 298.15 -5179.159+117.854*T-22.096*T*LN(T)-.0048407*T**2; 1728 Y
    -27840.655+279.135*T-43.1*T*LN(T)+1.12754E+31*T**(-9); 6000 N !
FUNCTION GBCCTI 298.15 -1272.064+134.71418*T-25.5768*T*LN(T)-6.63845E-04*T**2-2.78803E-07*T**3+7208*T**(-1); 1155 Y
    +6667.385+105.366379*T-22.3771*T*LN(T)+.00121707*T**2-8.4534E-07*T**3-2002750*T**(-1); 1941 Y
    +26483.26-182.426471*T+19.0900905*T*LN(T)-.02200832*T**2+1.228863E-06*T**3+1400501*T**(-1); 6000 N !
TYPE_DEFINITION % SEQ * !
DEFINE_SYSTEM_DEFAULT ELEMENT 2 !
DEFAULT_COMMAND DEF_SYS_ELEMENT VA !
TYPE_DEFINITION & GES A_P_D BCC_A2 MAGNETIC -1.0 0.4 !
PHASE BCC_A2 %& 2 1 3 !
CONSTITUENT BCC_A2 :NI,TI:VA: !
PARAMETER TC(BCC_A2,NI:VA;0) 298.15 +575; 6000 N !
PARAMETER BMAGN(BCC_A2,NI:VA;0) 298.15 +.85; 6000 N !
PARAMETER G(BCC_A2,NI:VA;0) 298.15 +8715.084-3.556*T+GHSERNI#; 6000 N !
PARAMETER G(BCC_A2,TI:VA;0) 298.15 +GBCCTI#; 6000 N !
PARAMETER G(BCC_A2,NI,TI:VA;0) 298.15 -97427+12.112*T; 6000 N !
PARAMETER G(BCC_A2,NI,TI:VA;1) 298.15 -32315; 6000 N !
TYPE_DEFINITION ' GES AMEND_PHASE_DESCRIPTION BCC_B2 DIS_PART BCC_A2 !
PHASE BCC_B2 %' 3 .5 .5 3 !
CONSTITUENT BCC_B2 :NI,TI:NI,TI:VA: !
PARAMETER G(BCC_B2,NI:TI:VA;0) 298.15 -33193.7+10.284*T; 6000 N !
PARAMETER G(BCC_B2,TI:NI:VA;0) 298.15 -33193.7+10.284*T; 6000 N !
PARAMETER G(BCC_B2,NI,TI:NI:VA;0) 298.15 -55288.8+25.4416*T; 6000 N !
PARAMETER G(BCC_B2,NI,TI:NI:VA;2) 298.15 +6010.11+3.95974*T; 6000 N !
PARAMETER G(BCC_B2,NI:NI,TI:VA;0) 298.15 -55288.8+25.4416*T; 6000 N !
PARAMETER G(BCC_B2,NI:NI,TI:VA;2) 298.15 +6010.11+3.95974*T; 6000 N !
PARAMETER G(BCC_B2,NI,TI:TI:VA;0) 298.15 +60723.7-15.4024*T; 6000 N !
PARAMETER G(BCC_B2,TI:NI,TI:VA;0) 298.15 +60723.7-15.4024*T; 6000 N !
"""


def niti_bcc_b2_compset(eta, x_ni, T):
    "Ni-Ti BCC_B2 composition set at X(NI) = x_ni and T, with order parameter eta (Ni moved onto the first sublattice)."
    dbf = Database(NITI_BCC_B2_TDB)
    comps = ["NI", "TI", "VA"]
    conds = {v.N: 1, v.P: 101325, v.T: T, v.X("NI"): x_ni}
    prf = PhaseRecordFactory(dbf, comps, conds, instantiate_models(dbf, comps, ["BCC_B2"]))
    compset = CompositionSet(prf["BCC_B2"])
    # (NI, TI) on each of the two equivalent sublattices, then VA
    compset.update(np.array([x_ni + eta, 1 - x_ni - eta, x_ni - eta, 1 - x_ni + eta, 1.0]), 1.0,
                   np.array([conds[sv] for sv in prf.state_variables], dtype=float))
    return compset, conds


def solve_niti_bcc_b2(eta):
    "Solver.solve on one Ni-Ti BCC_B2 composition set at X(NI) = 0.51 and 300 K, starting from order parameter eta."
    compset, conds = niti_bcc_b2_compset(eta, 0.51, 300)
    return Solver().solve([compset], conds), compset


@pytest.mark.solver
@pytest.mark.xfail(strict=True, reason="The disordered configuration is a saddle point with zero gradient: "
                                       "every descent step is zero")
def test_solver_ordered_b2_from_disordered_configuration():
    "Ni-Ti BCC_B2 at fixed composition below its ordering temperature, from the disordered configuration (gh-670)."
    result, compset = solve_niti_bcc_b2(0.0)
    assert result.converged
    # The order parameter is the only freedom at fixed composition; G is maximal when disordered, minimal at full order
    assert_allclose(compset.energy, -43131.3475, atol=1e-3)


@pytest.mark.solver
def test_solver_ordered_b2_from_partially_ordered_configuration():
    "Ni-Ti BCC_B2 at fixed composition below its ordering temperature, from slight order on sublattice 1 (gh-670)."
    result, compset = solve_niti_bcc_b2(0.05)
    assert result.converged
    assert_allclose(compset.energy, -43131.3475, atol=1e-3)
    # G is concave in the order parameter from disorder to near full order, so a downhill path keeps Ni on the first
    # sublattice; the other ordered variant, equal in energy, is only reached by climbing over the disordered maximum
    y = np.asarray(compset.dof)[3:]
    assert y[0] > y[2]


@pytest.mark.solver
def test_solver_derivative_of_ordered_b2_inside_composition_spinodal():
    "Heat capacity (HM.T) of BCC_B2 alone at fixed composition, where it is unstable at fixed chemical potentials."
    # The solution minimizes G over the order parameter at X(NI) = 0.3, but G - MU.X curves down at its chemical
    # potentials (composition spinodal): Newton steps need the curvature correction, the derivative the exact matrix
    compset, conds = niti_bcc_b2_compset(0.1, 0.3, 1100)
    result = Solver().solve([compset], conds)
    assert result.converged
    cpm = as_property("HM.T").compute_property([compset], conds, result.chemical_potentials)
    hm = []
    for T in (1099.9, 1100.1):
        result = Solver().solve([compset], {**conds, v.T: T})
        hm.append(as_property("HM").compute_property([compset], {**conds, v.T: T}, result.chemical_potentials))
    assert_allclose(cpm, (hm[1] - hm[0]) / 0.2, rtol=1e-4)


@pytest.mark.solver
def test_dormant_ordered_b2_driving_force_from_partially_ordered_configuration():
    "DormantPhase driving force of Ni-Ti BCC_B2 from slight order, at the chemical potentials of disordered BCC_A2."
    compset, conds = niti_bcc_b2_compset(0.05, 0.5, 600)
    wks = Workspace(Database(NITI_BCC_B2_TDB), ["NI", "TI", "VA"], ["BCC_A2"], conds)
    # Not 0 J/mol (the disordered configuration, i.e. BCC_A2 itself): the dormant iterations descend to full order
    assert_allclose(wks.get(DormantPhase(compset, None).driving_force), 12919.997, atol=0.01)
