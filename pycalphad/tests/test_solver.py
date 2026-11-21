"""
The solver test module verifies the solver from explicit composition sets, independent of the starting point.
"""

import numpy as np
from numpy.testing import assert_allclose
import pytest
from pycalphad import variables as v
from pycalphad.codegen.phase_record_factory import PhaseRecordFactory
from pycalphad.core.composition_set import CompositionSet
from pycalphad.core.solver import Solver
from pycalphad.core.utils import instantiate_models
from pycalphad.tests.fixtures import load_database, select_database


@pytest.mark.solver
@select_database("COST507.tdb")
def test_solver_phase_entering_at_phase_rule_limit_replaces_phase_by_mass_balance(load_database):
    "A phase entering at the Gibbs phase rule limit replaces the stable phase the mass balance removes first."
    dbf = load_database()
    comps = ["AL", "CU", "MG", "SI", "ZN", "VA"]
    conds = {v.N: 1, v.P: 101325, v.T: 550, v.X("CU"): 0.029, v.X("MG"): 0.063, v.X("SI"): 0.527, v.X("ZN"): 0.187}
    e = 1e-14
    # Five stable composition sets (the phase rule limit) and metastable BCC_B2 and LAVES_C14. BCC_B2 gains a positive
    # driving force and must replace ALCU_ETA; removing the first stable set (FCC_A1) instead cascades to a failed solve.
    start = [
        ("FCC_A1", 0.16, [0.9, e, e, e, 0.1, 1]),  # (AL,CU,MG,SI,ZN)(VA)
        ("ALCU_ETA", 0.09, [1, e, 0.63, 0.37]),  # (AL,CU)(CU,ZN)
        ("MG2SI", 0.05, [1, 1]),  # (MG)(SI)
        ("DIAMOND_A4", 0.51, [e, 1, e]),  # (AL,SI,ZN)
        ("MG2ZN11", 0.19, [1, 0.03, 0.006, 0.964]),  # (MG)(AL,CU,ZN)
        ("BCC_B2", 0, [0.37, e, e, e, 0.63, 0.37, e, e, e, 0.63, 1]),  # (AL,CU,MG,SI,ZN)(AL,CU,MG,SI,ZN)(VA)
        ("LAVES_C14", 0, [e, 0.017, e, 0.983, e, e, 1, e]),  # (AL,CU,MG,ZN)(AL,CU,MG,ZN)
    ]
    prf = PhaseRecordFactory(dbf, comps, conds, instantiate_models(dbf, comps, [phase for phase, _, _ in start]))
    statevars = np.array([conds[sv] for sv in prf.state_variables], dtype=float)
    compsets = []
    for phase, amount, site_fractions in start:
        compsets.append(CompositionSet(prf[phase]))
        compsets[-1].update(np.array(site_fractions, dtype=float), amount, statevars)
    result = Solver().solve(compsets, conds)
    assert result.converged
    assert {cs.phase_record.phase_name for cs in compsets} == {"BCC_B2", "DIAMOND_A4", "FCC_A1", "MG2SI", "MG2ZN11"}
    # component (alphabetical) order: AL, CU, MG, SI, ZN
    assert_allclose(result.chemical_potentials, [-18094.262, -70468.946, -50329.821, -12176.943, -25563.178], rtol=1e-6)
