"""Unit tests for internal Solver class"""

from pycalphad.core.solver import Solver
from pycalphad import Model
from pycalphad.core.composition_set import CompositionSet
import pycalphad.variables as v
from pycalphad.tests.fixtures import select_database, load_database
from pycalphad.codegen.phase_record_factory import PhaseRecordFactory
from pycalphad.core.utils import instantiate_models
from pycalphad.property_framework import ModelComputedProperty

from collections import OrderedDict

import pytest
import numpy as np

# Mark every test in this module as a solver test
pytestmark = [pytest.mark.solver,]

@select_database("alzn_mey.tdb")
def test_non_unity_N_conditions(load_database):
    dbf = load_database()
    components = ["AL", "VA"]
    phase_name = "FCC_A1"
    phases = [phase_name]
    conditions = {v.N: 2.0, v.P: 1.0e5, v.T: 300.0}
    str_conditions = OrderedDict(N=conditions[v.N], P=conditions[v.P], T=conditions[v.T])

    models = instantiate_models(dbf, components, phases)
    prf = PhaseRecordFactory(dbf, components, conditions, models)

    compset = CompositionSet(prf[phase_name])
    # FCC_A1 in this database reduces to a single site fraction for pure AL.
    compset.update(np.array([1.0]), 1.0, np.array([conditions[v.P], conditions[v.T]]))

    solver = Solver()
    result = solver.solve([compset], str_conditions)

    expected_GM = -8496.605669599447  # computed by hand
    GM = ModelComputedProperty('GM').compute_property([compset], str_conditions, result.chemical_potentials)
    G = ModelComputedProperty('G').compute_property([compset], str_conditions, result.chemical_potentials)

    print(GM, G)
    print(f"NP: {compset.NP}")
    np.testing.assert_allclose(result.chemical_potentials, expected_GM, atol=1e-5)
    np.testing.assert_allclose(compset.NP, conditions[v.N])  # assumes 1 mole of f.u. per 1 mole atoms
    np.testing.assert_allclose(GM, expected_GM, atol=1e-5)
    np.testing.assert_allclose(G, GM * conditions[v.N])


@select_database("alzn_mey.tdb")
def test_per_component_moles_conditions(load_database):
    """Per-component amount conditions N(i) give the same equilibrium as X + total N.

    For a single FCC_A1 (AL, ZN) phase at fixed (P, T), the condition set
    {N(AL)=0.3, N(ZN)=0.7} (two extensive amounts, no X and no total N) must
    converge to the same intensive state as {X(AL)=0.3, N=1}.
    """
    dbf = load_database()
    components = ["AL", "ZN", "VA"]
    phase_name = "FCC_A1"
    phases = [phase_name]
    models = instantiate_models(dbf, components, phases)
    prf = PhaseRecordFactory(dbf, components, {v.P, v.T}, models)

    def solve(conditions):
        compset = CompositionSet(prf[phase_name])
        # Initial guess: site fractions [Y(AL), Y(ZN)]; state variables [P, T]
        compset.update(np.array([0.3, 0.7]), 1.0,
                       np.array([conditions[v.P], conditions[v.T]]))
        result = Solver().solve([compset], conditions)
        assert result.converged
        return compset, result

    ref_conds = {v.X("AL"): 0.3, v.N: 1.0, v.P: 1.0e5, v.T: 600.0}
    amt_conds = {v.Moles("AL"): 0.3, v.Moles("ZN"): 0.7, v.P: 1.0e5, v.T: 600.0}

    cs_ref, res_ref = solve(ref_conds)
    cs_amt, res_amt = solve(amt_conds)

    al_idx = cs_amt.phase_record.nonvacant_elements.index("AL")
    # Same intensive state (chemical potentials and composition)
    np.testing.assert_allclose(res_amt.chemical_potentials, res_ref.chemical_potentials, atol=1e-5)
    np.testing.assert_allclose(np.asarray(cs_amt.X), np.asarray(cs_ref.X), atol=1e-6)
    np.testing.assert_allclose(cs_amt.X[al_idx], 0.3, atol=1e-5)
    # Total system amount N = sum_i NP_i * sum(X_i) = N(AL) + N(ZN) = 1.0
    np.testing.assert_allclose(cs_amt.NP * np.sum(cs_amt.X), 1.0, atol=1e-6)
