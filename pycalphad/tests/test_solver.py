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
    # FCC_A1 in this database reduces to a single site fraction for pure AL
    compset.update(np.array([1.0]), 1.0, np.array([conditions[v.N], conditions[v.P], conditions[v.T]]))

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
