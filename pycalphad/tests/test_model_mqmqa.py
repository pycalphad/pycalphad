"""Solver-independent invariants for the MQMQA model.

Purpose: Verify coordinate, composition, charge, and property invariants of
    ModelMQMQA within explicitly bounded fixture states.
Created: 2026-10-10.
Authoring agent/model: gpt-5.6-sol.
Rationale: Separate model-local correctness checks from parser and equilibrium
    regressions.
Inputs: Packaged MQMQA DAT fixtures and explicit quadruplet populations.
Outputs: Pytest pass/fail results; no side effects.
"""

import numpy as np

from pycalphad import variables as v
from pycalphad.models.model_mqmqa import ModelMQMQA
from pycalphad.tests.fixtures import load_database, select_database


MAX_CHARGE_RESIDUAL_PER_FORMULA_UNIT = 0.0


def _model_states(load_database):
    dbf = load_database()
    model = ModelMQMQA(dbf, ["FE", "SB", "O", "S"], "SLAG-LIQ")
    quadruplet_fractions = tuple(
        model._X_ijkl(*quadruplet) for quadruplet in model._quadruplets
    )
    interior = {
        fraction: 1.0 / len(quadruplet_fractions)
        for fraction in quadruplet_fractions
    }
    interior[v.T] = 1000.0
    return dbf, model, quadruplet_fractions, interior


def _value(expression, substitutions):
    return float(expression.subs(substitutions).n(real=True))


@select_database("Shishin_Fe-Sb-O-S_slag.dat")
def test_mqmqa_coordinate_normalization_and_limits(load_database):
    _, model, quadruplet_fractions, interior = _model_states(load_database)

    constraints = model.get_internal_constraints()
    assert len(constraints) == 1
    np.testing.assert_allclose(_value(constraints[0], interior), 0.0, atol=1e-14)
    np.testing.assert_allclose(
        _value(
            sum(model._X_ik(cation, anion) for cation, anion in model._pairs),
            interior,
        ),
        1.0,
        atol=1e-14,
    )
    for sublattice in (model.cations, model.anions):
        np.testing.assert_allclose(
            _value(sum(model._Y_i(species) for species in sublattice), interior),
            1.0,
            atol=1e-14,
        )

    cation, anion = model.cations[0], model.anions[0]
    pure_quadruplet = model._X_ijkl(cation, cation, anion, anion)
    pure_state = {
        fraction: float(fraction == pure_quadruplet)
        for fraction in quadruplet_fractions
    }
    np.testing.assert_allclose(_value(constraints[0], pure_state), 0.0)
    np.testing.assert_allclose(
        _value(model._X_ik(cation, anion), pure_state), 1.0
    )
    np.testing.assert_allclose(_value(model._Y_i(cation), pure_state), 1.0)
    np.testing.assert_allclose(_value(model._Y_i(anion), pure_state), 1.0)

    other_cation, other_anion = model.cations[1], model.anions[1]
    assert model._X_ijkl(
        cation, other_cation, anion, other_anion
    ) == model._X_ijkl(other_cation, cation, other_anion, anion)


@select_database("Shishin_Fe-Sb-O-S_slag.dat")
def test_mqmqa_element_and_charge_accounting(load_database):
    dbf, model, _, interior = _model_states(load_database)

    np.testing.assert_allclose(
        _value(
            sum(model.moles(element) for element in ("FE", "SB", "O", "S")),
            interior,
        ),
        1.0,
        atol=1e-14,
    )
    charge_per_formula_unit = sum(
        species.charge * model._n_i(dbf, species)
        for species in model.cations + model.anions
    )
    assert abs(_value(charge_per_formula_unit, interior)) <= (
        MAX_CHARGE_RESIDUAL_PER_FORMULA_UNIT
    )


@select_database("Shishin_Fe-Sb-O-S_slag.dat")
def test_mqmqa_thermodynamic_properties_are_finite_at_interior_state(load_database):
    _, model, _, interior = _model_states(load_database)

    properties = {
        name: _value(getattr(model, name), interior)
        for name in ("GM", "SM", "HM", "CPM")
    }
    assert all(np.isfinite(value) for value in properties.values())
    np.testing.assert_allclose(
        properties["HM"],
        properties["GM"] + interior[v.T] * properties["SM"],
        atol=1e-10,
    )
