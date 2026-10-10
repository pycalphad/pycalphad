"""Benchmark provenance contract tests.

Purpose: Hash-bind selected fixtures and validate benchmark evidence metadata.
Created: 2026-10-10.
Authoring agents/models: gpt-5.6-luna (initial draft); gpt-5.6-sol (review).
Rationale: Establish an explicit, auditable public reliability-corpus contract.
Inputs: Packaged test fixtures and referenced pytest evidence functions.
Outputs: Pytest pass/fail results; no side effects.
"""

from dataclasses import FrozenInstanceError, dataclass
import hashlib
import importlib
from importlib.resources import files

import pytest


@dataclass(frozen=True)
class BenchmarkFixture:
    fixture: str
    expected_sha256: str
    model_family: str
    source: str
    oracle_or_reference: str
    evidence_status: str
    distribution_status: str
    claim_boundary: str
    evidence_tests: tuple[str, ...]


MANIFEST = (
    BenchmarkFixture(
        fixture="Shishin_Fe-Sb-O-S_slag.dat",
        expected_sha256="UNBOUND",
        model_family="MQMQA-SUBQ",
        source="doi:10.1016/j.calphad.2019.01.016",
        oracle_or_reference="Thermochimica and FactSage point-energy values",
        evidence_status="qualified-state-points",
        distribution_status="explicit license not recorded",
        claim_boundary=(
            "model energy only at recorded state points; one FactSage comparison "
            "approximates unprinted trace fractions"
        ),
        evidence_tests=(
            "test_energy::test_MQMQA_SUBQ_Q_mixing_1000K",
            "test_energy::test_MQMQA_SUBQ_Q_mixing_1000K_FACTSAGE",
        ),
    ),
    BenchmarkFixture(
        fixture="2026-Dixon-Na-K-Cl-I.dat",
        expected_sha256="UNBOUND",
        model_family="MQMQA-SUBQ-reciprocal",
        source="doi:10.1080/00295639.2026.2653362",
        oracle_or_reference="FactSage 800 K equilibrium Gibbs energy",
        evidence_status="qualified-state-point",
        distribution_status="explicit license not recorded",
        claim_boundary="one reciprocal parser/model/equilibrium state point only",
        evidence_tests=(
            "test_equilibrium::test_MQMQA_reciprocal_interaction_parameter_Dixon",
        ),
    ),
    BenchmarkFixture(
        fixture="TiO-17Yan.tdb",
        expected_sha256="UNBOUND",
        model_family="charged-CEF",
        source="Yang et al., J. Alloys Compd. 722 (2017) 365-374",
        oracle_or_reference="internal charged-stoichiometric equilibrium regression",
        evidence_status="provisional-no-independent-oracle",
        distribution_status="explicit license not recorded",
        claim_boundary="charged stoichiometric equilibrium convergence only",
        evidence_tests=(
            "test_equilibrium::test_charged_stoichiometric_phases_converge",
        ),
    ),
    BenchmarkFixture(
        fixture="alfeo.tdb",
        expected_sha256="UNBOUND",
        model_family="charged-CEF",
        source="fixture attribution: Greta Lindwall, Xuan L. Liu, Zi-Kui Liu (2015)",
        oracle_or_reference="Thermo-Calc equilibrium point values",
        evidence_status="provisional-source-publication-not-recorded",
        distribution_status="explicit license not recorded",
        claim_boundary="charged Al-Fe-O equilibrium point values only",
        evidence_tests=("test_equilibrium::test_eq_charge_alfeo",),
    ),
)


def test_manifest_metadata_is_explicit_and_immutable():
    assert MANIFEST
    for item in MANIFEST:
        assert item.fixture and item.model_family and item.source
        assert item.oracle_or_reference
        assert item.evidence_status and item.distribution_status and item.claim_boundary
        assert item.expected_sha256
        with pytest.raises(FrozenInstanceError):
            setattr(item, "fixture", "mutated")


@pytest.mark.parametrize("item", MANIFEST, ids=lambda item: item.fixture)
def test_fixture_hash_matches_manifest(item):
    resource = files("pycalphad.tests.databases").joinpath(item.fixture)
    digest = hashlib.sha256(resource.read_bytes()).hexdigest()
    assert digest == item.expected_sha256


@pytest.mark.parametrize("item", MANIFEST, ids=lambda item: item.fixture)
def test_manifest_evidence_functions_resolve(item):
    for reference in item.evidence_tests:
        module_name, function_name = reference.split("::", 1)
        module = importlib.import_module(f"pycalphad.tests.{module_name}")
        assert callable(getattr(module, function_name))
