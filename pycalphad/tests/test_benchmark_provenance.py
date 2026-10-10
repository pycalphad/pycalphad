"""Benchmark provenance contract tests.

Purpose: Canonically hash-bind selected text fixtures and validate benchmark
    evidence metadata across platform-specific Git checkouts.
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
import re

import pytest


@dataclass(frozen=True)
class BenchmarkFixture:
    fixture: str
    expected_canonical_sha256: str
    model_family: str
    source: str
    oracle_or_reference: str
    evidence_status: str
    distribution_status: str
    claim_boundary: str
    evidence_tests: tuple[str, ...]


EVIDENCE_STATUSES = frozenset(
    {
        "qualified-state-point",
        "qualified-state-points",
        "provisional-no-independent-oracle",
        "provisional-source-publication-not-recorded",
    }
)
DISTRIBUTION_STATUSES = frozenset({"explicit license not recorded"})
EXPECTED_CLASSIFICATIONS = {
    "Shishin_Fe-Sb-O-S_slag.dat": (
        "qualified-state-points",
        "explicit license not recorded",
    ),
    "2026-Dixon-Na-K-Cl-I.dat": (
        "qualified-state-point",
        "explicit license not recorded",
    ),
    "TiO-17Yan.tdb": (
        "provisional-no-independent-oracle",
        "explicit license not recorded",
    ),
    "alfeo.tdb": (
        "provisional-source-publication-not-recorded",
        "explicit license not recorded",
    ),
}


MANIFEST = (
    BenchmarkFixture(
        fixture="Shishin_Fe-Sb-O-S_slag.dat",
        expected_canonical_sha256=(
            "d17d6f34bf704b57762422265c2119b873a2fd4fc28077c1c80898152718ec60"
        ),
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
        expected_canonical_sha256=(
            "83e1b1c067b43058ae3a06f21238fb2304ea495fc9d19723381888279e86e075"
        ),
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
        expected_canonical_sha256=(
            "f79af7f26633b1d664d2a44a87efea098b4ae288e3ac3eb1676c1a996cd08647"
        ),
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
        expected_canonical_sha256=(
            "bb4e3a57aa59b9149529f5851689ba2106e965b99bd853c6d959b0e1bf5c2119"
        ),
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
    assert {item.fixture for item in MANIFEST} == set(EXPECTED_CLASSIFICATIONS)
    for item in MANIFEST:
        assert item.fixture and item.model_family and item.source
        assert item.oracle_or_reference
        assert item.evidence_status in EVIDENCE_STATUSES
        assert item.distribution_status in DISTRIBUTION_STATUSES
        assert (
            item.evidence_status,
            item.distribution_status,
        ) == EXPECTED_CLASSIFICATIONS[item.fixture]
        assert item.claim_boundary
        assert re.fullmatch(
            r"[0-9a-f]{64}", item.expected_canonical_sha256
        )
        with pytest.raises(FrozenInstanceError):
            setattr(item, "fixture", "mutated")


def _canonical_text_sha256(content: bytes) -> str:
    """Hash text bytes after normalizing Git checkout line endings to LF."""
    canonical_content = content.replace(b"\r\n", b"\n")
    return hashlib.sha256(canonical_content).hexdigest()


def test_canonical_text_hash_is_line_ending_independent():
    lf_content = b"line one\nline two\n"
    crlf_content = b"line one\r\nline two\r\n"
    assert _canonical_text_sha256(lf_content) == _canonical_text_sha256(
        crlf_content
    )


@pytest.mark.parametrize("item", MANIFEST, ids=lambda item: item.fixture)
def test_fixture_canonical_hash_matches_manifest(item):
    resource = files("pycalphad.tests.databases").joinpath(item.fixture)
    digest = _canonical_text_sha256(resource.read_bytes())
    assert digest == item.expected_canonical_sha256


@pytest.mark.parametrize("item", MANIFEST, ids=lambda item: item.fixture)
def test_manifest_evidence_functions_resolve(item):
    for reference in item.evidence_tests:
        module_name, function_name = reference.split("::", 1)
        module = importlib.import_module(f"pycalphad.tests.{module_name}")
        test_function = getattr(module, function_name)
        assert callable(test_function)

        database_marks = tuple(
            mark
            for mark in getattr(test_function, "pytestmark", ())
            if mark.name == "parametrize"
            and mark.args[0] == "load_database"
            and mark.kwargs.get("indirect") is True
        )
        assert len(database_marks) == 1
        assert tuple(database_marks[0].args[1]) == (item.fixture,)
