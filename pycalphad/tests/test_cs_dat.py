"""Tests for the ChemSage DAT parser.

Purpose: Verify DAT syntax handling and selected parser semantic contracts.
Created: Pre-existing; provenance header added 2026-10-10.
Authoring agent/model: Original authors unknown; header and MQMQA semantic
    contracts added by gpt-5.6-sol.
Rationale: Keep parser-level meaning independently testable from model and solver
    behavior.
Inputs: Synthetic token streams and packaged ChemSage DAT fixtures.
Outputs: Pytest pass/fail results; no side effects.
"""

import math

import pytest

from pycalphad import variables as v
from pycalphad.io.cs_dat import TokenParser, TokenParserError, parse_header
from pycalphad.tests.fixtures import load_database, select_database


EXPECTED_DIXON_RECIPROCAL_ENERGY = "UNBOUND"
EXPECTED_SWITCHED_COORDINATIONS = "UNBOUND"


def _find_parameter(dbf, phase_name, parameter_type, constituent_array):
    matches = [
        parameter
        for parameter in dbf._parameters.all()
        if parameter["phase_name"] == phase_name
        and parameter["parameter_type"] == parameter_type
        and parameter["constituent_array"] == constituent_array
    ]
    assert len(matches) == 1
    return matches[0]


@pytest.mark.parametrize(
    ("extras", "blank_lines"),
    [("", ""), ("  7  8", ""), ("\t7   8  9   ", "\n  \t")],
)
def test_parse_header_discards_only_second_line_extra_tokens(extras, blank_lines):
    toks = TokenParser(
        "title\n"
        f"2 1 1 0{extras}\n"
        f"{blank_lines}AL NI\n"
        "26.9815 58.6934\n"
        "2 1 2\n"
        "2 3 4\n"
    )

    assert toks.parse(str) == "title"
    header = parse_header(toks)

    assert header.list_soln_species_count == [1]
    assert header.num_stoich_phases == 0
    assert header.pure_elements == ["AL", "NI"]
    assert header.pure_elements_mass == [26.9815, 58.6934]
    assert header.gibbs_coefficient_idxs == [1, 2]
    assert header.excess_coefficient_idxs == [3, 4]


def test_parse_header_extras_do_not_hide_later_malformed_line():
    toks = TokenParser("title\n1 1 1 0 7 8\nAL\nnot-a-mass\n")

    assert toks.parse(str) == "title"
    with pytest.raises(TokenParserError, match="not-a-mass"):
        parse_header(toks)


@select_database("2026-Dixon-Na-K-Cl-I.dat")
def test_subq_parser_preserves_reciprocal_parameter_semantics(load_database):
    dbf = load_database()
    phase = dbf.phases["MSCL"]
    species = {item.name: item for item in dbf.species}
    na = species["NA+1+1.0"]
    k = species["K+1+1.0"]
    cl = species["CL-1.0"]
    iodine = species["I-1.0"]

    assert phase.model_hints["mqmqa"]["type"] == "SUBQ"
    assert phase.constituents == (frozenset({na, k}), frozenset({cl, iodine}))
    assert {
        item.name: (item.charge, item.constituents)
        for item in (na, k, cl, iodine)
    } == {
        "NA+1+1.0": (1.0, {"NA": 1.0}),
        "K+1+1.0": (1.0, {"K": 1.0}),
        "CL-1.0": (-1.0, {"CL": 1.0}),
        "I-1.0": (-1.0, {"I": 1.0}),
    }

    kcl = _find_parameter(dbf, "MSCL", "MQMG", ((k,), (cl,)))
    assert kcl["zeta"] == 6.0
    assert kcl["stoichiometry"] == [1.0, 1.0, 0.0, 0.0, 0.0]
    for temperature in (400.0, 1200.0, 2400.0):
        expected = (
            -443767.726
            + 406.397574 * temperature
            - 73.59656 * temperature * math.log(temperature)
        )
        assert float(kcl["parameter"].subs({v.T: temperature})) == pytest.approx(
            expected
        )

    reciprocal = _find_parameter(
        dbf, "MSCL", "MQMX", ((na, k), (cl, iodine))
    )
    assert reciprocal["mixing_code"] == "R"
    assert reciprocal["exponents"] == [1, 0, 0, 0]
    assert reciprocal["parameter"] == EXPECTED_DIXON_RECIPROCAL_ENERGY
    assert reciprocal["additional_mixing_constituent"] == [na, cl]
    assert reciprocal["additional_mixing_exponent"] == [1, 0, 0, 0]


@select_database("KF-NIF2_switched.dat")
def test_subg_parser_canonically_reorders_coordination_numbers(load_database):
    dbf = load_database()
    phase = dbf.phases["LIQUID2"]
    species = {item.name: item for item in dbf.species}
    k = species["K+1.0"]
    nickel = species["NI+2.0"]
    fluorine = species["F-1.0"]

    assert phase.model_hints["mqmqa"]["type"] == "SUBG"
    assert phase.constituents == (
        frozenset({k, nickel}),
        frozenset({fluorine}),
    )
    assert {
        item.name: (item.charge, item.constituents)
        for item in (k, nickel, fluorine)
    } == {
        "K+1.0": (1.0, {"K": 1.0}),
        "NI+2.0": (2.0, {"NI": 1.0}),
        "F-1.0": (-1.0, {"F": 1.0}),
    }

    mixed_coordination = _find_parameter(
        dbf,
        "LIQUID2",
        "MQMZ",
        ((k, nickel), (fluorine, fluorine)),
    )
    assert mixed_coordination["coordinations"] == EXPECTED_SWITCHED_COORDINATIONS
