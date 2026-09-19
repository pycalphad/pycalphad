import pytest

from pycalphad.io.cs_dat import TokenParser, TokenParserError, parse_header


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
