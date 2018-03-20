"""
Test variables module.
"""

from pycalphad import variables as v


def test_species_parse_unicode_strings():
    """Species should properly parse unicode strings."""
    s = v.Species(u"MG")
