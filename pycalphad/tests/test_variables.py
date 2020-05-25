"""
Test variables module.
"""

import numpy as np
from pycalphad import Database
from pycalphad import variables as v
from .datasets import CUO_TDB


def test_species_parse_unicode_strings():
    """Species should properly parse unicode strings."""
    s = v.Species(u"MG")


def test_mole_and_mass_fraction_conversions():
    """Test mole <-> mass conversions work as expected."""
    # Passing database as a mass dict works
    dbf = Database(CUO_TDB)
    mole_fracs = {v.X('O'): 0.5}
    mass_fracs = v.get_mass_fractions(mole_fracs, v.Species('CU'), dbf)
    assert np.isclose(mass_fracs[v.W('O')], 0.20113144)  # TC
    # Conversion back works
    round_trip_mole_fracs = v.get_mole_fractions(mass_fracs, 'CU', dbf)
    assert all(np.isclose(round_trip_mole_fracs[mf], mole_fracs[mf]) for mf in round_trip_mole_fracs.keys())

    # Using Thermo-Calc's define components to define Al2O3 and TiO2
    # Mass dict defined by hand
    md = {'AL': 26.982, 'TI': 47.88, 'O': 15.999}
    alumina = v.Species('AL2O3')
    mass_fracs = {v.W(alumina): 0.81, v.W("TIO2"): 0.13}
    mole_fracs = v.get_mole_fractions(mass_fracs, 'O', md)
    assert np.isclose(mole_fracs[v.X('AL2O3')], 0.59632604)  # TC
    assert np.isclose(mole_fracs[v.X('TIO2')], 0.12216562)  # TC
    # Conversion back works
    round_trip_mass_fracs = v.get_mass_fractions(mole_fracs, v.Species('O'), md)
    assert all(np.isclose(round_trip_mass_fracs[mf], mass_fracs[mf]) for mf in round_trip_mass_fracs.keys())
