"""
Test variables module.
"""
import copy
import pickle
import numpy as np
from pycalphad import variables as v
from pycalphad.tests.fixtures import select_database, load_database


def test_species_parse_unicode_strings():
    """Species should properly parse unicode strings."""
    s = v.Species(u"MG")


@select_database("cuo.tdb")
def test_mole_and_mass_fraction_conversions(load_database):
    """Test mole <-> mass conversions work as expected."""
    # Passing database as a mass dict works
    dbf = load_database()
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


def test_component_and_species_repr_str_methods():
    comp = v.Component("O2", {"O": 2})
    assert repr(comp) == "Component('O2', 'O2')"
    assert str(comp) == "O2"

    comp = v.Component("*", {})
    assert repr(comp) == "Component('*')"
    assert str(comp) == "*"

    comp = v.Component(None)
    assert repr(comp) == "Component(None)"
    assert str(comp) == ""

    sp = v.Species("O2-4", {"O": 2}, charge=-4)
    assert repr(sp) == "Species('O2-4', 'O2', charge=-4)"
    assert str(sp) == "O2-4"

    sp = v.Species("*", {})
    assert repr(sp) == "Species('*')"
    assert str(sp) == "*"

    sp = v.Species(None)
    assert repr(sp) == "Species(None)"
    assert str(sp) == ""

def test_deepcopy():
    '''
    Tests that deepcopy of variables produce the same variables
    This addresses an unreported issue where copying the chemical potential
    would use the name attribute rather than the species (this resulted in deepcopy(v.MU('A')) -> v.MU(v.MU('A')))
    '''
    assert copy.deepcopy(v.NP('*')) == v.NP('*')
    assert copy.deepcopy(v.NP('A')) == v.NP('A')

    assert copy.deepcopy(v.X('B')) == v.X('B')
    assert copy.deepcopy(v.X('A','B')) == v.X('A','B')

    assert copy.deepcopy(v.W('B')) == v.W('B')
    assert copy.deepcopy(v.W('A','B')) == v.W('A','B')

    assert copy.deepcopy(v.Y('A',0,'B')) == v.Y('A',0,'B')
    assert copy.deepcopy(v.T) == v.T
    assert copy.deepcopy(v.P) == v.P
    assert copy.deepcopy(v.MU('A')) == v.MU('A')


def test_copy_with_cached_new():
    """Test that copy.copy creates independent objects even with cached __new__.

    The @lru_cache on StateVariable.__new__ could cause copy.copy() to return
    the same cached object instead of a new one. This test verifies that the
    __copy__ method correctly bypasses the cache.
    """
    # Test singleton types (TemperatureType, PressureType, SystemMolesType)
    # These have __reduce__ returning (cls, ()) so are most susceptible to cache issues
    t_copy = copy.copy(v.T)
    assert t_copy is not v.T, "copy of T should be a different object"
    assert t_copy == v.T, "copy of T should be equal to original"

    p_copy = copy.copy(v.P)
    assert p_copy is not v.P, "copy of P should be a different object"

    n_copy = copy.copy(v.N)
    assert n_copy is not v.N, "copy of N should be a different object"

    # Test that modifying copy doesn't affect original
    original_units = v.T.display_units
    t_copy.display_units = 'degC'
    assert v.T.display_units == original_units, "modifying copy should not affect original"

    # Test other StateVariable subclasses
    x = v.X('AL')
    x_copy = copy.copy(x)
    assert x_copy is not x, "copy of MoleFraction should be a different object"
    assert x_copy == x, "copy of MoleFraction should be equal to original"

    y = v.Y('FCC', 0, 'AL')
    y_copy = copy.copy(y)
    assert y_copy is not y, "copy of SiteFraction should be a different object"
    assert y_copy == y, "copy of SiteFraction should be equal to original"


def test_moles_construction_and_dispatch():
    """Moles() dispatches to the total SystemMolesType; Moles(species) and
    Moles(phase, species) build component and phase-local amounts."""
    # Total moles: Moles() returns the SystemMolesType singleton (== v.N)
    assert v.Moles() is v.N
    assert isinstance(v.N, v.SystemMolesType)
    assert isinstance(v.N, v.Moles)
    assert str(v.N) == 'N'
    # Total moles deliberately exposes no species/phase_name attributes (duck-typing)
    assert getattr(v.N, 'species', None) is None
    assert getattr(v.N, 'phase_name', None) is None
    assert not hasattr(v.N, 'species')

    # Component moles
    n_al = v.Moles('AL')
    assert str(n_al) == 'N_AL'
    assert n_al.species == v.Component('AL')
    assert n_al.phase_name is None
    assert isinstance(n_al, v.Moles)
    assert not isinstance(n_al, v.SystemMolesType)

    # Phase-local moles: valid to construct (not yet a solver condition)
    n_fcc_al = v.Moles('FCC_A1', 'AL')
    assert str(n_fcc_al) == 'N_FCC_A1_AL'
    assert n_fcc_al.phase_name == 'FCC_A1'
    assert n_fcc_al.species == v.Component('AL')
    assert isinstance(n_fcc_al, v.Moles)

    # Singleton/caching behavior
    assert v.Moles('AL') is v.Moles('AL')


def test_moles_copy_and_reduce():
    """Moles instances survive copy.copy (cache-bypassing __new__) and pickling-style reduce."""
    n_al = v.Moles('AL')
    n_al_copy = copy.copy(n_al)
    assert n_al_copy is not n_al
    assert n_al_copy == n_al
    assert copy.deepcopy(n_al) == n_al
    assert copy.deepcopy(v.Moles('FCC_A1', 'AL')) == v.Moles('FCC_A1', 'AL')

    # Total moles copy
    n_copy = copy.copy(v.N)
    assert n_copy is not v.N
    assert n_copy == v.N

    # Unit variant via __getitem__ -> __copy__ must not raise or mutate original
    n_al_mol = v.Moles('AL')['mol']
    assert n_al_mol.display_units == 'mol'
    assert n_al_mol == n_al


def test_getitem_units_does_not_mutate_original():
    """Test that using __getitem__ for unit conversion doesn't mutate the original.

    v.T['degC'] should return a new object with different display_units,
    leaving v.T unchanged.
    """
    original_units = v.T.display_units

    t_celsius = v.T['degC']
    assert t_celsius.display_units == 'degC'
    assert v.T.display_units == original_units, "v.T should not be mutated by __getitem__"

    # Verify they are different objects
    assert t_celsius is not v.T

    # But they should be equal (same underlying symbol)
    assert t_celsius == v.T


def test_issue557_state_variable_unit_changes_survive_roundtrip_pickle():
    T2 = copy.deepcopy(v.T)
    T2.display_units = 'degC'
    T2_roundtrip = pickle.loads(pickle.dumps(T2))
    print(f'T2: {T2.display_units}')
    print(f'T2_roundtrip: {T2_roundtrip.display_units}')
    assert T2.display_units == T2_roundtrip.display_units
    T3 = pickle.loads(pickle.dumps(v.T["degC"]))
    assert T2.display_units == T3.display_units