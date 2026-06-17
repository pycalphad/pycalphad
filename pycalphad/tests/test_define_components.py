"""Tests for the redefined-component-basis ("define components") feature.

A component basis lets conditions and outputs be expressed in terms of components
that are linear combinations of the non-vacant pure elements (e.g. AL2O3, SIO2, O2).
The change-of-basis matrix S (components x non-vacant elements) is built and validated
in PhaseRecordFactory; for a pure-element component list S is the identity ("trivial"
basis) and behavior is unchanged.
"""
import numpy as np
import pytest
from pycalphad import Workspace, variables as v
from pycalphad.core.errors import ConditionError
from pycalphad.tests.fixtures import select_database, load_database


# ---------------------------------------------------------------------------
# Phase 1: change-of-basis matrix construction, trivial detection, validation
# ---------------------------------------------------------------------------

@select_database("alzn_mey.tdb")
def test_trivial_basis_is_identity(load_database):
    """A pure-element component list yields the identity basis (trivial)."""
    dbf = load_database()
    wks = Workspace(dbf, ['AL', 'ZN', 'VA'], ['FCC_A1'])
    prf = wks.phase_record_factory
    assert prf.nonvacant_elements == ['AL', 'ZN']
    assert prf.basis_is_trivial is True
    np.testing.assert_array_equal(prf.component_basis, np.eye(2))
    np.testing.assert_array_equal(prf.component_basis_inv_T, np.eye(2))


@select_database("alzn_mey.tdb")
def test_synthetic_binary_basis_matrix(load_database):
    """A synthetic molecular component basis builds the expected non-identity S."""
    dbf = load_database()
    # ALZN = {AL:1, ZN:1}; basis components sorted by name -> [ALZN, ZN] over [AL, ZN]
    wks = Workspace(dbf, ['ALZN', 'ZN', 'VA'], ['FCC_A1'])
    prf = wks.phase_record_factory
    assert prf.nonvacant_elements == ['AL', 'ZN']
    assert [str(c) for c in prf.basis_components] == ['ALZN', 'ZN']
    expected_S = np.array([[1.0, 1.0],
                           [0.0, 1.0]])
    np.testing.assert_array_equal(prf.component_basis, expected_S)
    np.testing.assert_allclose(prf.component_basis_inv_T, np.linalg.inv(expected_S.T))
    assert prf.basis_is_trivial is False
    # component molar masses: MW(ALZN) = m_AL + m_ZN; MW(ZN) = m_ZN
    m_al, m_zn = prf.molar_masses  # sorted [AL, ZN]
    np.testing.assert_allclose(prf.component_molar_masses, [m_al + m_zn, m_zn])


@select_database("alcrni.tdb")
def test_synthetic_ternary_basis_matrix(load_database):
    """A 3x3 synthetic basis is built and inverted."""
    dbf = load_database()
    wks = Workspace(dbf, ['ALCR', 'CRNI', 'NI', 'VA'], ['FCC_A1'])
    prf = wks.phase_record_factory
    assert prf.nonvacant_elements == ['AL', 'CR', 'NI']
    assert [str(c) for c in prf.basis_components] == ['ALCR', 'CRNI', 'NI']
    expected_S = np.array([[1.0, 1.0, 0.0],
                           [0.0, 1.0, 1.0],
                           [0.0, 0.0, 1.0]])
    np.testing.assert_array_equal(prf.component_basis, expected_S)
    np.testing.assert_allclose(prf.component_basis_inv_T, np.linalg.inv(expected_S.T))
    assert prf.basis_is_trivial is False


@select_database("alcrni.tdb")
def test_fractional_basis_matrix(load_database):
    """Components with stoichiometric coefficients > 1 and shared elements (oxide-like)
    build the expected S. Uses a metallic database so the model builds cleanly; the
    matrix math is independent of the phase."""
    dbf = load_database()
    # AL2CR={AL:2,CR:1}, CRNI3={CR:1,NI:3}, NI={NI:1}
    wks = Workspace(dbf, ['AL2CR', 'CRNI3', 'NI', 'VA'], ['FCC_A1'])
    prf = wks.phase_record_factory
    assert prf.nonvacant_elements == ['AL', 'CR', 'NI']
    assert [str(c) for c in prf.basis_components] == ['AL2CR', 'CRNI3', 'NI']
    # rows: AL2CR, CRNI3, NI ; cols: AL, CR, NI
    expected_S = np.array([[2.0, 1.0, 0.0],
                           [0.0, 1.0, 3.0],
                           [0.0, 0.0, 1.0]])
    np.testing.assert_array_equal(prf.component_basis, expected_S)
    np.testing.assert_allclose(prf.component_basis_inv_T, np.linalg.inv(expected_S.T))
    assert prf.basis_is_trivial is False


@select_database("alcrni.tdb")
def test_incomplete_basis_raises(load_database):
    """Too few components to span the element space raises immediately, before solving."""
    dbf = load_database()
    # ALCR spans {AL, CR} but is a single component -> 1 < 2
    with pytest.raises(ConditionError, match="incomplete"):
        Workspace(dbf, ['ALCR', 'VA'], ['FCC_A1'])


@select_database("alzn_mey.tdb")
def test_overdetermined_components_fall_back_to_element_basis(load_database):
    """An over-determined component list (more distinct components than elements) is a
    normal calculation, not a redefined basis, and falls back to the trivial element
    basis without error. This is the same shape as ionic/expanded-species lists from
    `calculate`, which must not be rejected."""
    dbf = load_database()
    wks = Workspace(dbf, ['AL', 'ZN', 'ALZN', 'VA'], ['FCC_A1'])
    prf = wks.phase_record_factory
    assert prf.basis_is_trivial is True
    np.testing.assert_array_equal(prf.component_basis, np.eye(2))


@select_database("alzn_mey.tdb")
def test_linearly_dependent_basis_raises(load_database):
    """A square but singular basis (linearly dependent components) raises."""
    dbf = load_database()
    # AL2ZN2 = 2*ALZN -> rows are proportional -> rank 1 < 2
    with pytest.raises(ConditionError, match="linearly dependent"):
        Workspace(dbf, ['ALZN', 'AL2ZN2', 'VA'], ['FCC_A1'])


# ---------------------------------------------------------------------------
# Phase 2: property outputs (MU(component) forward S; non-basis output raises)
# ---------------------------------------------------------------------------

@select_database("alcrni.tdb")
def test_mu_component_output_forward_S(load_database):
    """MU(component) output sums element chemical potentials over constituents and works
    for components that are NOT part of the (here trivial, pure-element) basis."""
    dbf = load_database()
    wks = Workspace(dbf, ['AL', 'CR', 'NI', 'VA'], ['FCC_A1', 'BCC_A2', 'LIQUID'],
                    {v.T: 1200, v.P: 1e5, v.N: 1, v.X('AL'): 0.2, v.X('CR'): 0.3})
    mu_al = float(np.squeeze(wks.get('MU(AL)')))
    mu_cr = float(np.squeeze(wks.get('MU(CR)')))
    np.testing.assert_allclose(float(np.squeeze(wks.get('MU(ALCR)'))), mu_al + mu_cr, atol=1e-6)
    np.testing.assert_allclose(float(np.squeeze(wks.get('MU(AL2CR)'))), 2 * mu_al + mu_cr, atol=1e-6)


@select_database("alzn_mey.tdb")
def test_component_amount_output_not_in_basis_raises(load_database):
    """N/X/W of a multi-element component that is not part of the basis is ill-defined
    and raises a clear error (here the basis is the pure elements)."""
    dbf = load_database()
    wks = Workspace(dbf, ['AL', 'ZN', 'VA'], ['FCC_A1'],
                    {v.T: 600, v.P: 1e5, v.N: 1, v.X('ZN'): 0.3})
    with pytest.raises(ConditionError, match="not part of the calculation's component basis"):
        wks.get('X(ALZN)')


# ---------------------------------------------------------------------------
# Phase 3: amount and fraction conditions in the component basis (equivalence)
# ---------------------------------------------------------------------------

def _assert_props_match(wks_a, wks_b, props, atol=1e-6):
    for prop in props:
        np.testing.assert_allclose(
            np.squeeze(np.asarray(wks_a.get(prop), dtype=float)),
            np.squeeze(np.asarray(wks_b.get(prop), dtype=float)),
            atol=atol, err_msg=f"property {prop} differs between component and element bases")


# Collision-free synthetic bases (component names differ from element names, as real
# oxide/fluoride bases do), so element outputs are unambiguous.
#   binary  [AL2ZN, ALZN2] over [AL, ZN]:  n_AL = 2a+b, n_ZN = a+2b
#   ternary [AL2CR, CRNI2, NI3] over [AL, CR, NI]:  n_AL=2p, n_CR=p+q, n_NI=2q+3r

@select_database("alzn_mey.tdb")
def test_binary_amount_conditions_equivalence(load_database):
    """N(component) conditions reproduce the equivalent pure-element equilibrium."""
    dbf = load_database()
    phases = ['FCC_A1', 'HCP_A3', 'LIQUID']
    base = {v.T: 600.0, v.P: 1e5}
    a, b = 0.2, 0.1  # n_AL = 2a+b = 0.5, n_ZN = a+2b = 0.4
    wks_comp = Workspace(dbf, ['AL2ZN', 'ALZN2', 'VA'], phases,
                         {**base, v.Moles('AL2ZN'): a, v.Moles('ALZN2'): b})
    wks_elem = Workspace(dbf, ['AL', 'ZN', 'VA'], phases,
                         {**base, v.Moles('AL'): 2 * a + b, v.Moles('ZN'): a + 2 * b})
    _assert_props_match(wks_comp, wks_elem, ['GM', 'MU(AL)', 'MU(ZN)', 'X(AL)', 'X(ZN)', v.N])
    # component conditions satisfied (outputs use (S^T)^-1)
    np.testing.assert_allclose(np.squeeze(wks_comp.get(v.Moles('AL2ZN'))), a, atol=1e-6)
    np.testing.assert_allclose(np.squeeze(wks_comp.get(v.Moles('ALZN2'))), b, atol=1e-6)


@select_database("alcrni.tdb")
def test_ternary_amount_conditions_equivalence(load_database):
    """3x3 basis [AL2CR, CRNI2, NI3]: n_AL=2p, n_CR=p+q, n_NI=2q+3r."""
    dbf = load_database()
    phases = ['FCC_A1', 'BCC_A2', 'LIQUID']
    base = {v.T: 1200.0, v.P: 1e5}
    p, q, r = 0.1, 0.2, 0.1  # n_AL=0.2, n_CR=0.3, n_NI=0.7
    wks_comp = Workspace(dbf, ['AL2CR', 'CRNI2', 'NI3', 'VA'], phases,
                         {**base, v.Moles('AL2CR'): p, v.Moles('CRNI2'): q, v.Moles('NI3'): r})
    wks_elem = Workspace(dbf, ['AL', 'CR', 'NI', 'VA'], phases,
                         {**base, v.Moles('AL'): 2 * p, v.Moles('CR'): p + q, v.Moles('NI'): 2 * q + 3 * r})
    _assert_props_match(wks_comp, wks_elem,
                        ['GM', 'MU(AL)', 'MU(CR)', 'MU(NI)', 'X(AL)', 'X(CR)', 'X(NI)', v.N])


@select_database("alzn_mey.tdb")
def test_binary_mole_fraction_condition_roundtrip(load_database):
    """X(component) (homogeneous form) round-trips and produces an equilibrium reproducible
    from its resulting element composition. (X(component)=n_c/sum n_c' is not X(element).)"""
    dbf = load_database()
    phases = ['FCC_A1', 'HCP_A3', 'LIQUID']
    base = {v.T: 600.0, v.P: 1e5, v.N: 1.0}
    k = 0.3
    wks_comp = Workspace(dbf, ['AL2ZN', 'ALZN2', 'VA'], phases, {**base, v.X('AL2ZN'): k})
    np.testing.assert_allclose(np.squeeze(wks_comp.get('X(AL2ZN)')), k, atol=1e-6)
    x_al = float(np.squeeze(wks_comp.get('X(AL)')))
    wks_elem = Workspace(dbf, ['AL', 'ZN', 'VA'], phases, {**base, v.X('AL'): x_al})
    _assert_props_match(wks_comp, wks_elem, ['GM', 'MU(AL)', 'MU(ZN)', 'X(ZN)'])


@select_database("alzn_mey.tdb")
def test_binary_mass_fraction_condition_roundtrip(load_database):
    """W(component) condition round-trips and yields an equilibrium reproducible from the
    resulting element composition (so the homogeneous mass-fraction constraint is correct)."""
    dbf = load_database()
    phases = ['FCC_A1', 'HCP_A3', 'LIQUID']
    base = {v.T: 600.0, v.P: 1e5, v.N: 1.0}
    k = 0.35
    wks_comp = Workspace(dbf, ['AL2ZN', 'ALZN2', 'VA'], phases, {**base, v.W('AL2ZN'): k})
    np.testing.assert_allclose(np.squeeze(wks_comp.get('W(AL2ZN)')), k, atol=1e-6)
    x_al = float(np.squeeze(wks_comp.get('X(AL)')))
    wks_elem = Workspace(dbf, ['AL', 'ZN', 'VA'], phases, {**base, v.X('AL'): x_al})
    _assert_props_match(wks_comp, wks_elem, ['GM', 'MU(AL)', 'MU(ZN)', 'X(ZN)'])


@select_database("alzn_mey.tdb")
def test_component_name_shadows_element(load_database):
    """When a basis component is named like an element (e.g. ZN in [ALZN, ZN]), N/X of that
    name resolve to the *component*, not the element."""
    dbf = load_database()
    phases = ['FCC_A1', 'HCP_A3', 'LIQUID']
    # [ALZN, ZN]: N(ALZN)=0.3, N(ZN_component)=0.4 -> n_AL=0.3, n_ZN=0.7
    wks = Workspace(dbf, ['ALZN', 'ZN', 'VA'], phases,
                    {v.T: 600.0, v.P: 1e5, v.Moles('ALZN'): 0.3, v.Moles('ZN'): 0.4})
    np.testing.assert_allclose(np.squeeze(wks.get(v.Moles('ZN'))), 0.4, atol=1e-6)  # component
    np.testing.assert_allclose(np.squeeze(wks.get(v.Moles('ALZN'))), 0.3, atol=1e-6)
    np.testing.assert_allclose(np.squeeze(wks.get(v.N)), 1.0, atol=1e-6)  # total atoms 0.3+0.7


@select_database("alzn_mey.tdb")
def test_element_condition_under_component_basis_raises(load_database):
    """No mixing: a pure-element condition is rejected under a component basis."""
    dbf = load_database()
    with pytest.raises(ConditionError):
        Workspace(dbf, ['AL2ZN', 'ALZN2', 'VA'], ['FCC_A1'],
                  {v.T: 600.0, v.P: 1e5, v.X('AL'): 0.3, v.N: 1.0})


# ---------------------------------------------------------------------------
# Phase 4: linear combination conditions over basis components
# ---------------------------------------------------------------------------

@select_database("alzn_mey.tdb")
def test_linear_combination_condition_over_components(load_database):
    """A linear combination of component mole fractions is satisfied at equilibrium."""
    dbf = load_database()
    phases = ['FCC_A1', 'HCP_A3', 'LIQUID']
    base = {v.T: 600.0, v.P: 1e5, v.N: 1.0}
    expr = 0.5 * v.X('AL2ZN') - 7 * v.X('ALZN2')
    wks = Workspace(dbf, ['AL2ZN', 'ALZN2', 'VA'], phases, {**base, expr: 0.1})
    result = 0.5 * float(np.squeeze(wks.get('X(AL2ZN)'))) - 7 * float(np.squeeze(wks.get('X(ALZN2)')))
    np.testing.assert_allclose(result, 0.1, atol=1e-6)
    # the same expression evaluated as an output property agrees
    np.testing.assert_allclose(np.squeeze(wks.get(expr)), 0.1, atol=1e-6)


@select_database("alzn_mey.tdb")
def test_linear_combination_ratio_over_components(load_database):
    """A molar ratio of component mole fractions is satisfied at equilibrium."""
    dbf = load_database()
    phases = ['FCC_A1', 'HCP_A3', 'LIQUID']
    base = {v.T: 600.0, v.P: 1e5, v.N: 1.0}
    target = 2.5
    wks = Workspace(dbf, ['AL2ZN', 'ALZN2', 'VA'], phases,
                    {**base, v.X('AL2ZN') / v.X('ALZN2'): target})
    ratio = float(np.squeeze(wks.get('X(AL2ZN)'))) / float(np.squeeze(wks.get('X(ALZN2)')))
    np.testing.assert_allclose(ratio, target, atol=1e-6)
    # change of basis is exact: ratio 2.5 over [AL2ZN, ALZN2] gives n_AL = (4/3) n_ZN -> X(AL)=4/7
    np.testing.assert_allclose(float(np.squeeze(wks.get('X(AL)'))), 4.0 / 7.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Phase 5: MU(component) conditions (fixed linear combination of chemical potentials)
# ---------------------------------------------------------------------------

@select_database("alzn_mey.tdb")
def test_mu_component_condition_enforced(load_database):
    """A MU(component) condition fixes sum_e constituents[e]*mu_e and is satisfied at
    equilibrium, with element chemical potentials recovered consistently.

    Note: like any chemical-potential condition, MU(component) can have multiple composition
    solutions; we assert the constraint is enforced and self-consistent, not a unique
    composition (see test_mu_component_with_composition_is_unique for a pinned case)."""
    dbf = load_database()
    phases = ['FCC_A1', 'HCP_A3', 'LIQUID']
    base = {v.T: 900.0, v.P: 1e5}
    ref = Workspace(dbf, ['AL2ZN', 'ALZN2', 'VA'], phases,
                    {**base, v.Moles('AL2ZN'): 0.2, v.Moles('ALZN2'): 0.1})
    mu_target = float(np.squeeze(ref.get('MU(AL2ZN)')))
    total_n = float(np.squeeze(ref.get(v.N)))
    test = Workspace(dbf, ['AL2ZN', 'ALZN2', 'VA'], phases,
                     {**base, v.MU('AL2ZN'): mu_target, v.N: total_n})
    np.testing.assert_allclose(np.squeeze(test.get('MU(AL2ZN)')), mu_target, atol=1e-3)
    # element chemical potentials are solved and consistent: MU(AL2ZN) == 2*MU(AL) + MU(ZN)
    mu_al = float(np.squeeze(test.get('MU(AL)')))
    mu_zn = float(np.squeeze(test.get('MU(ZN)')))
    np.testing.assert_allclose(2 * mu_al + mu_zn, mu_target, atol=1e-3)
    assert np.isfinite(float(np.squeeze(test.get('GM'))))


@select_database("alcrni.tdb")
def test_mu_single_element_multiple_component(load_database):
    """MU(component) for a single-element multiple (NI3 = {NI:3}) equals 3*MU(NI)."""
    dbf = load_database()
    phases = ['FCC_A1', 'BCC_A2', 'LIQUID']
    base = {v.T: 1200.0, v.P: 1e5}
    ref = Workspace(dbf, ['AL2CR', 'CRNI2', 'NI3', 'VA'], phases,
                    {**base, v.Moles('AL2CR'): 0.1, v.Moles('CRNI2'): 0.2, v.Moles('NI3'): 0.1})
    np.testing.assert_allclose(np.squeeze(ref.get('MU(NI3)')),
                               3 * float(np.squeeze(ref.get('MU(NI)'))), atol=1e-4)


# ---------------------------------------------------------------------------
# Phase 6: comprehensive coverage, string parsing, out-of-basis outputs, realism
# ---------------------------------------------------------------------------

def test_unpack_components_parses_formula_strings():
    """Multi-element component strings parse to their elemental constituents (including
    fractional stoichiometry)."""
    by_name = {str(c): c for c in v.unpack_components(['AL2O3', 'SIO2', 'SI1O2'])}
    assert by_name['AL2O3'].constituents == {'AL': 2.0, 'O': 3.0}
    assert by_name['SIO2'].constituents == {'SI': 1.0, 'O': 2.0}
    assert by_name['SI1O2'].constituents == {'SI': 1.0, 'O': 2.0}
    # fractional stoichiometry via explicit construction (the '/N' suffix denotes charge)
    assert v.Component('ALO3', {'AL': 1.0, 'O': 1.5}).constituents == {'AL': 1.0, 'O': 1.5}


def test_manual_component_construction_for_adjacent_single_letters():
    """Adjacent single-letter element symbols (e.g. K, F in 'KF') are ambiguous to the
    formula parser (it greedily reads two letters as one symbol), so such components must be
    constructed explicitly. This is the documented fallback."""
    assert v.Component('KF').constituents == {'KF': 1.0}  # parser limitation
    kf = v.Component('KF', {'K': 1.0, 'F': 1.0})          # explicit construction works
    assert kf.constituents == {'K': 1.0, 'F': 1.0}


@select_database("alcrni.tdb")
def test_outputs_outside_basis(load_database):
    """Under a redefined basis, pure-element and non-basis-component MU/X outputs are all
    computable (element X; MU as a forward stoichiometric sum)."""
    dbf = load_database()
    phases = ['FCC_A1', 'BCC_A2', 'LIQUID']
    wks = Workspace(dbf, ['AL2CR', 'CRNI2', 'NI3', 'VA'], phases,
                    {v.T: 1200.0, v.P: 1e5, v.Moles('AL2CR'): 0.1, v.Moles('CRNI2'): 0.2, v.Moles('NI3'): 0.1})
    for el in ['AL', 'CR', 'NI']:
        assert np.isfinite(float(np.squeeze(wks.get(f'X({el})'))))
        assert np.isfinite(float(np.squeeze(wks.get(f'MU({el})'))))
    # MU of a component not in the basis = forward sum over its constituents
    mu_cr = float(np.squeeze(wks.get('MU(CR)')))
    mu_ni = float(np.squeeze(wks.get('MU(NI)')))
    np.testing.assert_allclose(float(np.squeeze(wks.get('MU(CRNI)'))), mu_cr + mu_ni, atol=1e-4)
    np.testing.assert_allclose(float(np.squeeze(wks.get('MU(NI2)'))), 2 * mu_ni, atol=1e-4)


@select_database("Ocadiz-Flores.dat")
def test_realistic_oxide_fluoride_equivalence(load_database):
    """Realistic ionic database: a NiF2-KF-F2 component basis reproduces the equivalent
    pure-element (Ni, K, F) equilibrium off the pseudo-binary line. Components are built
    explicitly because 'KF' has adjacent single-letter elements."""
    dbf = load_database()
    NIF2 = v.Component('NIF2', {'NI': 1.0, 'F': 2.0})
    KF = v.Component('KF', {'K': 1.0, 'F': 1.0})
    F2 = v.Component('F2', {'F': 2.0})
    phases = list(dbf.phases.keys())
    base = {v.T: 1300.0, v.P: 1e5}
    a, b, c = 0.2, 0.5, 0.1  # off the pseudo-binary (excess F via F2); n_F=2a+b+2c, n_K=b, n_NI=a
    wks_comp = Workspace(dbf, [NIF2, KF, F2, 'VA'], phases,
                         {**base, v.Moles(NIF2): a, v.Moles(KF): b, v.Moles(F2): c})
    assert wks_comp.phase_record_factory.basis_is_trivial is False
    assert wks_comp.phase_record_factory.nonvacant_elements == ['F', 'K', 'NI']
    wks_elem = Workspace(dbf, ['NI', 'K', 'F', 'VA'], phases,
                         {**base, v.Moles('NI'): a, v.Moles('K'): b, v.Moles('F'): 2 * a + b + 2 * c})
    _assert_props_match(wks_comp, wks_elem,
                        ['GM', 'MU(NI)', 'MU(K)', 'MU(F)', 'X(NI)', 'X(K)', 'X(F)', v.N], atol=1e-5)
    # component amounts and a non-basis component output (MU(F2) = 2*MU(F))
    np.testing.assert_allclose(np.squeeze(wks_comp.get(v.Moles(NIF2))), a, atol=1e-5)
    np.testing.assert_allclose(np.squeeze(wks_comp.get(v.Moles(KF))), b, atol=1e-5)
    np.testing.assert_allclose(np.squeeze(wks_comp.get(v.MU(F2))),
                               2 * float(np.squeeze(wks_comp.get('MU(F)'))), atol=1e-4)
