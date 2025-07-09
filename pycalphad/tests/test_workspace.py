from numpy.testing import assert_allclose
import numpy as np
from pycalphad import Database, Workspace, variables as v
from pycalphad.core.conditions import ConditionError
from pycalphad.core.composition_set import CompositionSet
from pycalphad.property_framework import as_property, ComputableProperty, T0, IsolatedPhase, DormantPhase
from pycalphad.property_framework.units import Q_
from pycalphad.tests.fixtures import load_database, select_database, ConvergenceFailureSolver
import pytest
from collections import Counter

@pytest.mark.solver
@select_database("alzn_mey.tdb")
def test_workspace_creation(load_database):
    dbf = load_database()
    Workspace() # test empty workspace creation
    wks = Workspace(database=dbf, components=['AL', 'ZN', 'VA'], phases=['FCC_A1', 'HCP_A3', 'LIQUID'],
                    conditions={v.N: 1, v.P: 1e5, v.T: (300, 1000, 10), v.X('ZN'): 0.3})
    wks2 = Workspace(dbf, ['AL', 'ZN', 'VA'], ['FCC_A1', 'HCP_A3', 'LIQUID'],
                     {v.N: 1, v.P: 1e5, v.T: (300, 1000, 10), v.X('ZN'): 0.3})
    assert_allclose(wks.eq.GM, wks2.eq.GM)

@select_database("alzn_mey.tdb")
def test_workspace_dependency_init_order(load_database):
    dbf = load_database()
    attrcounter = Counter()
    class TestWorkspace(Workspace):
        def __setattr__(self, attr, val):
            attrcounter[attr] += 1
            super().__setattr__(attr, val)
    twks = TestWorkspace(database=dbf, components=['AL', 'ZN', 'VA'], phases=['FCC_A1', 'HCP_A3', 'LIQUID'],
                         conditions={v.N: 1, v.P: 1e5, v.T: (300, 1000, 10), v.X('ZN'): 0.3})
    del attrcounter['_suspend_dependency_updates']
    firstcase = set(attrcounter.values())
    models = twks.models
    phase_record_factory = twks.phase_record_factory
    attrcounter.clear()
    twks = TestWorkspace(database=dbf, components=['AL', 'ZN', 'VA'], phases=['FCC_A1', 'HCP_A3', 'LIQUID'],
                         conditions={v.N: 1, v.P: 1e5, v.T: (300, 1000, 10), v.X('ZN'): 0.3},
                         models=models, phase_record_factory=phase_record_factory)
    del attrcounter['_suspend_dependency_updates']
    secondcase = set(attrcounter.values())
    assert len(firstcase) == 1 and next(iter(firstcase)) == 1
    assert len(secondcase) == 1 and next(iter(secondcase)) == 1

@select_database("alzn_mey.tdb")
def test_workspace_conditions_change_clear_result(load_database):
    dbf = load_database()
    wks = Workspace(dbf, ['AL', 'ZN', 'VA'], ['FCC_A1', 'HCP_A3', 'LIQUID'],
                    {v.N: 1, v.P: 1e5, v.T: (300, 1000, 100), v.X('ZN'): 0.3})
    assert wks._eq is None
    # Attribute access will trigger calculation
    assert wks.eq is not None
    assert wks._eq is wks.eq
    assert len(wks.eq.coords['T']) == 7
    # Conditions change should clear previous result
    wks.conditions[v.T] = 600
    # Check private member so that calculation is not triggered again
    assert wks._eq is None
    # New calculation result will have different shape
    assert len(wks.eq.coords['T']) == 1

@select_database("alzn_mey.tdb")
def test_workspace_conditions_specify_units(load_database):
    dbf = load_database()
    wks = Workspace(dbf, ['AL', 'ZN', 'VA'], ['FCC_A1', 'HCP_A3', 'LIQUID'],
                    {v.N: 1, v.P: 1e5, v.T['degC']: (0, 100, 1), v.X('ZN'): 0.3})
    assert_allclose(wks.conditions[v.T], np.arange(0., 100., 1.) + 273.15)
    wks.conditions[v.T['degC']] = (10, 300, 5)
    assert_allclose(wks.conditions[v.T], np.arange(10., 300., 5.) + 273.15)
    assert_allclose(wks.get(v.T), np.arange(10., 300., 5.) + 273.15)
    assert_allclose(wks.get(v.T['degC']), np.arange(10., 300., 5.))

@select_database("alzn_mey.tdb")
def test_meta_property_creation(load_database):
    dbf = load_database()
    wks = Workspace(database=dbf, components=['AL', 'ZN', 'VA'], phases=['FCC_A1', 'HCP_A3', 'LIQUID'],
                    conditions={v.N: 1, v.P: 1e5, v.T: (300, 1000, 10), v.X('ZN'): 0.3})
    my_tzero = T0('FCC_A1', 'HCP_A3', wks=wks)
    assert isinstance(my_tzero, ComputableProperty)

@select_database("alzn_mey.tdb")
def test_tzero_property(load_database):
    dbf = load_database()
    wks = Workspace(database=dbf, components=['AL', 'ZN', 'VA'], phases=['FCC_A1', 'HCP_A3', 'LIQUID'],
                    conditions={v.N: 1, v.P: 1e5, v.T: 600, v.X('ZN'): (0.01,1-0.01,0.01)})
    my_tzero = T0('FCC_A1', 'HCP_A3', wks=wks)
    my_tzero.maximum_value = 1700 # ZN reference state in this database is not valid beyond this temperature
    assert isinstance(my_tzero, ComputableProperty)
    assert my_tzero.property_to_optimize == v.T
    t0_values = wks.get(my_tzero)
    assert_allclose(np.nanmax(t0_values), 1686.814152)
    wks.conditions[v.X('ZN')] = 0.3
    my_tzero.property_to_optimize = v.X('ZN')
    my_tzero.minimum_value = 0.0
    my_tzero.maximum_value = 1.0
    t0_composition = wks.get(my_tzero)
    assert_allclose(t0_composition, 0.86119, atol=my_tzero.residual_tol)

@select_database("alzn_mey.tdb")
def test_isolated_phase_with_suspended_parent(load_database):
    dbf = load_database()
    wks = Workspace(database=dbf, components=['AL', 'ZN', 'VA'], phases=['FCC_A1'],
                    conditions={v.N: 1, v.P: 1e5, v.T: 600, v.X('ZN'): 0.3})
    # liquid is not entered in the parent workspace
    iph = IsolatedPhase('LIQUID', wks=wks)
    energy = wks.get(iph('GM'))
    assert not np.any(np.isnan(energy))

@select_database("alzn_mey.tdb")
def test_jansson_derivative_binary_temperature(load_database):
    dbf = load_database()
    wks = Workspace(database=dbf, components=['AL', 'ZN', 'VA'], phases=['FCC_A1', 'HCP_A3', 'LIQUID'],
                    conditions={v.N: 1, v.P: 1e5, v.T: 300, v.X('ZN'): 0.3})
    x, y_dot = wks.get('T', 'MU(AL).T')
    # Checked by finite difference
    assert_allclose(y_dot, -28.775364)

@select_database("alnipt.tdb")
def test_jansson_derivative_with_invalid_mass_conditions(load_database):
    """
    CPF values including Jansson derivatives computed for conditions that are invalid should produce NaN.
    """
    dbf = load_database()
    wks = Workspace(dbf, ["AL", "NI", "PT"], ["LIQUID"], {v.T: 298.15, v.P: 101325, v.N: 1, v.X("AL"): 0.6, v.X("PT"): 0.6})
    T = wks.get("T")
    assert np.isnan(T)
    GM = wks.get("GM")
    assert np.isnan(GM)
    dGM_dT = wks.get("GM.T")
    assert np.isnan(dGM_dT)

@select_database("alzn_mey.tdb")
def test_condition_zero_length(load_database):
    dbf = load_database()
    wks = Workspace(database=dbf, components=['AL', 'ZN', 'VA'], phases=['FCC_A1', 'HCP_A3', 'LIQUID'],
                    conditions={v.N: 1, v.P: 1e5, v.X('ZN'): 0.3})
    # Note that 100 < 300; this expands to a zero-length array
    with pytest.raises(ConditionError):
        wks.conditions[v.T] = (300, 100, 10)

@select_database("alzn_mey.tdb")
def test_get_composition_sets(load_database):
    dbf = load_database()
    wks = Workspace(database=dbf, components=['AL', 'ZN', 'VA'], phases=['FCC_A1', 'HCP_A3', 'LIQUID'],
                    conditions={v.N: 1, v.P: 1e5, v.T: 300, v.X('ZN'): 0.3})
    compsets = wks.get_composition_sets()
    assert len(compsets) == 2
    assert all(isinstance(c, CompositionSet) for c in compsets)

    wks.conditions[v.T] = (300, 1000, 10)
    # this function only works for point calculations
    with pytest.raises(ConditionError):
        wks.get_composition_sets()

@select_database("alzn_mey.tdb")
def test_condition_nonexistent_component(load_database):
    dbf = load_database()
    wks = Workspace(database=dbf, components=['AL', 'ZN', 'VA'], phases=['FCC_A1', 'HCP_A3', 'LIQUID'],
                    conditions={v.N: 1, v.P: 1e5, v.T: 300})
    with pytest.raises(ConditionError):
        wks.conditions[v.X('FE')] = 0.3
    with pytest.raises(ConditionError):
        wks.conditions[v.W('FE')] = 0.3
    with pytest.raises(ConditionError):
        wks.conditions[v.Y('FCC_A1', 0, 'FE')] = 0.3

@select_database("alzn_mey.tdb")
def test_jansson_derivative_binary_composition(load_database):
    dbf = load_database()
    wks = Workspace(database=dbf, components=['AL', 'ZN', 'VA'], phases=['FCC_A1', 'HCP_A3', 'LIQUID'],
                    conditions={v.N: 1, v.P: 1e5, v.T: 600, v.X('ZN'): 0.1})
    x, y_dot = wks.get('X(ZN)', 'MU(AL).X(ZN)')
    # Checked by finite difference
    assert_allclose(y_dot, -2806.93592)

@select_database("alzn_mey.tdb")
def test_mass_fraction_binary_condition(load_database):
    dbf = load_database()
    wks = Workspace(database=dbf, components=['AL', 'ZN', 'VA'], phases=['FCC_A1', 'HCP_A3', 'LIQUID'],
                    conditions={v.N: 1, v.P: 1e5, v.T: 300, v.W('AL'): 0.1})
    results = wks.get('W(AL)', 'W(ZN)', 'W(FCC_A1,AL)', 'W(HCP_A3,AL)', 'W(LIQUID,AL)',
                      'W(FCC_A1,ZN)', 'W(HCP_A3,ZN)', 'W(LIQUID,ZN)')
    truth = [0.1, 0.9, 0.98650697, 6.64406221e-05, np.nan, 0.01349303, 0.99993356, np.nan]
    np.testing.assert_almost_equal(results, truth, decimal=5)

@select_database("alzn_mey.tdb")
def test_mass_fraction_binary_dilute(load_database):
    dbf = load_database()
    wks = Workspace(database=dbf, components=['AL', 'ZN', 'VA'], phases=['FCC_A1', 'HCP_A3', 'LIQUID'],
                    conditions={v.N: 1, v.P: 1e5, v.T: 300, v.W('AL'): 0})
    result = wks.get('W(AL)')
    np.testing.assert_almost_equal(result, 0)

@select_database("alzn_mey.tdb")
def test_lincomb_binary_condition(load_database):
    dbf = load_database()
    wks = Workspace(database=dbf, components=['AL', 'ZN', 'VA'], phases=['FCC_A1', 'HCP_A3', 'LIQUID'],
                    conditions={v.T: 300, v.P: 1e5, 0.5*v.X('ZN') - 7*v.X('AL'): 0.1})
    result = 0.5 * wks.get('X(ZN)') - 7 * wks.get('X(AL)')
    np.testing.assert_almost_equal(result, 0.1, decimal=8)
    result2 = wks.get(0.5*v.X('ZN') - 7*v.X('AL'))
    np.testing.assert_almost_equal(result2, result, decimal=8)

@select_database("alzn_mey.tdb")
def test_lincomb_binary_condition_rhs_negative(load_database):
    dbf = load_database()
    wks = Workspace(database=dbf, components=['AL', 'ZN', 'VA'], phases=['FCC_A1', 'HCP_A3', 'LIQUID'],
                    conditions={v.T: 300, v.P: 1e5, v.X('ZN') - v.X('AL'): -0.5})
    result = wks.get('X(ZN)') - wks.get('X(AL)')
    np.testing.assert_almost_equal(result, -0.5, decimal=8)
    result2 = wks.get(v.X('ZN') - v.X('AL'))
    np.testing.assert_almost_equal(result2, result, decimal=8)
    del wks.conditions[v.X('ZN') - v.X('AL')]
    wks.conditions[v.X('ZN') - v.X('AL')] = -1.5
    result3 = wks.get(v.X('ZN') - v.X('AL'))
    assert np.isnan(result3)


@select_database("alzn_mey.tdb")
def test_lincomb_ratio_binary_condition(load_database):
    dbf = load_database()
    wks = Workspace(database=dbf, components=['AL', 'ZN', 'VA'], phases=['FCC_A1', 'HCP_A3', 'LIQUID'],
                    conditions={v.T: 300, v.P: 1e5, v.X('AL')/v.X('ZN'): [0.25, 1, 1.5]})
    result = wks.get('X(AL)') / wks.get('X(ZN)')
    np.testing.assert_almost_equal(result, [0.25, 1, 1.5], decimal=8)

@select_database("alzn_mey.tdb")
def test_phaselocal_binary_sitefrac_condition(load_database):
    dbf = load_database()
    wks = Workspace(database=dbf, components=['AL', 'ZN', 'VA'], phases=['FCC_A1', 'LIQUID'],
                    conditions={v.X('ZN'): 0.1, v.T: (890, 1000, 20), v.P: 1e5,
                                v.Y('LIQUID', 0, 'ZN'): 0.3, v.N: 1})
    result = wks.get('Y(LIQUID,0,ZN)')[0]
    np.testing.assert_almost_equal(result, np.full_like(result, 0.3), decimal=8)

@select_database("alzn_mey.tdb")
def test_phaselocal_binary_molefrac_condition(load_database):
    dbf = load_database()
    wks = Workspace(database=dbf, components=['AL', 'ZN', 'VA'], phases=['FCC_A1', 'LIQUID'],
                    conditions={v.X('ZN'): 0.1, v.T: (890, 1000, 20), v.P: 1e5,
                                v.X('LIQUID', 'ZN'): 0.3, v.N: 1})
    result = wks.get('X(LIQUID,ZN)')[0]
    np.testing.assert_almost_equal(result, np.full_like(result, 0.3), decimal=8)

@select_database("alzn_mey.tdb")
def test_miscibility_gap_cpf_specifier(load_database):
    dbf = load_database()
    # these conditions should be at an fcc miscibility gap
    wks = Workspace(database=dbf, components=['AL', 'ZN', 'VA'], phases=['FCC_A1'],
                    conditions={v.X('ZN'): 0.3, v.T: 580, v.P: 1e5, v.N: 1})
    # should return two values: one for each fcc composition set
    result_one = wks.get('X(FCC_A1,ZN)')
    # phase wildcard should return both composition sets
    # note that all other phases are suspended, or else we'd get the metastable ones too
    result_two = wks.get('X(*,ZN)')
    fcc_1 = wks.get('X(FCC_A1#1,ZN)')
    fcc_2 = wks.get('X(FCC_A1#2,ZN)')
    np.testing.assert_almost_equal(result_one, [0.181492, 0.538458], decimal=6)
    np.testing.assert_equal(result_one, result_two)
    np.testing.assert_equal(result_one, np.r_[fcc_1, fcc_2])
    # this composition set doesn't exist
    fcc_3 = wks.get('X(FCC_A1#3,ZN)')
    assert np.isnan(fcc_3)

@select_database("cumg.tdb")
def test_component_wildcards(load_database):
    """Wildcards of properties that depend on components should expand without phase constituent Species like vacancies"""
    dbf = load_database()
    components = ["CU", "MG", "VA"]
    phases = list(dbf.phases.keys())
    wks = Workspace(dbf, components, phases, {v.N:1, v.P:1e5, v.T:300, v.X("MG"): 0.25})
    # expansions for component properties
    X_fracs = wks.get("X(*)")
    np.testing.assert_almost_equal([0.75, 0.25], X_fracs)
    chempots = wks.get("MU(*)")
    np.testing.assert_almost_equal([-9949.7314137, -43634.7588925], chempots)
    # site fraction expansions _should_ have species
    Y_fracs = wks.get("Y(FCC_A1,0,*)")  # FCC CU,MG
    np.testing.assert_almost_equal([0.998168, 0.001832], Y_fracs)
    Y_fracs = wks.get("Y(FCC_A1,*,*)")  # FCC CU,MG,VA
    np.testing.assert_almost_equal([0.998168, 0.001832, 1.0], Y_fracs)
    # change phases to simplify phase expansion
    wks.phases = ["FCC_A1", "LIQUID", "CU2MG"]
    Y_fracs = wks.get("Y(*,*,*)")
    np.testing.assert_almost_equal([
        1.0, 1.6126622e-12, 1.1836675e-06, 9.9999882e-01,  # CU2MG CU,MG:CU,MG
        0.998168, 0.001832, 1.0,                           # FCC_A1 CU,MG:VA
        np.nan, np.nan,                                           # LIQUID CU,MG
        ], Y_fracs)

@pytest.mark.solver
@select_database("cumg.tdb")
def test_site_fraction_conditions(load_database):
    "No numerical errors from site fraction conditions near limits."
    components = ["CU", "MG"]
    dbf = load_database()
    phases = ['LIQUID']
    wks = Workspace(dbf, components, phases, {v.N:1, v.P:1e5, v.T:1080})
    wks.conditions.update({v.Y('LIQUID', 0, 'MG'): 0, v.X('MG'): 0})
    gm, phase_amt = wks.get('GM(*)', 'NP(*)') # should have only one phase
    np.testing.assert_almost_equal(gm, [-48941.69181887])
    np.testing.assert_almost_equal(phase_amt, np.array([1.0]), decimal=10)
    wks.conditions.update({v.Y('LIQUID', 0, 'MG'): 1, v.X('MG'): 1})
    gm, phase_amt = wks.get('GM(*)', 'NP(*)') # should have only one phase
    np.testing.assert_almost_equal(gm, [-53295.58547987])
    np.testing.assert_almost_equal(phase_amt, np.array([1.0]), decimal=10)

@select_database("cumg.tdb")
def test_jansson_derivative_chempot_condition(load_database):
    components = ["CU", "MG"]
    dbf = load_database()
    phases = ['LIQUID']
    wks = Workspace(dbf, components, phases, {v.N:1, v.P:1e5, v.T:1080})
    wks.conditions[v.X('MG')] = 0.3
    chempot1, result1 = wks.get('MU(CU)', 'MU(CU).X(MG)')
    wks.conditions[v.X('MG')] = wks.conditions[v.X('MG')] + 1e-6
    chempot2 = wks.get('MU(CU)')
    np.testing.assert_almost_equal(result1, (chempot2 - chempot1) / 1e-6, decimal=1)

    del wks.conditions[v.X('MG')]
    wks.conditions[v.MU('CU')] = chempot1
    molefrac1, result2 = wks.get('X(MG)', 'X(MG).MU(CU)')
    wks.conditions[v.MU('CU')] = chempot1 + 1.0
    molefrac2 = wks.get('X(MG)')
    np.testing.assert_almost_equal(molefrac1, 0.3)
    np.testing.assert_almost_equal(result2, (molefrac2 - molefrac1) / 1.0, decimal=2)

@pytest.mark.filterwarnings("ignore:No valid points found for phase SPINEL*:UserWarning")  # Filter out an expected warning so we don't fail the test
def test_issue_503_suspend_phase_infeasible_internal_constraints():
    "Phases that cannot satisfy internal constraints are correctly suspended (gh-503)"
    TDB = """
    ELEMENT /-   ELECTRON_GAS              0.0000E+00  0.0000E+00  0.0000E+00!
    ELEMENT VA   VACUUM                    0.0000E+00  0.0000E+00  0.0000E+00!
    ELEMENT O    1/2_MOLE_O2(G)            1.5999E+01  4.3410E+03  1.0252E+02!
    ELEMENT ZR   BLANK                     0.0000E+00  0.0000E+00  0.0000E+00!
    SPECIES O-2                         O1/-2!
    SPECIES ZR+4                        ZR1/+4!
    PHASE SPINEL %  4 1 2 2 4 !
    CONSTITUENT SPINEL : ZR+4 : VA : VA : O-2 :  !
    PHASE GAS:G %  1  1.0  !
    CONSTITUENT GAS:G :O,ZR :  !
    """
    # SPINEL phase cannot charge balance, so even though it contains ZR, O, and VA, it must be suspended
    wks = Workspace(TDB, ['O', 'ZR', 'VA'], ['SPINEL', 'GAS'], {v.P: 1e5, v.X('O'): 0.5, v.T: 1000})
    wks.verbose = True
    print(wks.get_dict('X(*,*)'))
    print(wks.get_dict('NP(*)'))
    assert np.isnan(wks.get('NP(SPINEL)'))
    np.testing.assert_almost_equal(wks.get('NP(GAS)'), 1.0)


@select_database("cumg.tdb")
def test_workspace_convergence_failures(load_database):
    """Convergence failures should propagate NaNs through to computed properties"""
    dbf = load_database()
    components = ["CU", "MG", "VA"]
    phases = ["LIQUID", "FCC_A1"]
    wks = Workspace(dbf, components, phases, {v.N:1, v.P:1e5, v.T:300, v.X("MG"): 0.5}, solver=ConvergenceFailureSolver())
    assert len(wks.get_composition_sets()) == 0  # convergence failures should have no stable composition sets
    assert np.isnan(wks.get("X(MG)"))
    assert np.isnan(wks.get("X(FCC_A1,MG)"))
    assert np.isnan(wks.get("Y(FCC_A1,0,MG)"))
    assert np.isnan(wks.get("GM"))
    assert np.isnan(wks.get(IsolatedPhase("LIQUID", wks=wks)("GM")))
    assert np.isnan(wks.get(DormantPhase("CUMG2", wks=wks)("GM")))
    assert np.isnan(wks.get("MU(CU)"))
    chempots = wks.get("MU(*)")
    assert np.asarray(chempots).shape == (2,)
    assert np.all(np.isnan(chempots))
    assert np.isnan(wks.get("MU(CU).X(MG)"))
    phase_fractions = wks.get("NP(*)")
    assert np.asarray(phase_fractions).shape == (len(phases),)
    assert np.all(np.isnan(phase_fractions))


@select_database("al2o3_nd2o3_zro2.tdb")
def test_components_are_updated_when_database_changes(load_database):
    dbf = load_database()
    phases = list(set(dbf.phases.keys()) - {"I_LIQUID"})
    wks = Workspace(dbf, phases=phases)
    # by default: all pure elements
    assert set(wks.components) == set({v.Component("AL"), v.Component("ND"), v.Component("ZR"), v.Component("O"), v.Component("VA")})
    wks.database = Database()
    assert set(wks.components) == set()
    wks.database = dbf
    assert set(wks.components) == set({v.Component("AL"), v.Component("ND"), v.Component("ZR"), v.Component("O"), v.Component("VA")})


@select_database("al2o3_nd2o3_zro2.tdb")
def test_constituents_are_updated_when_components_change(load_database):
    dbf = load_database()
    # NOTE: I_LIQUID is currently broken in this database (no vacancies in I2SL
    # constituents), so we remove that from the phases list. We should be able
    # to rely on the default constructor once that is fixed.
    phases = list(set(dbf.phases.keys()) - {"I_LIQUID"})
    wks = Workspace(dbf, phases=phases)
    # by default: all pure elements
    assert set(wks.components) == set({v.Component("AL"), v.Component("ND"), v.Component("ZR"), v.Component("O"), v.Component("VA")})
    all_expected_species = {
        v.Species('AL'),
        v.Species('AL+3', charge=3),
        v.Species('ALO3/2', {"AL": 1, "O": 1.5}),
        v.Species('ND'),
        v.Species('ND+3', charge=3),
        v.Species('O'),
        v.Species('O-2', charge=-2),
        v.Species('O2', {"O": 2}),
        v.Species('VA'),
        v.Species('ZR'),
        v.Species('ZR+4', charge=4)
    }

    assert set(wks.constituents) == all_expected_species

    # Remove Zr
    wks.components = ["AL", "ND", "O", "VA"]
    assert set(wks.components) == set({v.Component("AL"), v.Component("ND"), v.Component("O"), v.Component("VA")})
    noZR_expected_species = {
        v.Species('AL'),
        v.Species('AL+3', charge=3),
        v.Species('ALO3/2', {"AL": 1, "O": 1.5}),
        v.Species('ND'),
        v.Species('ND+3', charge=3),
        v.Species('O'),
        v.Species('O-2', charge=-2),
        v.Species('O2', {"O": 2}),
        v.Species('VA'),
    }
    assert set(wks.constituents) == noZR_expected_species


@select_database("crtiv_ghosh.tdb")
def test_multicomponent_jansson_derivative_dependent_component(load_database):
    "Jansson derivatives with respect to an independent composition should have only one dependent component."
    dbf = load_database()
    conds = {v.T: 2000.0, v.P: 101325, v.X("TI"): 0.2, v.X("V"): 0.3, v.N: 1}
    wks = Workspace(dbf, ["CR", "TI", "V", "VA"], ["LIQUID"], conditions=conds)

    dGM_dXTi = wks.get("GM.X(TI)")
    # With Cr as the dependent component,
    # dGM / dX(Ti) = MU(Ti) - MU(Cr)
    np.testing.assert_allclose(dGM_dXTi, wks.get("MU(TI)") - wks.get("MU(CR)"))
    np.testing.assert_allclose(dGM_dXTi, -26856.725962)

@select_database("cfe_broshe.tdb")
def test_unit_conversion(load_database):
    dbf = load_database()
    conds = {v.P: 101325, v.T: (1530, 1570, 10), v.N: 1, v.W("C"): 0.02}
    wks = Workspace(dbf, ["FE", "C", "VA"], ["LIQUID", "FCC_A1"], conditions=conds)

    # Test that it did not raise any exception (gh-609)
    wks.get(v.T, as_property("enthalpy")["J/g"])
