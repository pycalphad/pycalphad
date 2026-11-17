"""
Tests for the ThermML XML database reader and writer (pycalphad.io.thermml).
"""

import warnings
import numpy as np
import pytest
from importlib.resources import files
import pycalphad.tests.databases
from pycalphad import Database, Model, calculate, variables as v
from pycalphad.tests.fixtures import select_database, load_database


# Minimal, hand-written components-only document.
COMPONENTS_ONLY = """<?xml version="1.0" encoding="UTF-8"?>
<database xmlns="http://calphad.org/thermml/v0"
          xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" name="MiniDB" version="0.1.0">
    <systemComponents>
        <systemComponent symbol="Pb" refstate="" molarMass="207.2" h298="0" s298="0"/>
        <systemComponent symbol="Sn" refstate="FCC_A1" molarMass="118.71" h298="1.0" s298="2.0"/>
    </systemComponents>
</database>
"""


# ---------------------------------------------------------------------------
# System components
# ---------------------------------------------------------------------------


def test_thermml_format_is_registered():
    "The 'xml' format is registered for reading and writing."
    from pycalphad.io.database import format_registry

    assert "xml" in format_registry
    assert format_registry["xml"].read is not None
    assert format_registry["xml"].write is not None


def test_thermml_components_from_string():
    "System components map to elements, species, and reference states."
    dbf = Database.from_string(COMPONENTS_ONLY, fmt="xml")
    assert dbf.elements == {"PB", "SN"}
    assert {s.name for s in dbf.species} == {"PB", "SN"}
    assert dbf.refstates["SN"]["phase"] == "FCC_A1"
    assert dbf.refstates["SN"]["mass"] == pytest.approx(118.71)
    assert dbf.refstates["SN"]["H298"] == pytest.approx(1.0)
    assert dbf.refstates["SN"]["S298"] == pytest.approx(2.0)


def test_thermml_extension_autodetect():
    "A .xml filename is auto-detected as the ThermML format."
    path = files(pycalphad.tests.databases).joinpath("Pb-Sn.xml")
    dbf = Database(str(path))  # no explicit fmt=
    assert "LIQUID" in dbf.phases


# ---------------------------------------------------------------------------
# CEF structure + G endmembers + L interactions + global functions
# ---------------------------------------------------------------------------


@select_database("Pb-Sn.xml")
def test_thermml_cef_structure_and_params(load_database):
    "CEF phases, sublattices, constituents, G/L params, and functions load."
    dbf = load_database()
    assert dbf.elements == {"PB", "SN"}
    assert "LIQUID" in dbf.phases
    assert len(dbf.symbols) == 10  # FunctionTypeExpr globals

    liquid = dbf.phases["LIQUID"]
    assert liquid.sublattices == (1.0,)
    assert {sp.name for sp in liquid.constituents[0]} == {"PB", "SN"}

    params = dbf._parameters.all()
    assert {p["parameter_type"] for p in params} == {"G", "L"}
    g_params = [p for p in params if p["parameter_type"] == "G"]
    l_params = [p for p in params if p["parameter_type"] == "L"]
    assert len(g_params) == 10
    # LIQUID interaction has rank 0 and rank 1 entries
    liquid_l_orders = sorted(
        p["parameter_order"] for p in l_params if p["phase_name"] == "LIQUID"
    )
    assert liquid_l_orders == [0, 1]


@select_database("Pb-Sn.xml")
def test_thermml_expression_references_resolve(load_database):
    "Brace function references ({Pb#F1}) resolve so a Model evaluates."
    dbf = load_database()
    mod = Model(dbf, ["PB", "SN"], "LIQUID")
    assert v.T in mod.GM.free_symbols
    res = calculate(dbf, ["PB", "SN"], "LIQUID", T=800, P=101325, output="GM")
    gm = np.asarray(res.GM)
    assert np.all(np.isfinite(gm))
    assert gm.min() < 0  # liquid Gibbs energy is negative at 800 K


# ---------------------------------------------------------------------------
# Function names preserved verbatim; gas constant resolved
# ---------------------------------------------------------------------------


@select_database("CrNi-16Tan.xml")
def test_thermml_function_names_preserved(load_database):
    "ThermML function names (with ':' '#') are kept verbatim as symbol keys."
    dbf = load_database()
    # These names contain characters SymEngine's string parser rejects; they
    # must survive unmangled so an emitted ThermML file matches the ingress.
    assert "Cr:Va#FCC_A1" in dbf.symbols
    assert "GHSERNI" in dbf.symbols
    # And a parameter that references such a name must still resolve in a Model.
    Model(dbf, ["CR", "NI", "VA"], "FCC_A1")


def test_thermml_gas_constant_reference_resolved():
    "A brace reference to the gas constant {R} resolves to its value, not a symbol."
    # {R} is a Thermo-Calc built-in that databases use without declaring; the
    # reader resolves it like a bare R (8.3145), so no free 'R' symbol survives.
    doc = (
        _DB_OPEN + "<globalExpressions>"
        '<expression xsi:type="FunctionTypeExpr" name="W1">'
        '<range low="298.15" high="6000">-860*{R}</range></expression>'
        "</globalExpressions></database>"
    )
    dbf = Database.from_string(doc, fmt="xml")
    expr = dbf.symbols["W1"]
    from symengine import Symbol

    assert Symbol("R") not in expr.free_symbols  # resolved, not left dangling
    # value folds to -860 * 8.3145
    val = float(expr.subs({v.T: 1000}))
    assert val == pytest.approx(-860 * 8.3145, rel=1e-9)


# ---------------------------------------------------------------------------
# IHJ magnetic model (phase hints + M endmember + TCL/BML)
# ---------------------------------------------------------------------------


@select_database("CrNi-16Tan.xml")
def test_thermml_magnetic_model(load_database):
    "IHJ magnetic hints and TC/BMAGN params match the TDB sign convention."
    dbf = load_database()
    fcc = dbf.phases["FCC_A1"]
    # afm_factor = -1/AFMFactor; structure factor passes through.
    assert fcc.model_hints["ihj_magnetic_afm_factor"] == pytest.approx(-3.0)
    assert fcc.model_hints["ihj_magnetic_structure_factor"] == pytest.approx(0.28)

    def _const(phase, ptype, array):
        for p in dbf._parameters.all():
            arr = tuple(tuple(s.name for s in subl) for subl in p["constituent_array"])
            if (
                p["phase_name"] == phase
                and p["parameter_type"] == ptype
                and arr == array
            ):
                return float(p["parameter"].subs({v.T: 500}).n())
        raise AssertionError(f"missing {ptype} {array} in {phase}")

    # Neel (Cr:Va): TC = temp*afm = 369.667*-3 = -1109; BMAGN = 0.82*-3 = -2.46
    assert _const("FCC_A1", "TC", (("CR",), ("VA",))) == pytest.approx(
        -1109.0, abs=1e-2
    )
    assert _const("FCC_A1", "BMAGN", (("CR",), ("VA",))) == pytest.approx(
        -2.46, abs=1e-3
    )
    # Curie (Ni:Va): stored directly.
    assert _const("FCC_A1", "TC", (("NI",), ("VA",))) == pytest.approx(633.0)
    assert _const("FCC_A1", "BMAGN", (("NI",), ("VA",))) == pytest.approx(0.52)
    # TCL/BML interaction terms map straight through (sign already baked in).
    assert _const("FCC_A1", "TC", (("CR", "NI"), ("VA",))) == pytest.approx(-3605.0)
    assert _const("FCC_A1", "BMAGN", (("CR", "NI"), ("VA",))) == pytest.approx(-1.91)


# ---------------------------------------------------------------------------
# Order/disorder linkage
# ---------------------------------------------------------------------------


@select_database("CrNi-16Tan.xml")
def test_thermml_order_disorder(load_database):
    "Ordered phases link to their disordered parent via model hints (both sides)."
    dbf = load_database()
    fcc_4sl = dbf.phases["FCC_4SL"]
    assert fcc_4sl.model_hints.get("ordered_phase") == "FCC_4SL"
    assert fcc_4sl.model_hints.get("disordered_phase") == "A1_FCC"
    # The disordered parent carries the same hints (matches the TDB importer).
    assert dbf.phases["A1_FCC"].model_hints.get("ordered_phase") == "FCC_4SL"
    assert dbf.phases["A1_FCC"].model_hints.get("disordered_phase") == "A1_FCC"
    Model(dbf, ["CR", "NI", "VA"], "FCC_4SL")  # ordered phase builds


# ---------------------------------------------------------------------------
# Full CEF coverage -- clean (warning-free) load
# ---------------------------------------------------------------------------


def test_thermml_full_cef_loads_without_warnings():
    "CrNi loads with no warn-skips (every emitted CEF construct is handled)."
    path = str(files(pycalphad.tests.databases).joinpath("CrNi-16Tan.xml"))
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warn-skip becomes an error
        dbf = Database(path)
    assert len(dbf.phases) == 9


# ---------------------------------------------------------------------------
# Parity: ThermML import vs the TDB it was generated from (same energies)
# ---------------------------------------------------------------------------


@select_database("CrNi-16Tan.xml")
def test_thermml_matches_tdb_energy(load_database):
    "FCC_A1 Gibbs energy from the ThermML import matches the TDB import."
    xml_dbf = load_database()
    tdb_dbf = Database(str(files(pycalphad.tests.databases).joinpath("CrNi-16Tan.tdb")))
    pts = np.array([[0.4, 0.6, 1e-12]])  # x(Cr,Ni,Va) on (Cr,Ni)(Va)
    comps = ["CR", "NI", "VA"]
    for T in (600.0, 1000.0, 1500.0):
        gx = calculate(xml_dbf, comps, "FCC_A1", T=T, P=101325, points=pts, output="GM")
        gt = calculate(tdb_dbf, comps, "FCC_A1", T=T, P=101325, points=pts, output="GM")
        assert float(gx.GM.squeeze()) == pytest.approx(float(gt.GM.squeeze()), abs=1e-3)


# ---------------------------------------------------------------------------
# Robustness against emitter quirks (species naming, volume props, commas)
# ---------------------------------------------------------------------------


def _cef_doc(
    species, constituents, endmember_const, disordered=None, extra_property=""
):
    "Build a tiny single-sublattice CEF database document for edge-case tests."
    dis_attr = f' disorderedPhase="{disordered}"' if disordered else ""
    species_xml = "".join(f'<specie name="{n}" composition="{c}"/>' for n, c in species)
    const_xml = "".join(f'<const species="{s}"/>' for s in endmember_const)
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<database xmlns="http://calphad.org/thermml/v0"
          xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" name="EdgeDB" version="0.1.0">
    <systemComponents>
        <systemComponent symbol="Al" molarMass="26.98"/>
        <systemComponent symbol="O" molarMass="16.0"/>
    </systemComponents>
    <phases>
        <phase xsi:type="CEFPhaseType" name="TESTPH" state="gas"{dis_attr}>
            <species>{species_xml}</species>
            <structure>
                <sublattices multiplicities="1">
                    <site constituents="{constituents}"/>
                </sublattices>
            </structure>
            <endmembers>
                <endmember name="EM">
                    <constituents><site>{const_xml}</site></constituents>
                    <property xsi:type="G"><expr>-1000+T</expr></property>
                    {extra_property}
                </endmember>
            </endmembers>
        </phase>
    </phases>
</database>
"""


def test_thermml_species_names_used_verbatim():
    "Species names are used exactly as written (constituents match the <species> block)."
    # The emitter now writes the same name in <species> and in constituents
    # (e.g. 'Al1O1' in both), so the reader uses them verbatim -- no trailing-1
    # normalization, no prefix resolution.
    doc = _cef_doc(
        species=[("Al1O1", "Al1O1"), ("Al", "Al")],
        constituents="Al Al1O1",
        endmember_const=["Al1O1"],
    )
    dbf = Database.from_string(doc, fmt="xml")
    names = {s.name for s in dbf.species}
    assert "AL1O1" in names  # kept verbatim (just upper-cased)
    assert "AL1O" not in names
    g = [p for p in dbf._parameters.all() if p["parameter_type"] == "G"]
    assert len(g) == 1
    assert {s.name for s in g[0]["constituent_array"][0]} == {"AL1O1"}


def test_thermml_volume_property_dropped_with_warning():
    "Volume-model properties warn and are dropped (no parameter stored)."
    doc = _cef_doc(
        species=[("Al", "Al")],
        constituents="Al",
        endmember_const=["Al"],
        extra_property='<property xsi:type="MolarVolume"><value>1.0e-5</value></property>',
    )
    with pytest.warns(UserWarning, match="volume-model property"):
        dbf = Database.from_string(doc, fmt="xml")
    types = {p["parameter_type"] for p in dbf._parameters.all()}
    assert types == {"G"}  # only the Gibbs energy survived


# ---------------------------------------------------------------------------
# MQM / SUBQ (ModifiedQuasichemicalPhaseType -> MQMQA)
# ---------------------------------------------------------------------------


def _load_db_quietly(filename):
    """
    Load a vendored database, suppressing warnings (pytest runs with
    ``filterwarnings = error`` and the MQM fixtures warn about unsupported
    reciprocal excess terms).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Database(str(files(pycalphad.tests.databases).joinpath(filename)))


# A small, hand-written MQM document used for the two interaction flavors the
# vendored Shishin fixture does not contain: an asymmetric-ternary ``<selected>``
# corner and a reciprocal ``MQM-L-RM`` term. The structure mirrors a real SUBQ
# phase (a charged-species salt) but the coefficients are invented -- it carries
# no data from any IP-protected database.
SYNTHETIC_MQM = """<?xml version="1.0" encoding="UTF-8"?>
<database xmlns="http://calphad.org/thermml/v0"
          xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" name="SynthMQM" version="0.1.0">
    <systemComponents>
        <systemComponent symbol="Li" molarMass="6.94"/>
        <systemComponent symbol="Na" molarMass="22.99"/>
        <systemComponent symbol="K" molarMass="39.10"/>
        <systemComponent symbol="F" molarMass="19.00"/>
        <systemComponent symbol="Cl" molarMass="35.45"/>
    </systemComponents>
    <phases>
        <phase xsi:type="ModifiedQuasichemicalPhaseType" name="SALT">
            <species>
                <specie name="Li" composition="Li" charge="1" group="1"/>
                <specie name="Na" composition="Na" charge="1" group="1"/>
                <specie name="K" composition="K" charge="1" group="1"/>
                <specie name="F" composition="F" charge="-1" group="2"/>
                <specie name="Cl" composition="Cl" charge="-1" group="2"/>
            </species>
            <interactions>
                <interaction name="Li,Na,K:F">
                    <constituents>
                        <site><const species="Li"/><const species="Na"/><const species="K"/></site>
                        <site><const species="F"/></site>
                    </constituents>
                    <property xsi:type="MQM-L-PF">
                        <expr i="0" j="0" k="1">+1234</expr>
                        <selected>Na</selected>
                    </property>
                </interaction>
                <interaction name="Li,Na:F,Cl">
                    <constituents>
                        <site><const species="Li"/><const species="Na"/></site>
                        <site><const species="F"/><const species="Cl"/></site>
                    </constituents>
                    <property xsi:type="MQM-L-RM"><expr>-600</expr></property>
                </interaction>
            </interactions>
        </phase>
    </phases>
</database>
"""


def test_thermml_mqm_structure():
    "MQM phase maps to cations/anions, chemical groups, and MQMG/MQMZ params."
    dbf = _load_db_quietly("Shishin_Fe-Sb-O-S_slag.xml")
    phase = dbf.phases["SLAG-LIQ"]
    assert phase.sublattices == (1.0,)
    assert phase.model_hints["mqmqa"]["type"] == "SUBQ"
    # cations (charge > 0) and anions (charge < 0), named like the .dat reader
    cations = {s.name for s in phase.constituents[0]}
    anions = {s.name for s in phase.constituents[1]}
    assert cations == {"FE2++2.0", "FE3++3.0", "SB3++3.0"}
    assert anions == {"O-2.0", "S-2.0"}
    cg = phase.model_hints["mqmqa"]["chemical_groups"]
    assert {s.name: g for s, g in cg["cations"].items()}["SB3++3.0"] == 1
    assert {s.name: g for s, g in cg["anions"].items()}["O-2.0"] == 2
    counts = {}
    for p in dbf._parameters.all():
        counts[p["parameter_type"]] = counts.get(p["parameter_type"], 0) + 1
    assert counts["MQMG"] == 6  # one per cation-anion pair
    assert counts["MQMZ"] == 6  # one per quadruplet (coordination numbers)


def test_thermml_mqm_matches_dat_subsystem():
    "MQMQA Gibbs energy matches the .dat import (Fe-O-S subsystem, PF excess)."
    xml_dbf = _load_db_quietly("Shishin_Fe-Sb-O-S_slag.xml")
    dat_dbf = _load_db_quietly("Shishin_Fe-Sb-O-S_slag.dat")
    # Fe-O-S activates the binary MQM-L-PF interaction (Fe2+:O,S, mixing_code 'G')
    # alongside the Fe2+,Fe3+:O quasichemical term.
    assert any(
        p["mixing_code"] == "G"
        for p in xml_dbf._parameters.all()
        if p["parameter_type"] == "MQMX"
    )
    comps = ["FE", "O", "S"]
    for T in (1800.0, 2200.0):
        gx = np.asarray(
            calculate(xml_dbf, comps, "SLAG-LIQ", T=T, P=101325, output="GM").GM
        )
        gd = np.asarray(
            calculate(dat_dbf, comps, "SLAG-LIQ", T=T, P=101325, output="GM").GM
        )
        assert np.allclose(gx, gd, atol=1e-4)


def test_thermml_mqm_quasichemical_matches_dat():
    "Quasichemical (MQM-L-Quasichemical -> code 'Q') excess matches the .dat import."
    xml_dbf = _load_db_quietly("Shishin_Fe-Sb-O-S_slag.xml")
    dat_dbf = _load_db_quietly("Shishin_Fe-Sb-O-S_slag.dat")
    # Fe-O activates the Fe2+,Fe3+:O quasichemical interaction (mixing_code 'Q').
    comps = ["FE", "O"]
    assert any(
        p["mixing_code"] == "Q"
        for p in xml_dbf._parameters.all()
        if p["parameter_type"] == "MQMX"
    )
    for T in (1800.0, 2200.0):
        gx = np.asarray(
            calculate(xml_dbf, comps, "SLAG-LIQ", T=T, P=101325, output="GM").GM
        )
        gd = np.asarray(
            calculate(dat_dbf, comps, "SLAG-LIQ", T=T, P=101325, output="GM").GM
        )
        assert np.allclose(gx, gd, atol=1e-4)


def test_thermml_mqm_selected_ternary():
    "An asymmetric-ternary <selected> PF term maps to an MQMX with the extra corner."
    # Li,Na,K:F with selected=Na -> binary {Li,K} pair, Na as the additional
    # mixing constituent at exponent k. (Shishin has no <selected> term; this uses
    # the hand-written synthetic fixture with invented coefficients.)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the doc also has a warn-skipped RM term
        dbf = Database.from_string(SYNTHETIC_MQM, fmt="xml")
    ternary = []
    for p in dbf._parameters.all():
        if p["parameter_type"] != "MQMX":
            continue
        amc = p["additional_mixing_constituent"]
        if isinstance(amc, list) or amc == v.Species(None):
            continue  # not an asymmetric-ternary term
        ca = tuple(tuple(s.name for s in subl) for subl in p["constituent_array"])
        ternary.append(
            (
                ca,
                amc.name,
                p["additional_mixing_exponent"],
                p["mixing_code"],
                float(p["parameter"]),
            )
        )
    assert ternary == [
        ((("LI+1.0", "K+1.0"), ("F-1.0", "F-1.0")), "NA+1.0", 1, "G", 1234.0)
    ]


def test_thermml_mqm_full_system_builds():
    "The full multicomponent MQM Model builds and evaluates to a finite energy."
    dbf = _load_db_quietly("Shishin_Fe-Sb-O-S_slag.xml")
    comps = ["FE", "SB", "O", "S"]
    Model(dbf, comps, "SLAG-LIQ")
    gm = np.asarray(calculate(dbf, comps, "SLAG-LIQ", T=1500, P=101325, output="GM").GM)
    assert np.all(np.isfinite(gm))


def test_thermml_mqm_reciprocal_excess_warn_skipped():
    "Reciprocal MQM-L-RM excess terms warn and are skipped (no MQMX produced)."
    # The MQMQA model cannot consume reciprocal excess params (it raises for the
    # .dat import too), so the reader warn-skips them to keep full-system Model
    # builds working: the reciprocal Li,Na:F,Cl term is dropped while the
    # asymmetric-ternary PF term still maps to an MQMX.
    with pytest.warns(UserWarning, match="MQM-L-RM"):
        dbf = Database.from_string(SYNTHETIC_MQM, fmt="xml")
    mqmx = [p for p in dbf._parameters.all() if p["parameter_type"] == "MQMX"]
    assert len(mqmx) == 1  # only the <selected> PF term survives; RM is skipped
    assert mqmx[0]["mixing_code"] == "G"


# ---------------------------------------------------------------------------
# Unsupported schema constructs: warn-or-raise behavior
# ---------------------------------------------------------------------------
# These constructs are part of the schema but not emitted by real producers or
# not consumed by pycalphad. The reader must inform the user rather than silently
# mis-load; each test pins the behavior so it cannot regress to a silent drop.

_DB_OPEN = (
    '<database xmlns="http://calphad.org/thermml/v0" '
    'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" name="D" '
    'version="0.1.0">'
)


# ---------------------------------------------------------------------------
# Schema version / namespace gating
# ---------------------------------------------------------------------------
# <database> carries a required SemVer 'version' attribute and lives in the
# versioned namespace '.../thermml/v0'. The reader accepts only the schema
# version it implements and rejects everything else rather than risk a silent
# mis-load (the meaning of constructs can change between schema versions).


def test_thermml_missing_version_raises():
    "A <database> without the required 'version' attribute raises clearly."
    doc = (
        '<database xmlns="http://calphad.org/thermml/v0" '
        'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" name="D">'
        "</database>"
    )
    with pytest.raises(ValueError, match="missing the required 'version'"):
        Database.from_string(doc, fmt="xml")


def test_thermml_unsupported_version_raises():
    "A <database> declaring a schema version the reader does not support raises."
    doc = (
        '<database xmlns="http://calphad.org/thermml/v0" '
        'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" name="D" '
        'version="0.2.0"></database>'
    )
    with pytest.raises(ValueError, match="Unsupported ThermML schema version"):
        Database.from_string(doc, fmt="xml")


def test_thermml_pure_phase_type_raises():
    "PurePhaseType (never emitted by real producers) raises clearly."
    doc = (
        _DB_OPEN
        + '<phases><phase xsi:type="PurePhaseType" name="X"/></phases></database>'
    )
    with pytest.raises(NotImplementedError, match="PurePhaseType"):
        Database.from_string(doc, fmt="xml")


def test_thermml_unknown_phase_type_warns():
    "An unrecognized phase xsi:type is warned about and skipped (phase dropped)."
    doc = (
        _DB_OPEN
        + '<phases><phase xsi:type="MysteryPhaseType" name="X"/></phases></database>'
    )
    with pytest.warns(
        UserWarning, match="phase type 'MysteryPhaseType'.*not.*recognized"
    ):
        dbf = Database.from_string(doc, fmt="xml")
    assert "X" not in dbf.phases


def test_thermml_alternative_global_expression_type_warns():
    "A non-FunctionTypeExpr global expression is warned about and skipped."
    doc = (
        _DB_OPEN + "<globalExpressions>"
        '<expression xsi:type="FunctionTypeTDB" name="GX">'
        '<range low="298.15" high="6000">0</range></expression>'
        "</globalExpressions></database>"
    )
    with pytest.warns(UserWarning, match="global expression type 'FunctionTypeTDB'"):
        dbf = Database.from_string(doc, fmt="xml")
    assert "GX" not in dbf.symbols


def test_thermml_unknown_cef_property_type_warns():
    "An unrecognized endmember property xsi:type is warned about and skipped."
    doc = _cef_doc(
        species=[("Al", "Al")],
        constituents="Al",
        endmember_const=["Al"],
        extra_property='<property xsi:type="MysteryProperty"><expr>0</expr></property>',
    )
    with pytest.warns(
        UserWarning, match="property type 'MysteryProperty'.*not supported"
    ):
        dbf = Database.from_string(doc, fmt="xml")
    # Only the recognized G parameter survived; the mystery property was dropped.
    assert {p["parameter_type"] for p in dbf._parameters.all()} == {"G"}


def test_thermml_ternary_interpolation_override_warns():
    "An explicit <ternaryInterpolations> override is warned about and dropped."
    interp = (
        "<ternaryInterpolations><interpolation><constituents><site>"
        '<const species="Al"/></site></constituents></interpolation>'
        "</ternaryInterpolations>"
    )
    doc = f"""<?xml version="1.0" encoding="UTF-8"?>
<database xmlns="http://calphad.org/thermml/v0"
          xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" name="D" version="0.1.0">
    <systemComponents><systemComponent symbol="Al" molarMass="26.98"/></systemComponents>
    <phases>
        <phase xsi:type="CEFPhaseType" name="TESTPH">
            <species><specie name="Al" composition="Al"/></species>
            <structure><sublattices multiplicities="1"><site constituents="Al"/></sublattices></structure>
            {interp}
        </phase>
    </phases>
</database>
"""
    with pytest.warns(UserWarning, match="ternary-interpolation override"):
        Database.from_string(doc, fmt="xml")
    # An empty <ternaryInterpolations/> (the common case) must NOT warn.
    doc_empty = doc.replace(interp, "<ternaryInterpolations/>")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Database.from_string(doc_empty, fmt="xml")


def test_thermml_neutral_mqm_species_warns():
    "A neutral (charge=0) MQM associate species is warned about and skipped."
    doc = (
        _DB_OPEN + "<phases>"
        '<phase xsi:type="ModifiedQuasichemicalPhaseType" name="SALT">'
        "<species>"
        '<specie name="Li" composition="Li" charge="1" group="1"/>'
        '<specie name="F" composition="F" charge="-1" group="2"/>'
        '<specie name="LiF" composition="LiF" charge="0" group="1"/>'
        "</species></phase></phases></database>"
    )
    with pytest.warns(UserWarning, match="neutral MQM species 'LiF'"):
        dbf = Database.from_string(doc, fmt="xml")
    # The neutral associate is not added as a phase constituent.
    constituent_names = {
        s.name for site in dbf.phases["SALT"].constituents for s in site
    }
    assert "LIF" not in constituent_names


# ---------------------------------------------------------------------------
# Writer (Database -> ThermML XML)
# ---------------------------------------------------------------------------
# The writer is the inverse of the reader: read(write(dbf)) == dbf. Round-trip
# tests assert whole-Database equality (the strongest check); feature-specific
# tests pin individual decisions.


def _roundtrip(dbf, **write_kwargs):
    "Write dbf to ThermML and read it back into a fresh Database."
    return Database.from_string(dbf.to_string(fmt="xml", **write_kwargs), fmt="xml")


def test_thermml_writer_is_registered():
    "The 'xml' format supports writing as well as reading."
    from pycalphad.io.database import format_registry

    assert format_registry["xml"].write is not None


def test_thermml_write_roundtrip_components_only():
    "A components-only database (no phases or functions) round-trips exactly."
    dbf = Database.from_string(COMPONENTS_ONLY, fmt="xml")
    assert _roundtrip(dbf) == dbf


@select_database("Pb-Sn.xml")
def test_thermml_write_roundtrip_cef(load_database):
    "A CEF database (Pb-Sn: G + L + functions) round-trips to an equal Database."
    dbf = load_database()
    assert _roundtrip(dbf) == dbf


@select_database("CrNi-16Tan.xml")
def test_thermml_write_roundtrip_magnetic_order_disorder(load_database):
    "Magnetic, order/disorder, and parameter references all round-trip exactly."
    dbf = load_database()
    assert _roundtrip(dbf) == dbf


def test_thermml_write_roundtrip_mqm():
    "An MQM/SUBQ database (Shishin) round-trips to an equal Database."
    dbf = _load_db_quietly("Shishin_Fe-Sb-O-S_slag.xml")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert _roundtrip(dbf) == dbf


def test_thermml_write_charged_cef_species_roundtrip():
    "Charged CEF species (ionic two-sublattice 'AL+3', 'O-2') are not mangled."
    doc = (
        _DB_OPEN + "<systemComponents>"
        '<systemComponent symbol="Al" molarMass="26.98"/>'
        '<systemComponent symbol="O" molarMass="16.0"/>'
        "</systemComponents>" + "<phases>"
        '<phase xsi:type="CEFPhaseType" name="IONIC">'
        "<species>"
        '<specie name="Al+3" composition="Al" charge="3"/>'
        '<specie name="O-2" composition="O" charge="-2"/>'
        '<specie name="Va" composition="Va"/>'
        "</species>"
        '<structure><sublattices multiplicities="2 3">'
        '<site constituents="Al+3"/><site constituents="O-2 Va"/>'
        "</sublattices></structure>"
        '<endmembers><endmember name="Al+3:O-2">'
        '<constituents><site><const species="Al+3"/></site>'
        '<site><const species="O-2"/></site></constituents>'
        '<property xsi:type="G"><expr>-1680000+T</expr></property>'
        "</endmember></endmembers>"
        "</phase></phases></database>"
    )
    dbf = Database.from_string(doc, fmt="xml")
    assert _roundtrip(dbf) == dbf
    assert {"AL+3", "O-2"} <= {s.name for s in dbf.species}  # not stripped to AL/O


def test_thermml_write_preserves_parameter_reference():
    "A <ref> on a property is written back and round-trips."
    doc = (
        _DB_OPEN
        + '<systemComponents><systemComponent symbol="Al" molarMass="26.98"/></systemComponents>'
        + '<phases><phase xsi:type="CEFPhaseType" name="FCC">'
        '<species><specie name="Al" composition="Al"/></species>'
        '<structure><sublattices multiplicities="1"><site constituents="Al"/></sublattices></structure>'
        '<endmembers><endmember name="Al">'
        '<constituents><site><const species="Al"/></site></constituents>'
        '<property xsi:type="G"><ref>91Din</ref><expr>-1000</expr></property>'
        "</endmember></endmembers>"
        "</phase></phases></database>"
    )
    dbf = Database.from_string(doc, fmt="xml")
    assert "<ref>91Din</ref>" in dbf.to_string(fmt="xml")
    assert {p["reference"] for p in _roundtrip(dbf)._parameters.all()} == {"91Din"}


def test_thermml_write_magnetic_emitted_as_tcl_bml_not_m():
    "Magnetic params are written as raw TCL/BML, not as ChemSage M (Curie/Neel)."
    dbf = Database(str(files(pycalphad.tests.databases).joinpath("CrNi-16Tan.xml")))
    xml = dbf.to_string(fmt="xml")
    assert 'xsi:type="TCL"' in xml and 'xsi:type="BML"' in xml
    assert 'xsi:type="M"' not in xml
    assert 'type="Neel"' not in xml and 'type="Curie"' not in xml


@select_database("CrNi-16Tan.xml")
def test_thermml_write_function_names_verbatim(load_database):
    "Function names with ':' '#' are emitted brace-wrapped, unmangled."
    dbf = load_database()
    assert "{Cr:Va#FCC_A1}" in dbf.to_string(fmt="xml")


def test_thermml_write_roundtrip_mqm_selected_ternary():
    "The asymmetric-ternary <selected> corner survives a write/read round-trip."
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # synthetic doc also has a warn-skipped RM term
        dbf = Database.from_string(SYNTHETIC_MQM, fmt="xml")
        rt = _roundtrip(dbf)
    sel = [
        p
        for p in rt._parameters.all()
        if p["parameter_type"] == "MQMX"
        and p["additional_mixing_constituent"] != v.Species(None)
        and p["additional_mixing_constituent"].name
    ]
    assert len(sel) == 1
    assert sel[0]["additional_mixing_constituent"].name == "NA+1.0"
    assert sel[0]["additional_mixing_exponent"] == 1
    constituent_names = tuple(
        tuple(s.name for s in subl) for subl in sel[0]["constituent_array"]
    )
    assert constituent_names == (("LI+1.0", "K+1.0"), ("F-1.0", "F-1.0"))


def test_thermml_write_multibranch_param_hoisted():
    "A multi-range Piecewise parameter is hoisted into a synthesized function."
    from symengine import Piecewise, And

    dbf = Database()
    dbf.elements.update(["CR"])
    dbf.species.add(v.Species("CR", {"CR": 1}))
    dbf.refstates["CR"] = {"phase": "BCC_A2", "mass": 52.0, "H298": 0.0, "S298": 0.0}
    dbf.add_phase("TESTP", {}, [1.0])
    dbf.add_phase_constituents("TESTP", [["CR"]])
    T = v.T
    pw = Piecewise(
        (-1000.0 + T, And(298.15 <= T, T < 1000.0)),
        (-2000.0 + 2 * T, And(1000.0 <= T, T < 3000.0)),
        (0, True),
    )
    dbf.add_parameter("G", "TESTP", [["CR"]], 0, pw)
    rt = _roundtrip(dbf)
    # The parameter is replaced by a reference to a synthesized global function.
    assert [k for k in rt.symbols if k.startswith("_PYCALPHAD_EXPR")]
    # Both temperature branches evaluate to the original energy.
    for T_val, expected in ((500.0, -500.0), (2000.0, 2000.0)):
        g0 = float(
            calculate(dbf, ["CR"], "TESTP", T=T_val, P=101325, output="GM").GM.squeeze()
        )
        g1 = float(
            calculate(rt, ["CR"], "TESTP", T=T_val, P=101325, output="GM").GM.squeeze()
        )
        assert g0 == pytest.approx(expected, abs=1e-9)
        assert g1 == pytest.approx(expected, abs=1e-9)


def _db_with_unsupported_params():
    "A database carrying a mobility (MQ) and a volume (V0) parameter."
    dbf = Database()
    dbf.elements.update(["CR"])
    dbf.species.add(v.Species("CR", {"CR": 1}))
    dbf.add_phase("TESTP", {}, [1.0])
    dbf.add_phase_constituents("TESTP", [["CR"]])
    dbf.add_parameter("G", "TESTP", [["CR"]], 0, -5000.0)
    dbf.add_parameter("MQ", "TESTP", [["CR"]], 0, -1.0, diffusing_species="CR")
    dbf.add_parameter("V0", "TESTP", [["CR"]], 0, 7.0e-6)
    return dbf


def test_thermml_write_unsupported_param_warns_and_skips():
    "Parameter types ThermML 0.1 cannot hold (mobility, volume) warn and are dropped."
    dbf = _db_with_unsupported_params()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rt = _roundtrip(dbf)
    messages = [str(w.message) for w in caught]
    assert any("parameter type 'MQ'" in m for m in messages)
    assert any("parameter type 'V0'" in m for m in messages)
    assert {p["parameter_type"] for p in rt._parameters.all()} == {"G"}


@pytest.mark.filterwarnings("error")
def test_thermml_write_if_incompatible_ignore():
    "if_incompatible='ignore' drops unrepresentable parameters without warning."
    dbf = _db_with_unsupported_params()
    rt = Database.from_string(
        dbf.to_string(fmt="xml", if_incompatible="ignore"), fmt="xml"
    )
    assert {p["parameter_type"] for p in rt._parameters.all()} == {"G"}


def test_thermml_write_if_incompatible_raise():
    "if_incompatible='raise' turns an unrepresentable parameter into an error."
    from pycalphad.io.database import DatabaseExportError

    with pytest.raises(DatabaseExportError, match="MQ"):
        _db_with_unsupported_params().to_string(fmt="xml", if_incompatible="raise")


def test_thermml_write_bad_if_incompatible_raises():
    "An invalid if_incompatible value raises ValueError."
    with pytest.raises(ValueError):
        Database().to_string(fmt="xml", if_incompatible="not_a_valid_option")


@select_database("CrNi-16Tan.tdb")
def test_thermml_tdb_to_thermml_to_tdb_functional(load_database):
    "TDB -> ThermML -> pycalphad -> TDB preserves Gibbs energy (the functional path)."
    tdb_dbf = load_database()
    xml_dbf = _roundtrip(tdb_dbf)
    tdb2 = Database.from_string(xml_dbf.to_string(fmt="tdb"), fmt="tdb")
    pts = np.array([[0.4, 0.6, 1e-12]])
    comps = ["CR", "NI", "VA"]
    for T in (600.0, 1000.0, 1500.0):
        g_orig = float(
            calculate(
                tdb_dbf, comps, "FCC_A1", T=T, P=101325, points=pts, output="GM"
            ).GM.squeeze()
        )
        g_xml = float(
            calculate(
                xml_dbf, comps, "FCC_A1", T=T, P=101325, points=pts, output="GM"
            ).GM.squeeze()
        )
        g_tdb2 = float(
            calculate(
                tdb2, comps, "FCC_A1", T=T, P=101325, points=pts, output="GM"
            ).GM.squeeze()
        )
        assert g_xml == pytest.approx(g_orig, abs=1e-6)
        assert g_tdb2 == pytest.approx(g_orig, abs=1e-6)


def test_thermml_write_title_description_metadata():
    "Optional title/description are emitted as <metadata>; nothing otherwise."
    dbf = Database.from_string(COMPONENTS_ONLY, fmt="xml")
    assert "<metadata>" not in dbf.to_string(fmt="xml")  # minimal by default
    xml = dbf.to_string(fmt="xml", title="MyDB", description="hello")
    assert "<title>MyDB</title>" in xml
    assert "<description>hello</description>" in xml


def test_thermml_write_from_dat_source_gm_parity():
    "ChemSage-DAT -> pycalphad -> ThermML -> pycalphad preserves Gibbs energy."
    # The writer is source-agnostic: an MQMQA phase imported from the .dat reader
    # exports as ThermML and re-reads to the same energy.
    dat = _load_db_quietly("Shishin_Fe-Sb-O-S_slag.dat")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        xml = _roundtrip(dat)
    for comps in (["FE", "O"], ["FE", "O", "S"]):
        for T in (1800.0, 2200.0):
            gd = np.asarray(
                calculate(dat, comps, "SLAG-LIQ", T=T, P=101325, output="GM").GM
            )
            gx = np.asarray(
                calculate(xml, comps, "SLAG-LIQ", T=T, P=101325, output="GM").GM
            )
            assert np.allclose(gd, gx, atol=1e-6)
