"""
Read and write the ThermML XML thermodynamic database format
(namespace ``http://calphad.org/thermml/v0``, schema version ``0.1.0``).

This module registers the ``xml`` format on :class:`Database` alongside
:mod:`pycalphad.io.tdb` and :mod:`pycalphad.io.cs_dat`. :func:`read_thermml` and
:func:`write_thermml` are inverses: ``read(write(dbf)) == dbf``. Constructs the
reader does not support are warned about and skipped; constructs the writer
cannot represent are warned about and dropped (see ``thermml_format_gaps.md``).

ThermML stores energies as TDB-flavored strings (``+1*{GHSERNI}-3.556*T``), so
both directions reuse the TDB string <-> SymEngine pipeline
(:func:`pycalphad.io.tdb._sympify_string`). Function references are
brace-delimited (``{name}``) and may contain ``:``/``#`` that SymEngine's string
parser rejects; the reader swaps each reference for a placeholder, parses, then
restores the exact name, so names round-trip unmangled.
"""

import math
import re
import warnings
import xml.etree.ElementTree as ET

from symengine import Add, And, E, Mul, Piecewise, Pow, S, Symbol, log, sympify

import pycalphad.variables as v
from pycalphad.io.database import Database, DatabaseExportError
from pycalphad.io.tdb import _sympify_string, get_supported_variables, to_interval

# Reuse the ChemSage-DAT species renaming so MQMQA species names match the .dat reader.
from pycalphad.io.cs_dat import rename_element_charge


# ThermML XML namespace. The schema major version is encoded in the namespace
# (the ``.../thermml/v0`` segment), so a major bump implies a new namespace.
TM_NS = "http://calphad.org/thermml/v0"
XSI_NS = "http://www.w3.org/2001/XMLSchema-instance"
_TM = "{%s}" % TM_NS
_XSI_TYPE = "{%s}type" % XSI_NS

# The full schema version this reader/writer targets, carried by the
# ``<database version="...">`` attribute (distinct from ``<metadata><version>``,
# which versions the database content, not the schema). The reader rejects any
# other version.
SUPPORTED_SCHEMA_VERSION = "0.1.0"

# Volume-model properties. TODO how these map/align to TDB-based V0, ...
_VOLUME_PROPERTY_TYPES = frozenset(
    {
        "MolarVolume",
        "ThermalExpansion",
        "Compressibility",
        "BulkModulusDerivative",
    }
)


# ---------------------------------------------------------------------------
# Small XML helpers
# ---------------------------------------------------------------------------


def _tag(elem):
    """Return the local (namespace-stripped) tag name of an element."""
    tag = elem.tag
    if tag.startswith(_TM):
        return tag[len(_TM) :]
    return tag


def _xsi_type(elem):
    """Return the local ``xsi:type`` of an element, or ``None``."""
    xtype = elem.get(_XSI_TYPE)
    if xtype is None:
        return None
    # xsi:type may carry a namespace prefix (e.g. "tm:CEFPhaseType"); strip it.
    return xtype.split(":")[-1]


def _find(elem, local_name):
    """Find a direct child by local name in the ThermML namespace."""
    return elem.find(_TM + local_name)


def _findall(elem, local_name):
    """Find all direct children by local name in the ThermML namespace."""
    return elem.findall(_TM + local_name)


def _warn_unsupported_ternary_interpolations(phase_elem, phase_name):
    """
    Warn about a non-empty ``<ternaryInterpolations>`` block.

    These are explicit per-ternary Kohler/Muggianu/Toop extrapolation overrides.
    pycalphad does not support explicit overrides, so -- like the
    ChemSage-DAT importer -- we drop them with a warning.
    """
    elem = _find(phase_elem, "ternaryInterpolations")
    if elem is None:
        return
    overrides = _findall(elem, "interpolation")
    if not overrides:
        return
    warnings.warn(
        f"ThermML: phase {phase_name!r} specifies {len(overrides)} explicit "
        f"ternary-interpolation override(s) (Kohler/Toop/Muggianu); pycalphad "
        f"does not support explicit overrides, so they are dropped (the default "
        f"extrapolation still applies, probably breaking correctness)."
    )


def _text(elem, default=""):
    """Return stripped element text, or ``default`` if missing/empty."""
    if elem is None or elem.text is None:
        return default
    return elem.text.strip()


# ---------------------------------------------------------------------------
# Expression handling
# ---------------------------------------------------------------------------

_FUNC_REF_RE = re.compile(r"\{([^}]*)\}")


def _expr_text_to_symengine(text):
    """
    Convert one ThermML expression string into a SymEngine object.

    Brace references (``{GHSERNI}``, ``{Cr:Va#FCC_A1}``) are swapped for
    placeholder identifiers so the SymEngine parser accepts the ``:``/``#``
    characters, then restored to the exact name. ``^`` is normalized to ``**``;
    everything else is handled by :func:`_sympify_string`."""
    s = text.strip()
    if not s:
        return sympify(0)

    placeholders = {}

    def _stash(match):
        name = match.group(1).strip()
        placeholder = "__TMREF{}__".format(len(placeholders))
        placeholders[placeholder] = name
        return placeholder

    s = _FUNC_REF_RE.sub(_stash, s)
    s = s.replace("^", "**")
    expr = _sympify_string(s)
    if placeholders:
        expr = expr.xreplace(
            {Symbol(ph): Symbol(name) for ph, name in placeholders.items()}
        )
        # Resolve brace-referenced built-ins (the gas constant R, and T/P).
        expr = expr.xreplace(get_supported_variables())
    return expr


def _ranges_to_piecewise(range_elems):
    """
    Build a ``Piecewise`` in T from ``<range low high>`` elements.
    """
    expr_cond_pairs = []
    for rng in range_elems:
        low = float(rng.get("low"))
        high = float(rng.get("high"))
        expr = _expr_text_to_symengine(rng.text or "")
        expr_cond_pairs.append((expr, And(low <= v.T, v.T < high)))

    # add the catch-all branch for beyond T.high
    expr_cond_pairs.append((0, True))
    return Piecewise(*expr_cond_pairs)


# ---------------------------------------------------------------------------
# System components (elements / species / reference states)
# ---------------------------------------------------------------------------

_FORMULA_RE = re.compile(r"([A-Z][a-z]?)(\d*\.?\d*)")


def _parse_composition(composition):
    """Parse a simple composition string (e.g. ``"Al2Si1"``) into a dict."""
    constituents = {}
    if not composition:
        return constituents
    for element, count in _FORMULA_RE.findall(composition):
        constituents[element.upper()] = float(count) if count else 1.0
    return constituents


def _read_system_components(dbf, components_elem):
    """Populate ``dbf.elements``, ``dbf.species`` and ``dbf.refstates``."""
    for comp in _findall(components_elem, "systemComponent"):
        symbol = comp.get("symbol").upper()
        dbf.elements.add(symbol)
        dbf.species.add(v.Species(symbol, {symbol: 1}))
        dbf.refstates[symbol] = {
            "phase": comp.get("refstate", "") or "",
            "mass": float(comp.get("molarMass", 0.0) or 0.0),
            "H298": float(comp.get("h298", 0.0) or 0.0),
            "S298": float(comp.get("s298", 0.0) or 0.0),
        }


# ---------------------------------------------------------------------------
# Global expressions (FUNCTIONs)
# ---------------------------------------------------------------------------


def _read_global_expressions(dbf, exprs_elem):
    """Populate ``dbf.symbols`` from ``<globalExpressions>`` (names verbatim)."""
    for expr in _findall(exprs_elem, "expression"):
        name = expr.get("name")  # preserved exactly; no mangling
        xtype = _xsi_type(expr)
        if xtype == "FunctionTypeExpr":
            dbf.symbols[name] = _ranges_to_piecewise(_findall(expr, "range"))
        else:
            # Other flavors (RangedTemperatureExpr, HSCPTemperatureExpr,
            # FunctionTypeTDB/CSdat) are in the schema but will be phased out.
            warnings.warn(
                f"ThermML: global expression type {xtype!r} not supported; "
                f"skipping function {name!r}."
            )


# ---------------------------------------------------------------------------
# Constituent arrays
# ---------------------------------------------------------------------------


def _read_constituent_array(constituents_elem):
    """
    Read a ``<constituents>`` block (one ``<site>`` per sublattice, each with
    one or more ``<const species=.../>``) into the nested list form expected by
    :meth:`Database.add_parameter`. Constituent names are used verbatim (upper-
    cased); they match the ``<species>`` block exactly.
    """
    array = []
    for site in _findall(constituents_elem, "site"):
        species = [c.get("species").upper() for c in _findall(site, "const")]
        array.append(species)
    return array


# ---------------------------------------------------------------------------
# CEF phases (structure, G, L, magnetic)
# ---------------------------------------------------------------------------


def _read_magnetic_model(structure_elem, model_hints):
    """
    Translate a ``<magnetic>`` element into ``ihj_magnetic_*`` model hints and
    return the pycalphad ``afm_factor`` (or ``None``).

    ThermML stores the antiferromagnetic factor as ``-1/afm_factor``, so FCC
    ``AFMFactor=1/3`` maps to pycalphad ``afm_factor=-3``. The Xiong formalism
    (``IHXMagneticType``) uses ``afm_factor == 0``.
    """
    magnetic = _find(structure_elem, "magnetic")
    if magnetic is None:
        return None
    mtype = _xsi_type(magnetic)
    structure_factor = float(_text(_find(magnetic, "structureFactorP"), "0.0"))
    afm_text = _text(_find(magnetic, "AFMFactor"), "0.0")
    # IHXMagneticType is the Xiong formalism, practically never emitted but 
    # supported by the schema.
    if mtype == "IHXMagneticType":
        # Improved (Xiong) formalism: pycalphad signals it with afm_factor == 0.
        afm_factor = 0.0
    else:
        afm_xml = float(afm_text)
        afm_factor = -1.0 / afm_xml if afm_xml != 0.0 else 0.0
    model_hints["ihj_magnetic_afm_factor"] = afm_factor
    model_hints["ihj_magnetic_structure_factor"] = structure_factor
    return afm_factor


def _read_cef_structure(dbf, phase_elem, phase_name):
    """
    Add the phase and its sublattice constituents from ``<structure>``.

    Returns the magnetic ``afm_factor`` (or ``None``) so endmember magnetic
    (``M``) properties can be converted to pycalphad's signed ``TC``/``BMAGN``
    convention.
    """
    structure = _find(phase_elem, "structure")
    model_hints = {}

    afm_factor = _read_magnetic_model(structure, model_hints)

    sublattices_elem = _find(structure, "sublattices")
    multiplicities = [float(x) for x in sublattices_elem.get("multiplicities").split()]
    site_elems = _findall(sublattices_elem, "site")

    dbf.add_phase(phase_name, model_hints, multiplicities)
    constituents = [
        [c.upper() for c in site.get("constituents").split()] for site in site_elems
    ]
    dbf.add_phase_constituents(phase_name, constituents)
    return afm_factor


def _read_cef_species(dbf, phase_elem):
    """Register phase species (site occupiers) into ``dbf.species``."""
    species_elem = _find(phase_elem, "species")
    if species_elem is None:
        return
    for specie in _findall(species_elem, "specie"):
        name = specie.get("name").upper()
        composition = _parse_composition(specie.get("composition", ""))
        charge = specie.get("charge")
        charge_val = int(charge) if charge else 0
        dbf.species.add(v.Species(name, composition, charge=charge_val))


def _add_magnetic_endmember_params(
    dbf, phase_name, constituent_array, prop_elem, afm_factor, ref
):
    """
    Map an endmember ``<property xsi:type="M">`` onto ``TC``/``BMAGN`` params.

    Curie ordering is stored directly; Neel (antiferromagnetic) ordering is
    stored as a negative ``TC`` and ``BMAGN`` scaled by ``afm_factor`` -- the
    convention pycalphad's IHJ ``magnetic_energy`` expects (it recovers the
    physical values via division by ``afm_factor`` for the ``<= 0`` branch).
    """
    temp_elem = _find(prop_elem, "temperature")
    moment_elem = _find(prop_elem, "moment")
    temperature = float(_text(temp_elem, "0.0"))
    moment = float(_text(moment_elem, "0.0"))
    ordering = (
        (temp_elem.get("type") or "Curie").lower() if temp_elem is not None else "curie"
    )

    if ordering == "neel":
        if afm_factor is None or afm_factor == 0.0:
            warnings.warn(
                f"ThermML: Neel magnetic property on phase {phase_name!r} has "
                f"no usable IHJ afm_factor; storing physical values unscaled."
            )
            scale = 1.0
        else:
            scale = afm_factor
        tc_value = temperature * scale
        bmagn_value = moment * scale
    else:  # Curie
        tc_value = temperature
        bmagn_value = moment

    dbf.add_parameter(
        "TC",
        phase_name,
        constituent_array,
        0,
        sympify(tc_value),
        ref=ref,
        force_insert=False,
    )
    dbf.add_parameter(
        "BMAGN",
        phase_name,
        constituent_array,
        0,
        sympify(bmagn_value),
        ref=ref,
        force_insert=False,
    )


def _read_cef_property(dbf, phase_name, constituent_array, prop_elem, afm_factor):
    """
    Dispatch a single endmember/interaction ``<property>`` element to
    ``add_parameter`` call(s) based on its ``xsi:type``.
    """
    ptype = _xsi_type(prop_elem)
    ref = _text(_find(prop_elem, "ref")) or None

    if ptype == "G":
        expr = _expr_text_to_symengine(_text(_find(prop_elem, "expr")))
        dbf.add_parameter(
            "G", phase_name, constituent_array, 0, expr, ref=ref, force_insert=False
        )
    elif ptype == "L":
        # One <expr rank="n"> per Redlich-Kister order.
        for expr_elem in _findall(prop_elem, "expr"):
            order = int(expr_elem.get("rank"))
            expr = _expr_text_to_symengine(expr_elem.text or "")
            dbf.add_parameter(
                "L",
                phase_name,
                constituent_array,
                order,
                expr,
                ref=ref,
                force_insert=False,
            )
    elif ptype == "M":
        _add_magnetic_endmember_params(
            dbf, phase_name, constituent_array, prop_elem, afm_factor, ref
        )
    elif ptype in ("TCL", "BML"):
        # Magnetic excess terms; the stored values carry the sign convention, so
        # they map straight to TC/BMAGN by rank.
        target = "TC" if ptype == "TCL" else "BMAGN"
        for expr_elem in _findall(prop_elem, "expr"):
            order = int(expr_elem.get("rank"))
            expr = _expr_text_to_symengine(expr_elem.text or "")
            dbf.add_parameter(
                target,
                phase_name,
                constituent_array,
                order,
                expr,
                ref=ref,
                force_insert=False,
            )
    elif ptype in _VOLUME_PROPERTY_TYPES:
        warnings.warn(
            f"ThermML: volume-model property {ptype!r} on phase {phase_name!r} "
            f"is not supported by pycalphad; dropping its coefficients."
        )
    else:
        # MQM-* properties are handled by the MQM phase reader; anything else here
        # is genuinely unknown.
        warnings.warn(
            f"ThermML: property type {ptype!r} on phase {phase_name!r} not "
            f"supported; parameter skipped."
        )


def _read_cef_phase(dbf, phase_elem, phase_name):
    """Read a CEFPhaseType / CEFOrderedPhaseType phase."""
    _read_cef_species(dbf, phase_elem)
    afm_factor = _read_cef_structure(dbf, phase_elem, phase_name)

    endmembers_elem = _find(phase_elem, "endmembers")
    if endmembers_elem is not None:
        for em in _findall(endmembers_elem, "endmember"):
            constituent_array = _read_constituent_array(_find(em, "constituents"))
            for prop in _findall(em, "property"):
                _read_cef_property(dbf, phase_name, constituent_array, prop, afm_factor)

    interactions_elem = _find(phase_elem, "interactions")
    if interactions_elem is not None:
        for inter in _findall(interactions_elem, "interaction"):
            constituent_array = _read_constituent_array(_find(inter, "constituents"))
            for prop in _findall(inter, "property"):
                _read_cef_property(dbf, phase_name, constituent_array, prop, afm_factor)

    _warn_unsupported_ternary_interpolations(phase_elem, phase_name)


def _read_pure_phase(dbf, phase_elem, phase_name):
    """Read a PurePhaseType (stoichiometric) phase."""
    # Real producers emit stoichiometric phases as single-sublattice CEFPhaseType,
    # so this only appears in hand-written schema examples and is not implemented.
    raise NotImplementedError("ThermML: PurePhaseType is not supported.")


# MQM-L-* excess xsi:type -> MQMX ``mixing_code`` (verified against cs_dat):
# PF/SP are pair-fraction / simple-polynomial (identical math) -> "G";
# Quasichemical -> "Q". Reciprocal families (RM/RS/Reciprocal) and RK are not
# mapped -- the MQMQA model cannot consume reciprocal excess params (it raises for
# the cs_dat import too), so they are warn-skipped.
_MQM_MIXING_CODE = {"MQM-L-PF": "G", "MQM-L-SP": "G", "MQM-L-Quasichemical": "Q"}


def _pair_stoichiometry(cation_charge, anion_charge):
    """
    Charge-neutral pair stoichiometry ``[n_cation, n_anion, 0, 0, 0]``.

    Neutrality of A(q+)/X(q-) gives reduced counts ``|q_X|/g`` and ``q_A/g``
    (e.g. Fe3+/O2- -> [2, 3]). The MQMQA model only consumes ``stoichiometry[0]``.
    """
    qc = int(round(cation_charge))
    qa = int(round(abs(anion_charge)))
    g = math.gcd(qa, qc) or 1
    return [float(qa // g), float(qc // g), 0.0, 0.0, 0.0]


def _read_mqm_phase(dbf, phase_elem, phase_name):
    """
    Read a ModifiedQuasichemicalPhaseType (SUBQ/MQM) phase onto pycalphad's
    MQMQA model (``pycalphad/models/model_mqmqa.py``).

    Mapping (validated against the ChemSage-DAT importer of the same systems):

    * species: split into cations (charge > 0) and anions (charge < 0); each is
      renamed with :func:`rename_element_charge` so names match the .dat reader.
      ``group`` becomes the ``chemical_groups`` model hint.
    * one sublattice with ratio ``[1.0]``; ``constituents = [cations, anions]``.
    * endmember ``MQM-G`` -> ``MQMG`` param ``[[cation], [anion]]`` with ``zeta``
      and a charge-derived ``stoichiometry`` (the per-endmember
      ``coordinationNumbers`` are charges, not coordinations, and are ignored).
    * ``<quadruplets>`` -> ``MQMZ`` params (the authoritative coordination source;
      the emitter lists every quadruplet, pure pairs included).
    * interaction ``MQM-L-PF`` -> ``MQMX`` params (``mixing_code="G"``,
      ``exponents=[i, j, 0, 0]``), duplicating the common ion into a quadruplet.
    """
    # --- species: cation/anion split + renaming + chemical groups ---
    name_map = {}  # XML species name -> renamed (rename_element_charge)
    species_charge = {}  # renamed name -> signed charge
    species_objs = {}  # renamed name -> Species object
    cations, anions = [], []
    chemical_groups = {"cations": {}, "anions": {}}
    for sp in _findall(_find(phase_elem, "species"), "specie"):
        xml_name = sp.get("name")
        charge = float(sp.get("charge", "0"))
        if charge == 0:
            # Neutral associates are not supported.
            warnings.warn(
                f"ThermML: neutral MQM species {xml_name!r} in phase "
                f"{phase_name!r} is not supported; skipping."
            )
            continue
        renamed = rename_element_charge(xml_name.upper(), charge)
        composition = _parse_composition(sp.get("composition", "")) or {
            xml_name.upper(): 1.0
        }
        species_obj = v.Species(renamed, composition, charge=charge)
        dbf.species.add(species_obj)
        name_map[xml_name] = renamed
        species_charge[renamed] = charge
        species_objs[renamed] = species_obj
        group = int(sp.get("group", "1"))
        if charge > 0:
            cations.append(renamed)
            chemical_groups["cations"][species_obj] = group
        else:
            anions.append(renamed)
            chemical_groups["anions"][species_obj] = group

    cation_set, anion_set = set(cations), set(anions)

    # SUBQ vs SUBG is not distinguishable from the XML; default to SUBQ.
    model_hints = {"mqmqa": {"type": "SUBQ", "chemical_groups": chemical_groups}}
    dbf.add_phase(phase_name, model_hints, sublattices=[1.0])
    dbf.add_structure_entry(phase_name, phase_name)
    dbf.add_phase_constituents(phase_name, [cations, anions])

    def _mapped_consts(site_elem):
        return [name_map[c.get("species")] for c in _findall(site_elem, "const")]

    # --- endmembers -> MQMG ---
    endmembers_elem = _find(phase_elem, "endmembers")
    if endmembers_elem is not None:
        for em in _findall(endmembers_elem, "endmember"):
            consts = [
                s
                for site in _findall(_find(em, "constituents"), "site")
                for s in _mapped_consts(site)
            ]
            cation = next(c for c in consts if c in cation_set)
            anion = next(c for c in consts if c in anion_set)
            gprop = next(
                (p for p in _findall(em, "property") if _xsi_type(p) == "MQM-G"), None
            )
            if gprop is None:
                continue
            expr = _expr_text_to_symengine(_text(_find(gprop, "expr")))
            zeta = float(_text(_find(gprop, "zeta"), "0.0"))
            stoich = _pair_stoichiometry(species_charge[cation], species_charge[anion])
            dbf.add_parameter(
                "MQMG",
                phase_name,
                [[cation], [anion]],
                param_order=None,
                param=expr,
                zeta=zeta,
                stoichiometry=stoich,
                force_insert=False,
            )

    # --- quadruplets -> MQMZ (coordination numbers) ---
    quads_elem = _find(phase_elem, "quadruplets")
    if quads_elem is not None:
        for qd in _findall(quads_elem, "quadruplet"):
            sites = {tag: _find(qd, tag) for tag in ("a", "b", "x", "y")}
            # Sort each sublattice by species name (with its Z) -- the MQMZ query
            # in the model looks up the canonically sorted constituent array.
            cat_pairs = sorted(
                (name_map[sites[t].get("species")], float(sites[t].get("Z")))
                for t in ("a", "b")
            )
            an_pairs = sorted(
                (name_map[sites[t].get("species")], float(sites[t].get("Z")))
                for t in ("x", "y")
            )
            constituent_array = [
                [cat_pairs[0][0], cat_pairs[1][0]],
                [an_pairs[0][0], an_pairs[1][0]],
            ]
            coordinations = [
                cat_pairs[0][1],
                cat_pairs[1][1],
                an_pairs[0][1],
                an_pairs[1][1],
            ]
            dbf.add_parameter(
                "MQMZ",
                phase_name,
                constituent_array,
                param_order=None,
                param=None,
                coordinations=coordinations,
                force_insert=False,
            )

    # --- interactions -> MQMX ---
    interactions_elem = _find(phase_elem, "interactions")
    if interactions_elem is not None:
        for inter in _findall(interactions_elem, "interaction"):
            sites = _findall(_find(inter, "constituents"), "site")
            cat_site = _mapped_consts(sites[0])
            an_site = _mapped_consts(sites[1])
            for prop in _findall(inter, "property"):
                _read_mqm_interaction_property(
                    dbf, phase_name, cat_site, an_site, prop, name_map, species_objs
                )

    _warn_unsupported_ternary_interpolations(phase_elem, phase_name)


def _read_mqm_interaction_property(
    dbf, phase_name, cat_site, an_site, prop, name_map, species_objs
):
    """
    Add the ``MQMX`` parameter(s) for one MQM interaction ``<property>``.

    Handles binary ``MQM-L-PF``/``MQM-L-SP`` (``mixing_code="G"``) and
    ``MQM-L-Quasichemical`` (``"Q"``), including the asymmetric-ternary
    ``<selected>`` corner. The mixing sublattice (the one with >1 constituent)
    has exponents ``[i, j]``; the common ion on the other sublattice is
    duplicated to form the quadruplet. A ``<selected>`` corner becomes the
    ``additional_mixing_constituent`` (with exponent ``k``).

    Reciprocal families (``MQM-L-RM``/``RS``/``Reciprocal``) and ``MQM-L-RK``
    are warn-skipped (see ``_MQM_MIXING_CODE``).
    """
    ptype = _xsi_type(prop)
    mixing_code = _MQM_MIXING_CODE.get(ptype)
    if mixing_code is None:
        warnings.warn(
            f"ThermML: MQM interaction property {ptype!r} in phase {phase_name!r} "
            f"is not supported; skipping."
        )
        return

    selected_elem = _find(prop, "selected")
    selected = name_map[_text(selected_elem)] if selected_elem is not None else None

    # Identify the mixing sublattice (cations or anions) by which one mixes.
    if len(cat_site) > 1 and len(an_site) == 1:
        members, common, mixing_is_cation = list(cat_site), an_site[0], True
    elif len(an_site) > 1 and len(cat_site) == 1:
        members, common, mixing_is_cation = list(an_site), cat_site[0], False
    else:
        warnings.warn(
            f"ThermML: reciprocal/higher-order MQM interaction in phase "
            f"{phase_name!r} is not supported; skipping."
        )
        return

    if selected is not None:
        members = [m for m in members if m != selected]
    if len(members) != 2:
        warnings.warn(
            f"ThermML: unexpected MQM interaction arity in phase {phase_name!r} "
            f"(members={members}, selected={selected}); skipping."
        )
        return

    A, B = members
    if mixing_is_cation:
        constituent_array = [[A, B], [common, common]]
    else:
        constituent_array = [[common, common], [A, B]]

    for ex in _findall(prop, "expr"):
        i = int(ex.get("i", "0"))
        j = int(ex.get("j", "0"))
        k = int(ex.get("k", "0"))
        expr = _expr_text_to_symengine(ex.text or "")
        if selected is not None:
            addl_constituent = species_objs[selected]  # real Species (comp + charge)
            addl_exponent = k
        else:
            addl_constituent = v.Species(None)
            addl_exponent = 0
        dbf.add_parameter(
            "MQMX",
            phase_name,
            constituent_array,
            param_order=None,
            param=expr,
            mixing_code=mixing_code,
            exponents=[i, j, 0, 0],
            additional_mixing_constituent=addl_constituent,
            additional_mixing_exponent=addl_exponent,
            force_insert=False,
        )


# Phase xsi:type -> handler dispatch.
_PHASE_READERS = {
    "CEFPhaseType": _read_cef_phase,
    "CEFOrderedPhaseType": _read_cef_phase,
    "PurePhaseType": _read_pure_phase,
    "ModifiedQuasichemicalPhaseType": _read_mqm_phase,
}


def _read_phases(dbf, phases_elem):
    for phase_elem in _findall(phases_elem, "phase"):
        phase_name = phase_elem.get("name").upper()
        xtype = _xsi_type(phase_elem)
        reader = _PHASE_READERS.get(xtype)
        if reader is None:
            warnings.warn(
                f"ThermML: phase type {xtype!r} (phase {phase_name!r}) is not "
                f"recognized; skipping."
            )
            continue
        reader(dbf, phase_elem, phase_name)


def _finalize_order_disorder(dbf, phases_elem):
    """
    Link ordered phases to their disordered parent. Done as a post-pass (after
    all phases exist) so declaration order in the document does not matter.

    Both phases receive the same ``ordered_phase``/``disordered_phase`` hints,
    matching the TDB importer. The endmembers of the ordered phase are listed
    explicitly in ThermML, so no symmetric-parameter generation is performed
    (unlike the TDB path's ``add_phase_symmetry_ordering_parameters``).
    """
    for phase_elem in _findall(phases_elem, "phase"):
        disordered = phase_elem.get("disorderedPhase")
        if disordered is None:
            continue
        ordered_name = phase_elem.get("name").upper()
        disordered_name = disordered.upper()
        hint = {"ordered_phase": ordered_name, "disordered_phase": disordered_name}
        if ordered_name in dbf.phases:
            dbf.phases[ordered_name].model_hints.update(hint)
        if disordered_name in dbf.phases:
            dbf.phases[disordered_name].model_hints.update(hint)
        else:
            warnings.warn(
                f"ThermML: ordered phase {ordered_name!r} references disordered "
                f"phase {disordered_name!r}, which was not found."
            )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _check_schema_version(root):
    """
    Validate the ``<database version="...">`` schema version.

    The attribute is mandatory in the schema (``SemanticVersionType``) and the
    reader only understands :data:`SUPPORTED_SCHEMA_VERSION`. A missing or
    mismatched version is a hard error.
    """
    version = root.get("version")
    if version is None:
        raise ValueError(
            "ThermML <database> is missing the required 'version' attribute; "
            f"this reader supports schema version {SUPPORTED_SCHEMA_VERSION!r}. "
            f'Add version="{SUPPORTED_SCHEMA_VERSION}" to the <database> element.'
        )
    if version != SUPPORTED_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported ThermML schema version {version!r}; this reader "
            f"supports version {SUPPORTED_SCHEMA_VERSION!r}."
        )


def read_thermml(dbf, fd):
    """
    Parse a ThermML XML file into a pycalphad Database object.

    Parameters
    ----------
    dbf : Database
        A pycalphad Database.
    fd : file-like
        File descriptor.
    """
    tree = ET.parse(fd)
    root = tree.getroot()
    if _tag(root) != "database":
        raise ValueError(
            f"Expected a ThermML <database> root element in namespace {TM_NS!r}, "
            f"got {root.tag!r}."
        )
    _check_schema_version(root)

    components_elem = _find(root, "systemComponents")
    if components_elem is not None:
        _read_system_components(dbf, components_elem)

    # Global expressions are read before phases so endmember/interaction
    # parameters can reference them, though pycalphad resolves symbols lazily.
    exprs_elem = _find(root, "globalExpressions")
    if exprs_elem is not None:
        _read_global_expressions(dbf, exprs_elem)

    phases_elem = _find(root, "phases")
    if phases_elem is not None:
        _read_phases(dbf, phases_elem)
        _finalize_order_disorder(dbf, phases_elem)

    dbf.process_parameter_queue()


# ===========================================================================
# Writer (Database -> ThermML XML)
# ===========================================================================
# The writer emits only what the Database holds; it does not synthesize optional
# ThermML content. Parameter types ThermML v0 cannot represent (mobility,
# property models, ...) are warn-skipped (see thermml_format_gaps.md). The result
# is intentionally not textually identical to a ChemSage file (see the magnetic
# handling), but round-trips through the reader exactly.

# Parameter types the writer can serialize; everything else is warn-skipped.
_WRITABLE_PARAM_TYPES = frozenset({"G", "L", "TC", "BMAGN", "MQMG", "MQMZ", "MQMX"})

# Temperature bounds used only when a Piecewise branch is unbounded (e.g. a
# TDB-derived ``0.01 <= T``); files with explicit bounds never hit these.
_DEFAULT_T_LOW = 298.15
_DEFAULT_T_HIGH = 6000.0

# MQM mixing_code -> ThermML excess xsi:type. "G" covers both MQM-L-PF and
# MQM-L-SP (the reader maps them identically); we emit PF, which re-reads to "G".
_MQM_XTYPE = {"G": "MQM-L-PF", "Q": "MQM-L-Quasichemical"}

# Bare symbols emitted without a brace wrapper (state variables).
_STATE_VAR_NAMES = frozenset({"T", "P"})

# Bare symbols brace-referenced by ThermML but not in ``dbf.symbols`` -- the gas
# constant R -- emitted as ``{R}`` without an "unknown function" warning.
_KNOWN_BRACE_REFS = frozenset({"R"})


def _format_number(value):
    """
    Format a numeric constant for a ThermML expression / attribute. Integral
    values print as integers; everything else uses ``repr`` of the underlying
    double so re-reading reconstructs the identical float (full precision).
    """
    fx = float(value)
    if fx == int(fx) and abs(fx) < 1e15:
        return str(int(fx))
    return repr(fx)


def _xml_species_name(species):
    """
    Recover the ``<specie name>`` for a Species. Neutral species pass through;
    MQM species had a ``rename_element_charge`` suffix appended on read
    (``Fe2+`` -> ``FE2++2.0``), which is stripped back off. The float-formatted
    suffix (``+2.0``) is what distinguishes a rename from a raw charged CEF
    species like ``AL+3`` (whose suffix would be ``+3.0``, not matching ``+3``).
    """
    if species.charge == 0:
        return species.name
    suffix = rename_element_charge("", float(species.charge))  # e.g. "+2.0" / "-2.0"
    if suffix and species.name.endswith(suffix):
        return species.name[: -len(suffix)]
    return species.name


def _composition_string(constituents):
    """
    Serialize a constituents dict (``{'FE': 1.0, 'O': 1.0}``) to a ``composition``
    string (``FeO``). Elements are Title-cased so :func:`_parse_composition`
    re-parses them (``FE`` would tokenize as F+E); a count of 1 is omitted.
    """
    parts = []
    for el in sorted(constituents):
        count = constituents[el]
        symbol = el[0].upper() + el[1:].lower()
        parts.append(symbol if count == 1 else symbol + _format_number(count))
    return "".join(parts)


def _array_label(constituent_array):
    """A display ``name`` for an endmember/interaction (cosmetic; the reader uses
    ``<constituents>``)."""
    return ":".join(",".join(s.name for s in subl) for subl in constituent_array)


class _ThermMLWriter:
    """
    Serialize a :class:`Database` into a ThermML ``<database>`` ElementTree.

    :meth:`_print_expr` is the inverse of the reader's
    :func:`_expr_text_to_symengine`: it walks a SymEngine expression and emits
    TDB-flavored text, brace-wrapping function references so they round-trip.
    """

    def __init__(self, dbf, if_incompatible="warn", title=None, description=None):
        if if_incompatible not in ("warn", "raise", "ignore"):
            raise ValueError(
                "Incorrect option for 'if_incompatible'. Valid args are "
                "'warn', 'raise', or 'ignore'."
            )
        self.dbf = dbf
        self.if_incompatible = if_incompatible
        self.title = title
        self.description = description
        # Names brace-wrapped on output: global symbols plus any multi-range
        # parameter hoisted into a synthesized function.
        self.function_names = set(dbf.symbols.keys())
        self.extra_functions = {}  # synthesized name -> Piecewise
        self._hoist_counter = 0
        self._warned = set()

    def _incompatible(self, message):
        """Warn, raise, or ignore an unrepresentable construct per ``if_incompatible``."""
        if self.if_incompatible == "raise":
            raise DatabaseExportError(message)
        elif self.if_incompatible == "warn" and message not in self._warned:
            self._warned.add(message)
            warnings.warn(message)
        # 'ignore' -> silently skip

    # -- expression serialization (inverse of _expr_text_to_symengine) ------
    def _print_expr(self, expr):
        """Convert a SymEngine expression to ThermML/TDB expression text."""
        expr = S(expr)
        if isinstance(expr, Add):
            terms = [self._print_expr(arg) for arg in expr.args]
            result = ""
            for term in terms:
                result += term if term.startswith("-") else "+" + term
            return result
        if isinstance(expr, Mul):
            parts = []
            for arg in expr.args:
                text = self._print_expr(arg)
                if isinstance(arg, Add):
                    text = "(" + text + ")"
                parts.append(text)
            return "*".join(parts)
        if isinstance(expr, Pow):
            base, exponent = expr.args
            if self._is_e(base):
                # exp(x) arrives as a Pow with an e-valued base (E was
                # numericalized on read); emit EXP so it round-trips exactly.
                return "EXP(" + self._print_expr(exponent) + ")"
            base_text = self._print_expr(base)
            if isinstance(base, (Add, Mul, Pow)):
                base_text = "(" + base_text + ")"
            return base_text + "**(" + self._print_expr(exponent) + ")"
        if isinstance(expr, log):
            return "LN(" + self._print_expr(expr.args[0]) + ")"
        if isinstance(expr, Symbol):
            name = expr.name
            if name in _STATE_VAR_NAMES:
                return name
            # Any other bare symbol is a function reference; warn if it is neither
            # a defined function nor the gas constant R.
            if name not in self.function_names and name not in _KNOWN_BRACE_REFS:
                self._incompatible(
                    f"ThermML: expression references {name!r}, which is not a "
                    f"known function symbol; emitting it as a brace reference."
                )
            return "{" + name + "}"
        return _format_number(expr)

    @staticmethod
    def _is_e(base):
        if base == E:
            return True
        try:
            return abs(float(base) - math.e) < 1e-12
        except (TypeError, RuntimeError, ValueError):
            return False

    def _real_branches(self, piecewise):
        """The non-default ``(expr, cond)`` branches of a Piecewise."""
        return [
            (val, cond)
            for val, cond in zip(piecewise.args[::2], piecewise.args[1::2])
            if not ((cond == S.true) and (val == S.Zero))
        ]

    def _param_expr_text(self, expr):
        """
        Text for one parameter expression. A single-branch Piecewise (the usual
        TDB shape) prints its inner expression; a multi-range Piecewise has no
        home in one ThermML ``<expr>`` and is hoisted into a global function.
        """
        expr = S(expr)
        if isinstance(expr, Piecewise):
            real = self._real_branches(expr)
            if len(real) == 0:
                return "0"
            if len(real) == 1:
                return self._print_expr(real[0][0])
            return "{" + self._hoist(expr) + "}"
        return self._print_expr(expr)

    def _hoist(self, piecewise):
        """Register a multi-branch Piecewise as a synthesized global function."""
        name = "_PYCALPHAD_EXPR{}".format(self._hoist_counter)
        self._hoist_counter += 1
        self.extra_functions[name] = piecewise
        self.function_names.add(name)
        return name

    # -- range emission (inverse of _ranges_to_piecewise) -------------------
    def _bound(self, value):
        if value == S.Infinity:
            return _format_number(_DEFAULT_T_HIGH)
        if value == S.NegativeInfinity:
            return _format_number(_DEFAULT_T_LOW)
        return _format_number(value)

    def _write_ranges(self, expr_elem, expr):
        """Emit ``<range low high>`` blocks for a (Piecewise) global function."""
        expr = S(expr)
        if isinstance(expr, Piecewise):
            branches = []
            for val, cond in self._real_branches(expr):
                interval = to_interval(cond)
                low, high = interval.args[0], interval.args[1]
                branches.append((low, high, val))
            if not branches:
                branches = [(S(_DEFAULT_T_LOW), S(_DEFAULT_T_HIGH), S.Zero)]
            # Sort by lower bound so ranges read low -> high like a TDB function.
            branches.sort(
                key=lambda b: (
                    float(b[0]) if b[0] != S.NegativeInfinity else float("-inf")
                )
            )
        else:
            branches = [(S(_DEFAULT_T_LOW), S(_DEFAULT_T_HIGH), expr)]
        for low, high, val in branches:
            rng = ET.SubElement(
                expr_elem,
                _TM + "range",
                {"low": self._bound(low), "high": self._bound(high)},
            )
            rng.text = self._print_expr(val)

    # -- XML element helpers ------------------------------------------------
    @staticmethod
    def _text_el(parent, local_name, text):
        elem = ET.SubElement(parent, _TM + local_name)
        elem.text = text
        return elem

    def _write_specie(self, parent, species, group=None):
        attrs = {
            "name": _xml_species_name(species),
            "composition": _composition_string(species.constituents),
        }
        if species.charge != 0:
            attrs["charge"] = _format_number(species.charge)
        if group is not None:
            attrs["group"] = str(group)
        ET.SubElement(parent, _TM + "specie", attrs)

    def _write_constituents(self, parent, constituent_array, xml_names=False):
        """Emit a ``<constituents>`` block (one ``<site>`` per sublattice)."""
        cons = ET.SubElement(parent, _TM + "constituents")
        for subl in constituent_array:
            site = ET.SubElement(cons, _TM + "site")
            for species in subl:
                name = _xml_species_name(species) if xml_names else species.name
                ET.SubElement(site, _TM + "const", {"species": name})
        return cons

    # -- top-level document -------------------------------------------------
    def build(self):
        ET.register_namespace("", TM_NS)
        ET.register_namespace("xsi", XSI_NS)
        root = ET.Element(
            _TM + "database",
            {"name": self.title or "", "version": SUPPORTED_SCHEMA_VERSION},
        )
        self._write_metadata(root)
        self._write_system_components(root)
        phases_elem = ET.SubElement(root, _TM + "phases")
        for name in sorted(self.dbf.phases):
            self._write_phase(phases_elem, self.dbf.phases[name])
        self._write_global_expressions(root)
        return root

    def _write_metadata(self, root):
        # Minimal by design: the Database has no metadata store, so we only emit
        # what the caller explicitly supplies (title/description kwargs).
        if not (self.title or self.description):
            return
        meta = ET.SubElement(root, _TM + "metadata")
        if self.title:
            self._text_el(meta, "title", self.title)
        if self.description:
            self._text_el(meta, "description", self.description)

    def _write_system_components(self, root):
        sc = ET.SubElement(root, _TM + "systemComponents")
        for el in sorted(self.dbf.elements):
            if el in ("VA", "/-"):
                # Vacancies / the electron are emitted in site constituents but
                # are not system components (mirrors what the reader tolerates).
                continue
            ref = self.dbf.refstates.get(el, {})
            ET.SubElement(
                sc,
                _TM + "systemComponent",
                {
                    "symbol": el,
                    "refstate": ref.get("phase", "") or "",
                    "molarMass": _format_number(ref.get("mass", 0.0)),
                    "h298": _format_number(ref.get("H298", 0.0)),
                    "s298": _format_number(ref.get("S298", 0.0)),
                },
            )

    def _write_global_expressions(self, root):
        functions = dict(self.dbf.symbols)
        functions.update(self.extra_functions)  # hoisted multi-range params
        if not functions:
            return
        exprs = ET.SubElement(root, _TM + "globalExpressions")
        for name in sorted(functions):
            expr_elem = ET.SubElement(
                exprs, _TM + "expression", {_XSI_TYPE: "FunctionTypeExpr", "name": name}
            )
            self._write_ranges(expr_elem, functions[name])

    # -- phase dispatch -----------------------------------------------------
    def _write_phase(self, phases_elem, phase):
        if "mqmqa" in phase.model_hints:
            self._write_mqm_phase(phases_elem, phase)
        else:
            self._write_cef_phase(phases_elem, phase)

    def _phase_params(self, phase_name):
        return [p for p in self.dbf._parameters.all() if p["phase_name"] == phase_name]

    def _phase_species(self, phase):
        """
        Every Species relevant to a phase's ``<species>`` block: the site
        constituents plus any species referenced only by a parameter (some
        emitters reference a species in a parameter without declaring it a site
        constituent). When a name appears more than once we keep the variant
        that actually carries a composition, so the round-trip preserves it.
        """
        species = {}

        def add(sp):
            existing = species.get(sp.name)
            if existing is None or (not existing.constituents and sp.constituents):
                species[sp.name] = sp

        for subl in phase.constituents or ():
            for sp in subl:
                add(sp)
        for param in self._phase_params(phase.name):
            for subl in param["constituent_array"]:
                for sp in subl:
                    add(sp)
            amc = param.get("additional_mixing_constituent")
            if amc is not None and amc != v.Species(None) and getattr(amc, "name", ""):
                add(amc)
        return sorted(species.values(), key=lambda s: s.name)

    # -- CEF phases ---------------------------------------------------------
    def _write_cef_phase(self, phases_elem, phase):
        hints = phase.model_hints
        is_ordered = hints.get("ordered_phase") == phase.name
        attrs = {
            _XSI_TYPE: "CEFOrderedPhaseType" if is_ordered else "CEFPhaseType",
            "name": phase.name,
        }
        if is_ordered:
            attrs["disorderedPhase"] = hints["disordered_phase"]
        phase_elem = ET.SubElement(phases_elem, _TM + "phase", attrs)

        dropped = set()
        species_elem = ET.SubElement(phase_elem, _TM + "species")
        for species in self._phase_species(phase):
            self._write_specie(species_elem, species)

        structure = ET.SubElement(phase_elem, _TM + "structure")
        subl_elem = ET.SubElement(
            structure,
            _TM + "sublattices",
            {"multiplicities": " ".join(_format_number(m) for m in phase.sublattices)},
        )
        for subl in phase.constituents or ():
            ET.SubElement(
                subl_elem,
                _TM + "site",
                {"constituents": " ".join(sorted(s.name for s in subl))},
            )
        if "ihj_magnetic_afm_factor" in hints:
            self._write_magnetic(structure, hints)

        # Bucket params into endmembers (every site length 1) and interactions.
        endmembers, interactions = {}, {}
        for param in self._phase_params(phase.name):
            ptype = param["parameter_type"]
            if ptype not in _WRITABLE_PARAM_TYPES:
                self._incompatible(
                    f"ThermML: parameter type {ptype!r} on phase {phase.name!r} "
                    f"has no ThermML v0 representation; skipping "
                    f"(see thermml_format_gaps.md)."
                )
                dropped.add(ptype)
                continue
            ca = param["constituent_array"]
            bucket = endmembers if all(len(s) == 1 for s in ca) else interactions
            bucket.setdefault(ca, []).append(param)

        if dropped:
            # Record the loss in the phase description (first child).
            desc = ET.Element(_TM + "description")
            desc.text = (
                "Parameter types dropped (no ThermML representation): "
                + ", ".join(sorted(dropped))
                + "."
            )
            phase_elem.insert(0, desc)

        if endmembers:
            ems = ET.SubElement(phase_elem, _TM + "endmembers")
            for ca, params in endmembers.items():
                em = ET.SubElement(ems, _TM + "endmember", {"name": _array_label(ca)})
                self._write_constituents(em, ca)
                self._write_cef_properties(em, params)
        if interactions:
            inters = ET.SubElement(phase_elem, _TM + "interactions")
            for ca, params in interactions.items():
                inter = ET.SubElement(
                    inters, _TM + "interaction", {"name": _array_label(ca)}
                )
                self._write_constituents(inter, ca)
                self._write_cef_properties(inter, params)

    def _write_magnetic(self, structure, hints):
        afm = hints["ihj_magnetic_afm_factor"]
        structure_factor = hints.get("ihj_magnetic_structure_factor", 0.0)
        if afm == 0:
            # Xiong (improved) formalism: pycalphad signals it with afm == 0.
            mag = ET.SubElement(
                structure, _TM + "magnetic", {_XSI_TYPE: "IHXMagneticType"}
            )
            afm_xml = 0.0
        else:
            mag = ET.SubElement(
                structure, _TM + "magnetic", {_XSI_TYPE: "IHJMagneticType"}
            )
            afm_xml = -1.0 / afm  # inverse of the reader's afm = -1/AFMFactor
        self._text_el(mag, "AFMFactor", _format_number(afm_xml))
        self._text_el(mag, "structureFactorP", _format_number(structure_factor))

    @staticmethod
    def _write_ref(prop, param):
        ref = param.get("reference")
        if ref:
            elem = ET.SubElement(prop, _TM + "ref")
            elem.text = ref

    def _write_ranked_property(self, parent, xtype, params, comment=None):
        prop = ET.SubElement(parent, _TM + "property", {_XSI_TYPE: xtype})
        self._write_ref(prop, params[0])
        if comment:
            self._text_el(prop, "comment", comment)
        for param in sorted(params, key=lambda p: p["parameter_order"]):
            expr = ET.SubElement(
                prop, _TM + "expr", {"rank": str(param["parameter_order"])}
            )
            expr.text = self._param_expr_text(param["parameter"])

    def _write_cef_properties(self, parent, params):
        by_type = {}
        for param in params:
            by_type.setdefault(param["parameter_type"], []).append(param)

        for param in by_type.get("G", []):
            prop = ET.SubElement(parent, _TM + "property", {_XSI_TYPE: "G"})
            self._write_ref(prop, param)
            self._text_el(prop, "expr", self._param_expr_text(param["parameter"]))

        if "L" in by_type:
            self._write_ranked_property(parent, "L", by_type["L"])
        # TC/BMAGN are stored signed; emit as TCL/BML by rank (the Curie/Neel
        # split of a ChemSage M property is not retained in the Database).
        comment = "Signed value by rank; no Curie/Neel split is retained."
        if "TC" in by_type:
            self._write_ranked_property(parent, "TCL", by_type["TC"], comment)
        if "BMAGN" in by_type:
            self._write_ranked_property(parent, "BML", by_type["BMAGN"], comment)

    # -- MQM / SUBQ phases --------------------------------------------------
    def _write_mqm_phase(self, phases_elem, phase):
        """Inverse of :func:`_read_mqm_phase`: species + groups, MQMG endmembers,
        MQMZ quadruplets, and MQMX interactions."""
        phase_elem = ET.SubElement(
            phases_elem,
            _TM + "phase",
            {_XSI_TYPE: "ModifiedQuasichemicalPhaseType", "name": phase.name},
        )
        chemical_groups = phase.model_hints["mqmqa"].get("chemical_groups", {})
        group_of = {}
        for sub in ("cations", "anions"):
            for species, group in chemical_groups.get(sub, {}).items():
                group_of[species.name] = group

        species_elem = ET.SubElement(phase_elem, _TM + "species")
        for species in self._phase_species(phase):
            self._write_specie(species_elem, species, group=group_of.get(species.name))

        params = self._phase_params(phase.name)
        mqmg = [p for p in params if p["parameter_type"] == "MQMG"]
        mqmz = [p for p in params if p["parameter_type"] == "MQMZ"]
        mqmx = [p for p in params if p["parameter_type"] == "MQMX"]
        other = [
            p for p in params if p["parameter_type"] not in ("MQMG", "MQMZ", "MQMX")
        ]
        for param in other:
            self._incompatible(
                f"ThermML: parameter type {param['parameter_type']!r} on MQM "
                f"phase {phase.name!r} has no ThermML v0 representation; "
                f"skipping (see thermml_format_gaps.md)."
            )

        if mqmg:
            ems = ET.SubElement(phase_elem, _TM + "endmembers")
            for param in mqmg:
                self._write_mqmg_endmember(ems, param)
        if mqmz:
            quads = ET.SubElement(phase_elem, _TM + "quadruplets")
            for param in mqmz:
                self._write_quadruplet(quads, param)
        if mqmx:
            inters = ET.SubElement(phase_elem, _TM + "interactions")
            self._write_mqmx_interactions(inters, phase.name, mqmx)

    def _write_mqmg_endmember(self, parent, param):
        ca = param["constituent_array"]
        cation, anion = ca[0][0], ca[1][0]
        em = ET.SubElement(
            parent,
            _TM + "endmember",
            {"name": _xml_species_name(cation) + ":" + _xml_species_name(anion)},
        )
        self._write_constituents(em, ca, xml_names=True)
        prop = ET.SubElement(em, _TM + "property", {_XSI_TYPE: "MQM-G"})
        self._text_el(prop, "expr", self._param_expr_text(param["parameter"]))
        self._text_el(prop, "zeta", _format_number(param["zeta"]))
        # No <coordinationNumbers>: they are redundant (the real coordinations
        # live in MQMZ/<quadruplets>) and no longer part of the emitted schema.

    def _write_quadruplet(self, parent, param):
        (a, b), (x, y) = param["constituent_array"]
        za, zb, zx, zy = param["coordinations"]
        qd = ET.SubElement(
            parent,
            _TM + "quadruplet",
            {"name": _array_label(param["constituent_array"])},
        )
        for tag, species, z in (("a", a, za), ("b", b, zb), ("x", x, zx), ("y", y, zy)):
            ET.SubElement(
                qd,
                _TM + tag,
                {"species": _xml_species_name(species), "Z": _format_number(z)},
            )

    def _reconstruct_mqmx(self, param):
        """
        Recover the XML mixing layout of an MQMX param: which sublattice mixes,
        its (ordered) members, the common ion on the other sublattice, and any
        asymmetric-ternary ``<selected>`` corner. Inverse of the reader's
        :func:`_read_mqm_interaction_property`.
        """
        ca = param["constituent_array"]
        if ca[1][0] == ca[1][1] and ca[0][0] != ca[0][1]:
            mixing_is_cation, members, common = True, list(ca[0]), ca[1][0]
        elif ca[0][0] == ca[0][1]:
            mixing_is_cation, members, common = False, list(ca[1]), ca[0][0]
        else:
            return None  # reciprocal (A!=B and X!=Y) never enters the Database
        selected = param.get("additional_mixing_constituent")
        if selected is not None and selected != v.Species(None) and selected.name:
            members = members + [selected]
        else:
            selected = None
        return mixing_is_cation, members, common, selected

    def _write_mqmx_interactions(self, parent, phase_name, mqmx):
        # Group params first by reconstructed interaction sites, then by
        # (mixing_code, selected) into properties carrying one <expr> each.
        interactions = {}
        for param in mqmx:
            code = param.get("mixing_code")
            if code not in _MQM_XTYPE:
                self._incompatible(
                    f"ThermML: MQM mixing_code {code!r} on phase {phase_name!r} "
                    f"(reciprocal/Redlich-Kister) has no ThermML v0 "
                    f"representation; skipping (see thermml_format_gaps.md)."
                )
                continue
            layout = self._reconstruct_mqmx(param)
            if layout is None:
                self._incompatible(
                    f"ThermML: reciprocal MQM interaction on phase "
                    f"{phase_name!r} cannot be represented; skipping."
                )
                continue
            mixing_is_cation, members, common, selected = layout
            if mixing_is_cation:
                cat_site = [_xml_species_name(s) for s in members]
                an_site = [_xml_species_name(common)]
            else:
                cat_site = [_xml_species_name(common)]
                an_site = [_xml_species_name(s) for s in members]
            key = (tuple(cat_site), tuple(an_site))
            interactions.setdefault(key, []).append((param, selected))

        for (cat_site, an_site), entries in interactions.items():
            inter = ET.SubElement(
                parent,
                _TM + "interaction",
                {"name": ",".join(cat_site) + ":" + ",".join(an_site)},
            )
            cons = ET.SubElement(inter, _TM + "constituents")
            for site_species in (cat_site, an_site):
                site = ET.SubElement(cons, _TM + "site")
                for name in site_species:
                    ET.SubElement(site, _TM + "const", {"species": name})
            # group entries into <property> blocks by (xsi:type, selected)
            prop_groups = {}
            for param, selected in entries:
                sel_name = _xml_species_name(selected) if selected is not None else None
                pkey = (_MQM_XTYPE[param["mixing_code"]], sel_name)
                prop_groups.setdefault(pkey, []).append((param, selected))
            for (xtype, sel_name), items in prop_groups.items():
                prop = ET.SubElement(inter, _TM + "property", {_XSI_TYPE: xtype})
                for param, selected in items:
                    exponents = param.get("exponents", [0, 0, 0, 0])
                    attrs = {"i": str(int(exponents[0])), "j": str(int(exponents[1]))}
                    if selected is not None:
                        attrs["k"] = str(
                            int(param.get("additional_mixing_exponent", 0))
                        )
                    expr = ET.SubElement(prop, _TM + "expr", attrs)
                    expr.text = self._param_expr_text(param["parameter"])
                if sel_name is not None:
                    self._text_el(prop, "selected", sel_name)


def write_thermml(dbf, fd, if_incompatible="warn", title=None, description=None):
    """
    Write a ThermML XML document for a pycalphad Database.

    The document targets namespace ``http://calphad.org/thermml/v0`` (schema
    version ``0.1.0``) and is the inverse of :func:`read_thermml`:
    ``read(write(dbf))`` reproduces ``dbf``. Only what the Database holds is
    emitted; constructs ThermML v0 cannot
    represent (atomic mobility, property models, two-state, volume, ...) are
    warn-skipped and recorded in ``pycalphad/io/thermml_format_gaps.md``.

    Parameters
    ----------
    dbf : Database
        A pycalphad Database.
    fd : file-like
        Text file descriptor to write to.
    if_incompatible : str, optional
        ``'warn'`` (default), ``'raise'`` or ``'ignore'`` -- governs what
        happens when a parameter or expression cannot be represented in ThermML.
    title : str, optional
        Emitted as ``<metadata><title>`` (no metadata is emitted otherwise; the
        Database has no metadata store).
    description : str, optional
        Emitted as ``<metadata><description>``.
    """
    writer = _ThermMLWriter(
        dbf, if_incompatible=if_incompatible, title=title, description=description
    )
    root = writer.build()
    tree = ET.ElementTree(root)
    ET.indent(tree, space="    ")
    fd.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    fd.write(ET.tostring(root, encoding="unicode"))
    fd.write("\n")


# Register the ThermML format for both reading and writing.
Database.register_format("xml", read=read_thermml, write=write_thermml)
