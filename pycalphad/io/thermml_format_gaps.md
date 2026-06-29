# Where TDB, ChemSage-DAT and ThermML do *not* overlap

Companion to `thermml_plan.md` (reader) and `thermml_writer_plan.md` (writer).
This is the canonical, evidence-backed list of feature gaps discovered while
building the pycalphad ThermML reader **and** writer and round-tripping the full
vendored example corpus (the 48 `xml-from-{tdb,dat}` files, plus the `*.dat` and
`*.tdb` sources in `C:\Users\GTT\ChemSage\udo\test-data`).

The writer emits a warning that points here whenever it meets a parameter it
cannot represent. The headline ask for the ThermML maintainers is at the bottom
(§5): to make ThermML a *true superset* of everything expressible in TDB,
ChemSage-DAT and ThermML itself, it needs a **kinetic/mobility section** and a
**property-model section**, and pycalphad's reader needs to start *consuming* the
volume/metadata sections ThermML already defines.

The mental model for "overlap" is three sources and one in-memory target:

```
        TDB ───┐
   ChemSage ───┼──►  pycalphad.Database  ◄──►  ThermML XML  (this reader+writer)
     ThermML ──┘
```

A feature only survives a *round-trip* if it can live in all the formats on the
path **and** in `Database`. Most gaps below are one link in that chain being the
weakest.

---

## 0. What *does* round-trip (the baseline that works)

Verified by `read(write(dbf)) == dbf` (elements, species incl. composition &
charge, refstates, phases, sublattices/constituents, `symbols`, and every stored
parameter) over **all 48** corpus files plus the three vendored fixtures, with
**GM parity to 0.0** on Pb-Sn (CEF), CrNi (magnetic + order/disorder) and Shishin
(MQM):

* CEF structure: sublattices, site constituents, per-phase species (incl.
  **charged** ionic species like `Al+3`/`O-2` and molecular species like `Al1O`).
* `G` endmembers, `L` (Redlich-Kister) interactions of any rank.
* IHJ magnetism (`ihj_magnetic_*` hints) and order/disorder linkage.
* Global functions (`FunctionTypeExpr`) with **verbatim** names (`Cr:Va#FCC_A1`).
* MQM / SUBQ: cation/anion split, chemical groups, `MQM-G` endmembers,
  `<quadruplets>` (`MQMZ`), binary `MQM-L-PF`/`-SP` (`G`) and
  `MQM-L-Quasichemical` (`Q`), and the asymmetric-ternary `<selected>` corner.
* `TDB → ThermML → pycalphad → TDB` reproduces Gibbs energy exactly for the
  CEF + magnetic + order/disorder systems.

Everything below is what falls *outside* that baseline.

---

## 1. In TDB / ChemSage-DAT but with **no home in ThermML 0.1** (extend ThermML)

These parameter types live in `tdb_keywords.TDB_PARAM_TYPES` and can be present in
a `Database` loaded from TDB. The writer **warns once per type and skips them**
(gated by `if_incompatible`; `'raise'` turns the warning into a
`DatabaseExportError`). They are listed in `_WRITABLE_PARAM_TYPES`'s complement.

| pycalphad type(s) | meaning | proposed ThermML extension |
|---|---|---|
| `MQ`, `MF`, `DQ`, `DF`, `VS` | atomic mobility / diffusivity (activation enthalpy, pre-exponential, volume term) | **`<kinetics>` / `<mobility>` section.** The single largest gap: ThermML has *no* kinetic model, so any DICTRA-style mobility database loses all of its physics on write. Mobilities carry a *diffusing species* (`PARAMETER MQ(PHASE&Species,...)`) that also needs a home. |
| `VISC`, `ELRS`, `THCD`, `SIGM`, `XI` | viscosity, electrical resistivity, thermal conductivity, surface tension, surface-tension damping | **`<propertyModel>` section** for non-energetic physical properties. |
| `GD` | Gibbs-energy difference for a two-state (liquid ↔ amorphous) model | a `<twoState>` construct. |
| `THETA` | Einstein temperature | no dedicated home; could fold into a future heat-capacity model. |
| `NT` | Néel temperature kept *separate* from a signed `TC` | see §3 (magnetism). pycalphad's IHJ already folds Néel into a signed `TC`, so a TDB-loaded `NT` is warn-skipped. |

> These were *decoded but not built* — there is nothing to emit until ThermML
> defines the target elements. The reader has no counterpart either, so even if
> we invented an encoding it could not round-trip.

## 2. Per-parameter temperature ranges (a structural mismatch, worked around)

Both TDB (`PARAMETER G(...) 298.15 <expr1>; 1000 <expr2>; 6000 N`) and ChemSage
let a *single parameter* be piecewise in temperature. ThermML's parameter
`<expr>` is **one expression per Redlich-Kister rank** — it carries no
`<range>` children (only `<globalExpressions>` do).

**Workaround (implemented, lossless for energy):** a multi-branch `Piecewise`
parameter is *hoisted* into a synthesized global function
(`_PYCALPHAD_EXPR{n}`, a `FunctionTypeExpr` with the real `<range>` blocks) and
the parameter is emitted as a brace reference to it. This round-trips Gibbs
energy exactly (verified in both temperature branches), but it does change the
*shape* of the `Database`: the once-inline piecewise becomes `param = {func}` +
a new symbol. Single-branch piecewise parameters (the overwhelmingly common
`(expr, 0.01<=T),(0,True)` TDB shape) are emitted inline with the trivial guard
dropped.

**Recommendation:** allow `<range>` children inside a parameter `<expr>` (or a
`rank`-and-range matrix), so a piecewise parameter can be expressed directly
without a synthetic helper function.

## 3. Magnetism: ThermML keeps *more* than pycalphad (lossy at the Database, not at ThermML)

A ChemSage `M` property on an endmember carries a **`<temperature type="Curie|Neel">`
plus a `<moment>`** — i.e. the ordering type is explicit. pycalphad's IHJ model
stores only signed `TC`/`BMAGN`: it bakes the antiferromagnetic factor into the
sign and **discards the Curie/Néel label**. Therefore the writer **cannot
faithfully reconstruct an `M` property** and instead emits every `TC`/`BMAGN` as
a raw **`TCL`/`BML`** property (signed value by rank), tagged with a `<comment>`
explaining the choice. This round-trips through pycalphad *exactly* but is **not
textually identical** to a ChemSage-emitted file (which would use `M` on
endmembers and `TCL`/`BML` only on interactions).

This is a **`Database` limitation, not a ThermML one** — ThermML already models
the distinction. Closing it is a *reader/Model* milestone: teach pycalphad to
retain the ordering type (e.g. a `magnetic_ordering` flag) so an `M` property can
be rebuilt. Until then the TCL/BML representation is the faithful minimum.

## 4. In ThermML (and ChemSage-DAT) but **not in TDB**, or **dropped by pycalphad**

### 4a. Features TDB cannot express at all
* **Quasichemical (MQM / SUBQ / SUBG).** TDB has no quasichemical formalism, so a
  ThermML or ChemSage MQM phase **cannot be written to TDB** — only the CEF
  phases of such a database survive a `… → TDB` export. (Confirmed: Shishin's
  `SLAG-LIQ` has no TDB representation; its stoichiometric solids do.) This is the
  main "format does not overlap" the user flagged: TDB ⊉ ChemSage.
* **Explicit ternary-interpolation overrides** (`Kohler`/`Toop`/`Muggianu` per
  binary edge, with a Toop `<constant>` corner). ThermML and ChemSage carry these
  per-ternary; pycalphad derives the default extrapolation from `chemical_groups`
  and **does not consume explicit overrides** (the cs_dat importer only warns,
  too — 0 parameters produced). The writer therefore has nothing to emit for them
  and the reader only warns. *Low value until pycalphad implements explicit
  overrides in `model_mqmqa.py`.*

### 4b. ThermML sections pycalphad's reader currently **drops** (so write→read can't preserve)
* **Volume models** — `MolarVolume`, `ThermalExpansion`, `Compressibility`,
  `BulkModulusDerivative`. **ThermML defines these and ChemSage emits them**
  (168 / 79 / 79 / 79 occurrences in the corpus), but pycalphad has no volume
  model, so the **reader warns and drops the coefficients**. Emitting them on
  write is therefore pointless until the reader consumes them — this is a
  *reader* milestone, not a writer gap. (The TDB side `V0`/`VA`/`VC`/`VK` is the
  §1 mirror of the same hole.)
* **Metadata** — `<title>/<version>/<authors>/<revisions>/<references>`. `Database`
  has no metadata store, so the reader drops it and the writer can only emit a
  minimal `<title>`/`<description>` from explicit `write_thermml(... title=,
  description=)` kwargs. **Upstream ask: add `dbf.metadata` / `dbf.references`.**
* **Alternative function & expression flavors** — `RangedTemperatureExpr`,
  `HSCPTemperatureExpr` (h298/s298 + Cp terms), `FunctionTypeTDB`,
  `FunctionTypeCSdat`, and `PurePhaseType`. Only `FunctionTypeExpr` and
  `CEF*PhaseType` are emitted by real producers and consumed here; the others are
  schema-completeness items (reader warns/raises; writer never needs them because
  the `Database` only holds `FunctionTypeExpr`-shaped piecewise symbols).

### 4c. MQM corners decoded but not yet round-trippable
* **Reciprocal excess** (`MQM-L-RM` → `R`, `MQM-L-RS` → `B`, `MQM-L-Reciprocal`).
  ThermML *can* express these and the reader/writer mapping is decoded, but the
  **MQMQA model raises** consuming them (a pre-existing bug that also breaks the
  cs_dat import — `'Species' object is not iterable`, `model_mqmqa.py:696`). They
  never enter the `Database`, so the writer simply never sees them; if one is
  present it is warn-skipped. *Blocked by the model, not the formats.*
* **`MQM-L-RK`** (Redlich-Kister, rank-based MQM excess). No cs_dat-readable
  ground-truth file exists (only in `large_SUBQ`, which cs_dat cannot parse), so
  it is warn-skipped pending a readable fixture.
* **SUBG vs SUBQ** is **not distinguishable from the XML** (the writer always
  emits, and the reader always reads, `type="SUBQ"`). A `<model>`/`subtype`
  attribute on `ModifiedQuasichemicalPhaseType` would disambiguate.
* **Neutral associates** (charge-0 MQM species) are not supported (reader warns
  and skips; writer never emits them because they are not in the Database).

## 5. ThermML *self-consistency* / emitter issues — RESOLVED upstream

These were ambiguities/lossiness in an earlier ChemSage→ThermML emitter that the
reader originally had to *repair*. **The emitter has since been fixed**, so the
corresponding reader work-arounds have been **removed** ("tightened"): the reader
now takes species names and constituents verbatim, and the writer emits clean
files with no compensating logic. They are kept here as a record of what the
emitter must continue to guarantee.

1. **Truncated site constituents** — *fixed.* The emitter used to truncate the
   species name in the `<sublattices>` site `constituents` attribute
   (`Cr2Al11`→`Cr2Al`, `Fe31Al61`→`Fe31Al6`, `Na+1`→`Na+`) while the `<species>`
   block kept the full name. It now writes the full name in both, so the
   constituent token matches a declared species exactly. Reader no longer does
   prefix resolution (`_resolve_constituent` removed); writer emits the full name.
2. **Reduced float precision on global-function constants** — partially. Some
   constants were emitted at ~15 sig figs / short scientific form (`-2.1543e6`
   where the `.dat` has `-2154339.61`, ~40 J/mol), capping GM fidelity. The
   **writer prints every constant at full round-tripping double precision**
   (`repr(float)`), so a pycalphad→ThermML→pycalphad chain loses nothing; the
   emitter should likewise preserve full precision wherever it still rounds.
3. **Trailing count-of-1 mismatch** — *fixed.* The emitter used to write `Al1O1`
   in `<species>` but `Al1O` in constituents. It now writes the **same** name in
   both (`Al1O1` everywhere), so they match verbatim. Reader no longer strips a
   trailing `1` (`_normalize_species_name` removed); names are used as-is.
4. **Trailing-comma `disorderedPhase`** tokens (`"BCC_A2,,,"`) — *fixed.* No
   longer emitted; reader no longer strips them.
5. **Redundant per-endmember `<coordinationNumbers>`** — *fixed.* On an `MQM-G`
   endmember these were the **species charges**, not coordination numbers (the
   real coordinations live in `<quadruplets>`/`MQMZ`). The element has been
   **removed from the emitted schema**; the reader never read it and the writer no
   longer emits it.

---

## 6. One-line summary table (round-trip survivability)

| Feature | TDB | ChemSage-DAT | ThermML 0.1 | pycalphad `Database` | Round-trips here? |
|---|:--:|:--:|:--:|:--:|:--:|
| CEF G / L / structure | ✅ | ✅ | ✅ | ✅ | **yes** |
| IHJ magnetism (signed TC/BMAGN) | ✅ | ✅ | ✅ | ✅ | **yes** (as TCL/BML; §3) |
| Curie/Néel + moment label | ✅ | ✅ | ✅ | ❌ | no (Database drops it) |
| Order/disorder | ✅ | ✅ | ✅ | ✅ | **yes** |
| Quasichemical (MQM/SUBQ) | ❌ | ✅ | ✅ | ✅ | **yes** (not via TDB) |
| MQM reciprocal / RK excess | ❌ | ✅ | ✅ | ⚠ model bug | no |
| Ternary interp. overrides | ⚠ | ✅ | ✅ | ❌ | no (warn only) |
| Volume models | ✅ (`V0`…) | ✅ | ✅ | ❌ | no (reader drops) |
| Mobility / diffusion | ✅ (`MQ`…) | ✅ | ❌ | ✅ | no (ThermML gap) |
| Property models (visc, …) | ✅ | ✅ | ❌ | ✅ | no (ThermML gap) |
| Two-state (`GD`), Einstein (`THETA`) | ✅ | ✅ | ❌ | ✅ | no (ThermML gap) |
| Per-parameter T-ranges | ✅ | ✅ | ⚠ via hoist | ✅ | **yes** (hoisted; §2) |
| Metadata / references | ⚠ | ✅ | ✅ | ❌ | no (Database has no store) |

Legend: ✅ native · ⚠ partial/lossy · ❌ absent.

**To make ThermML a true superset:** add (a) a kinetic/mobility section, (b) a
property-model section, (c) a two-state construct and an Einstein-temperature
home, and (d) per-parameter temperature ranges. On the pycalphad side, to make
the round-trip *complete* rather than *energy-faithful*, the reader needs to
consume volume models and metadata and to retain the magnetic ordering label.
