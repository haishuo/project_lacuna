# Lacuna-Survey — Architecture & Program (Master Document)

*Single source of truth for the current architecture phase. **Synthesis + design of record.** Status:
**living document; specs below are pending approval (no code yet).** Consolidates the ~17 docs of the
2026-06 architecture phase. Where this conflicts with an earlier spec's text, **this document's framing
is current** (the earlier specs predate the last three framing updates). Governed by `NORTH-STAR.md`.*

**Status legend:** ✅ decided/approved · 🟡 pending approval (no code) · ⏸️ deferred.

---

## 0. One paragraph

Lacuna-Survey asks: **with column *names stripped*, can the observed data of survey tables — pooled across
many — tell us which columns deserve a sensitivity analysis and how strong a δ-prior they warrant?** It is
a **governance** tool, not a mechanism oracle: it never claims to identify δ from data alone. The current
architecture is a **two-stage, column-primary** design — (A) a distribution-agnostic **MCAR-vs-not gate**,
then (B) a **per-column distribution encoder (φ)** feeding a **calibrated δ-prior + detectability +
abstention** — built on the validated semi-synthetic scaffold (generators, oracle, leakage gate,
calibration, held-out ladder). The old BERT backbone is **replaced as the δ spine** (proven mismatched);
the scaffold is **kept**.

---

## 1. The object (what Lacuna-Survey estimates) ✅

> **observed-data law (identified)  +  named survey-manifold prior  →  calibrated δ-prior + detectability +
> abstention.**

Three layers (`ARCHITECTURE-OBJECT-revised.md`):
- **Layer 1 — the missingness CONSEQUENCE** (estimable): the distributional distortion of the observed
  data relative to a plausible reference. The network's real job; a two-distribution comparison, *not*
  classification.
- **Layer 2 — the identification BRIDGE** (consequence→δ): many-to-one (Molenberghs); **carried by the
  manifold prior**, not learned.
- **Layer 3 — the OUTPUT**: a calibrated δ-prior + a detectability state + abstention.

The **reference is the δ=0 slice of the named prior** — a prior object, *not* an identified MAR
expectation (`INFERENCE-OBJECT-CRITIQUE.md`). "Deviation" is an *optional internal parameterization*
(Level 2), not the object.

## 2. The canonical question & deliverable ✅

`SURVEY-REALISM-and-the-core-question.md`:
> **Strip the column names from a survey table — pool many — can the data *alone* say which columns
> deserve a sensitivity analysis, and how strong a δ-prior?**

This is exactly the **no-metadata** design (φ reads values, never names). Consequences:
- **Primary deliverable = column triage / flagging** (which columns warrant sensitivity analysis), not
  δ-magnitude. δ-magnitude is secondary.
- **Honest caveat:** "data alone" still leaks coarse semantic *type* via distributional shape →
  **"near-semantics-free," not semantics-free**. State this whenever reporting the data-alone result.

## 3. Identification, the comparison class, and the "lab-coat" hypothesis ✅

`COMPARISON-CLASS-and-two-stage-design.md`, `SURVEY-REALISM-and-the-core-question.md`:

- **"Detectable" is always relative to a *named plausible class*, never "air" or all-of-math.** Our oracle
  already compares an MNAR mechanism against the **best-fitting MAR-on-observed-predictors** (a plausible
  family), not the infinity. So:
  - own-value self-censoring → **"flat" because its MAR competitor (demographic-driven nonresponse) is
    genuinely plausible** and reproduces the footprint.
  - top-coding → **"detectable" because its only MAR mimic is off-manifold/contrived** (= M3, manifold
    near-injectivity).
- **Binding phrasing:** every (non-)detectability claim **names its comparison class** (profiled
  MAR-on-predictors / `P_prior` support); drop unqualified "non-identifiable." "Self-censoring isn't
  detectable" → **"indistinguishable from the best *plausible* MAR-on-predictors at matched rate."**
- **Open research question (M3):** is profiled-MAR-on-predictors the *right* plausible class? Substantive,
  re-foregrounded as a thesis contribution.
- **The "lab-coat" hypothesis:** if the semantics-stripped data is uninformative for an idiom, then
  sensitivity analysis on it is driven by *domain knowledge*, not data — "domain knowledge in a lab coat."
  **Endorsed, but (a) idiom-dependent** (top-coding detectable; sensitive-item self-censoring flat — and
  the flat ones are exactly the sensitive items that dominate real MNAR worry), **(b) oracle-gated** (a
  "no" is a result only from the Bayes-optimal oracle, not a weak model), **(c) best as a MEASUREMENT**:
  quantify the **data-vs-semantics ("lab-coat") fraction** = Level-1 (data-alone) recovery vs the metadata
  channel. The own-value flat result is a *measurement*, not a failure — potentially the project's most
  important contribution (epistemology of sensitivity analysis).

## 4. The empirical chain that earned the architecture ✅

| step | result | doc |
|---|---|---|
| Architecture-fitness | encoder is row-primary/averaging; δ needs within-column order-statistics | `ARCHITECTURE-FITNESS-delta-estimation.md` |
| Distributional-stream **patch** | live + used (Gate 1) but did NOT move held-out (Gate 2: LOD mean 0.589 vs ECDF 0.577, both ≪ LR 0.754) | `ARCHITECTURE-CHANGE-AUDIT-distributional-stream.md` |
| §8 representation probe | old backbone reps don't carry/preserve the signal (~0.52, R²≤0, random *and* trained) | `probe-encoder-representation-findings.md` |
| **Stage 0** | a learned **column-primary φ recovers it** (0.735) where the backbone loses it (0.548), same regime | `probe-stage0-column-primary-findings.md` |
| Architecture investigation | **Gate I met** (replace the spine); **Gate II** (does a neural deviation module beat raw-ECDF) untested | `ARCHITECTURE-INVESTIGATION-lacuna-survey.md` |

**Verdict:** the BERT backbone is mismatched as the δ spine → **replace it with a column-primary spine**;
the scaffold is salvageable. φ-alone only *matches* raw-ECDF (no within-column learning headroom), so a
heavier neural architecture must earn itself via Layer-2 deviation (Gate II, deferred).

## 5. The architecture (design of record) 🟡

**Two stages** (`COMPARISON-CLASS-and-two-stage-design.md`) + **three product levels**
(`ARCHITECTURE-OBJECT-revised.md`):

- **Stage A — MCAR-vs-not gate** (the *identifiable* axis, §2 solid ground; old-Lacuna's real strength).
  Distribution-agnostic learned detector (Little's test assumes multivariate normality, violated by real
  survey data; prior art: Jamshidian-Jalal 2010 / Kim-Bentler — Lacuna = learned/calibrated/manifold/
  integrated). **If MCAR → ignorable → no sensitivity analysis needed (triage answer); stop.**
- **Stage B — only if not-MCAR** → the column-primary δ-prior pipeline:
  - **Level 0** — per-column φ distribution encoder (order-statistic/ECDF pooling; *proven* — Stage 0).
  - **Level 1 (MVP)** — φ + mask-topology + named-prior + calibration + **detectability** + **abstention**.
    Adds **governance**, not accuracy.
  - **Level 2 (⏸️ gated, Gate II)** — optional conditional reference/deviation module (historically
    fragile: `transfer_features` failed; predictor-referencing). Built only if it beats raw-ECDF OOF.

**Per-column output:** **{ MCAR-departure, δ-prior, detectability, UNKNOWN }** where **"unknown" = off-
manifold / abstain** ("a mechanism we haven't learned") — a **first-class label**, distinct from
in-manifold-flat (which returns the prior, *confidently uncertain*). Most common in natural missingness.

**Load-bearing minimum** (`ARCHITECTURE-INVESTIGATION` §D): within-column φ + calibrated output +
(for Level 2) a deviation mechanism. φ-alone = raw-ECDF, so Level 2's deviation module is the only thing
that would justify a *neural* architecture over a one-line raw-ECDF baseline.

## 6. Idiom vocabulary — survey-realism ✅

`SURVEY-REALISM-and-the-core-question.md`:
- **"LOD" is an instrument idiom (Lacuna-Instrument), NOT survey.** Our step mechanism is survey-realistic
  as **top-coding / bracketing** (CPS/ACS income confidentiality). **Rename `lod_top_coding → top_coding`;
  assay-LOD out of scope.**
- **Idiom map:** own-value self-censoring = **social-desirability / sensitive-item** (survey ✓, *flat*);
  step = **top-coding** (survey ✓, *detectable*).
- **Survey vocabulary only:** item nonresponse, skip logic/branching, top-coding/bracketing, social-
  desirability self-censoring, attrition/unit-nonresponse, DK-vs-refuse. Instrument concepts excluded.
- **NHANES projection restriction:** questionnaire + demographic items only; **exclude lab/exam assays**
  (instrument-LOD).

## 7. The named prior & data-role discipline ✅

`PROPOSAL-Level1-design-spec.md` §9.0, §2.2:
- **`P_prior` = {real survey datasets} × {survey idiom vocabulary} × {δ-grid} × {matched-rate} × {φ bias}**
  — the explicit, auditable semi-synthetic *training measure* (NOT a hand-authored density or LLM opinion).
  The model returns the posterior under `P_prior`; recorded in a manifest `named_prior` block incl.
  `prior_marginal`.
- **Three data roles (binding):** **A natural** (real missingness, no δ truth — manifold/OOD/detectability
  validation + generation, *never* supervised) · **B complete projection** (the *sole* supervised δ
  corpus) · **C synthetic worlds** (training amplifier; never a validation substitute). Ingestion is
  **dual-output** (archive A + emit B). **Never train on natural missingness.**

## 8. Data: inventory, projection, scaling ✅ / 🟡

- **Ground-truth inventory** (`DATA-INVENTORY-ground-truth.md`): catalog scans only `raw/` → 12 `survey_*`
  (role B, 0 natural missingness) + 31 generic ML. **Physically present but un-wired:** NHANES (full +
  modules), ESS R11, PISA, GSS — mostly **role A** (natural missingness). Three layers: (a) physical ≫ 12,
  (b) catalog-wired = 43, (c) training-eligible = 12 **→ 9 genuine** (drop cars93/computers/survey ✅).
- **Role-B projection feasibility** (`DATA-ROLE-B-PROJECTION-feasibility.md`): on-disk assets → **9 →
  ~13–15 genuinely/weakly-distinct worlds across ~6–8 domains** without downloads; the **win is DOMAIN
  coverage** (demographics, health, education, attitudes), not raw count. NHANES-modules / ESS-countries
  are **correlated blocks**, not independent worlds — do **not** count slices as scaling evidence.
- **Evidence standards** (`PROPOSAL-Level1-design-spec.md` §9.A): judge at the **dissertation/grant**
  standard, not deployment. **Dissertation success** = reproduce Stage-0 signal in-pipeline *and* metrics
  improve with diversity. **Grant success** = the cross-domain learning-curve slope is **positive at the
  catalog edge** ⇒ acquisition justified. The curve is **the deliverable scientific result**, not a gate.
- **Cross-domain learning curve** (`PROPOSAL-cross-domain-corpus-plan.md`): x-axis = cumulative **domains**;
  **block-aware leave-DOMAIN-out** split (NHANES = one block, ESS = one block — leakage guard *and* the real
  transportability test); two readings (add-domain-improves-OOF; leave-one-domain-out transfer).

## 9. Plan of record (sequence) ✅ direction; 🟡 each step

1. **Build the Level-1 φ-spine** (🟡 — needs Stage-1 spec + 4 sub-decisions).
2. **Project on-disk role-A → role-B bases** (🟡 — needs the role-B projection spec): NHANES demographics +
   income (top-coding) + weight (top-coding/LOD-as-top-coding) + ESS pooled; PISA ⏸️.
3. **Run the cross-domain learning curve** (needs 1 + 2).
4. **Decide acquisition from the slope.**
5. **Synthetic generation (⏸️) only then.**

Steps 1 and 2 are independent and parallelizable once approved.

## 10. What is salvaged / replaced / removed ✅

**Keep** (scaffold): generators, δ-bins/RPS/metrics/temperature, **leakage gate**, **oracle** (now also the
detectability target source + the comparison-class engine), manifest, `consequence_features`,
`masked_quantile_pool`. **Replace** (spine): tokenization-for-δ → column-major; **BERT encoder → φ
column-primary spine** (not the δ spine; may survive as an auxiliary mask stream only); mean/attention
pooling → order-statistic pooling. **Remove** (from δ path): encoder-coupled heads (`DeltaPriorModel`,
`TargetConditionedDeltaModel`, `rep_ecdf`), `transfer_features`, the v1.0 MoE/3-class/decision.

## 11. Non-goals & guardrails ✅

No metadata channel yet (⏸️) · no Level-2 reference/deviation yet (⏸️, Gate II) · no training on natural
missingness · no claim Level 1 beats raw-ECDF on *discrimination* (value = governance) · no 3-class
mechanism objective · no assay-LOD (instrument scope) · no claim to beat non-identifiability · never an
unqualified "non-identifiable" (name the comparison class) · count **domains/independent worlds**, not
files.

## 12. Open decisions (pending PI) 🟡

- **Stage-1 sub-decisions:** `e_col`=32; full 7-bin grid + binary δ0-vs-δ2.5 slice; 5/2/2 split (keep
  cps1985/workinghours test); new `level1_train.py`. *(`PROPOSAL-Stage1-implementation-spec.md`)*
- **Role-B projection / cross-domain:** NHANES modules; ESS pooled-vs-country; leave-one-domain-out; flagged
  separate registry; sentinel/τ params; parallel sequencing. *(`PROPOSAL-cross-domain-corpus-plan.md`)*
- **Framing updates to fold into the specs on approval:** two-stage MCAR-gate (elevate MCAR-departure to
  Stage A); "unknown" as a first-class label; `lod_top_coding → top_coding` rename + NHANES-questionnaire-
  only; detectability defined comparison-class-relative; "lab-coat fraction" as a headline aim.

## 13. Document index (supporting detail)

**Charter/history:** `NORTH-STAR.md` · `DECISION-MEMO-P2.2c-consolidation.md` · `PHASE-SUMMARY-P2.2-P2.2b.md`
**Architecture analysis:** `ARCHITECTURE-FITNESS-delta-estimation.md` · `ARCHITECTURE-CHANGE-AUDIT-
distributional-stream.md` · `probe-encoder-representation-findings.md` · `probe-stage0-column-primary-
findings.md` · `ARCHITECTURE-INVESTIGATION-lacuna-survey.md`
**Object/foundations:** `INFERENCE-OBJECT-CRITIQUE.md` · `ARCHITECTURE-OBJECT-revised.md` ·
`SURVEY-REALISM-and-the-core-question.md` · `COMPARISON-CLASS-and-two-stage-design.md`
**Specs/plan:** `PROPOSAL-OptionC-column-primary-architecture-spec.md` · `PROPOSAL-Level1-design-spec.md`
(incl. §9 data sufficiency) · `PROPOSAL-Stage1-implementation-spec.md` · `PROPOSAL-cross-domain-corpus-
plan.md`
**Data:** `DATA-INVENTORY-ground-truth.md` · `DATA-ROLE-B-PROJECTION-feasibility.md` ·
`lacuna_survey/DATA_ACQUISITION.md` (v1.0-era acquisition guide)
**Superseded (history):** `PROPOSAL-distributional-consequence-stream-audit.md` (the patch — now Option B,
not built on)

---

*If a future statement about Lacuna-Survey contradicts this document or the North Star, the statement is
wrong, not this. This document is updated as decisions are made; the per-topic docs carry the detailed
reasoning.*
