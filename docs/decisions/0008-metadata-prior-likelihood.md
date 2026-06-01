# 0008 — Metadata-prior × data-likelihood missingness instrument (experiment pre-registration)

- **Date:** 2026-06-01
- **Status:** Accepted as experiment design / pre-registration — implementation and results pending
- **Predecessors:** ADR-0007 (composition-posterior estimand); the Stage A–E composition arc; **Stage F**
  (subtype detectability averages away into the dataset composition); the **metadata-prior bakeoff**
  (`docs/experiments/2026-06-01-metadata-prior-bakeoff.md`).
- **Relationship to ADR-0007:** **Extends**, does not supersede. The composition estimand and its honest
  seam stand. This ADR (a) **cashes out ADR-0007 commitment 4** ("deployment may supply a real-world
  prior") with a concrete mechanism, and (b) **revisits ADR-0007's demotion of Q3**: the per-column layer
  returns — but as a *subtype detector + prior*, not the Stage-5 per-column 3-way classifier that collapsed.

This is a **pre-registration**, not a retrospective: it records what we commit to *before* running, so the
outcome — a useful combined instrument, a wide honest band, or a null — is attributable rather than post-hoc.
Per the folder convention this record is immutable; if results force a redesign, a later ADR supersedes it.

## Context — why a second channel is now warranted

The composition arc (A–E) and **Stage F** established, repeatedly and cleanly, that **MAR-vs-MNAR is not
identifiable from the observed data alone** (Molenberghs): the data instrument *reads structure and
abstains on mechanism*. Stage F is the sharpest statement — even a *loud* threshold/detection MNAR
subtype, which is detectable per-column (Stage 4), **averages away** at the dataset composition because its
surviving aggregate footprint is co-missingness that reads MAR-ward. No amount of model/calibration work on
the data channel moves this; it is a property of the information set, not the model.

But **a real analyst never works from the data alone.** They know what the columns *mean* — and column
meaning carries a strong, legitimate prior over the mechanism: "income" → self-censoring MNAR (heavy at the
extremes, especially zero); a lab assay with a detection limit → detection MNAR; a skip-gated follow-up →
MAR-by-design; a randomized booklet rotation → MCAR-by-design. Molenberghs, read correctly, is not a wall
but a **map of which component requires outside information** — and the prior supplies exactly that
component. The **metadata-prior bakeoff** then showed this prior is *cheaply authorable by a small, local,
HIPAA-safe model*: Qwen2.5-14B (4-bit, 16 GB card) recovers the grounded prior at 0.92, gets every signature
case right — **including the PISA MCAR-by-design blind spot the data channel provably cannot fix** — with no
frontier model in the loop.

## The estimand (what this ADR commits to)

Lacuna's output becomes a **posterior over the missingness composition formed by combining two explicitly
separable channels**, per column, then aggregated to the dataset by the LOCKED by-cell denominator (ADR-0007):

- **Likelihood — the data fingerprint.** `P(observed footprint | mechanism)`: the existing calibrated
  composition/footprint instrument. Bounded by Molenberghs; resolves the identifiable random-vs-structured
  axis, abstains on the MAR-vs-MNAR split.
- **Prior — the column metadata.** `P(mechanism | column metadata)`: authored from a **frozen, auditable
  metadata→prior table** (LLM as a feature-extractor over `name + codebook description + domain`).
- **Posterior ∝ prior × likelihood**, with an explicit **"indeterminate" mass** when *both* the prior is
  uninformative (opaque metadata) *and* the likelihood is at the Molenberghs band.

**Both channels are reported, always** — the data-only posterior (the measurement) *and* the
knowledge-informed posterior (the decision), plus the **disagreement** between them. This is what keeps the
instrument honest and distinguishes it from a column-name lookup table (see commitment 1).

## The Molenberghs reframe (load-bearing for the defense)

We do **not** claim to beat Molenberghs. Molenberghs is a precise, correct statement about identifiability
*from the observed-data distribution*; its value here is that it tells us *exactly which component is
unidentified* and therefore *must* be supplied from outside. The prior is that supply, made **explicit,
auditable, and reusable** — the formalization of the substantive assumption a statistician already brings to
a sensitivity analysis by hand. Framed this way the limit is the justification, not the adversary.

## Design commitments (pre-registered)

1. **Two separable channels; the disagreement is a first-class output; the data must be able to override
   the prior.** The instrument reports `P(mech | data)` and `P(mech | data, metadata)` side by side. The
   **override test** is a pre-registered acceptance criterion, not a nicety: there must exist datasets where
   a strong data fingerprint moves the posterior *off* the prior (e.g. prior says "income → MNAR" but the
   observed column shows no truncation / value-independent missingness → the posterior pulls back and the
   disagreement is flagged). **If the prior is structurally un-overridable, the instrument is a lookup table
   and fails this ADR.**
2. **The LLM is a feature-extractor, not an oracle.** It is asked "what *kind* of variable is this?" (a small
   semantic ontology), never "is this MNAR?" — LLM stated probabilities are miscalibrated and sycophantic.
   The **calibrated mapping** from semantic class to prior strength (and the **abstention threshold**) is
   *fit* on grounded data/consensus, not taken from the model. (The bakeoff confirmed raw abstention is
   uncalibrated — 3B over-abstains, ≥8B under-abstains — so the threshold must be fit.)
3. **Frozen, auditable, versioned prior table.** The LLM authors the metadata→prior map **offline, once,
   under human review**; inference reads the table. This buys **determinism** (Coding Bible Rule 6),
   **reproducibility** (a pinned local model + a frozen table reproduce forever), and **HIPAA safety**
   (schema-only metadata, a local model, nothing in the hot path, nothing leaves the box). The LLM is *how
   the prior is authored*, not a live dependency.
4. **Graceful degradation.** Opaque / uninformative metadata (`V37`, `x3`) → a flat prior → the instrument
   falls back to pure data-driven. It never fabricates a prior from nothing. (Bakeoff: models ≥3B correctly
   abstained on cryptic columns.)
5. **Local model only — no frontier in the loop.** The bakeoff fixes the operating point: **Qwen2.5**, with
   the family-robust floor at ~9–14B (Qwen2.5-3B suffices for Qwen specifically; cross-family the floor is
   ~8–9B). The endpoint may be a distilled small classifier with a fitted abstention threshold. Frontier
   models are used, if at all, only to *bootstrap* the offline table or as a ceiling yardstick — never
   deployed.
6. **Per-column is the natural granularity — this is how Q3 returns.** The prior is *inherently* per-column
   (it is about what each column means), and Stage F showed the loud-subtype data signal is *also* per-column.
   So the per-column layer is restored as a **subtype detector + prior** (threshold / detection / self-censor
   / skip / planned-random / … + an explicit indeterminate class), NOT the per-column MCAR/MAR/MNAR 3-way
   classifier that Stage 5 showed collapses at matched rate. Aggregation to the dataset composition is by
   missing cell, as in ADR-0007.

## Supervision and evaluation

- **The likelihood:** semi-synthetic, calibrated — unchanged from the composition arc.
- **The prior:** a **face-validity object**, validated against *grounded* truth, not measured per-dataset.
  The grounded benchmark (`scripts/metadata_prior/benchmark.json`) anchors it: NHANES published
  limit-of-detection flags → detection MNAR; explicit skip-logic → MAR-by-design; randomized
  administration / PISA rotation / random subsamples → MCAR-by-design; sensitive-item survey-anchor
  consensus → self-censoring MNAR; opaque columns → indeterminate.
- **Headline metric — CALIBRATION of the combined posterior** (per ADR-0007). The combined posterior must:
  (a) beat **data-only** on the non-identifiable axis where the prior carries signal (the whole point);
  (b) **not degrade** data-only calibration on the identifiable axis where the data carries signal;
  (c) pass the **override audit** — demonstrate cases where the data overrides a deliberately-wrong prior.
- **Define "tanked" before running.** The instrument is tanked if any of: the combined posterior is no
  better-calibrated than data-only on the prior-informed axis; the prior is never overridable (lookup-table
  failure, commitment 1); or the frozen prior table cannot be validated against the grounded anchors.

## Experiment stages (one variable at a time)

- **Stage P0 — feasibility bakeoff. DONE** (2026-06-01). A local model authors the grounded prior without a
  frontier model; cross-family confirmed; runtime/quant shown not to matter; Qwen selected.
- **Stage P1 — the prior table.** Author the `metadata → semantic class → (mechanism prior, strength)`
  mapping: a curated table on the anchor domains first (to prove the machinery with zero LLM), then
  LLM-scaled with the chosen local Qwen model, human-reviewed and frozen/versioned. Fit the
  semantic→strength map and the abstention threshold on the grounded benchmark.
- **Stage P2 — the combiner + calibration.** Per-column `prior × likelihood` → posterior; aggregate by
  missing cell; calibrate (the ADR-0007 temperature machinery); report **both channels + disagreement**;
  run the **override test** on constructed prior-vs-data conflicts.
- **Stage P3 — face validity & the override audit on real data.** The PISA MCAR-by-design fix (the prior
  resolves what the footprint can't), the NHANES LOD / skip-logic anchors, the real survey anchors; report
  the indeterminate mass and where prior and data disagree on real anchors.

## Consequences

**Enables:** traction on the non-identifiable MAR-vs-MNAR axis *without* claiming to beat Molenberghs —
because the lean is explicitly attributed to a stated, auditable prior, with the data-only measurement
preserved alongside. Fixes the documented PISA MCAR-by-design blind spot. Restores a *useful*, stable
per-column layer (subtype detection) that the Stage-5 classifier could not deliver.

**Forecloses / costs:** an LLM + metadata dependency (mitigated to a frozen, reviewed, local table); a shift
of some belief onto assumptions (mitigated by always reporting the data-only channel and the disagreement);
the prior is only as good as the metadata (mitigated by graceful degradation). A narrative risk — that the
LLM looks like the contribution — is mitigated by keeping the data fingerprint the scientific core and the
prior a principled, well-understood Bayesian add-on.

**Conditions to revisit / supersede:**
- The override never fires (the prior dominates on every case) → the instrument is a lookup table; narrow
  the claim to "encodes a documented prior" and drop the inferential framing.
- The frozen prior table cannot be validated against the grounded anchors → degrade to data-only (an honest,
  smaller claim — the composition arc as it stands).
- A reviewer rejects combining a knowledge prior with the data likelihood as outside the estimand → report
  the two channels separately and let the analyst combine (the channels are separable by construction).

## For statisticians using this tool

The data channel is a **measurement** bounded by Molenberghs (identifiability from observed data). The prior
channel encodes **substantive assumptions** about the data-generating process, the same kind a sensitivity
analysis requires — here surfaced, made auditable, and reusable instead of left in the analyst's head. The
combined posterior is a *belief under stated assumptions*; the data-only posterior is reported alongside so
the assumptions' contribution, and any tension with the data, is always visible. We do **not** claim to
identify MAR-vs-MNAR from data; we claim to combine the data's evidence with an explicit prior, honestly.

## Cross-references

- Estimand & denominator this builds on: `docs/decisions/0007-composition-estimand.md`.
- Why a second channel: `docs/experiments/2026-06-01-stageF-subtype-detectability.md` (data-only averages
  Pillar 1 away) and `docs/experiments/2026-05-30-composition-arc-synthesis.md`.
- Feasibility of the local-model prior: `docs/experiments/2026-06-01-metadata-prior-bakeoff.md`;
  benchmark `scripts/metadata_prior/benchmark.json`; harness `scripts/metadata_prior_bakeoff.py`.
- Data instrument to reuse: `lacuna/data/missingness_footprint.py`, `lacuna/models/composition_head.py`,
  `lacuna/training/composition_calibration.py`. Real anchors: `lacuna_survey/anchors.py`.
- Superseded primitive (the per-column 3-way classifier that collapsed): the Stage-5 result in
  `docs/experiments/2026-05-29-stage5-true-mixtures.md`.
