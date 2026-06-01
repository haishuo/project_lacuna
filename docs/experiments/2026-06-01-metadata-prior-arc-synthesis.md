# The metadata-prior arc (ADR-0008) — synthesis: Stages P0–P6

- **Date:** 2026-06-01
- **ADR:** docs/decisions/0008 (metadata-prior × data-likelihood instrument).
- **Branch:** `experiment/metadata-prior` (off `experiment/composition-stageb`). This doc ties the
  per-stage findings into one narrative; per-stage docs are `2026-06-01-{metadata-prior-bakeoff,
  stageP1-prior-table,stageP2-override-audit,stageP3-anchors-combined,stageP5-combined-calibration,
  stageP6-abstention}.md`.

## The question

Stage F (composition arc) proved the data channel reads STRUCTURE and abstains on MECHANISM: MAR-vs-MNAR is
non-identifiable from observed data (Molenberghs), and even a *loud* MNAR subtype averages away into the
dataset composition. But a real analyst is never working from the data alone — column *meaning* carries a
legitimate prior (income → self-censoring MNAR; a detection-limit assay → detection MNAR; a rotated booklet
→ MCAR-by-design). ADR-0008 asks: can we author that prior cheaply and locally, combine it with the data
likelihood, and get **honest** traction on the non-identifiable axis — *without* claiming to beat
Molenberghs, and with the data able to override a wrong prior?

## The arc at a glance

| stage | question | headline result |
|---|---|---|
| **P0** bakeoff | does authoring the prior need a frontier model? | **No.** Qwen2.5-14B (4-bit, 16 GB) hits 0.92 on grounded columns; codebook descriptions are load-bearing; not Qwen-specific (Gemma2-9B ties); family-robust floor ~8–9B |
| **P1** prior table | freeze the metadata→prior map | prior = Dirichlet pseudo-counts per semantic class, pooled with the likelihood by **summing evidence**; LLM-driven 0.92 strong / 1.0 consensus; flat prior = graceful degradation |
| **P2** combiner + override audit | combine, and can the data override? | **combine in RAW evidence** (calibrated-space over-flattens); data overrides on the identifiable axis (survival 0.44) < prior decides on the split (0.56); a correct prior cuts split-MAE 0.25→0.16, a wrong one inflates it to 0.42 (data can't rescue the non-id axis) |
| **P3** real anchors | end-to-end on real survey data | Qwen reads design facts mechanism-blind; the **PISA MCAR-by-design blind spot is fixed**; sensitive items lean MNAR; disagreements surfaced (`chile`: income → prior MNAR, data MCAR, consensus MAR) |
| **tiered fix** | strength by epistemic tier | fact tier 0.85 / gut tier 0.65 → **PISA-2018 flips to MCAR** (blind spot fully fixed) while the gut tier stays overridable |
| **P4** confidence gate | mitigate misclassification amplification | self-consistency gate: PISA preserved (planned_random 7/7), `bfi` de-rated (skip_gated 4/7 → over-drive removed); limit = confidently-wrong labels |
| **P5** combined calibration | calibrate the combined posterior | a temperature calibrates it for *any* prior reliability; a reliable prior improves split-axis Brier 0.150→0.117 — **calibration fixes confidence, not correctness** |
| **P6** abstention | the "unable to determine" output | calibrated selective prediction: data-only abstains on the non-identifiable axis; a reliable prior lifts coverage@80%-acc 0.66→1.0 |

## The through-line

**Two separable channels, honestly combined.** The data likelihood reads what is identifiable (structure)
and is allowed to override the prior where it is informative; the metadata prior supplies the
non-identifiable MAR-vs-MNAR lean where the data is silent. Every stage is an expression of this:

- **The data overrides where it can; the prior decides where it can't.** P2's survival ratios (0.44
  identifiable vs 0.56 split) and P6's commit-rates (data-only 100% on MCAR, abstains on the split) are the
  same seam from two angles. A wrong prior is harmless on the identifiable axis (the data overrides it —
  P2/P6 MCAR stays correct) but uncorrectable on the non-identifiable axis (the data has nothing to override
  with — P2 split-MAE 0.42, P3 `chile`).
- **The prior's value is real but contingent and uncheckable.** A *reliable* prior adds genuine resolution
  on the split (P2 0.25→0.16; P5 Brier 0.150→0.117; P6 coverage 0.66→1.0 with acc 0.81–0.86). An
  *unreliable* one amplifies error (P2 0.42) and the data can't catch it — so the prior must be auditable,
  reported as a separate channel, and its strength scaled to its epistemic tier (tiered fix) and gated on
  classifier confidence (P4).
- **The instrument is honest by construction.** It calibrates regardless of prior accuracy (P5), abstains
  by default on the non-identifiable axis (P6), and surfaces every prior-vs-data disagreement (P3). It does
  not claim to identify MAR-vs-MNAR from data; it claims to combine the data's evidence with an explicit,
  bounded, overridable prior — and to say "unable to determine" when neither channel supports a call.

## What the honest/negative results establish (findings, not gaps)

- **No frontier model needed (P0)** — and it isn't wanted in a HIPAA-shaped, reproducibility-bound loop;
  the prior table is frozen and authored offline by a local model.
- **Combine in raw evidence, calibrate after (P2→P5)** — the calibration must come *after* the combination,
  or τ-flattening erases the data's differential informativeness and the prior dominates both axes.
- **Strength must be tiered and gated (tiered fix + P4)** — a uniform cap left design facts too weak to fix
  the PISA blind spot; a uniform strong prior amplified misclassifications. Fact tier strong (overrides a
  misreading data channel), gut tier capped/overridable, both gated on classifier self-consistency.
- **Calibration ≠ correctness (P5), and abstention is the honest floor (P6)** — the user's "priors amplify
  error" made precise: the instrument is always reliable, useful only in proportion to the prior's accuracy,
  and explicitly declines to commit on the non-identifiable axis without a reliable prior.

## What was built (additive; the v1.0 encoder + generator registry untouched)

- Prior channel: `lacuna/priors/metadata_prior.py` — Dirichlet prior per semantic class, tiered + gated
  strength, evidence-pooling combiner, by-cell aggregation, channel-disagreement, selective decision (42 tests).
- Bakeoff + benchmark: `scripts/metadata_prior_bakeoff.py`, `scripts/metadata_prior/benchmark.json`.
- Stage scripts: `scripts/stageP{1,2,3,4,5,6}_*.py` (+ the frozen `prior_table.json`).
- Reused unchanged: the composition encoder/head/calibration (`lacuna/{models,training}/composition_*`) and
  the survey anchors. Suite 1712 → 1754 passed / 1 skipped throughout; registry bit-identical.

## Defensible claims vs scoped caveats (for the write-up)

**Claim:** a metadata-authored prior, combined with the calibrated data likelihood in raw evidence space,
(a) is authorable by a small local model, (b) lets the data override a wrong prior on the identifiable axis,
(c) supplies calibrated resolution on the non-identifiable axis *when the prior is reliable*, (d) fixes the
documented PISA MCAR-by-design blind spot, (e) is calibrated and (f) abstains honestly when it cannot tell —
all while reporting the two channels and their disagreement separately, and *without* claiming to identify
MAR-vs-MNAR from data.

**Caveats (in neon):** the prior's *accuracy* on real data is validated only by face validity (no mechanism
ground truth; the consensus labels are themselves elicited judgement — semi-circular except on the fact tier
of LOD/skip/rotation); prior reliability in P2/P5/P6 is *simulated* (ρ injected) because the real prior is
mismatched on semi-synthetic; the confidence gate catches *unstable* misclassifications, not confidently-
wrong ones; survey-scoped; the demonstrated granularity is the 3-way mechanism composition, not the subtype.

## Open leads (characterised, not chased)

- **The subtype layer** (restored Q3 as a per-column subtype detector + prior): property 3's literal "20%
  threshold MNAR" lives here; the P1–P6 machinery transfers directly.
- **A confidently-wrong fact label** needs a second metadata-side check (P4 limit) — a data-side one would
  re-couple the channels.
- **Online τ_c / prior-reliability fit** once any labelled real-mechanism data exists.
