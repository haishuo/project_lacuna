# P2.2b — Cardinality / Strong-δ Probe — Findings (investigation-closing)

*Branch `p2/delta-prior-rearchitecture`. Note: `feasibility-p2p2b-cardinality-probe-note.md`.
This closes the P2.2/P2.2b "learn δ from the real-X footprint" investigation. Stop-for-review —
the recommended next step is a STRATEGIC decision, not another model-tuning diagnostic.*

> **FRAMING CORRECTION (2026-06-05, PI ruling — supersedes parts of this doc).** Two precisions, now
> binding (see NORTH-STAR §8):
> 1. **These experiments are SEMI-SYNTHETIC, not "real missingness."** Every run here is
>    *semi-synthetic missingness imposed on real survey X* — real values, holes WE punched, δ KNOWN.
>    That is the legitimate supervised test (NORTH-STAR §8.1 rung 2), not a real-data performance
>    claim. Read every "real-X floor" below as "the footprint→δ channel is flat *even on
>    semi-synthetic data with known δ on real survey X*."
> 2. **The result is NARROW and does NOT license replacing Lacuna with a metadata/LLM prior.** It
>    marks *own-value smooth self-censoring, single column, matched rate, current footprint* as a
>    **flat-likelihood idiom** (NORTH-STAR §8.2) — one idiom, not Lacuna. §2 predicts other idioms
>    (LOD/top-coding, skip logic) are more detectable; untested. The "re-scope to prior-driven
>    governance / metadata-authored prior" recommendation in the §"Strategic implication" and
>    §"Recommendation" sections below **OVERREACHED and is withdrawn.** A metadata-prior channel is
>    admissible ONLY if itself calibrated against held-out semi-synthetic answer sheets
>    (NORTH-STAR §8.3) — never as unevaluated opinion. The corrected fork (test another idiom vs build
>    the evaluated metadata channel) is in the session report; the technically-correct *findings*
>    (the tables and what is ruled out) stand as written.

## Headline: the own-value δ footprint is effectively INVISIBLE on real survey X at matched rate

Two facts, both decisive:
1. **Cardinality does not rescue it.** A target-conditioned 7-bin δ-prior trained with
   cardinality-stratified sampling floors on **all** strata; high-cardinality/continuous targets
   (wage, price, weight) are **no better** than low-cardinality coded items — if anything slightly
   worse.
2. **Strong δ is unreadable.** Even the crudest, strongest contrast — δ=0 (no censoring) vs δ=2.5
   (extreme self-censoring) — is at chance, AUC ≈ 0.50, including on high-cardinality continuous
   targets (0.584, barely above chance).

So this is the decision-rule branch: *neither high-cardinality nor strong-δ learns ⇒ real-X
own-value self-censoring at matched rate is effectively invisible under the current observable
footprint.* The signal that is crisp and learnable on clean synthetic Gaussian X (rung 1, adj-acc
0.90) does **not survive** on real survey columns — and it is **not** a discreteness, resolution,
proxy, scale, or localization problem.

## Part A — cardinality-stratified 7-bin (uniform RPS = 0.1905)

Safeguard (printed pre-train): genuine cardinality range — low `n_unique∈[3,8]` (n=10), medium
`[10,59]` (n=14), high `[67,5970]` (n=11; wage 5970, price 808, weight 238).

| stratum | test RPS | (uni−RPS)/SE | bin | adj | entropy (bits) | P(δ=0) |
|---|---|---|---|---|---|---|
| low (≤8) | 0.1898 | +0.08 | 0.175 | 0.483 | 2.740 | 0.122 |
| medium (9–60) | 0.1936 | −0.37 | 0.150 | 0.425 | 2.707 | 0.117 |
| high (>60) | 0.1955 | −0.61 | 0.108 | 0.425 | 2.729 | 0.126 |

All at the floor; **no cardinality gradient** (high ≤ low). Discreteness is not the bottleneck.

## Part B — strong-δ contrast δ∈{0, 2.5} (DIAGNOSTIC; AUC of P(δ>0), chance = 0.5)

| stratum | AUC | bin | (the RPS column uses the 7-bin uniform ref and is not meaningful for a 2-value grid) |
|---|---|---|---|
| low (≤8) | **0.497** | 0.517 | — |
| medium (9–60) | **0.481** | 0.483 | — |
| high (>60) | **0.584** | 0.525 | — |

Even no-censoring vs extreme-self-censoring is ~indistinguishable from the observable footprint on
real X. Not a resolution/granularity issue — the binary, strongest-effect version is also invisible.
(Labeled `kind="ablation"`; binary never used as a P2 objective.)

## The complete ladder (everything ruled out)

| hypothesis for the real-X floor | verdict | evidence |
|---|---|---|
| δ-prior concept / RPS / 7-bin / objective | **ruled out** | rung 1 PASS (synthetic, adj 0.90) |
| target localization / pooling interface | ruled out | rung 3 (conditioning no help) |
| per-example evidence scale | ruled out | rung 3b (1024 rows floored) |
| observed-column proxy absorption | ruled out | proxy sweep (floors at R²≈0.02) |
| target discreteness / cardinality | **ruled out** | Part A (no cardinality gradient) |
| ordinal resolution (7-bin too fine) | **ruled out** | Part B (strong-δ binary also ≈ chance) |

**Remaining and now affirmatively supported: the own-value self-censoring footprint, at matched
rate on a single real survey column, carries essentially no recoverable information about δ.** The
data likelihood for δ is, for this idiom, nearly FLAT.

## Interpretation — this is genuine non-identifiability, and it VALIDATES the North Star

The synthetic case learned precisely because it was a clean, shared, fully-specified generative
process — a semi-artificial setting where the δ-distortion of the conditional is crisp and the
baseline is learnable across examples. Real survey data exhibits the **genuine non-identifiability
the charter is built on** (Molenberghs): from a single observed (censored, matched-rate) dataset,
the δ-induced distortion of `p(observed target | predictors)` is not separable from the unknown
true conditional. The model's uniform max-entropy output on real-X is, for this idiom, the
**honest** answer — *but* it is uniform because real-X is uniformly unidentifiable here, so there
is no within-real-X identifiability gradient to calibrate a useful abstention signal against.

Crucially: per our own falsifiability discipline, we may now say the real-X floor is **not a
representation bug we failed to fix** — we removed proxy, scale, localization, discreteness, and
resolution, and a STRONG effect is still invisible. The remaining explanation is that **the
information is not in the matched-rate single-column footprint.** That is a property of the data +
idiom, not of the head, loss, or encoder.

## Strategic implication (the real decision — not another tuning run)

The "learn δ from the real-X footprint" frame is the wrong frame for real survey data with this
idiom. The North Star already anticipated this: Lacuna is not a mechanism *identifier*; it is a
**calibrated prior + sensitivity report**. The empirical result now forces the architecture to
match that:

1. **Prior-driven, not likelihood-driven, for this idiom.** When the data likelihood is flat, the
   δ-prior must come from *outside* the footprint — the **metadata-authored prior** (the
   metadata-prior / ADR-0008 direction in memory: an LLM/metadata-authored prior over δ combined
   with whatever weak data likelihood exists), with the data providing little-to-no update. The
   product is then the **deterministic sensitivity / tipping-point report** (PROPOSAL §7), honestly
   labeled "δ not identifiable from data; here is the assumption-indexed sweep."
2. **Abstention becomes the default, not a trigger.** For own-value self-censoring on real survey X,
   the calibrated output is a wide prior + sensitivity sweep essentially everywhere — and that is
   correct, not a failure, *as long as we say so explicitly* and do not dress flat-likelihood
   behavior as confident inference.
3. **New information is required to do better.** Recovering δ on real data would need signal beyond
   the matched-rate single-column footprint — a different idiom (skip-logic, attrition,
   unit-nonresponse), auxiliary/covariate information, or relaxing the matched-rate defusing (the
   rate itself, though a weak/unreliable cue per P1). These are separate research arcs, not tweaks.

## Recommendation

**Do not run further model/feature/scale diagnostics on the own-value-self-censoring + matched-rate
+ single-column idiom** — the ladder has shown the information is not there. Bring this to a
strategic decision:
- **(A) Re-scope P2 to prior-driven governance:** metadata-authored δ-prior + flat-likelihood-aware
  combination + deterministic sensitivity/tipping-point reporting (estimand layer). This is the
  North-Star-faithful product and uses what we built (generator, answer sheets, δ-bins, calibration,
  leakage gate, manifest) for *evaluation/calibration of the prior*, not for footprint extraction.
- **(B) Open a separate research arc** on whether OTHER idioms or auxiliary information carry a
  recoverable data signal — kept distinct from the governance product so it cannot quietly become a
  "beat non-identifiability" effort.

## Discipline / status

Safeguards honored (genuine cardinality range; stratified training so high-card was learnable);
leakage clean on both runs; manifests validated; binary strictly a labeled diagnostic; no
architecture/feature changes were made to reach this conclusion. Full suite green (1296 passed,
1 skipped). **Investigation closed: own-value δ is not recoverable from the real-X matched-rate
single-column footprint. Recommend the strategic pivot to a prior-driven, sensitivity-reporting P2.
Stop for review.**
