# Stage Q — the per-column SUBTYPE layer (ADR-0008 commitment 6, the restored Q3)

- **Date:** 2026-06-01
- **ADR:** docs/decisions/0008-metadata-prior-likelihood.md, **commitment 6** (Q3 returns as a per-column
  *subtype detector + prior*, NOT the Stage-5 per-column 3-way classifier that collapsed).
- **Branch:** `experiment/subtype-layer` (off `experiment/metadata-prior`).
- **Modules:** `lacuna/priors/{dirichlet_evidence,subtype_ontology,subtype_prior}.py`,
  `lacuna/data/subtype_targets.py`, `lacuna/models/subtype_likelihood.py` (+124 tests).
  **Scripts:** `scripts/stageQ_subtype_layer.py`, `scripts/stageQ_feature_attribution.py`.
  **Reports:** `runs/stage0_general_baseline/stageQ_{subtype_layer,feature_attribution}.json`.

## What Q builds

The whole ADR-0008 prior × likelihood arc (P0–P6) ran at MECHANISM granularity (MCAR/MAR/MNAR). Q
extends the *same* machinery to SUBTYPE granularity — where property 3's literal output lives ("fairly
certain 20% is THRESHOLD MNAR … unable to determine for the rest"). Two separable channels, fused per
column, aggregated to a dataset subtype-composition by missing cell, with a calibrated abstain mass.

- **Ontology O (the fused Dirichlet support, K=5):** `threshold_mnar`, `detection_mnar` (the LOUD,
  data-detectable MNAR subtypes), `self_censoring_mnar` (the quiet/non-identifiable MNAR catch-all),
  `mcar`, `mar` (data-silent per column). Parent mechanisms roll the subtype-composition up to the
  ADR-0007 mechanism-composition.
- **Likelihood (data):** a per-column readout over `L = {threshold, detection, reject}`. It detects
  the loud fingerprints and folds the *entire* non-identifiable region (quiet self-censoring MNAR +
  MAR + MCAR) into one **reject** class. This is the design move that avoids the Stage-5 trap: the
  readout never makes the per-column MAR-vs-MNAR call that collapsed; it makes the *identifiable*
  loud-vs-reject call. The L→O embedding sends reject mass to **zero evidence** (the data DEFERS to the
  prior on the non-detectable classes), so a column the data rejects does not pull the posterior off a
  prior's self-censoring lean.
- **Prior (metadata):** the subtype prior (`subtype_prior.py`): `lab_lod → detection_mnar`,
  `planned_random → mcar`, `skip_gated → mar`, `sensitive_disclosure → self_censoring_mnar`,
  `indeterminate → flat`; tiered + confidence-gated as the mechanism prior. `threshold_mnar` has **no**
  semantic class — a sharp cutoff is a DATA fingerprint, not a codebook fact — so the prior is flat on
  it and the data leads there (the honest seam at subtype granularity).
- **Fuse → calibrate → abstain:** combine in raw evidence (P2), aggregate by missing cell (ADR-0007),
  temperature-calibrate the combined subtype-composition (P5), per-column selective decision with a
  risk-coverage commit threshold (P6). On semi-synthetic the prior reliability is *simulated* at a
  known ρ (P5/P6 method) to separate calibration from correctness; the real frozen spec is checked for
  consistency against the grounded benchmark.

Regime: matched miss rate, full diversity (`mnar_diverse + mar_diverse`; diverse pools ⇒
`compensate_rate` is a no-op — the Stage-5 cue-free regime; MNAR even realises *lower* miss rate, so no
rate cue favours it). 5 seeds, mean ± sd.

## Result 1 — the data channel: stable detection, but WEAK, with honest abstain (5 seeds)

| likelihood metric | mean ± sd |
|---|---|
| loud-vs-reject ROC-AUC | **0.819 ± 0.018** |
| loud-vs-reject AP (base rate 0.112) | 0.421 ± 0.043 |
| loud recall @ precision 0.6 | 0.214 ± 0.082 |
| loud recall @ precision 0.8 | 0.091 ± 0.054 |
| reject-correct on the silent region | **0.989 ± 0.006** |
| threshold↔detection argmax confusion | 0.075 / 0.071 |

- **The Stage-5 instability is gone — the reframe works.** The per-column 3-way MCAR/MAR/MNAR
  classifier was seed-unstable at matched rate (winner-take-all; Stage-5 per-class recall sd
  **0.30–0.43**). The loud-vs-reject detector is **seed-stable: AUC sd 0.018** (per-seed 0.80–0.85).
  Folding the non-identifiable region into one reject class — and reading it out with a stable model
  (random forest) on the deployable features — removes the collapse. This is the methodological point:
  ask the data only the *identifiable* question.
- **Honest abstain holds.** The data correctly rejects **99%** of the silent/MAR/MCAR region — it does
  not fabricate loud subtypes where there is no cliff.
- **But the detector is WEAK.** AUC 0.82 / AP 0.42 is real signal (~3.8× the base rate), yet usable
  recall is low: it confidently flags only ~21% of loud columns at 60% precision (~9% at 80%). **The
  Stage-4 single-mechanism "threshold/detection detectable at 0.9–1.0" does NOT transfer** to the
  loud-vs-reject detector on matched-rate mixtures — self-censoring MNAR's truncation footprint
  overlaps the loud cliff (the loud-vs-self-censoring RF AUC is only ~0.73), so most loud columns are
  not separable from the quiet region by the 5 deployable features. The data confidently identifies a
  *minority* of loud columns and abstains on the rest.

## Result 2 — attribution: the signal is in the features; the encoder DILUTES it (3 seeds)

`scripts/stageQ_feature_attribution.py`, loud-vs-reject ceiling (RF), matched rate, loud base 0.106:

| input to the RF | ROC-AUC | AP | recall @P.5 |
|---|---|---|---|
| deployable distributional features (5-D) | **0.808 ± 0.021** | 0.366 | 0.212 |
| frozen-encoder per-column token reps (64-D) | 0.649 ± 0.020 | 0.205 | 0.050 |
| features + encoder reps | 0.756 ± 0.021 | 0.298 | 0.070 |

The loud signal lives in the deployable observed-value features; the frozen dataset-task encoder reps
are barely above chance and **dilute** the features when concatenated (0.81 → 0.76). This is why the
canonical `ColumnReadoutHead` (encoder reps + features) collapsed to all-reject in early runs, and why
the Q likelihood is a deployable-feature detector with **no encoder, no oracle**. It echoes Stage C's
"the encoder is the bottleneck / under-represents the footprint", now at subtype granularity. (Among
feature models: RF/GBM ~0.85–0.86 ≫ histogram-GBM ~0.79 ≫ a small MLP ~0.78 — the RF is the lens.)

## Result 3 — fusion: prior-led, calibrated, safe; the override is weak (5 seeds)

Per-column posterior → dataset subtype-composition (by missing cell) + per-column abstention. Three
conditions: data-only, a *reliable* simulated prior (ρ=0.85), a *near-chance* one (ρ=0.45); favoured
prob 0.70 (the commitment-1 overridable cap).

| condition | comp L1 | query-ECE | Brier (prior-only 0.119) | coverage @acc.8 | committed-acc | abstain mass |
|---|---|---|---|---|---|---|
| data-only | 0.905 | 0.086 | 0.116 | **~0.00** | (1.0 on the few) | **~1.00** |
| reliable prior | **0.644** | **0.042** | **0.079** | **0.999** | 0.832 | 0.001 |
| near-chance prior | 0.854 | 0.039 | 0.109 | ~0.005 | — | 0.996 |

committed subtype-accuracy split (at the target): reliable → loud 0.812 / silent 0.834; near-chance →
**loud 0.948** / silent 0.0.

- **Data-only abstains on essentially everything at subtype granularity** ("unable to determine" is the
  honest default): at an 80%-committed-accuracy target the data alone commits on ~0% of columns,
  because it cannot reliably name *which* subtype (threshold↔detection unresolved, most loud columns
  not confidently flagged). Its subtype-composition barely beats the prior-only base-rate predictor
  (Brier 0.116 vs 0.119) — the composition is dominated by the silent classes the data cannot resolve.
- **A reliable prior supplies exactly the resolution the data lacks** — coverage@80% lifts ~0 → 1.0,
  committed-accuracy 0.83, the combined composition is calibrated (ECE 0.042) and beats prior-only on
  the proper score (Brier 0.079 vs 0.119). This is the prior × likelihood payoff at subtype
  granularity (the P5/P6 result transfers).
- **A near-chance prior manufactures no false confidence** (coverage stays ~0), and **the data still
  protects the loud axis**: where the data commits to a loud subtype even under a bad prior, it is
  **0.948** correct — the P6 safety property at subtype granularity.
- **But the override (commitment 1) is WEAK here.** Across all loud columns where the prior is wrong,
  the data overrides it only **~0.10** of the time (vs the strong override at mechanism granularity).
  The reason is Result 1: the per-column data evidence is weak, so it can drag the posterior off a
  wrong subtype prior only on the confident minority (where it does — committed-loud 0.95). **The
  subtype layer is therefore PRIOR-DOMINATED**, with the data's override-safety limited to its
  confident minority. (Override rises with the data-evidence concentration `kappa_like`, but inflating
  it would manufacture confidence the weak detector does not have — so the honest operating point keeps
  the override weak.)
- The real frozen subtype-prior spec is **1.0 consistent** with the benchmark's `gold_subtype` (n=43) —
  a P1-style consistency check (the spec faithfully encodes the grounded semantic→subtype map), not an
  independent accuracy claim.

## Verdict (pre-registered success criteria)

**PARTIAL — an honest seam at subtype granularity.** Against the pre-registered bar:

- *Detect the loud subtypes at HIGH recall with honest abstain on the quiet ones* — **partial.**
  Detection is **stable** (the Stage-5 collapse is solved by the detector-with-reject reframe) and the
  abstain is **honest** (99% reject on the silent region), but recall is **low** (~21% @60% precision):
  the single-mechanism detectability does not survive matched-rate mixtures.
- *The prior adds calibrated resolution where the data is silent, and the data overrides where
  informative* — **yes on resolution + calibration** (reliable prior: coverage 0→1.0, Brier beats
  prior-only, ECE 0.042; data protects the loud axis under a bad prior), **weak on override** (~0.10
  aggregate; only the confident minority overrides).
- *The dataset subtype-composition + abstain mass is calibrated* — **yes** (combined ECE ~0.04; the
  abstain mass is the property-3 "unable to determine" fraction, ~1.0 data-only and ~0 with a reliable
  prior).

So the layer is **buildable, calibrated, and stable**, and the P-arc machinery (tiered/gated prior,
raw-evidence combine, calibration, selective abstention, by-cell aggregation) transfers cleanly to
subtype granularity — but the per-column DATA channel is too weak at matched rate to detect the loud
subtypes at high recall or to strongly override a wrong prior. The honest output is therefore a
**prior-led** subtype-composition with a large, calibrated "unable to determine" mass when the prior is
absent — fully consistent with Stage 5 (the matched-rate non-identifiability wall) and the P-arc
(the prior's value is real, reliability-contingent, and uncheckable on the non-identifiable subtypes).

## Caveats

- **Semi-synthetic; simulated prior reliability** (ρ injected, as P5/P6) — the real prior's subtype
  accuracy is a face-validity question (P1/P3), not measured here. The real spec is only checked for
  *consistency* against the grounded benchmark.
- **Small val pool** (7 source datasets → ~800 subsample/mixture instances): enough for the cal/test
  split and the 5-seed sd, but the dataset *diversity* is limited.
- **`kappa_like` is a fixed operating point** (20). It sets the prior:data evidence ratio; calibration
  absorbs the scale, but the override rate is sensitive to it (reported at the honest, non-inflated value).
- **Detector AUC ~0.82** is the RF ceiling on the *current* 5 deployable features; richer per-column
  distributional / MNAR-axis features (the Stage-5 "future" lead) could lift it — untested here.

## Reproduce

```
python scripts/stageQ_subtype_layer.py --seeds 5            # the layer (likelihood + fusion + abstention)
python scripts/stageQ_feature_attribution.py --seeds 3      # the encoder-dilution attribution
```

## Files

- Ontology + generic ops: `lacuna/priors/{subtype_ontology,dirichlet_evidence}.py`.
- Prior + targets + detector: `lacuna/priors/subtype_prior.py`, `lacuna/data/subtype_targets.py`,
  `lacuna/models/subtype_likelihood.py` (+ tests under `tests/unit/{priors,data,models}/`).
- Scripts + reports: `scripts/stageQ_{subtype_layer,feature_attribution}.py`;
  `runs/stage0_general_baseline/stageQ_{subtype_layer,feature_attribution}.json`.
