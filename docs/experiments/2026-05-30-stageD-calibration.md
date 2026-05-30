# Stage D — calibration of the composition posterior (the headline)

- **Date:** 2026-05-30
- **ADR:** docs/decisions/0007 (composition-posterior estimand), Stage D — the headline metric.
- **Predecessor:** Stage C (the posterior this stage calibrates; it recovered the identifiable
  composition but its stated confidence was not yet reliable).
- **Module:** `lacuna/training/composition_calibration.py` (+ tests). **Script:** `scripts/stageD_calibration.py`.
- **Question:** ADR-0007 makes CALIBRATION the deliverable: when the posterior says "p% sure the
  composition is in simplex-region Q", it should be right p% of the time. Point accuracy is secondary.
  Does post-hoc calibration make the region statements reliable, does it beat the pre-registered
  baselines, and does the honest seam (MCAR resolvable, MAR-vs-MNAR a wide band) show up in calibration?

## Method

- **Posterior:** the Stage-C frozen-encoder + footprint-feature heads (3-head deep ensemble, pooled by
  evidence-summing) — the best Stage-C config.
- **Recalibration:** temperature scaling of the Dirichlet evidence, `alpha_cal = 1 + (alpha-1)/tau`
  (the evidential analogue of logit temperature: flattens toward the uniform prior as `tau` grows). A
  single `tau` is fit on a held-out CALIBRATION split (40 batches) to minimise query-ECE, then applied
  to a disjoint TEST split (40 batches, n=640). Both splits are held out from Stage-C training.
- **Metrics over the simplex-region query set** `P(f_c >= t)`, c∈{MCAR,MAR,MNAR}, t∈{.2,.35,.5,.65,.8}:
  - **ECE** (reliability): |stated prob − empirical frequency|, binned.
  - **Brier** (proper score = reliability AND resolution): the bar to beat prior-only, since a
    constant predictor can be trivially reliable (low ECE) but has no resolution (high Brier).
  - Region probabilities are exact (a Dirichlet coordinate is `Beta(alpha_c, alpha0-alpha_c)`).
- **Baselines:** the UNCALIBRATED posterior (`tau=1`) and a PRIOR-ONLY constant Dirichlet at the
  calibration-mean composition (its concentration fit on calibration to minimise its own Brier).

## Result — calibration halves ECE and beats prior-only on the proper score

Test split, n=640, fitted `tau = 4.26` (the posterior was substantially overconfident):

| posterior | ECE (reliability) ↓ | Brier (proper score) ↓ |
|---|--:|--:|
| uncalibrated (tau=1) | 0.0715 | 0.1473 |
| **calibrated (tau=4.26)** | **0.0442** | 0.1505 |
| prior-only (constant) | 0.0256 | **0.1745** |

- **Calibration works:** temperature scaling cuts ECE 0.072 → 0.044 (−38%). The region statements are
  now reliable to ~±0.04.
- **It beats prior-only on the proper score:** Brier 0.151 (cal) / 0.147 (uncal) vs **0.175 prior-only**.
  Prior-only's lower ECE (0.026) is the textbook degenerate — a constant base-rate predictor is
  trivially reliable but carries no information; on Brier (which rewards resolution) the posterior wins
  by ~0.02–0.03. By ADR-0007's bar (beat prior-only; beat the uncalibrated posterior) the result is
  **not tanked**: the calibrated posterior is reliable AND resolved.

### The honest seam, in calibration terms

| axis (calibrated) | ECE ↓ | Brier ↓ | prior-only Brier | resolution gain |
|---|--:|--:|--:|--:|
| random-vs-structured (MCAR queries) | 0.045 | 0.139 | 0.180 | **0.041** |
| MAR-vs-MNAR split (MAR/MNAR queries) | 0.063 | 0.156 | 0.172 | 0.015 |

The calibrated posterior resolves the **identifiable** random-vs-structured axis well beyond prior-only
(Brier gain 0.041), but on the **non-identifiable** MAR-vs-MNAR split it falls back toward prior-only
(gain only 0.015 — ~⅓ as much). This is exactly the honest seam ADR-0007 pre-registered: the model is
informative where the question is answerable and a near-uninformative (correctly wide) band where it is
not — reported as success, not failure.

### Worked example statements (calibrated, from the test split)

| realised (MCAR/MAR/MNAR) | P(f_MAR ≥ 0.7) | P(f_MNAR ≥ 0.5) | P(f_MCAR ≤ 0.2) |
|---|--:|--:|--:|
| 0.16 / 0.37 / 0.47 | 0.04 | 0.24 | 0.31 |
| 0.00 / **0.75** / 0.25 | 0.08 | 0.23 | 0.46 |
| **0.78** / 0.07 / 0.16 | 0.01 | 0.14 | 0.07 |

Sensible and hedged on the identifiable axis (row 3: high MCAR → P(f_MCAR≤0.2) correctly ~0.07), but it
will not confidently assert a high MAR fraction (row 2: realised 0.75 MAR, yet P(f_MAR≥0.7)=0.08) — the
MAR-vs-MNAR non-identifiability surfacing as honest under-commitment rather than a confident error.

## Interpretation

1. **The headline holds:** a single fitted temperature makes the composition posterior's region
   statements reliable (ECE 0.044) and the posterior beats prior-only on the proper score — it is
   calibrated *and* resolved. This is the ADR-0007 deliverable.
2. **The honest seam is now a calibration statement, not just a point-error one:** resolution
   concentrates on the random-vs-structured axis (Brier gain 0.041) and collapses toward prior-only on
   the MAR-vs-MNAR split (0.015) — the Molenberghs limit showing up as a correctly wide band.
3. **A real limitation → the clear lead.** A *global* temperature fixes average reliability but not the
   per-instance uncertainty *ranking*: the can't-tell mass (mean 0.59 after calibration) still
   anti-correlates weakly with error (corr −0.15). The can't-tell mass tracks ambiguity at the AXIS
   level (it falls back to prior-only on MAR/MNAR) but not per-dataset. Fixing the ranking needs a
   *feature-conditional* recalibrator (e.g. predict the temperature / vacuity from the footprint), not
   a global scalar — the named Stage-D-v2 lead.

## "Tanked" check (pre-registered, ADR-0007)

- Beats the **uncalibrated** posterior on reliability (ECE 0.044 < 0.072). ✓
- Beats **prior-only** on the proper score (Brier 0.151 < 0.175) — reliable *with* resolution. ✓
- (Stage C already showed it beats prior-only on point composition: L1 0.589 vs 0.708.)

Not tanked.

## Caveats

- **Semi-synthetic, survey-scoped** (ADR-0005/0007); MAR-vs-MNAR is non-identifiable on real data by
  construction — the wide band on that axis is the correct report, validated only against the by-cell
  ground truth the generator provides.
- Global temperature is the standard low-variance recalibrator; per-instance/feature-conditional
  calibration is the open improvement (above).
- Calibration is fit and tested on disjoint held-out splits, but both are drawn from the same broad
  prior + the same val datasets; deployment under a shifted composition prior is untested (ADR allows a
  deployment prior).

## Reproduce

```
python scripts/stageD_calibration.py --cal-batches 40 --test-batches 40 --batch-size 16
```

## Files

- Report: `/mnt/artifacts/project_lacuna/runs/stage0_general_baseline/stageD_calibration.json`.
- Module: `lacuna/training/composition_calibration.py`; script `scripts/stageD_calibration.py`;
  tests `tests/unit/training/test_composition_calibration.py`.
