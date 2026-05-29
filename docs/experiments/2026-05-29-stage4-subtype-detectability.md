# Stage 4 — Per-column detectability stratifies sharply by generator subtype

- **Date:** 2026-05-29
- **ADR:** docs/decisions/0006, Stage 4. Script: `scripts/stage4_subtype_detectability.py`.
- **Setup:** full registry (`lacuna_tabular_110`, 113 gens), one generator per dataset
  (single-mechanism, full diversity), label each missing-bearing column by its class, per-column
  recall broken down BY GENERATOR. Frozen-probe (head-only on RUN-071 encoder + deployable feats).
- **Status:** frozen-probe check complete. Confirms the subtype-stratification hypothesis;
  full fine-tune/multi-seed run is the handoff below.

## Result (frozen probe, per-generator per-column recall, n≥20)

By class: **MNAR 0.806** (n=5880), **MCAR 0.222** (n=10608), **MAR 0.177** (n=2776); overall ≈ 0.39.

Per-subtype recall ranges the full [0, 1] within every class — detectability is **subtype-dependent**:

| band | examples |
|---|---|
| **0.93–1.00 MNAR** | Truncation, DetectUpper, DetectLower (0.93), ThreshTwoSided, SoftThresh (0.98), Q70 (0.94), CompEvents, LatentSES, ValDepStr, Privacy, SelfCensor-Weak (0.97) |
| 0.73–0.86 MNAR | UnderReport, SelfCensor-Extreme/High/Low, LatentHealth/Motiv, DemoDepend, RiskMonitor, Q80, DetectBoth |
| 0.50 MNAR | OutcomeDep, ThreshRight, AdaptSamp |
| 0.00 MAR | **Section, Branching, SkipLogic, ColBlocks, Conditional, CrossClass, ContPred** (structural / survey skip-logic) |
| 0.50–1.00 MAR | RealisticSingle (0.75), BinaryPred (0.70), DiscPred (0.63), Interactive (1.00, small n) |
| 0.1–0.3 MCAR | most Bernoulli/row/col MCAR variants (over-called MNAR here) |

## Findings

1. **The hypothesis is confirmed: per-column detectability stratifies sharply by subtype.** The
   threshold / detection-limit / quantile / truncation MNAR mechanisms are detected at **0.9–1.0
   per column** — the *easiest*, not the hardest. They leave a sharp, observable footprint.
2. **This deconfounds the Stage 0–3 "per-column MNAR is bounded" conclusion.** That was the product
   of three stacked confounds: a single MNAR subtype (logistic self-censoring — among the harder),
   a head trained on only 3 generators, and mixed-mechanism eval. Trained on the full registry and
   evaluated per generator, MNAR subtypes are largely detectable.
3. **MAR structural/skip-logic is the surprising 0.00 band.** "Missing-by-design" (skip-logic,
   sections) should be the *easiest* MAR (deterministic on an observed gateway) — its 0.0 here is
   almost certainly the frozen probe's MNAR lean swamping it, not genuine undetectability. The
   fine-tuned run should recover it; worth watching specifically.

## Critical caveats

- **Frozen probe has an MNAR-prediction lean** (MNAR recall 0.81 but MCAR/MAR ≈ 0.2; overall acc
  ≈ 0.39). The **relative** stratification is the trustworthy signal; **absolute per-class balance
  is not** until the fine-tuned run rebalances it.
- **Single-mechanism eval**, not mixtures (all missing columns in a dataset share one mechanism).
  Easier than the per-column mixtures of Stages 0–3; the two MNAR numbers are not directly comparable.
- Single seed. Current-code 10-feature baseline encoder.

## Handoff — full experiment (fresh session)

The frozen probe answers "does detectability stratify" (yes). The full experiment answers "what is
the *balanced* per-subtype detectability with a properly trained model":

```
python scripts/stage4_subtype_detectability.py --fine-tune --epochs 40 \
    --eval-batches 200 --seed <S> \
    --output .../stage4_subtype_finetune_s<S>.json    # repeat for ≥3 seeds
```

Then aggregate per-generator recall across seeds (mean ± sd) and read off the stratification with
balanced cross-class accuracy. Expected: sharp-footprint MNAR + missing-by-design MAR high and
stable; self-censoring/latent and weak-MAR lower. That per-subtype map is the deployable, honest
product spec — flag the detectable mechanisms confidently, report the rest as calibrated-uncertain.
