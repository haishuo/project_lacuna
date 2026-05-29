# Stage 3a — A deployable per-column MNAR signal (distributional features)

- **Date:** 2026-05-29
- **ADR:** docs/decisions/0006, Stage 3 (deployable-proxy search).
- **Setup:** the column head augmented with **deployable** per-column features — scale-free
  distributional statistics of the OBSERVED values only (`lacuna/models/column_deployable_features.py`:
  missing_rate, robust skew `|mean−median|/std`, excess kurtosis). No oracle, no reconstruction
  targets. `--deployable-features`. Frozen probe, a combined frozen run (+zeroed recon), and fine-tune.
- **Status:** complete — **positive result.**

## Result (vs the S1 baseline and the S2b oracle ceiling)

| config | acc | MCAR | MAR | MNAR | ECE | composition L1 |
|---|--:|--:|--:|--:|--:|--:|
| frozen (S1, encoder reps only) | 0.425 | 0.467 | 0.724 | 0.040 | 0.032 | 0.448 |
| frozen +recon-TRUE (**oracle** ceiling, S2b) | 0.410 | 0.484 | 0.499 | 0.232 | 0.062 | 0.447 |
| **frozen +deploy (S3a)** | 0.424 | 0.450 | 0.640 | **0.152** | **0.029** | **0.431** |
| frozen +deploy +recon-zeroed | 0.422 | 0.487 | 0.611 | 0.139 | 0.049 | 0.451 |
| fine-tune +deploy | 0.470 | 0.442 | 0.751 | 0.179 | 0.066 | 0.484 |

Confusion (frozen +deploy) MNAR row: `[444, 1510, 350]` — MNAR true-positives 92 (S1) → 350.

## Finding

**A deployable per-column signal recovers most of the (theory-bounded) MNAR signal.**
- Frozen MNAR recall **0.04 → 0.15 (~3.8×)**, i.e. **~65% of the oracle ceiling (0.23)** — using only
  observed-value statistics, with **no oracle target**. The truncation footprint that MNAR
  self-censoring leaves in the observed distribution is real and per-column usable.
- It is also the **best-calibrated** (ECE 0.029) and **lowest composition error** (L1 0.431) of every
  configuration tried, at a **milder MAR cost** than the oracle (MAR 0.64 vs the oracle's 0.50) and no
  loss of overall accuracy. So unlike the oracle's sharp MAR↔MNAR trade, the deployable features give a
  gentler, better-behaved operating point.

Two negatives sharpen the picture:
- **+zeroed-recon hurts** (0.152 → 0.139): the dataset-gate's zeroed reconstruction signal is noise
  per column (consistent with the Stage 2 null); deploy-alone is best.
- **fine-tune +deploy ≈ S1 fine-tune** (0.179 vs 0.184): explicit features help in the *frozen* probe,
  where the fixed encoder cannot relearn them; an unfrozen encoder already extracts what it can.

## Interpretation — approaching a low ceiling

The S2b oracle ceiling is itself low (MNAR recall 0.23, and only by trading MAR) because of MAR/MNAR
non-identifiability. The deployable proxy reaching 0.15 captures roughly two-thirds of that
already-bounded signal. So this is **progress that is near the theoretical limit**, not a full solve:
per-column MNAR is still substantially under-detected (≈85% of MNAR columns still read as MAR), but
the deployable distributional features close most of the gap that any method could close here. This is
the point where reality begins to push back — not with a null, but with diminishing headroom against a
non-identifiability-bounded ceiling.

## Practical upshot

For the product goal (calibrated per-column posterior + composition estimate, Q1/Q2/Q3), the deployable
features are a genuine, deployable improvement: best composition-L1 and best calibration observed, with
no oracle dependency. MCAR/MAR remain reliable; MNAR is partially recoverable and honestly uncertain.

## Next options (rabbit-hole status)

- **Artificial-reconstruction probe** — mask observed cells, reconstruct, error vs the known held-out
  values (deployable). Tests whether reconstruction adds signal beyond the distributional footprint.
  Bigger build (two encoder passes); headroom to the oracle is only ~0.08 MNAR recall.
- **Richer distributional features** (cheap) — per-column SMD-to-others (MAR axis), tail-asymmetry.
- **Conclude** — the deployable proxy sits near a theory-bounded ceiling; diminishing returns.
- **Build the product** — composition/calibration already improved; ship Q1/Q2/Q3 with honest MNAR caveats.

## Caveats / scope

- Current-code 10-feature baseline (RUN-071); canonical-generator subset; clean-MAR regime; frozen
  reconstruction heads; 15-epoch fine-tune. Deployable features are scale-free by design (raw-scale values).
