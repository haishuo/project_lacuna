# Stage 2 — Does wiring per-column reconstruction errors unlock per-column MNAR?

- **Date:** 2026-05-29
- **ADR:** docs/decisions/0006, Stage 2 (the lead handed down by Stage 1).
- **Setup:** the Stage 1 column head, augmented with **per-column reconstruction-error features**
  from the baseline's frozen reconstruction heads (`lacuna/models/column_recon_features.py`),
  via `scripts/stage1_column_head.py --use-recon-features`. Single-variable ablation: identical
  code path to Stage 1, recon features on/off. Reconstruction heads kept frozen (fixed signal
  source). Eval on 7 held-out datasets × mixed compositions.
- **Status:** complete; **negative result** with a clear next step.

## Result (2×2: recon features off = Stage 1, on = Stage 2)

| config | per-col acc | MCAR rec | MAR rec | MNAR rec | ECE | composition L1 |
|---|--:|--:|--:|--:|--:|--:|
| frozen | 0.425 | 0.467 | 0.724 | 0.040 | 0.032 | 0.448 |
| frozen +recon | 0.448 | 0.540 | 0.685 | **0.080** | 0.031 | 0.437 |
| fine-tune | 0.473 | 0.457 | 0.741 | 0.184 | 0.053 | 0.476 |
| fine-tune +recon | 0.473 | 0.439 | 0.780 | **0.158** | 0.069 | 0.492 |

Stage 2 confusion [true × predicted]:
- frozen +recon:    MNAR `[580, 1539, 185]`  (MNAR cols → mostly MAR)
- fine-tune +recon: MNAR `[389, 1550, 365]`  (MNAR cols → mostly MAR)

## Finding

**Adding the reconstruction-error features did not unlock per-column MNAR.** Frozen MNAR recall
nudged 0.04 → 0.08 (still near-zero); fine-tuned MNAR recall *fell* 0.184 → 0.158 and
composition-L1 worsened 0.476 → 0.492. MNAR columns remain overwhelmingly classified as MAR in
every configuration. There are small, real gains elsewhere (frozen MCAR recall 0.47 → 0.54;
frozen accuracy 0.43 → 0.45; frozen composition-L1 marginally better), but the headline — per-column
MNAR — does not move.

This **refines, and partly refutes, the Stage 1 diagnosis.** Stage 1 reasoned that per-column MNAR
failed because the head could not see the reconstruction signal. Giving it that signal (as the
dataset gate defines it) is not sufficient: per-column MNAR stays hard. The bottleneck is deeper
than "the head lacked the feature."

## The cross-stage arc (Stages 0→2)

- **Stage 0** — dataset level on mixtures: MNAR *over*-fires (a ~25%-MNAR mixture is labelled MNAR;
  the MNAR attractor).
- **Stage 1** — per-column from encoder reps: MNAR *under*-fires (collapses to MAR; recall ≤0.18).
- **Stage 2** — per-column + reconstruction features: MNAR still under-fires (recall ≤0.16).

Consistent thread: **the column is a viable unit for MCAR/MAR (well-calibrated, MAR recall ~0.7–0.8),
but per-column MNAR is the genuine frontier.** This matches the theory — MNAR is the non-identifiable
boundary, and per column there is even less evidence than at the dataset level.

## Why the recon features likely didn't help (and the next step)

The per-column features were computed faithfully to the dataset gate's signal: squared error of
each head's prediction against the batch's `original_values` on the column's missing cells. But
`original_values` is **zeroed at naturally-missing cells**, so the feature is effectively each
head's *prediction magnitude* at missing cells, not its *reconstruction accuracy*. That aggregates
into a usable dataset-level signal but is evidently too weak per column.

**Stage 2b (clear next step):** because the data is semi-synthetic, the *true* missing values are
known (`complete X`). Recompute per-column reconstruction error against the true values — genuine
"how badly does each head reconstruct this column's hidden values" — which is the real self-censoring
signal (MNAR values are hard to reconstruct from observed context). Requires threading `complete X`
through `build_mixed_batch` (add a `complete_values` field) and a true-target variant of the
feature extractor. If that *also* fails to lift per-column MNAR, the conclusion strengthens to
"per-column MNAR is not recoverable with this architecture," a substantive dissertation finding.

## Caveats / scope

- Control/encoder/recon-heads are the current-code 10-feature baseline (RUN-071).
- Canonical-generator subset, clean-MAR regime, 15-epoch fine-tune (not exhaustively tuned).
- Recon features used the zeroed-target (vs the dataset gate) operationalization; Stage 2b tests the
  true-target version before any architectural conclusion.
