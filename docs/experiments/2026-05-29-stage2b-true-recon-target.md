# Stage 2b — Per-column reconstruction error vs the TRUE missing values

- **Date:** 2026-05-29
- **ADR:** docs/decisions/0006, Stage 2b (follow-up to the Stage 2 null).
- **Setup:** identical to Stage 2, but the per-column reconstruction error is computed against the
  **true complete values** (`MixedBatch.complete_values`, knowable only because the data is
  semi-synthetic) instead of the zeroed `original_values`. `stage1_column_head.py
  --true-recon-target`. Single-variable ablation. Reconstruction heads frozen.
- **Status:** complete. The capstone of the Stage 0→2b arc.

## Result (full arc; frozen and fine-tune)

| config | acc | MCAR | MAR | MNAR | ECE | composition L1 |
|---|--:|--:|--:|--:|--:|--:|
| frozen (S1) | 0.425 | 0.467 | 0.724 | 0.040 | 0.032 | 0.448 |
| frozen +recon, zeroed (S2) | 0.448 | 0.540 | 0.685 | 0.080 | 0.031 | 0.437 |
| **frozen +recon, TRUE (S2b)** | 0.410 | 0.484 | 0.499 | **0.232** | 0.062 | 0.447 |
| fine-tune (S1) | 0.473 | 0.457 | 0.741 | 0.184 | 0.053 | 0.476 |
| fine-tune +recon, zeroed (S2) | 0.473 | 0.439 | 0.780 | 0.158 | 0.069 | 0.492 |
| **fine-tune +recon, TRUE (S2b)** | 0.437 | 0.302 | 0.778 | **0.189** | 0.071 | 0.522 |

S2b confusion [true × predicted]:
- frozen:    MNAR `[565, 1204, 535]` (MNAR true-positives 92→535 vs S1; MNAR precision ~0.38)
- fine-tune: MNAR `[188, 1681, 435]`

## Finding

**The per-column MNAR signal exists in the true reconstruction error — but recovering it trades
against MAR.** The frozen probe's MNAR recall rises **0.04 → 0.23 (6×)** with the true-target
features (true-positives 92 → 535). So the signal that separates self-censoring is genuinely there
in *reconstruction accuracy vs the hidden values* — neither encoder reps (S1) nor the zeroed-target
proxy (S2) exposed it. But it is **not a clean unlock**: MAR recall falls 0.72 → 0.50, MNAR precision
dips 0.46 → 0.38, and overall accuracy edges down. The head moved its operating point *along* the
MAR↔MNAR boundary, recovering real MNAR true-positives at MAR's expense — the per-column footprint of
the fundamental MAR/MNAR non-identifiability. Fine-tuning end-to-end with the true features (rather
than the frozen probe) did not improve MNAR and destabilized MCAR (0.46 → 0.30); the frozen probe is
the cleaner lens.

## Critical caveat — this is an ORACLE result

The true-target reconstruction error uses the actual missing values, which exist only because the
data is semi-synthetic. **A deployed per-column classifier cannot compute this** — at inference on
real data the missing values are, by definition, unavailable. So Stage 2b establishes that the
per-column MNAR signal is *present in principle* (an upper bound / ceiling), NOT that there is a
deployable method to extract it. The deployable proxy the dataset gate actually uses (zeroed target,
S2) does **not** carry per-column MNAR. Closing that gap — a deployable reconstruction-quality signal
that surfaces per-column self-censoring — is the open problem this arc exposes.

## Cross-stage conclusion (Stages 0 → 2b)

1. **The column is a reliable unit for MCAR and MAR.** Per-column MAR recall is ~0.7–0.8,
   well-calibrated — genuinely recovering the class the dataset-level model erased on mixtures (S0).
2. **Per-column MNAR is the frontier, and it is non-identifiability-bound.** At the dataset level
   MNAR over-fires on mixtures (S0 attractor); per-column it under-fires (S1/S2). The signal is
   recoverable from oracle reconstruction error (S2b, 6× recall) but only by trading against MAR —
   you can move along the MAR/MNAR boundary, not dissolve it.
3. **Composition estimate (Q2)** never drops below L1 ≈ 0.44, bottlenecked by the MNAR/MAR trade.

This is a coherent, defensible result: per-column classification *works* for the identifiable part
(MCAR/MAR) and runs into the theoretical MAR/MNAR wall for the rest — now localized (the signal is in
reconstruction accuracy) and quantified (the see-saw), rather than merely asserted.

## Next options

- **Conclude** the column-level experiment and write the synthesis (the arc is publishable as-is).
- **Deployable-proxy search**: can a reconstruction-quality signal that surfaces per-column MNAR be
  built *without* oracle targets (e.g. cross-fitting, calibrated residual models)?
- **Product path**: accept the MCAR/MAR-reliable, MNAR-traded operating point and ship the calibrated
  per-column posterior + composition estimate (the original Q1/Q2/Q3 goal), with honest MNAR caveats.

## Caveats / scope

- Oracle target (see above). Current-code 10-feature baseline (RUN-071); canonical-generator subset;
  clean-MAR regime; frozen reconstruction heads; 15-epoch fine-tune (not exhaustively tuned).
