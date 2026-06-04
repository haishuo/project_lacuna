# P2.2b — Learnability Ladder — RUNG 1 Findings

*Branch `p2/delta-prior-rearchitecture`. Spec: `docs/PROPOSAL-P2.2b-learnability-ladder-audit.md`.
Rung 1 only — stop-for-review per the approved plan.*

## Result: RUNG 1 PASSES — the δ-bin head + RPS loss learns ordered δ in the clean setting

Synthetic 2-column `ConditionalGaussian(ρ)` X + own-value self-censoring on the KNOWN target
column (col 1), matched rate 0.30, 7 δ-bins, in-distribution (learnability, not generalization).
112,105-param model (hidden 64 / evidence 32 / 2 layers, `max_cols=2`), n=1024 rows/example,
ρ∈{0, 0.3, 0.6}, 20 epochs, ~227 s CPU. `runs/p2p2b-rung1.json` (validated, `kind=main`).

| metric | rung 1 (after-T) | reference | reading |
|---|---|---|---|
| **test RPS** | **0.0599** | uniform/base-rate **0.1905** | **3.2× lower — beats both far outside CI** |
| bin accuracy | **0.479** | chance 0.143 | strong |
| adjacent accuracy (±1 bin) | **0.900** | — | **ordering genuinely captured (RPS working)** |
| E[δ] MAE | 0.205 | — | small |
| interval coverage 50/80/90 | 0.91 / 0.99 / 1.00 | nominal | well-covered (mildly conservative) |
| ECE (before/after T) | 0.065 / 0.087 | — | modest; T=0.862 |
| best val RPS | 0.0593 | — | matches test (no overfit) |
| **leakage_pass** | **True** | — | corr(δ,rate)=−0.02; rate-only 0.07 < base 0.11 |

## Interpretation (per the approved decision rules)

**Rung 1 succeeds**, so by the approved decision table:

> *If rung 1 succeeds, the δ-bin head/loss formulation is NOT the problem.*

This is the important diagnostic outcome. The P2.2 uniform-floor failure is **not** a broken head
or a wrong loss — given a clean, known-target, 2-column problem with adequate per-example evidence,
the exact same `DeltaBinHead` + `rps_loss` + temperature + metrics + leakage + manifest stack
learns an ordered 7-bin δ-prior with high adjacent accuracy and clean calibration. The RPS
objective is doing precisely what it was chosen for: adjacent-acc 0.90 means the mass concentrates
on or next to the true bin, not scattered.

**Therefore the P2.2 floor is caused by one (or more) of the DOWNSTREAM axes** the ladder isolates
next — ordinal resolution at real scale (rung 2), real-X covariance geometry (rung 3), table
width / target localization (rung 4), or per-example evidence/scale (rung 5). Rung 1 has removed
all of those at once and the model learns, so at least one of them is the binding constraint.

Note also: rung 1 used **n=1024 rows/example** vs the P2.2 pilots' 128–512. This is a clean,
controlled regime, so it does not by itself prove "more rows fixes P2.2" (the in-distribution
max_rows=512 P2.2 diagnostic was still at the floor on real wide X) — but it confirms the signal
is there and learnable when geometry+width+localization are removed.

## What this does NOT claim

Rung 1 is an in-distribution learnability check on synthetic 2-column data. It does **not** claim
real-X performance, out-of-family generalization, or that the P2.2 task is solved. It localizes
the bottleneck *away from* the head/loss and *toward* the downstream axes — which is exactly its
job.

## Machinery / discipline

Same harness as P2.2 (no new objective): RPS primary, no 3-class, no binary main; from-scratch
(`checkpoint_loaded=False`, `all_layers_trainable=True`); leakage gate evaluated and clean;
manifest validates as `kind=main`. The training loop was refactored to inject a pluggable
`ExampleSource` (audit §8, approved) so every rung shares one harness — full suite green
(1266 passed, 1 skipped); 12 new `example_source` tests.

## Next step (await review)

Per "stop after rung 1": **hold here.** On approval, proceed to **rung 2** (2-col coarse 3-bin —
isolates ordinal resolution) and **rung 3** (real survey X with the head-side target-conditioned
head — isolates real-X geometry from localization), per §3/§6 of the audit. Do not scale compute
yet.
