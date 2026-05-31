# Stage C-v2 — the head→ceiling "gap" is half an eval-split artifact + half the calibrated-head cost

- **Date:** 2026-05-30
- **ADR:** docs/decisions/0007, Stage C follow-up (the "NN head < RF footprint ceiling" lead).
- **Predecessor:** Stage C found a RandomForest on the 20-D footprint reaches composition L1 0.385
  while the composition head reaches 0.589 — a 0.20 "gap" flagged as a lead. This stage attributes it.
- **Scripts:** `scripts/stageC_composition_head.py` (new knobs: `--head-layers`, `--no-evidence`,
  `--offline-corpus`); offline diagnostics in the run log.
- **Question:** Why does the composition head (0.589) trail the RF footprint ceiling (0.385), and can it
  be closed? Hypotheses: head capacity, the frozen evidence input, the online training regime, the KL
  regulariser, or the model class.

## Method — ablate one variable at a time, and re-measure the ceiling honestly

The Stage-C "ceiling" (0.385) was a RandomForest trained on 2500 pooled footprint→composition samples
with a random ROW split — so train and test shared the SAME catalog datasets (in-distribution). The
head, by contrast, trains on `train_datasets` (21) and evals on HELD-OUT `val_datasets` (7). The first
fix is to compare like with like: re-measure the footprint ceiling under the SAME held-out-DATASETS
split the head uses. Then ablate the head knobs.

## Result — the gap decomposes cleanly; no head knob closes the residual

Footprint→composition ceiling, by evaluation split (RandomForest and an offline-fit MLP):

| reference | split | composition L1 |
|---|---|--:|
| RandomForest | in-distribution (pooled datasets, row split) | 0.365 ≈ the original "0.385 ceiling" |
| RandomForest | **held-out datasets** (the head's regime) | **0.487** |
| offline MLP (MSE) | held-out datasets | 0.486 |
| offline torch-MLP (EDL, kl=0) | in-distribution | 0.395 |

Composition head, held-out datasets (deep ensemble, n=640), across every knob:

| head configuration | L1 | note |
|---|--:|---|
| evidence + footprint, online, kl=0.5 (Stage C) | 0.589 | — |
| footprint-only, online, kl=0.5 | 0.596 | dropping the evidence: no effect |
| footprint-only, offline corpus, kl=0.5 | 0.610 | online→offline regime: no effect |
| footprint-only, offline, **kl=0** | 0.662 | KL off: *worse*, and calibration destroyed (can't-tell 0.06, ECE 0.21) |
| width∈{64,128,256}, depth∈{1,2} (ablation) | 0.63–0.65 | capacity/depth: no effect |

**The 0.20 apparent gap decomposes into two ~equal halves:**
1. **Evaluation-split artifact (≈0.10):** the 0.385 ceiling was in-distribution; the honest held-out
   ceiling is **0.487**. (My Stage-C interpretation overstated the headroom — a correction.)
2. **Calibrated-head cost (≈0.10):** the head (≈0.59) trails the held-out MSE ceiling (0.487) — and an
   offline MSE/EDL-kl0 MLP *reaches* that ceiling. So the model class, capacity, depth, evidence input,
   and training regime are all NON-factors; what remains is the **evidential Dirichlet objective vs plain
   MSE point regression.** The EDL Bayes-risk objective places the Dirichlet *mean* with a slight
   shrinkage toward uniform (the price of a proper distribution), and removing its KL term to chase the
   mean overfits and destroys calibration (kl=0 row). The ~0.10 is the cost of emitting a CALIBRATED
   DISTRIBUTION + can't-tell mass instead of a point estimate.

## Interpretation

1. **There was no architecture deficit to fix.** Capacity, depth, the frozen evidence, and the
   online/offline regime were ablated and are non-factors; the head already matches what an MLP/RF
   extract from the footprint *as a point regressor*. The "gap" was ~half a split artifact + ~half the
   intended cost of the calibrated-distribution estimand.
2. **Point accuracy is secondary (ADR-0007); calibration is the headline.** A plain MSE regressor would
   reach L1 0.487 but produce NO posterior, NO can't-tell mass, and NO calibration — the Stage-C/D/E
   deliverables. The calibrated head (kl=0.5) pays ~0.10 L1 for them; kl=0 recovers nothing (it is both
   worse on L1 and badly miscalibrated). So the calibrated head is the correct operating point.
3. **The dominant real limiter is generalisation to unseen dataset types** (RF 0.365 in-distribution →
   0.487 held-out): a bigger limiter than anything about the head. More training-dataset diversity would
   move the held-out ceiling more than any head change.

## "Tanked" framing

This lead does NOT produce a better head — because the premise (a closable architecture gap) was largely
a measurement artifact. The honest outcome: the gap is attributed (split artifact + calibrated-head
cost), every head knob is ruled out, and the calibrated head stands. A clean negative-with-correction.

## Lead (if point accuracy is ever wanted alongside calibration)

A **hybrid objective** — `dirichlet_edl_loss + λ·MSE(Dirichlet mean, target)` — could pull the mean to
the held-out ceiling (≈0.49) while the EDL/KL terms keep the concentration calibrated. Untested; it
trades a new hyperparameter and a possible calibration interaction for ~0.10 L1 on a secondary metric.

## Reproduce

```
# honest held-out ceiling vs in-distribution (the split artifact):
#   the offline diagnostics in this stage's run log
python scripts/stageC_composition_head.py --freeze-encoder --use-footprint-features --no-evidence \
       --offline-corpus 60 --epochs 40 --kl-max 0.5 --n-models 3   # footprint-only, offline (0.610)
```

## Files

- Metrics: `/mnt/artifacts/project_lacuna/runs/stage0_general_baseline/stageC_composition_{footprint-only,offline,offline_kl0}.json`.
- Knobs: `scripts/stageC_composition_head.py` (`--head-layers/--no-evidence/--offline-corpus`),
  `CompositionHead(n_hidden_layers, use_evidence)`; tests `tests/unit/models/test_composition_head.py`.
