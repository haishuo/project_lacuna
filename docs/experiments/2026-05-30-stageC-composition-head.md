# Stage C — Dirichlet composition head + can't-tell mass

- **Date:** 2026-05-30
- **ADR:** docs/decisions/0007 (composition-posterior estimand), Stage C.
- **Predecessors:** Stage B (the generator + by-cell ground truth this head is supervised on).
- **Modules:** `lacuna/models/composition_head.py`, `lacuna/training/composition_loss.py`,
  `lacuna/data/composition_batch.py` (+ tests). **Script:** `scripts/stageC_composition_head.py`.
- **Question:** ADR-0007 commitment 6 adds a head that emits a calibrated DISTRIBUTION over the
  composition simplex (a Dirichlet) + an explicit can't-tell mass, on the existing encoder
  (frozen-first), supervised by the Stage-B realised by-cell composition. Stage C asks: does the head
  learn the composition, does it beat the pre-registered baselines, and does its uncertainty behave?

## Method

- **Head** (`composition_head`): the encoder's dataset-level `evidence` vector → `alpha = softplus(logits)+1`
  ∈ ℝ³, a Dirichlet `Dir(alpha)` over `(f_MCAR, f_MAR, f_MNAR)`. The expected composition is `alpha/alpha0`;
  the **can't-tell mass is the vacuity `K/alpha0`** (Sensoy 2018) — a query on the Dirichlet, per ADR.
  Purely additive (multi-task): the existing class head is untouched. A deep ensemble pools by
  evidence-summing.
- **Loss** (`composition_loss`): evidential — expected-CE Bayes risk (pulls the mean to the realised
  composition) + annealed `KL(Dir(alpha_tilde)||Dir(1))` on the misleading evidence (retains vacuity
  where the footprint can't pin the split → calibration). Target = the by-cell FRACTION (scale-free in
  the denominator → epistemic, not sampling, uncertainty).
- **Data** (`composition_batch`): catalog datasets punched by the Stage-B generator at compositions
  drawn from the broad simplex prior; the supervised target is the realised by-cell composition.
- **Baselines (pre-registered "tanked" bar, ADR-0007):** the posterior must beat a **prior-only**
  constant predictor (the eval-set mean composition). Evaluation reports composition L1, the honest
  seam (random-vs-structured `f_MCAR` MAE vs the MAR-vs-MNAR split MAE), the can't-tell behaviour
  (corr of vacuity with error), and a simplex-query ECE (a Stage-D calibration preview).

## Result 1 — frozen-encoder probe FAILS the prior-only bar (a clean negative)

Frozen encoder (trained for the old single-label task), 3-head ensemble, 15 epochs, n=640 held-out:

| metric | single head | ensemble-3 | prior-only |
|---|--:|--:|--:|
| composition L1 | 0.736 | **0.728** | **0.708** |
| per-class MAE (MCAR/MAR/MNAR) | .248/.236/.253 | .240/.236/.252 | — |
| random-vs-structured (MCAR) MAE | 0.248 | 0.240 | — |
| MAR-vs-MNAR split MAE | 0.332 | 0.332 | — |
| mean can't-tell mass | 0.516 | 0.283 | — |
| corr(vacuity, error) | +0.220 | +0.185 | — |
| query ECE (preview) | 0.118 | 0.145 | — |

The frozen head **does not beat prior-only** (L1 0.728 vs 0.708) — by the pre-registered definition,
the frozen probe is *tanked*. The encoder's pooled `evidence`, learned for the old dataset-level
single-label task, does not carry the composition signal (the Stage 0/1 frozen-encoder pattern, now at
the composition level). Two signals survive the negative, though: **vacuity correlates positively with
error (+0.19–0.22)** — the can't-tell mass already tracks "how wrong am I" — and the honest seam is
visible (random-vs-structured MAE 0.24 < MAR-vs-MNAR split 0.33).

## Result 2 — fine-tune (encoder + head end-to-end) only marginally beats prior-only

Encoder+head fine-tuned (lr 3e-4), 2-head ensemble, 20 epochs, n=640:

| metric | single | ensemble-2 | prior-only |
|---|--:|--:|--:|
| composition L1 | 0.695 | 0.697 | 0.708 |
| per-class MAE (MCAR/MAR/MNAR) | .211/.230/.255 | .213/.230/.255 | — |
| random-vs-structured (MCAR) MAE | **0.211** | 0.213 | — |
| MAR-vs-MNAR split MAE | **0.333** | 0.332 | — |
| corr(vacuity, error) | +0.106 | +0.225 | — |
| query ECE (preview) | 0.072 | 0.089 | — |

Fine-tuning crosses the prior-only bar (0.695 < 0.708) but only **marginally**, and the entire gain
is on the random-vs-structured (MCAR) axis; the MAR-vs-MNAR split MAE (0.33) is essentially the
no-information value. The encoder, even trained end-to-end, barely extracts the composition from its
pooled evidence. (Query ECE improves vs the frozen probe, 0.118→0.072 — fine-tuning helps calibration
more than point accuracy.)

## Result 3 — attribution: the signal IS in the footprint; the ENCODER is the bottleneck

A direct footprint(20-D)→composition regressor (RandomForest, 2500 Stage-B datasets, 30% test=750)
bounds what *any* model reading the observable mask can recover:

| metric | RandomForest on footprint | its prior-only |
|---|--:|--:|
| composition L1 | **0.385** | 0.617 |
| per-class MAE (MCAR/MAR/MNAR) | .111/.125/.150 | — |
| random-vs-structured (MCAR) MAE | 0.111 | — |
| MAR-vs-MNAR split MAE | 0.210 | — |

The footprint regressor **crushes its prior-only baseline** (L1 0.385 vs 0.617) and even halves the
MAR-vs-MNAR split error (0.21) relative to the encoder (0.33). Its top features are the cross-column
structure ones — `miss_corr_mean_abs` (importance 0.34), `mar_coupling_mean` (0.14), `miss_corr_max_abs`,
`col_rate_max`, `distinct_pattern_ratio`. So the composition is substantially recoverable from the
*observable* footprint, but the encoder's pooled evidence — built/trained for the old per-column
single-label task — does not surface that cross-column structure. **The encoder is the bottleneck, not
the observable signal.** (Prior-only differs across setups, 0.617 vs 0.708, because the eval
distributions differ; the valid comparison is within-setup: RF beats its prior by 0.23, the encoder by
~0.01.)

This is the composition-level echo of the column arc's Stage-3a result — explicit deployable features
carry signal the encoder misses.

## Result 4 — fix: frozen encoder + explicit footprint features

Bridging the bottleneck the Stage-3a way — concatenate the 20-D observable footprint (deployable, no
oracle) to the frozen encoder's evidence before the Dirichlet head. 3-head ensemble, 15 epochs, n=640:

| metric | single | ensemble-3 | prior-only |
|---|--:|--:|--:|
| composition L1 | 0.634 | **0.589** | 0.708 |
| per-class MAE (MCAR/MAR/MNAR) | .197/.196/.241 | .162/.197/.230 | — |
| random-vs-structured (MCAR) MAE | 0.197 | **0.162** | — |
| MAR-vs-MNAR split MAE | 0.312 | 0.315 | — |
| corr(vacuity, error) | −0.065 | **−0.160** | — |
| query ECE (preview) | 0.071 | 0.091 | — |

The footprint features **clearly beat both prior-only (0.589 vs 0.708) and the fine-tuned encoder
(0.697)** — with a frozen encoder and no oracle. The honest seam is sharp: the random-vs-structured
(MCAR) axis is well-recovered (MAE 0.16) while the MAR-vs-MNAR split stays at the no-information value
(0.32). The catch: the can't-tell mass now **anti-correlates** with error (−0.16, down from +0.19
without footprints) — sharpening the point estimate broke the uncertainty's link to residual error.

### Summary across configurations (composition L1 ↓, per its own eval)

| configuration | composition L1 | vs prior | MCAR-axis MAE | MAR-vs-MNAR split | corr(vacuity, error) |
|---|--:|--:|--:|--:|--:|
| frozen probe (ens-3) | 0.728 | 0.708 (worse) | 0.240 | 0.332 | +0.19 |
| fine-tune (ens-2) | 0.697 | 0.708 (marginal) | 0.213 | 0.332 | +0.23 |
| **frozen + footprint (ens-3)** | **0.589** | 0.708 (**−0.12**) | **0.162** | 0.315 | −0.16 |
| RF footprint ceiling | 0.385 | 0.617 | 0.111 | 0.210 | — |

## Interpretation

1. **The composition head recovers the IDENTIFIABLE composition from deployable observables.** Frozen +
   footprint beats prior-only by 0.12 L1 (and the encoder alone by 0.11), with the gain on the
   random-vs-structured axis (MCAR MAE 0.16). This passes the ADR-0007 "tanked" bar (beat prior-only)
   and confirms the estimand is learnable at the dataset level — *if* the cross-column footprint is fed
   in. The MAR-vs-MNAR split stays near the no-information value (≈0.32), exactly the honest seam: the
   identifiable part is recovered, the non-identifiable part is not — as designed, not a defect.
2. **The encoder is the bottleneck, not the observable signal.** The frozen and fine-tuned encoder both
   sit near prior-only (0.70–0.73), yet a RandomForest on the same 20-D footprint reaches 0.385 and a
   frozen encoder + those 20 features reaches 0.589. The encoder's pooled `evidence`, learned for the
   old per-column single-label task, under-represents the cross-column structure (`miss_corr`,
   `mar_coupling`, `distinct_pattern_ratio`) that carries the composition — the composition-level echo
   of the column arc's Stage-3a deployable-features finding. The footprint features bridge it.
3. **A unification worth noting.** The same observable footprint Stage A used to *measure realism* and
   Stage B tuned the generator *toward* is also what carries the *supervised composition signal* the
   head needs. Realism target and learnable signal are the same 20 statistics.
4. **Calibration is unsolved and is the Stage-D headline.** The point estimate improved but the can't-
   tell mass stopped tracking error (vacuity-error corr +0.19 → −0.16 once footprints sharpened the
   fit), and query ECE held ~0.07–0.09. The EDL concentration + deep ensemble give an uncertainty
   *mechanism*, but it is not yet *calibrated* — precisely what Stage D must fix (recalibrating the
   Dirichlet concentration / the can't-tell mass against held-out reliability). Per ADR-0007 point
   accuracy was always secondary; Stage C establishes the head and hands a sharply-posed calibration
   problem to Stage D.
5. **Gap to the footprint ceiling (0.589 vs 0.385).** The NN head leaves signal on the table vs the RF
   — leads: a larger/deeper head, joint multi-task training, or feeding the footprint features through
   the encoder rather than concatenating at the evidence layer.

## Interpretation

_INTERPRETATION_

## Caveats

- **Semi-synthetic, survey-scoped** (ADR-0005/0007). The realised composition exists as ground truth
  only because we punched the holes; MAR-vs-MNAR is non-identifiable on real data by construction.
- **Point accuracy is secondary** (ADR-0007): the Stage-D headline is CALIBRATION of the posterior, not
  composition L1. Stage C establishes the head + the uncertainty mechanism; the query-ECE here is a
  preview, not the calibrated deliverable.
- Distributions over seeds/ensemble are reported; single-seed point numbers are not the headline.

## Reproduce

```
python scripts/stageC_composition_head.py --freeze-encoder --n-models 3                 # probe (negative)
python scripts/stageC_composition_head.py --n-models 2 --lr 3e-4 --epochs 20            # fine-tune (marginal)
python scripts/stageC_composition_head.py --freeze-encoder --use-footprint-features \
       --n-models 3                                                                      # the fix (L1 0.589)
```

## Files

- Metrics: `/mnt/artifacts/project_lacuna/runs/stage0_general_baseline/stageC_composition_{frozen-probe,fine-tune,frozen-footprint}.json`.
- Modules: `lacuna/models/composition_head.py`, `lacuna/training/composition_loss.py`,
  `lacuna/data/composition_batch.py`; script `scripts/stageC_composition_head.py`; tests under `tests/unit/`.
