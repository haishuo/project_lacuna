# The composition arc (ADR-0007) — synthesis: Stages A–E + follow-ups

- **Date:** 2026-05-30
- **ADR:** docs/decisions/0007 (composition-posterior estimand).
- **Branch:** `experiment/composition-stageb`. This doc ties the per-stage findings into one narrative.
- **Per-stage docs:** `2026-05-30-stage{A,B,C,D,E}-*.md`, `2026-05-30-stageD2-feature-calibration.md`,
  `2026-05-30-stageC2-ceiling-gap-attribution.md`.

## The estimand and the question

ADR-0007 reframed Lacuna's output as a **calibrated distribution over a dataset's missingness
COMPOSITION** `(f_MCAR, f_MAR, f_MNAR)` on the simplex, plus an explicit **can't-tell mass** — not
per-column hard labels (Stage 5 killed those as a primitive). The denominator is LOCKED as the
**missing cell**, and the **generator is the operational definition** of the estimand: its target, its
ground-truth tags, and the eval metric all share one denominator. The arc asks: can we build a
generator realistic enough to train on, a head that estimates the composition, and a posterior that is
*calibrated* — and how much of the composition is even knowable?

## The arc at a glance

| stage | question | headline result |
|---|---|---|
| **A** | how far is our synthetic missingness from real? | discriminator AUC **1.000** — trivially separable; the gap is cross-column STRUCTURE |
| **B** | build a composition-controlled, realism-tuned generator | GATE-1 realised==target by cell (L1≈0.08 at d≥13); realism AUC **1.000→0.985**, per-feature σ-gaps collapse (rate alone no longer separates; best single-feature AUC 0.97→0.85) |
| **C** | a Dirichlet composition head on the encoder | frozen encoder fails prior-only; the signal is in the FOOTPRINT (RF L1 0.385), the encoder under-represents it; footprint-feature head recovers the identifiable composition (L1 **0.589** vs prior 0.708) |
| **D** | calibrate the posterior (the headline) | temperature **halves ECE (0.072→0.044)**, beats prior-only on the proper score (Brier 0.151 vs 0.175); honest seam: resolution on the MCAR axis, prior-fallback on MAR/MNAR |
| **E** | face validity on real survey anchors | reads STRUCTURE (identifiable) and ABSTAINS on MECHANISM (non-identifiable); NHANES sensitive items read 0.75–0.83 structured; MAR-vs-MNAR at chance (a correct wide band); PISA MCAR-by-design blind spot documented |
| **D-v2** | can per-instance uncertainty be fixed? | **null** — composition error is not predictable from the footprint (held-out R²=−0.31); a global temperature is the best attainable |
| **C-v2** | can the head reach the 0.385 footprint ceiling? | the "gap" is ~half an eval-split artifact (honest held-out ceiling 0.487) + ~half the calibrated-head cost; capacity/depth/evidence/regime/KL/MSE-hybrid all ruled out |

## The through-line

**The instrument reads STRUCTURE and abstains on MECHANISM** — and this single sentence is consistent
across every stage:

- **The identifiable axis (random-vs-structured) is recoverable and transfers.** Stage B makes the
  generator produce realistic cross-column structure; Stage C shows the observable footprint carries
  the composition's identifiable content; Stage D resolves the MCAR axis well beyond prior-only; Stage E
  confirms real survey missingness reads as structured, tracking the footprint's actual co-missingness.
- **The non-identifiable axis (MAR-vs-MNAR) is a wide, calibrated band — by design, not by failure.**
  Stage D's resolution collapses toward prior-only on the MAR/MNAR split; Stage E reads MNAR-vs-MAR
  anchors at chance with a large can't-tell mass. This is the Molenberghs limit, reported honestly.
- **The uncertainty is honest at the levels where it can be.** Stage D calibrates the *global*
  reliability; the *axis-level* seam flags MAR/MNAR as unresolvable. D-v2 shows the *per-instance*
  ranking cannot be improved — the error is non-identifiable noise, homoscedastic in the footprint.

## What the negative results establish (they are findings, not gaps)

- **D-v2 (per-instance calibration): null, attributably.** Per-dataset composition error is not
  predictable from the observable footprint (held-out R²=−0.31). The footprint carries the COMPOSITION
  signal but not the per-instance ERROR signal; honest uncertainty therefore lives at the axis and
  global levels, not per-instance.
- **C-v2 (the head→ceiling gap): half artifact, half a principled cost.** The eye-catching "0.589 vs
  0.385" was ~half an in-distribution evaluation-split artifact (honest held-out ceiling 0.487) and
  ~half the point-accuracy cost of a calibrated Dirichlet head vs a plain MSE regressor. Every lever —
  capacity, depth, the frozen evidence, online/offline training, KL weight, and an MSE-hybrid loss —
  was tried; none close it. The 0.487 ceiling is attainable only by a point regressor with NO posterior.
- Both nulls were **gated by feasibility probes / ceiling re-measurements first**, so the outcome is
  attributable rather than a confounded "it didn't work."

## What was built (all on the frozen v1.0 encoder; registry bit-identical throughout)

- Generator: `lacuna/data/composition_{target,blocks,allocator,sampler,batch}.py`.
- Head + losses: `lacuna/models/composition_head.py`,
  `lacuna/training/composition_{loss,calibration,recalibration}.py`.
- Scripts: `scripts/stage{a_realism_gap,C_composition_head,D_calibration,D2_feature_calibration,
  E_face_validity}.py`.
- ~1712 tests green throughout; the v1.0 encoder and the generator registry were never modified
  (composition is a purely additive, multi-task head).

## Defensible claims vs scoped caveats (for the write-up)

**Claim:** a calibrated composition posterior that (a) hits its by-cell target on semi-synthetic, (b)
is reliable on simplex-region statements (ECE 0.044, beats prior-only on a proper score), (c) resolves
the identifiable random-vs-structured axis and honestly abstains on the non-identifiable MAR-vs-MNAR
split, and (d) behaves face-validly on real survey anchors.

**Caveats (scoped, in neon):** survey-specialised (the realism target is the NHANES/survey manifold);
semi-synthetic supervision (real data has no mechanism ground truth — MAR-vs-MNAR is non-identifiable
by construction); "MCAR" in the estimand means value-independent AND unstructured, so planned-missing
block designs (PISA rotation) are a documented blind spot; point accuracy is bounded both by
generalisation to unseen dataset types and by the calibrated-distribution cost.

## Open leads (characterised, not chased)

- A **planned-missing / value-dependence flag** to recover MCAR-by-design block designs (the PISA blind
  spot) — needs MCAR-block generation + a value-dependence test; only n=2 real anchors to validate on.
- **More training-dataset diversity** would move the held-out ceiling (the dominant real limiter:
  RF in-distribution 0.365 → held-out 0.487) more than any head change.
- A genuinely different readout (non-MLP, or a two-stage point-then-calibrate scheme) is the only
  untried route to the point ceiling WITH a posterior — but point accuracy is secondary to calibration.
