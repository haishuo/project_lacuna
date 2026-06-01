# Stage F — does MNAR-subtype detectability (Pillar 1) survive into the composition posterior?

- **Date:** 2026-06-01
- **ADR:** docs/decisions/0007 (composition-posterior estimand). Follow-up to the A–E arc.
- **Branch:** `experiment/composition-stagef` (off `experiment/composition-stageb`).
- **Script:** `scripts/stageF_subtype_detectability.py`. **Report:**
  `runs/stage0_general_baseline/stageF_subtype_detectability.json` (+ `_tau1.json`).
- **Question:** ADR-0007's **Pillar 1** — "not all mechanisms are equally unidentifiable" — was confirmed
  at the **single-mechanism / per-column** level (Stage 4): a threshold / detection-limit MNAR process
  leaves a visibly truncated ("cliffed") observed distribution and is detectable ~0.9–1.0 per column,
  while smooth self-censoring is near the floor. The composition model, however, reports an **aggregate
  dataset-level MNAR fraction** that averages detectable (loud) and undetectable (quiet) MNAR subtypes
  together. Stage E's real anchors were dominated by the hard (self-censoring/social) subtypes, so its
  "MAR-vs-MNAR at chance" is *consistent* with Pillar 1, not a clean test of it. **Does the composition
  posterior, holding the MNAR FRACTION fixed, lean MORE MNAR and/or report HIGHER confidence when the
  MNAR is the LOUD (threshold/detection) subtype vs the QUIET (self-censoring) one?**

## Design — change ONE variable (the MNAR subtype family)

Two corpora of catalog datasets composed at a **fixed by-cell target MCAR .2 / MAR .3 / MNAR .5**
(miss rate drawn ~U(0.1, 0.4)); the only difference is the per-column MNAR mechanism family:

- **Corpus L (LOUD):** MNAR columns drawn only from `{threshold_left, threshold_right,
  threshold_two_sided, soft_threshold, col_specific_thresh, detection_lower, detection_upper,
  detection_both}` — sharp cutoffs.
- **Corpus Q (QUIET):** only `{self_censoring, selfcensor_high/low/extreme/weak/strong}` — smooth
  sigmoid-on-own-value.

**MNAR is forced PER-COLUMN-ONLY** (`mnar_block_share=0`) so the contrast is not diluted by the joint
MNAR blocks (latent/attrition/module), which have no threshold variant. **The matched-corpus
invariant** (verified in `tests/unit/data/test_composition_sampler.py`): the SAME rng seed builds both
corpora, so datasets, the allocation plan, per-column rates, and the **MCAR and MAR cell masks are
byte-identical** between L and Q — only the MNAR per-column masks differ. The contrast is reported as
**paired deltas** (L_i − Q_i on matched datasets), the cleanest possible single-variable comparison.

New optional, default-bit-identical generator knobs enable this (registry hash unchanged; suite
1712→1731): `mnar_column_pool.sample_mnar_column_generator(subtypes=…)`,
`composition_allocator.plan_allocation(mnar_block_share=…)`, and `composition_{sampler,batch}` threading
plus `build_composition_batch(fixed_composition=…)`.

**Instrument:** the deployed composition model = encoder + the Stage-C frozen+footprint **3-head deep
ensemble** (64-wide) + the Stage-D global temperature. (The on-disk Stage-C checkpoint had been
overwritten by the Stage-C2 capacity ablation with a single 256×2 head; we regenerated the canonical
3-head ensemble — L1 0.610 vs prior 0.708, reproducing Stage C — and re-fit τ = **4.99** via Stage D,
reproducing its pattern: ECE halved 0.087→0.035, beats prior-only on Brier. The C2 head is preserved
as `…_c2-256x2-single.pt`.)

5 seeds × 160 datasets/corpus/seed; all quantities reported as **mean ± sd over seeds**.

## Step 0 — FOOTPRINT CEILING (model-free): can the observable footprint see loud vs quiet?

A held-out RandomForest classifying L vs Q on the 20-D footprint **separates them essentially
perfectly**:

| | value |
|---|---|
| RF L-vs-Q AUC (per seed) | [0.999, 1.0, 1.0, 1.0, 1.0] → **0.9998 ± 0.0003** |
| pooled AUC / accuracy | 0.999 / 0.975 |
| top RF features | `mar_coupling_mean` .11, `mar_coupling_max` .11, `miss_corr_mean_abs` .11, `distinct_pattern_ratio` .05, `obs_abs_skew_mean` .05 |

But **what** separates them is the surprise. The Pillar-1 signature — the observed-value cliff — barely
moves at the dataset aggregate, while the **co-missingness / coupling** features move a lot:

| footprint feature | L (loud) | Q (quiet) | gap |
|---|--:|--:|--:|
| `obs_abs_skew_mean` (the cliff) | 1.423 ± 0.126 | 1.397 ± 0.120 | ~0 |
| `obs_excess_kurt_mean` (the cliff) | 5.71 ± 1.02 | 5.50 ± 0.89 | small |
| **`mar_coupling_mean`** | **0.405 ± 0.012** | **0.354 ± 0.008** | **+0.051** |
| **`miss_corr_mean_abs`** | **0.120 ± 0.002** | **0.102 ± 0.004** | **+0.018** |

The cliff (`obs_abs_skew_mean`) is the *MNAR-specific* fingerprint, but it is **column-averaged over all
columns** — only ~8 of ~20 are MNAR, and the catalog's intrinsic skew dominates — so the per-column cliff
is diluted to near-invisibility in the aggregate. What actually distinguishes loud is that a **threshold
on a column's own value, when columns are correlated, censors the same extreme rows across columns** →
**coupled missingness that also correlates with *other* columns' observed values** = exactly the
observable **MAR-coupling** signature. Self-censoring's graded sigmoid couples less. **So the footprint
can perfectly tell loud from quiet — but via a structure/coupling signal that is observationally
MAR-like, not via an MNAR-diagnostic one.** (The coupling gap goes *opposite* to the realized-rate gap
below, so it is a mechanism-type effect, not a quantity artifact.)

## Step 1 — MODEL READ (calibrated instrument; mean ± sd over 5 seeds; paired L−Q)

| quantity | L (loud) | Q (quiet) | paired Δ (L−Q) |
|---|--:|--:|--:|
| **MNAR fraction** `f_MNAR` (metric a) | 0.315 ± 0.006 | 0.310 ± 0.005 | **+0.005 ± 0.003** (seed5 ≈ 0) |
| can't-tell mass (metric b) | 0.654 ± 0.002 | 0.646 ± 0.002 | **+0.008 ± 0.003** (loud *less* confident) |
| Dirichlet α₀ concentration (metric b) | 4.59 | 4.65 | **−0.055 ± 0.019** (loud *less* concentrated) |
| P(f_MNAR ≥ 0.4) (metric c) | 0.315 ± 0.010 | 0.306 ± 0.009 | +0.010 ± 0.005 |
| — *honest-seam disaggregation* — | | | |
| `f_MCAR` (random-vs-structured axis) | 0.379 ± 0.007 | 0.404 ± 0.005 | **−0.025 ± 0.007** (loud more STRUCTURED) |
| MNAR\|structured (the MAR-vs-MNAR split) | 0.509 ± 0.005 | 0.521 ± 0.004 | **−0.013 ± 0.002** (loud *less* MNAR-within) |

The disaggregation is decisive:

1. **The identifiable axis responds, correctly.** Loud reads as **more structured** (`f_MCAR` −0.025,
   robust across all 5 seeds) — tracking its higher co-missingness/coupling, exactly as the footprint
   shows.
2. **The non-identifiable MAR-vs-MNAR split does NOT lean MNAR for loud — it leans slightly the other
   way** (MNAR\|structured −0.013, robust). Loud's distinguishing signature is MAR-coupling, which a
   calibrated posterior (rightly, per Molenberghs) attributes toward **MAR**, not MNAR.
3. **Confidence goes the wrong way.** Loud is *less* confident (higher can't-tell, lower α₀), not more.
4. **The tiny +0.005 `f_MNAR` lean is a by-product of the structure shift** (`f_MNAR = f_structured ×
   MNAR|structured`; f_structured rose, MNAR|structured fell), not mechanism detection — and it is
   neither robust (seed 5 = 0) nor meaningful (< 0.02). The model's `f_MNAR` is in fact **pinned near
   ~0.32 for both corpora**, far below either realized truth (≈0.46–0.50) — near-blind to the MNAR
   fraction in this regime, and to the subtype.

**Robust to calibration.** With τ = 1 (uncalibrated), the effect is *stronger in the same direction*:
`f_MCAR` Δ −0.051, MNAR\|structured Δ −0.031. Calibration did not hide a lean — the raw posterior itself
reads loud's signature as MAR-ward; τ only compresses the (wrong-way) split toward zero.

## Verdict — NULL on the mechanism axis (the informative null)

> **The detectable-subtype signal does NOT survive aggregation into the dataset-level composition
> posterior's MNAR fraction.** The footprint can separate loud from quiet essentially perfectly
> (AUC ≈ 1.0), and the posterior *does* respond on the identifiable random-vs-structured axis (loud
> reads more structured). But loud's distinguishing signature, at the column-averaged aggregate, is
> co-missingness/coupling — observationally MAR-like — so the MAR-vs-MNAR split does **not** lean MNAR
> for loud (if anything, MAR-ward) and confidence does not rise. Pillar 1 lives at the per-column /
> per-mechanism level (Stage 4); the dataset-level fraction averages it away.

This is **consistent with**, not a refutation of, Pillar 1: the per-column cliff is real (Stage 4), but
(a) it is diluted by column-averaging in the 20-D footprint, and (b) the part of the loud-MNAR footprint
that *does* survive aggregation (coupled censoring of extreme rows) is the MAR-coupling fingerprint, not
the MNAR-cliff. The honest one-line summary of the arc — "the instrument reads STRUCTURE and abstains on
MECHANISM" — is **refined, not overturned**: it was suggested that the wide MAR-vs-MNAR band might
"sharpen for loud subtypes"; at the **dataset-aggregate composition** granularity it does not. The place
to express loud-MNAR detectability is the **per-column (Q3) layer** — a concrete argument for not having
demoted Q3 to optional, at least for the "which mechanism" question on loud subtypes.

## Confounds checked / caveats

- **Realized composition is matched in target, slightly mismatched in realization.** The plan and target
  are byte-identical; realized comp is L ≈ (.22, .32, .46) vs Q ≈ (.21, .30, .50), miss-rate .264 vs
  .283 — loud realizes a *lower* MNAR fraction (threshold hits its rate budget; self-censoring overshoots
  — the Stage-4b property). This works **against** the verdict being an artifact: loud has *less* MNAR
  yet *more* coupling, so the "loud → MAR-ward via coupling" conclusion is a mechanism-type effect, and
  the model under-reads MNAR for both (pinned near prior) regardless. It does mean the raw
  MNAR\|structured magnitude is mildly confounded by quantity (Q has more true MNAR); the robust,
  unconfounded signals are the coupling gap (opposite sign to the quantity gap) and the structure-axis
  response.
- **Single instrument, semi-synthetic, survey-scoped** (ADR-0005/0007). Same caveats as the rest of the
  arc; MAR-vs-MNAR is non-identifiable on real data by construction.
- **LOUD/QUIET are the per-column-targetable own-value families** (8 + 6 subtypes); the contrast is
  family-vs-family, both internally diverse (no monoculture — the Stage-4/5 lesson).
- **Paired design**: with matched corpora the per-seed Δ has sd ~0.002–0.007, so even tiny effects are
  statistically resolvable — which is *why* we report magnitudes (robustly-tiny ≠ meaningful) and the
  structure-vs-split disaggregation, not just significance.

## Reproduce

```
python scripts/stageF_subtype_detectability.py            # full (5 seeds); reads τ from stageD report
python scripts/stageF_subtype_detectability.py --tau 1.0  # uncalibrated robustness check
python scripts/stageF_subtype_detectability.py --quick     # fast smoke (2 seeds)
```

Prereq (if the canonical 3-head Stage-C checkpoint is not on disk):
`python scripts/stageC_composition_head.py --freeze-encoder --use-footprint-features --n-models 3`
then `python scripts/stageD_calibration.py`.

## Files

- Report: `runs/stage0_general_baseline/stageF_subtype_detectability.json` (+ `_tau1.json`).
- Script: `scripts/stageF_subtype_detectability.py`.
- Generator knobs + tests: `lacuna/data/{mnar_column_pool,composition_allocator,composition_sampler,
  composition_batch}.py`; `tests/unit/data/test_{mnar_column_pool,composition_allocator,
  composition_sampler,composition_batch}.py`.
</content>
</invoke>
