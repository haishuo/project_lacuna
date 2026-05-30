# Stage B — the composition-controlled, realism-tuned generator

- **Date:** 2026-05-30
- **ADR:** docs/decisions/0007 (composition-posterior estimand), Stage B.
- **Predecessor:** docs/experiments/2026-05-30-stageA-realism-gap.md (the gap this stage closes).
- **Modules:** `lacuna/data/composition_{target,blocks,allocator,sampler}.py` (+ their test files).
- **Gate script:** `scripts/stagea_realism_gap.py --generator composition` (the Stage-A discriminator-
  as-ruler, now pointed at the new generator; legacy default unchanged for comparison).
- **Question:** ADR-0007 makes the generator the *operational definition* of the estimand: it must
  (G1) realise a target **by-cell** composition `(f_MCAR, f_MAR, f_MNAR)` drawn from a broad simplex
  prior — target, ground-truth tags, and metric sharing ONE denominator (the missing cell) — and
  (G2) produce missingness whose **observable footprint** is materially less distinguishable from
  real survey missingness than the current generator (Stage A: discriminator AUC 1.000), while
  keeping the per-column fingerprints (Band 3) and the honest seam (MCAR random vs structured).

## Method — architecture

A target composition flows through four focused modules (Coding Bible Rule 3; each tested):

1. **`composition_target`** — draws `(f_MCAR, f_MAR, f_MNAR)` from a symmetric Dirichlet (concentration
   1 = uniform over the simplex, prior-agnostic) and an overall miss rate from a broad uniform
   (~the real 0.01–0.60 range). Composition and miss rate are orthogonal (a ratio is scale-free).
2. **`composition_allocator`** (pure planner) — partitions the working columns across classes by the
   target fractions (largest-remainder), then draws each unit's per-column rate from one
   `Beta(mean=μ, ν=1.2)` law and **rescales each class's rates (iterative water-fill, per-unit caps)
   to hit that class's exact by-cell budget** `f_c·miss_rate·d`. The rescale pins the composition to
   target *and* the overall miss rate, while multiplicative scaling preserves the per-column rate
   spread — so Band-2 rate variation (sd≈0.33, a heavily-missing tail) survives as a *free* by-product
   rather than a hidden confound (the Stage-5 lesson, made constructive). High rates are routed to the
   block units (and MCAR Bernoulli), which tolerate them; the diverse MAR/MNAR column pools are capped
   pool-safe (≤0.60).
3. **`composition_blocks`** — applies the re-admitted JOINT mechanisms to a block's columns via a
   sub-matrix view (so a block can only delete *within* its columns), tuned to a target rate:
   - MNAR: `MNARModuleRefusal` (sharp value-driven battery refusal — co-missingness, few patterns),
     `MNARAttrition` (monotone dropout, row-gated for rate), `MNARLatentHealth` (graded latent
     co-missingness, row-gated). MAR: `MARModuleSkip` (sharp gate-driven battery skip — the MAR twin).
   - All four reuse the registry-frozen generators unchanged (Rule 8; registry stays bit-identical).
4. **`composition_sampler`** — applies the plan (per-column units via the existing MCAR/MAR/MNAR column
   pools, blocks via `composition_blocks`), records the per-**cell** mechanism TAG matrix (the by-cell
   ground truth), and measures the realised composition. Every column is owned by exactly one unit, so
   each missing cell has an **unambiguous** tag — the LOCKED joint-cell convention realised with no
   cross-mechanism overlap to adjudicate.

**Honest seam, structural:** MCAR is per-column Bernoulli (the random anchor — the identifiable
"structured-vs-random" axis); MAR *and* MNAR both carry block structure, so blockiness ≠ MNAR and the
MAR-vs-MNAR split stays the genuinely non-identifiable part.

## Result G1 — realised == target by cell (the shared-denominator invariant holds)

The realised by-cell composition tracks the drawn target. On the actual catalog data in the gate
regime (d∈[4,25], broad prior, block_rate_share 0.85, n=400 draws), stratified by width:

| width | n | realised-composition L1 (mean) | p90 |
|---|--:|--:|--:|
| d 4–8 (coarse) | 110 | 0.146 | 0.265 |
| d 9–12 | 99 | 0.095 | 0.180 |
| **d 13–25 (real median d=13 and up)** | 191 | **0.078** | 0.152 |

Realised miss-rate error: mean 0.037. Fidelity is tight at adequate width (L1≈0.08 at d≥13) and
degrades at small d, where a **3-class composition is inherently coarsely quantised on few columns**
(a near-zero target class gets 0 columns) — an expected, documented granularity limit, not a control
failure. The unit test `test_composition_sampler.py` asserts the invariant at d≈18 over fixed and
random targets (mean L1 < 0.10, each seed < 0.20); the planner's *expected* composition is near-exact
(L1≈0.008 at d=20), confirming the residual is realisation noise + quantisation, not bias.

## Result G2 — realism gap: AUC 1.000 → 0.985 ± 0.005, every per-feature gap shrank

Discriminator ROC-AUC (5-fold CV, 413 real vs 250 synthetic, **5 seeds**, block_rate_share 0.85):
**0.985 ± 0.005** (per-seed 0.979 / 0.980 / 0.991 / 0.991 / 0.985), down from Stage A's **1.000**.

The AUC moved little — but that is the *insensitive* readout here (see below). The substantive result
is the per-feature standardised-gap collapse (real−synth, in pooled σ; Stage A vs Stage B):

**Band 1 — cross-column structure (the fundamental gap):**

| feature | Stage A synth (gap) | Stage B synth (gap) |
|---|--:|--:|
| `distinct_pattern_ratio` | 0.422 (−1.39σ) | 0.198 (−0.86σ) |
| `miss_corr_max_abs` | 0.253 (+2.84σ) | 0.727 (**+0.45σ**) |
| `miss_corr_mean_abs` | 0.055 (+1.77σ) | 0.126 (+1.19σ) |
| `frac_pairs_coupled` | 0.002 (+1.48σ) | 0.086 (+0.93σ) |
| `monotone_row_frac` | 0.211 (+1.69σ) | 0.407 (+0.93σ) |
| `top1_pattern_frac` | 0.120 (+1.66σ) | 0.191 (+1.22σ) |
| `top3_pattern_frac` | 0.229 (+2.08σ) | 0.382 (+1.39σ) |

**Band 2 — per-column rate distribution (essentially closed):**

| feature | Stage A synth (gap) | Stage B synth (gap) |
|---|--:|--:|
| `col_rate_max` | 0.368 (+3.34σ) | 0.679 (**+0.87σ**) |
| `frac_cols_high_miss` | 0.003 (+3.10σ) | 0.266 (**+0.80σ**) |
| `col_rate_sd` | 0.125 (+3.06σ) | 0.244 (**+0.97σ**) |
| `col_rate_mean` | 0.216 (+1.77σ) | 0.289 (+0.84σ) |

**Band 3 — mechanism fingerprints (≈ unchanged, as intended):**

| feature | Stage A synth (gap) | Stage B synth (gap) |
|---|--:|--:|
| `mar_coupling_max` | 0.555 (+1.04σ) | 0.511 (+1.24σ) |
| `obs_abs_skew_mean` | 1.892 (+0.63σ) | 1.780 (+0.72σ) |
| `obs_excess_kurt_mean` | 19.5 (+0.28σ) | 17.0 (+0.36σ) |

## Interpretation

1. **Band 2 is closed and Band 1 is roughly halved.** Every one of Stage A's *trivial* separators —
   the rate features at >3σ and the co-missingness/pattern features at 1.5–2.8σ — dropped to <1.4σ.
   The largest remaining gap is `top3_pattern_frac` at +1.39σ, versus `col_rate_max` at +3.34σ before.
   In Stage A a classifier could separate on **rate alone**; in Stage B no single footprint feature
   separates the two corpora — the discriminator now needs the full 20-D joint of many ~1σ residuals.
   **Best single-feature AUC falls 0.969 → 0.851**: for the legacy generator one feature
   (`frac_cols_high_miss`, i.e. rate) nearly separates the corpora on its own; for the composition
   generator the strongest single axis (`miss_corr_mean_abs`) reaches only 0.85, well below the joint
   0.985 — the separability has moved off any one axis onto the multivariate joint.
2. **AUC ≈ 1.0 is a saturated headline, not a null result.** With 413 real exemplars and a 20-D
   footprint, a linear discriminator finds a separating hyperplane from the *combination* of many
   small-but-consistent offsets, so AUC sits near 1 whenever any systematic multivariate difference
   remains. The σ-gap collapse (and the single-feature-AUC drop) is the metric that tracks the actual
   progress; AUC barely moving while gaps fall 3–4× is exactly that saturation.
3. **The residual is distributed and robust to the two cheap levers.** A focused diagnostic showed
   neither bigger blocks (max_block_width 8→20: AUC 0.985→0.987 — within-block correlation is already
   ~1; mean *cross*-block correlation is unaffected) nor a structured low-MCAR prior (AUC 0.985→0.978)
   meaningfully closes it. The gap is spread across pattern concentration (`top1/top3_pattern_frac`),
   mean co-missingness (`miss_corr_mean_abs` 0.13 vs real 0.34; `frac_pairs_coupled` 0.09 vs 0.28),
   monotonicity (0.41 vs 0.67), and `mar_coupling` — not a single tunable knob.
4. **Band 3 held (mechanism fingerprints preserved).** `mar_coupling_max` drifted slightly *worse*
   (+1.04σ→+1.24σ): block-dominated MAR (a battery gated on one in-block column) carries marginally
   less per-column MAR-coupling diversity than the legacy per-column MAR pool — a small, named cost of
   routing most MAR into skip blocks.

## Residual gap and leads (one variable at a time)

The dominant remaining separator is **mean pairwise miss-correlation** (real 0.34 vs synth 0.13) — and
bigger blocks don't move it because blocks only raise *within*-block correlation. The mechanism real
surveys have that the current generator lacks is **cross-block coupling**: the same respondents
(unit-nonresponse / burden) miss *multiple, different* modules, correlating missingness across the
whole row. The clear Stage-B-v2 lead is a **shared per-row nonresponse-propensity** that couples the
block row-selections (and a **nested skip hierarchy** for `monotone_row_frac` + pattern concentration).
Both were left out of Stage B deliberately: they add a mechanism that would have to be folded into the
by-cell budget (risking the tested G1 invariant) for a residual that is distributed, not single-feature.
A modest MAR per-column share could recover `mar_coupling` at a small cost to pattern structure.

## Caveats

- **Survey-scoped.** The real corpus is ~97% NHANES-sampled blocks (ADR-0005 specialisation); "realistic"
  means realistic *for surveys*, and the σ-gaps are measured against that corpus.
- **Observable floor, not mechanism proof** (ADR-0007). Matching the footprint is necessary, not
  sufficient — it validates the surface, not the mechanism *mix*. The by-cell tags are exact regardless
  (a generative fact); identifiability of MAR-vs-MNAR is a separate, Molenberghs-bound question.
- **G1 granularity.** Composition fidelity needs enough columns (L1≈0.08 at d≥13, ≈0.15 at d 4–8).
- **The broad training prior includes low-structure draws** (high-MCAR, balanced 3-way) that are
  intrinsically less survey-like; we keep it (prior-agnostic training, ADR-0007 commitment 4) and
  report the cost rather than narrowing the prior to flatter the metric.

## Reproduce

```
python scripts/stagea_realism_gap.py --generator composition --block-rate-share 0.85 \
    --n-synth 250 --n-seeds 5 --output /mnt/artifacts/project_lacuna/composition/stageb_gap.json
# baseline for comparison: --generator legacy (Stage-A path, AUC ~1.000)
```

## Files

- Gate report: `/mnt/artifacts/project_lacuna/composition/stageb_gap.json` (AUC mean±sd, per-seed,
  per-feature table).
- Real corpus (Stage A): `/mnt/artifacts/project_lacuna/composition/stagea_real_footprints.json`.
- Generator: `lacuna/data/composition_{target,blocks,allocator,sampler}.py`; tests under `tests/unit/data/`.
