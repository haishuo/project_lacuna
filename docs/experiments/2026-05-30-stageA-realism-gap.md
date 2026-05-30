# Stage A — The realism gap: our synthetic missingness is trivially distinguishable from real

- **Date:** 2026-05-30
- **ADR:** docs/decisions/0007 (composition-posterior estimand), Stage A.
- **Scripts:** `scripts/stagea_real_footprints.py` (real corpus), `scripts/stagea_realism_gap.py` (gap).
- **Question:** ADR-0007 commits to making the synthetic generators *observably realistic* (match real
  survey missingness on a fixed footprint), because Lacuna's calibration is only as trustworthy as the
  manifold it is trained on. Before rebuilding the generator (Stage B) we measure the gap: how far is
  our *current* synthetic missingness from real survey missingness, and **on which features**?

## Method

- **Footprint** (`lacuna.data.missingness_footprint`): 20 scale-free, observable, mechanism-agnostic
  statistics of a mask + observed values (column-rate distribution, row patterns, co-missingness
  correlation, pattern structure, monotonicity, MAR-coupling, MNAR observed-distortion).
- **Real corpus (n = 413):** the 14 labelled survey anchors + 399 analyst-realistic blocks sampled
  from the raw NHANES tables (`real_mask_sampler`; blocks 4–25 cols × 200–3000 rows, realised miss
  0.04–0.60). Weighted ~97% NHANES-sampled / ~3% curated anchors.
- **Synthetic corpus (n = 400):** our **current best** generator — `compose_mixed_missingness` at
  full diversity (`mnar_diverse + mar_diverse + compensate_rate`), punched into the complete catalog
  datasets, matched to the real (n, d) regime.
- **Discriminator-as-ruler** (ADR-0007 commitment 5 — we *read* it, we do not train against it):
  standardised logistic regression, real=1 / synth=0, 5-fold CV ROC-AUC + standardised coefficients.

## Result — AUC = 1.000 (perfectly separable)

A linear classifier tells real from synthetic **without error**. The per-feature gaps (real − synth,
in pooled-σ) decompose the gap into three cleanly-separated bands:

**Band 1 — cross-column STRUCTURE (the fundamental gap; per-column independence cannot make it):**

| feature | real | synth | gap |
|---|--:|--:|--:|
| `distinct_pattern_ratio` | 0.044 | **0.422** | −1.39σ (synth ~10× too many patterns) |
| `miss_corr_max_abs` | 0.874 | 0.253 | +2.84σ |
| `miss_corr_mean_abs` | 0.337 | 0.055 | +1.77σ |
| `frac_pairs_coupled` | 0.277 | **0.002** | +1.48σ |
| `monotone_row_frac` | 0.673 | 0.211 | +1.69σ |
| `top1_pattern_frac` | 0.449 | 0.120 | +1.66σ |
| `top3_pattern_frac` | 0.741 | 0.229 | +2.08σ |

**Band 2 — per-column RATE distribution (real, but easily fixable):**

| feature | real | synth | gap |
|---|--:|--:|--:|
| `col_rate_max` | 0.867 | 0.368 | +3.34σ |
| `frac_cols_high_miss` | 0.433 | 0.003 | +3.10σ |
| `col_rate_sd` | 0.332 | 0.125 | +3.06σ |
| `col_rate_mean` | 0.405 | 0.216 | +1.77σ |

**Band 3 — mechanism fingerprints (already the SMALLEST gaps):**

| feature | real | synth | gap |
|---|--:|--:|--:|
| `mar_coupling_max` | 0.806 | 0.555 | +1.04σ |
| `obs_abs_skew_mean` | 3.187 | 1.892 | +0.63σ |
| `obs_excess_kurt_mean` | 32.2 | 19.5 | +0.28σ |

## Interpretation

1. **Our synthetic missingness is "too random," exactly as predicted.** Every Band-1 feature is the
   signature of *per-column independence*: synthetic columns go missing on their own, so the mask has
   ~no co-missingness (`frac_pairs_coupled` 0.002 vs 0.277), ~10× too many distinct patterns
   (`distinct_pattern_ratio` 0.42 vs 0.04), and little monotone/dropout structure (0.21 vs 0.67). Real
   survey missingness is dominated by *modules/blocks/skip-logic* — whole groups of columns missing
   together for a subset of respondents, leaving a few dominant patterns. This is the gap.
2. **The gap is precisely what re-admitting joint mechanisms is for.** The Band-1 features map directly
   onto the joint mechanisms ADR-0007 brought back: module-refusal (block + few-pattern), attrition
   (monotone dropout), latent factors / wave-dropout (co-missingness). Stage A independently confirms
   that the joint-mechanism decision is not optional — it is the *only* way to close Band 1.
3. **The mechanism fingerprints (Band 3) are already close.** Our per-column mechanisms *do* produce
   MAR-coupling and MNAR observed-distortion — those are the smallest gaps. What's missing is the
   cross-column scaffolding to embed them in, not the per-column fingerprints themselves.
4. **The rate gap (Band 2) is real but trivial to fix.** Real columns have higher, more variable,
   sometimes near-total miss rates; our composer pins ~0.25–0.30. The Stage-B by-cell sampler draws a
   target composition and *varies* per-column rates by construction, which closes Band 2 for free.

## Stage B spec (what the rebuilt generator must produce)

To drive AUC toward 0.5 the composition-controlled generator must, in priority order:
1. **Co-missingness / blocks** — groups of columns that go missing together (modules); target
   `miss_corr` and `frac_pairs_coupled` into the real range.
2. **Few dominant patterns** — whole-block refusal for a subset of rows (skip-logic / module battery):
   pull `distinct_pattern_ratio` down (~0.04) and `top1/top3_pattern_frac` up.
3. **Monotone / dropout structure** — attrition / wave dropout: raise `monotone_row_frac`.
4. **Varied per-column rates** — including some near-total columns; raise `col_rate_sd`,
   `frac_cols_high_miss`. (Falls out of the by-cell sampler.)
…while still hitting the target by-cell composition and keeping the per-column mechanism fingerprints
(Band 3) it already produces.

## Caveats

- **Real corpus is survey-weighted (~97% NHANES).** "Realistic" here = realistic for surveys
  (consistent with ADR-0005 specialisation), not general tabular. AUC = 1.0 is so saturated that the
  exact corpus weighting does not change the conclusion.
- **Observable floor, not mechanism proof** (ADR-0007): matching the footprint is necessary, not
  sufficient — it cannot validate the mechanism *mix*, only the surface.
- AUC = 1.0 means the gap is currently un-subtle; once Band 1 is closed we expect to need the full
  discriminator (and possibly an expanded/ re-weighted real corpus) to resolve the residual.

## Files

- Real corpus: `runs/.../composition/stagea_real_footprints.json` (413 footprints).
- Gap report: `runs/.../composition/stagea_gap.json` (AUC, coefficients, per-feature table).
