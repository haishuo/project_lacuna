# 0001 — Remove `pointbiserial` and `distributional` missingness feature groups

- **Date:** 2026-04-18
- **Status:** Accepted
- **Supersedes:** — (first ADR)

## Context

`lacuna/data/missingness_features.py` originally computed five groups of
hand-rolled statistical features, fed into the MoE gating network alongside
the learned evidence vector and reconstruction errors. The five groups were:

| Group | Features | Intent |
|---|---:|---|
| `missing_rate_stats` | 4 | Mean / variance / range / max of per-column missing rates |
| `pointbiserial` | 3 | Correlation (mean / max / std) between missingness indicators and row-mean of observed values |
| `cross_column_corr` | 3 | Correlation between missingness indicators across columns (mean / max / fraction-high) |
| `distributional` | 4 | Skewness and kurtosis (mean / variance) of observed values per column |
| `littles_approx` | 2 | Median-split standardised mean difference — a heuristic occupying the "Little's test slot" |

Total: 16 scalar features. Groups were selected a priori from the missing-data
literature: missing-rate structure for MCAR vs. MAR, point-biserial for MAR
detection, cross-column correlations for shared-predictor mechanisms,
distributional distortions (skew/kurt) for MNAR self-censoring, and Little's
test as the canonical non-MCAR signal.

## Ablation evidence

On 2026-04-18 we ran a paired 5-seed ablation on the full semi-synthetic
configuration (`configs/training/ablation.yaml`: 24 tabular training datasets,
7 held-out validation datasets, 110 generators from `lacuna_tabular_110`, 100
epochs, batch size 16, 200 batches/epoch). For each of the five feature groups
we disabled exactly that group and left the other four enabled. A `baseline`
spec (all groups enabled) and an `all_disabled` spec (no missingness features)
framed the comparison.

**Headline accuracy deltas (variant − baseline, paired by seed, n=5):**

| Variant | Δ accuracy | 95% bootstrap CI | Wilcoxon p | Cohen d_z | Verdict |
|---|---:|---:|---:|---:|:---|
| `disable_missing_rate` | −0.084 | [−0.156, −0.027] | 0.125 | −1.05 | Contributory |
| `disable_littles` | −0.064 | [−0.169, −0.008] | 0.058 | −0.55 | Contributory |
| `disable_cross_column` | −0.023 | [−0.061, +0.027] | 0.438 | −0.39 | Borderline |
| **`disable_distributional`** | **−0.007** | **[−0.078, +0.046]** | **0.625** | **−0.09** | **Non-contributory** |
| **`disable_pointbiserial`** | **+0.018** | **[−0.031, +0.077]** | **1.000** | **+0.26** | **Non-contributory** |
| `all_disabled` | −0.099 | [−0.165, −0.030] | 0.125 | −1.13 | (context) |

**ECE deltas** (positive = variant worse-calibrated):

| Variant | Δ ECE | 95% CI |
|---|---:|---:|
| `disable_distributional` | +0.005 | [−0.043, +0.077] |
| `disable_pointbiserial` | −0.011 | [−0.067, +0.036] |

Both groups have 95% bootstrap CIs that straddle zero for both accuracy and
ECE. Point estimates are small (|Δ| ≤ 2%) and effect sizes are negligible
(|d_z| ≤ 0.3). Removing either group had no measurable effect on either
headline metric.

Statistical caveat worth noting on the record: n=5 Wilcoxon has a minimum
two-sided achievable p of 0.0625, so no test in this sweep could have reached
p < 0.05 by Wilcoxon alone. **The 95% bootstrap CIs are the decision
instrument, not p-values.** For the two groups we're removing, the CIs are
symmetric around zero.

Raw data: `/mnt/artifacts/project_lacuna/ablation/semisynth_5seed.csv`.

## Decision

Remove `pointbiserial` and `distributional` feature groups from the
missingness feature extractor. Specifically:

- Delete `compute_pointbiserial_correlations()` and
  `compute_distributional_stats()` from `lacuna/data/missingness_features.py`.
- Remove the `include_pointbiserial` and `include_distributional` flags from
  `MissingnessFeatureConfig`.
- Remove the `disable_pointbiserial` and `disable_distributional` entries
  from `DEFAULT_SPECS` in `lacuna/analysis/ablation_harness.py`.
- Update the `n_features` property and `get_feature_names()` accordingly.

The surviving feature suite is 3 groups totaling 9 scalars:
`missing_rate_stats` (4), `cross_column_corr` (3), `littles_approx` (2).

## Consequences

**Enabled:**
- Smaller gating-network input (16 → 9 scalars), which reduces the number of
  parameters in the MoE gate by roughly the gate's hidden-dim × 7.
- Less training compute (the two removed functions ran on every batch).
- Cleaner dissertation narrative: the retained feature groups each have a
  defensible contribution (`missing_rate_stats` and `littles_approx` have
  bootstrap CIs that exclude zero; `cross_column_corr` is borderline but
  consistent in direction).

**Foreclosed:**
- The removed groups cannot be re-enabled via the YAML path any more. If a
  future reviewer insists on reinstating them, the functions and flags have
  to be re-added from git history (commit where this ADR was accepted).
- Any existing trained checkpoints built with the full 16-feature gate are
  **not loadable** by the post-removal model — the gate input dimension
  changed. Old checkpoints become reference artifacts; fresh training is
  required.

**Conditions under which we'd revisit:**
- If a future ablation at higher n (≥ 10 seeds) and higher generator
  diversity (beyond `lacuna_tabular_110`) shows either group recovering a
  contribution, this ADR should be superseded by a new record that
  re-enables them.
- If the model architecture changes in a way that changes which features
  are redundant (e.g. the evidence vector stops capturing distributional
  information that made `distributional` redundant), re-enable.

## For statisticians using this tool

One of the points of removing these groups *with a published record* is to
make the tool's internals defensible to a statistical reader. A future user
looking at `missingness_features.py` will see three hand-rolled groups and
reasonably ask "why not also skewness / kurtosis?" The answer lives here:
*we tested it, it did not contribute measurably, and we removed it rather
than leave dead features in place to suggest otherwise*. That honesty is
load-bearing for the tool's credibility.

## Cross-references

- Raw ablation CSV: `/mnt/artifacts/project_lacuna/ablation/semisynth_5seed.csv`
- Ablation config: `configs/training/ablation.yaml`
- Ablation harness: `lacuna/analysis/ablation_harness.py`
- Feature extractor: `lacuna/data/missingness_features.py`
- Analysis primitives used: `lacuna/analysis/ablation_stats.py`
  (`paired_comparison`, `bootstrap_delta_ci`, `paired_permutation`)
