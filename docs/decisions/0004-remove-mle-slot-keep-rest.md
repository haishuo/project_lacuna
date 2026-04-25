# 0004 — Remove the cached-MLE Little's slot; keep the rest of the missingness-features pipeline

- **Date:** 2026-04-25
- **Status:** Accepted
- **Supersedes:** ADR 0002, ADR 0003

## Context

The post-2026-04-17 ablation arc produced two committee-grade
findings at n=30 paired power:

1. **`n30-mle-vs-disable` (2026-04-23):** the cached-MLE Little's
   slot *hurts* mechanism classification accuracy. Δ = +0.029 in
   favour of `disable_littles`, CI [+0.002, +0.054], wilcoxon
   p=0.036, 21/30 seeds favour disable. ECE also favours disable
   with CI excluding zero. See
   `docs/experiments/2026-04-23-n30-mle-vs-disable.md`.

2. **`full-5spec-canonical-n30` (2026-04-25):** the rest of the
   missingness-features pipeline (`missing_rate_stats` +
   `cross_column_corr`) *contributes* meaningfully. Joint
   contribution is ~6 pp accuracy: `all_disabled` vs
   `disable_littles` Δ = −0.059, CI [−0.095, −0.021], wilcoxon
   p=0.007, d_z=−0.56. See
   `docs/experiments/2026-04-25-canonical-n30.md`.

These findings together complete the picture across six MCAR-test
families (the `mcar-alternatives-bakeoff` 2026-04-21 screened MLE,
MoM, revived median-split SMD, RF/HGB propensity, HSIC kernel,
Jamshidian-Jalal MissMech) plus the two non-MCAR feature groups.
Empirically:

- The **MCAR slot specifically** is non-contributory across every
  family tested at n=10, and the MLE variant is harmful at n=30.
- The **non-MCAR feature groups** (column-wise missing-rate
  statistics; cross-column missingness correlations) contribute
  meaningfully at n=30 — removing them costs ~6 pp.

The MoE gating network can apparently use simple summary statistics
of the missingness *pattern* (variation in per-column missing rates;
between-column correlations of which cells are missing) but does
NOT benefit from a precomputed MCAR-test scalar — every test family
attempted either fails to add signal or actively pollutes the
feature vector.

## Decision

1. **Remove the cached-Little's slot from the default missingness
   feature configuration.** `MissingnessFeatureConfig.include_littles_approx`
   default becomes `False`. The `little_mcar_stat` /
   `little_mcar_pvalue` arguments to
   `extract_missingness_features` remain as kwargs for any caller
   that wants the legacy behaviour, but the default extractor no
   longer requires them. The `n_features` property drops from
   9 to 7 by default (4 missing_rate_stats + 3 cross_column_corr).

2. **Retain `missing_rate_stats` and `cross_column_corr` feature
   groups as defaults.** Both are committee-grade-confirmed
   contributory at n=30. No change to either group's
   implementation.

3. **Keep the schema-v3 cache (`littles_mcar_v3.json`) on disk and
   in the codebase.** It still powers research-mode use (loading
   `baseline`, `baseline_propensity` etc. for new experiments) and
   was expensive to build. The default training and evaluation
   path simply doesn't read from it.

4. **Reorganise `DEFAULT_SPECS` in
   `lacuna/analysis/ablation_harness.py`** so that `disable_littles`
   becomes the de-facto baseline for future ablation experiments,
   and `baseline` (cached-MLE on) is preserved as an explicit named
   spec for reproducing the historical comparisons.

5. **Document the bakeoff/canonical findings in the dissertation
   methods chapter.** Both the H3-confirmed bakeoff and the
   canonical's "non-MCAR features contribute, MCAR slot doesn't"
   decomposition are first-class findings.

## Rationale

The three sweeps that motivated the MLE-vs-disable comparison
(`mle-vs-mom`, `n10-followup`, bakeoff Stage 1) all showed
directional support for "MLE hurts" with CIs that narrowly included
zero at n≤10. `n30-mle-vs-disable` was the pre-registered powered
confirmation; it landed cleanly. ADR 0002 (which upgraded the slot
from heuristic → real Little's MLE) and ADR 0003 (which added the
MoM variant) were both based on evidence that has now been
superseded.

The canonical's `all_disabled` vs `disable_littles` comparison
specifically guards against an over-correction. A naive reading of
the bakeoff's "no MCAR family helps" might have suggested removing
the entire missingness-features pipeline; the canonical shows that
would cost ~6 pp accuracy. The right targeted action is to remove
*just the MCAR slot*, retaining the simpler pattern statistics that
do work.

## Consequences

**Code changes required (separate session, not part of this ADR
draft):**

- `lacuna/data/missingness_features.py::MissingnessFeatureConfig`:
  flip `include_littles_approx` default to `False`. Update
  `n_features` property's documentation. Existing callers that
  pass `include_littles_approx=True` explicitly still work
  unchanged.
- `lacuna/analysis/ablation_harness.py::DEFAULT_SPECS`: re-label
  the previous `disable_littles` spec to be the new baseline
  (preserve historical name as an alias for reproducibility);
  add a clearly-labelled `baseline_legacy_mle` spec for
  reproducing the harmful-MLE comparison.
- `scripts/train.py`, `scripts/evaluate.py`, `scripts/run_ablation.py`:
  no longer require `--littles-cache` for default training (the
  default config no longer reads from it). The argument stays
  accepted for opt-in use of `baseline_legacy_mle` /
  `baseline_propensity` / etc.
- Tests in `tests/unit/data/test_features.py`,
  `tests/unit/analysis/test_ablation_harness.py`: update default-
  shape assertions; add regression that
  `MissingnessFeatureConfig().include_littles_approx is False`.
- `lacuna/data/littles_cache.py`: no functional changes. Cache
  remains useful for research-mode experiments.

**Statistical / scientific consequences:**

- The dissertation's primary claim about Lacuna's missingness-
  features path: **summary statistics of the missingness pattern
  contribute (~6 pp); MCAR-test statistics do not.** This is
  counter to a decade of tabular-missingness practice that has
  treated Little 1988 (and modern variants) as the load-bearing
  detection test.
- The committee comparison table (the "ablation table") is the
  per-spec summary in `2026-04-25-canonical-n30.md`. No further
  ablation runs are needed for the methods chapter.

**What this ADR does NOT do:**

- Does not delete `pystatistics.mvnmle.little_mcar_test` or any
  other public API. Little 1988 remains a textbook test in
  pystatistics; this ADR is about Lacuna's *use* of it as a
  cached feature, not about the test's general validity.
- Does not delete `lacuna.analysis.mcar.*` (MoM, propensity, HSIC,
  MissMech). Those remain in the registry for research-mode
  experiments and for reproducing the bakeoff.
- Does not run the per-feature-group decomposition that would
  separately attribute the +6-pp lift to `missing_rate_stats` vs
  `cross_column_corr`. That's a separable follow-up if the
  dissertation needs it; the headline finding (pipeline
  contributes via the non-MCAR groups, MCAR slot doesn't) is
  established.

## Open questions deferred

- **Why does no MCAR-test family help at n=10?** Six families
  screened in the bakeoff; none cleared the advancement bar. The
  revived median-split SMD heuristic (which contributed at n=5 per
  ADR 0001) also failed at n=10. Possible explanations: the
  2026-04-17 +6 % finding was small-sample noise; the MoE router
  routes around any single MCAR slot; interaction with feature
  groups removed by ADR 0001. Flagged in `PLANNED.md` deferred
  queue for a dedicated post-defense analysis session — not
  blocking the dissertation.
- **Individual contribution of `missing_rate_stats` vs
  `cross_column_corr`.** Joint contribution is committee-grade
  confirmed; per-group split would need two new specs. Optional
  follow-up.
