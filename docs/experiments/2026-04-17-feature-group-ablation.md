# 2026-04-17 — Initial feature-group ablation (5 seeds)

- **Status:** Complete
- **Date run:** 2026-04-17 (started ~21:40; finished 2026-04-18 ~10:54, 13h 14m)
- **Commit state:** pre-ADR-0001; feature extractor contained 5 feature groups
  including `pointbiserial` and `distributional`; Little's-slot was the
  median-split standardised-mean-difference heuristic (`compute_littles_test_approx`).

## Goal

Measure which of the five hand-rolled missingness feature groups contribute
to mechanism classification accuracy and calibration, using paired
bootstrap CIs + Wilcoxon on per-seed deltas from a leave-one-group-out
ablation.

## Configuration

- **Config file:** `configs/training/ablation.yaml` (semi-synthetic:
  24 real UCI / OpenML / sklearn training datasets, 7 held-out real val
  datasets, `lacuna_tabular_110` mechanism registry, 100 epochs, batch 16,
  200 batches/epoch, 50 val batches, lr=3e-4)
- **Specs (7):** `baseline`, `disable_missing_rate`,
  `disable_pointbiserial`, `disable_cross_column`, `disable_distributional`,
  `disable_littles`, `all_disabled`
- **Seeds:** 5 (1, 2, 3, 4, 5)
- **Total runs:** 35
- **Mean runtime:** 22.7 min/run

## Results (paired Δ vs. baseline, n=5)

### Accuracy

| Variant | Δ accuracy | 95% bootstrap CI | Wilcoxon p | d_z | Verdict |
|---|---:|---:|---:|---:|---|
| `disable_missing_rate` | −0.084 | [−0.156, −0.027] | 0.125 | −1.05 | Contributory |
| `disable_littles` | −0.064 | [−0.169, −0.008] | 0.058 | −0.55 | Contributory |
| `disable_cross_column` | −0.023 | [−0.061, +0.027] | 0.438 | −0.39 | Borderline |
| `disable_distributional` | −0.007 | [−0.078, +0.046] | 0.625 | −0.09 | **Non-contributory** |
| `disable_pointbiserial` | +0.018 | [−0.031, +0.077] | 1.000 | +0.26 | **Non-contributory** |
| `all_disabled` | −0.099 | [−0.165, −0.030] | 0.125 | −1.13 | (context) |

### ECE (Δ positive = variant worse calibrated)

| Variant | Δ ECE | 95% CI |
|---|---:|---:|
| `disable_missing_rate` | +0.062 | [+0.011, +0.140] |
| `all_disabled` | +0.084 | [+0.031, +0.155] |
| (others: CIs spanned zero) | | |

## Interpretation

Three of five feature groups contribute; two do not:

- **Load-bearing:** `missing_rate_stats` and `littles_approx` — CIs exclude
  zero on accuracy for both, and `missing_rate_stats` also improves
  calibration (CI excludes zero on ECE).
- **Marginal / borderline:** `cross_column_corr` — small negative trend,
  CI spans zero.
- **Non-contributory:** `pointbiserial` and `distributional` — CIs symmetric
  around zero for both accuracy and ECE; effect sizes |d_z| ≤ 0.3.

Statistical caveat: n=5 Wilcoxon has a minimum two-sided achievable p of
0.0625, so no test in this sweep could have reached p < 0.05 by Wilcoxon
alone. **The 95% bootstrap CIs are the decision instrument**, not p-values.

## Decisions taken

- **ADR 0001 (2026-04-18):** Remove `pointbiserial` and `distributional`
  feature groups. Surviving suite: 3 groups / 9 scalars.
- **ADR 0002 (2026-04-18):** Since `littles_approx` was contributory,
  upgrade from the median-split heuristic to the real
  `pystatistics.mvnmle.little_mcar_test`, cached per
  (dataset, generator) pair.

## Raw data

- `/mnt/artifacts/project_lacuna/ablation/semisynth_5seed.csv`

## Follow-ups queued at the time

- Build the Little's MCAR cache (prerequisite for the real-test upgrade).
- Re-ablate with real Little's to verify the ~6% contribution persists
  when the statistic is well-founded rather than heuristic.

*(Second follow-up was run on 2026-04-19; see*
*[`2026-04-19-mle-vs-mom.md`](2026-04-19-mle-vs-mom.md).)*
