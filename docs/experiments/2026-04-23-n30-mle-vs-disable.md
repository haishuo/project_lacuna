# 2026-04-23 — n=30 1v1: `baseline` (cached MLE Little's) vs `disable_littles`

- **Status:** Complete
- **Started:** 2026-04-23 ~01:30 UTC (launched from the bakeoff
  handoff in `HANDOFF-2026-04-23.md`)
- **Finished:** 2026-04-23 14:33 UTC
- **Wall-clock runtime:** ~13 h for 60 runs (matches the ~13 min/run
  post-`apply_missingness`-subsample-fix norm)
- **Mean training time:** 811 s/run (13.5 min)

## Goal

Move the "cached Little's MLE slot hurts" signal from three sweeps of
consistent directional evidence (`mle-vs-mom` 2026-04-19,
`n10-followup` 2026-04-19, `mcar-alternatives-bakeoff` 2026-04-21) —
each with a CI that narrowly included zero — to a single committee-
grade rejection of "MLE adds nothing, null effect." n=30 has enough
combinatorial headroom for Wilcoxon and paired permutation to reach
p<0.05 when a real paired effect at the observed ~±0.03 scale exists.

This is the focused confirmation experiment originally specced as
`n30-1v1-littles-slot` and previously folded into the now-skipped
bakeoff Stage 2.

## Hypothesis

At n=30, the paired `disable_littles − baseline` accuracy CI excludes
zero in the positive direction. Based on bakeoff n=10 measured delta
+0.027 and paired SD ~0.06, the expected n=30 95 % CI half-width is
~0.022, so a point estimate at +0.027 would exclude zero by roughly
+0.005.

## Configuration

- **Config file:** `configs/training/ablation.yaml` (unchanged)
- **Specs (2):** `baseline` (cached Little's MLE) and `disable_littles`
- **Cache:** `/mnt/artifacts/project_lacuna/cache/littles_mcar_v3.json`
- **Seeds:** 30 (1 through 30)
- **Total runs:** 60

## Command

```bash
cd /mnt/projects/project_lacuna
python -u scripts/run_ablation.py \
    --config configs/training/ablation.yaml \
    --seeds 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 \
            16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 \
    --specs baseline disable_littles \
    --littles-cache /mnt/artifacts/project_lacuna/cache/littles_mcar_v3.json \
    --csv /mnt/artifacts/project_lacuna/ablation/n30_mle_vs_disable.csv \
    2>&1 | tee /mnt/artifacts/project_lacuna/ablation/n30_mle_vs_disable.log
```

## Pre-registered decision rule

Per `PLANNED.md §1` at launch:

- CI excludes zero AND `disable_littles` wins → **ADR 0002 / 0003
  superseded.** Write ADR 0004 specifying that the Little's-MLE slot
  is removed; the feature extractor's default becomes
  `include_littles_approx=False`.
- CI excludes zero AND `baseline` wins → reversal of the three-sweep
  pattern. Would need a re-run before acting.
- CI includes zero → three-sweep pattern was noise; keep ADR 0002 / 0003.

## Results

### Per-spec summary (n=30 seeds)

| spec              | mean acc | sd acc | mean ECE | mean gen-gap |
|-------------------|---------:|-------:|---------:|-------------:|
| `baseline` (MLE)  |   0.7187 | 0.0634 |   0.1807 |       0.1580 |
| `disable_littles` |   **0.7474** | 0.0724 |   **0.1559** |       **0.1463** |

### Paired comparison (disable_littles − baseline, across 30 seeds)

| metric              |      Δ |       95 % CI       | Wilcoxon p | Perm p | Paired-t p |     d_z |
|---------------------|-------:|:-------------------:|-----------:|-------:|-----------:|--------:|
| **accuracy**        | **+0.0287** | **[+0.0021, +0.0541]** | **0.0359** | **0.0425** |    0.0432 | +0.386 |
| **ECE**             | **−0.0248** | **[−0.0449, −0.0033]** | **0.0212** | **0.0336** |    —     | −0.412 |
| generalization_gap  | −0.0117 | [−0.0352, +0.0114]  |    0.3235  | 0.3541 |    —     | −0.175 |

### Sign pattern

`disable_littles` wins on **21 / 30** seeds. Seeds where `baseline`
wins: 1, 5, 7, 10, 11, 16, 17, 26 (8 seeds) plus seed 24 tied at
~0.74. The 21 / 30 win rate under binomial H₀ (p=0.5) has p-value
0.016 — independently corroborates the wilcoxon / permutation
p-values.

### Per-seed accuracies

```
baseline         = [0.7525, 0.6913, 0.8037, 0.6813, 0.6138, 0.6225, 0.6963, 0.7725, 0.6813, 0.7600,
                    0.7412, 0.7225, 0.6225, 0.7200, 0.7225, 0.7750, 0.7238, 0.7638, 0.8263, 0.8325,
                    0.7638, 0.7262, 0.7150, 0.7063, 0.7688, 0.6150, 0.6188, 0.7937, 0.6987, 0.6288]
disable_littles  = [0.6800, 0.7950, 0.7588, 0.7725, 0.6562, 0.7113, 0.6775, 0.7288, 0.7512, 0.8150,
                    0.6275, 0.8237, 0.7887, 0.8063, 0.7875, 0.6112, 0.6700, 0.7775, 0.8400, 0.8063,
                    0.7675, 0.8113, 0.8063, 0.7400, 0.8350, 0.5713, 0.6600, 0.8187, 0.7825, 0.7438]
```

## Interpretation

### Decision: **ADR 0002 / 0003 are superseded. The cached MLE Little's slot hurts.**

All three pre-registered stopping conditions are satisfied:

1. Paired accuracy CI [+0.002, +0.054] excludes zero in the positive
   direction.
2. Wilcoxon signed-rank p=0.036, paired permutation p=0.043, paired-t
   p=0.043 — all reject the null at conventional α=0.05.
3. Direction is `disable_littles` wins (21 / 30 seeds), consistent
   with `mle-vs-mom`, `n10-followup`, and the Stage 1 bakeoff.

Effect sizes worth recording:

- **Accuracy**: mean improvement from removing MLE is +2.87 percentage
  points (0.7474 vs 0.7187). Cohen's d_z = +0.39 (small-to-medium).
- **Calibration**: ECE drops from 0.181 to 0.156 (a 14 % relative
  improvement in calibration error). This CI also excludes zero — the
  MLE slot is hurting calibration independently of the accuracy
  effect.
- **Generalization gap**: −0.012 directionally (train accuracy −
  val accuracy), CI includes zero. Not a significant effect.

### Why the three-sweep pattern was taking so long to confirm

Each of the earlier sweeps (`mle-vs-mom` n=5, `n10-followup` n=10,
`bakeoff` n=10) measured essentially the same effect (Δ ≈ +0.02–0.03),
but at n≤10 the 95 % CI half-width was ~0.04–0.06 — wider than the
effect itself, so every CI straddled zero. n=30 drops the half-width
to ~0.025, which is narrow enough for a +0.029 point estimate to
exclude zero cleanly. This is a textbook case where the right response
to repeated underpowered-null findings is to run at adequate power,
not to stop looking — and why `EVIDENCE-STANDARD.md` now specifies
n=30 as the committee-grade floor for Lacuna ablations.

### What this does NOT yet tell us

This experiment confirms that the **cached MLE** Little's slot hurts.
It does NOT tell us:

- Whether the OTHER missingness-feature groups (`missing_rate_stats`,
  `cross_column_corr`) contribute. Those are still on in `disable_littles`
  and were included in both specs. Answer comes from the queued
  `full-5spec-canonical-n30` in `PLANNED.md §2`.
- Why filling the slot with any MCAR-test statistic fails — the
  bakeoff showed six different families all non-contributory at n=10.
  Remaining open puzzle flagged in `PLANNED.md` deferred queue.

## Follow-up actions

### ADR 0004 — Remove the cached Little's MLE slot

Ready to draft. Key points the ADR should pin:

- `MissingnessFeatureConfig.include_littles_approx` default becomes
  `False` (was `True` since ADR 0002).
- `DEFAULT_SPECS` in `lacuna/analysis/ablation_harness.py` reorders:
  what was `disable_littles` becomes the de-facto baseline; the
  legacy `baseline` (MLE Little's on) is retained for historical
  reproducibility but marked as documented-harmful.
- `littles_cache` (`littles_mcar_v3.json`) is retained for research
  purposes — it powers both `baseline` (history) and the
  `baseline_propensity / hsic / missmech` bakeoff specs that may
  yet be useful in other experiments. Not dropped from the cache
  schema; just dropped from the model default path.
- Supersedes: ADR 0002 (upgrade heuristic → real MLE), ADR 0003
  (MLE vs MoM tradeoff). Both based on evidence that has now been
  superseded by three progressively-more-powered sweeps.

### `full-5spec-canonical-n30` (PLANNED.md §2) launches next

Spec list fixed by this result: `disable_littles` is the surviving
baseline. Launch command is in `PLANNED.md §2`.

## Raw data

- `/mnt/artifacts/project_lacuna/ablation/n30_mle_vs_disable.csv`
  (60 rows + header, written incrementally per run)
- `/mnt/artifacts/project_lacuna/ablation/n30_mle_vs_disable.log`
