# 0003 — Dual-method cache (MLE + MoM) with model-side selector

- **Date:** 2026-04-18
- **Status:** Accepted (ablation pending)
- **Supersedes:** — (builds on ADR 0002)

## Context

ADR 0002 established the cached-Little's pipeline — one real MCAR test
statistic per (dataset, generator) pair, looked up at training time.
`pystatistics` 2.1.0 shipped two relevant additions:

1. **MoM MCAR test** (`mom_mcar_test`): a method-of-moments variant
   (Hawkins 1981-style moments on pairwise-deletion sample statistics).
   Consistent under MCAR but not asymptotically efficient. **30–50×
   faster** than the MLE-based Little's test at typical tabular data sizes
   (iris 8×, wine 27×, breast_cancer 53×).
2. **Batched EM + SQUAREM** for MLE: ~2× speedup on the MLE path itself.

On the iris × lacuna_minimal_6 smoke build, MLE rejected MCAR on 0 of 12
pairs while MoM rejected on 6 of 12. This is consistent with the
pystatistics documentation's observation that MoM is prone to over-rejecting
compared to MLE, and confirms they are *materially different features* from
the MoE gating network's perspective — not the same statistic at different
precision.

## Decision

**Cache both methods per (dataset, generator) pair. Let the model choose
which one feeds the MoE gate at training time.**

Schema version bumped 1 → 2. Each `LittlesCacheEntry` now carries:

- `mle_statistic`, `mle_p_value`, `mle_df`, `mle_rejected`
- `mom_statistic`, `mom_p_value`, `mom_df`, `mom_rejected`
- `n_used` (same for both — same sample)

`compute_entry` runs both tests on the same filtered array. Since MoM is
essentially free next to MLE, this adds no meaningful wall-clock to cache
construction.

`SemiSyntheticDataLoader` gains `littles_method: str = "mle"`. The loader
calls `cache.get(dataset, gen_id, method=self.littles_method)` and attaches
the selected pair to `TokenBatch.little_mcar_stat` / `little_mcar_pvalue`.
The model forward pass is unchanged — it receives a 2-scalar pair and
doesn't know which test produced it.

`AblationSpec` gains `littles_method: str = "mle"`. `DEFAULT_SPECS` gains
a new `baseline_mom` entry alongside the existing `baseline`. The targeted
ablation to answer "MoM vs MLE?" is:

```
python scripts/run_ablation.py --config configs/training/ablation.yaml \
    --seeds 1 2 3 4 5 \
    --specs baseline baseline_mom disable_littles \
    --littles-cache /path/to/littles_mcar_v2.json \
    --csv mom_vs_mle_ablation.csv
```

3 specs × 5 seeds = 15 runs ≈ 4–6 hours.

v1 caches (MLE only) are rejected on load with a clear "rebuild" error —
fail-loud per Coding Bible rule 1 rather than silently filling MoM fields
with zeros.

## Expected outcomes & decision rules

- **Bootstrap CI on the paired (MLE − MoM) accuracy delta excludes zero**
  with MLE winning by material margin (say |Δ| ≥ 2%): keep MLE as default,
  document MoM as a research curiosity.
- **CI excludes zero, MoM wins**: switch the default to MoM. Faster cache
  builds, same or better downstream accuracy. Write a follow-up ADR.
- **CI straddles zero**: prefer MoM on engineering grounds. Cache builds
  drop from ~30 min to ~3 min at ablation.yaml scale; no measurable
  downstream accuracy cost. A defensible dissertation story ("the MoM
  variant is 30× cheaper with no accuracy penalty") and a clean
  recommendation for anyone deploying Lacuna in batch-diagnostic contexts.

## Consequences

**Enabled:**
- One cache build covers both methods; comparison is paired by construction.
- Method choice is a data-loader flag, not a model-architecture flag — the
  model stays agnostic.
- Future additions (e.g., a pattern-permutation test, a likelihood-ratio
  variant) fit the same shape: add fields to the entry, bump the schema.

**Foreclosed:**
- Caches built before 2026-04-18 (v1) are no longer loadable. Rebuild; the
  new fast path means this is a ~3 min cost, not a day.
- Checkpoints trained under MLE and loaded under a MoM-configured loader
  (or vice versa) will receive different feature values and produce
  different predictions — the cached scalars are part of the model's
  runtime input contract. The checkpoint itself doesn't know which method
  it was trained against; surface this via the run's saved config.

## Open questions for future ADRs

- Does the train-val gap diagnostic (ADR-unnumbered, added 2026-04-18)
  differ between MLE and MoM? MoM's over-rejection might correlate with
  more memorisable signal. The ablation harness logs `generalization_gap`
  per run, so this is measurable in the same sweep that decides MLE vs MoM.

## Cross-references

- Module: `lacuna/data/littles_cache.py` (schema v2, `VALID_METHODS`)
- Loader: `lacuna/data/semisynthetic.py` (`littles_method` kwarg)
- Harness: `lacuna/analysis/ablation_harness.py`
  (`AblationSpec.littles_method`, `DEFAULT_SPECS` +1 entry)
- Builder: `scripts/build_littles_cache.py` (outputs v2 by default)
- Tests: `tests/unit/data/test_littles_cache.py` (12 tests, now covering
  per-method `get`, v1 rejection, dual-method sentinels)
- pystatistics reference: release notes for 2.1.0 (`mom_mcar_test`,
  batched EM, SQUAREM)
