# 0002 — Replace Little's MCAR heuristic with cached real Little's test

- **Date:** 2026-04-18
- **Status:** Accepted
- **Supersedes:** — (additive, builds on ADR 0001)

## Context

The 2026-04-18 ablation (ADR 0001) showed that the "Little's slot" feature
group — the 2-scalar pair fed to the MoE gating network — was contributory:
paired Δ accuracy = −0.064, 95% CI [−0.169, −0.008], d_z = −0.55. Removing
it cost ~6 points of mean accuracy across 5 seeds.

The statistic occupying the slot, however, was *not* Little's MCAR test.
It was `compute_littles_test_approx` in `lacuna/data/missingness_features.py`:
a median-split standardised-mean-difference heuristic, computed per-batch
as a pure tensor operation. Name aside, it does not resemble Little's test —
Little's test iterates over missingness patterns and fits a chi-squared via
EM, while the heuristic partitions rows by missingness *count* and compares
pooled column means.

The pure-tensor form was a speed compromise: `pystatistics.mvnmle.little_mcar_test`
is an EM-based CPU solver taking seconds per call, untenable per batch.

## Decision

**Replace the heuristic with the real `pystatistics.mvnmle.little_mcar_test`,
computed offline and cached by (dataset, generator) pair.**

Specifically:

- **Cache format:** JSON file keyed on `(dataset_name, generator_id)`, each
  entry storing `{statistic, p_value, df, n_used, rejected, dataset, generator_id,
  generator_name}`. Human-readable and diffable. Schema version tagged
  (`CACHE_SCHEMA_VERSION = 1`) so future format changes fail loud on load.
- **Cache builder:** `scripts/build_littles_cache.py`. Runs
  `little_mcar_test` once per (dataset, generator) pair at `sample_rows=1000`
  (clamped to dataset size). For `ablation.yaml` (31 datasets × 110 generators),
  this is ~3,400 one-time computations — roughly 1–4 hours on CPU, run once.
- **Data flow:** `SemiSyntheticDataLoader` accepts an optional `littles_cache`.
  When present, it validates that every reachable (dataset, generator) pair is
  covered (fail loud on a partial cache) and populates two new `TokenBatch`
  fields: `little_mcar_stat` and `little_mcar_pvalue`, each a `[B]` tensor.
- **Feature extractor:** `MissingnessFeatureExtractor.forward` now accepts
  `little_mcar_stat` / `little_mcar_pvalue` keyword arguments. When
  `config.include_littles_approx=True`, these MUST be supplied; the model
  raises `ValueError` otherwise. The heuristic is removed entirely — there is
  no fallback.

**Deleted:** `compute_littles_test_approx` in `lacuna/data/missingness_features.py`.

**Added:** `lacuna/data/littles_cache.py` (cache I/O + compute primitive +
`build_cache` orchestration), `scripts/build_littles_cache.py` (CLI runner),
`tests/unit/data/test_littles_cache.py` (10 tests covering normal / edge /
failure cases).

## Why cache, not per-batch?

Three options were considered:

1. **Per-batch real Little's test on batch data.** Prohibitive at 800k
   evaluations per ablation sweep; even on GPU (≥100ms each) would add
   22+ hours of pure Little's overhead per run.
2. **Per-batch GPU Little's.** `pystatistics.mvnmle` has a GPU backend for
   `mlest`, but `little_mcar_test` does not expose the `backend` parameter.
   Even with upstream plumbing, the batched EM over patterns doesn't
   vectorise well across independent datasets, and the solver is
   R-reference-tuned for CPU. Speedup would be modest (~3–10×). Still
   untenable for per-batch.
3. **Offline cache.** One-time ~1–4 hours of CPU. Per-batch cost is a
   dictionary lookup. Chosen.

## Why (dataset, generator), not (dataset, generator, batch instance)?

Each batch the semi-synthetic loader draws produces a fresh random
realisation of the generator's mechanism on the dataset's X. The cached
value is computed on *one* realisation at `sample_rows=1000` — which is
more statistically powerful than the per-batch heuristic was on 128 rows.
We lose no useful signal by caching per (dataset, generator) rather than
per-batch; we gain statistical power.

**Memorisation risk:** with ~3,400 unique (dataset, generator) pairs and
~900k parameters in the gate, the model could in principle memorise the
2-scalar→class mapping. The harness's train-val gap diagnostic (ADR not
written; added to `AblationResult` on 2026-04-18) is the instrument to
measure this. If a follow-up sweep shows a widened gap under the cached
feature vs. the heuristic, we'll add jitter (multiple cached realisations
per pair, sampled at training time). Not implemented now because it would
be optimising against a problem we haven't measured.

## Consequences

**Enabled:**
- The Little's-slot feature now contains an actual Little's MCAR result
  rather than a heuristic proxy. Any dissertation claim about "what the
  model learned about MCAR" is now defensible against a statistician.
- The statistic is computed at larger sample size than any individual
  batch, so it's more powerful against weak-signal mechanisms.
- Cache builder is separable from training, so cache construction can be
  parallelised, resumed, or inspected independently.

**Foreclosed:**
- Training now requires a pre-built cache whenever `include_littles_approx=True`
  (the default). `scripts/train.py` and `scripts/evaluate.py` accept
  `--littles-cache`; `scripts/run_ablation.py` accepts it too. Without the
  cache, these fail loud at loader construction. Build once per
  (dataset pool, generator registry) combination.
- Checkpoints from pre-ADR-0002 training are not compatible with post-ADR
  model forward — the feature slot's runtime interface changed (from
  tensor-computed to cache-provided), and the feature distribution
  (Little's chi-squared, range 0–∞) differs from the heuristic's (bounded
  standardised mean difference, clamped to [0, 10]).

**Conditions under which we'd revisit:**
- If a future sweep shows the cached feature hurts accuracy relative to
  the heuristic (unexpected — it's a strictly more informative statistic
  — but possible if the model was exploiting heuristic-specific noise).
- If memorisation measurements (train-val gap) show the cache enables
  shortcut-learning that wasn't possible before. In that case, implement
  the jittered-cache variant.
- If pystatistics ships a vectorised `little_mcar_test` that can run over
  a batch of datasets per call with GPU acceleration at ~ms per evaluation,
  the per-batch path becomes viable and the cache becomes an optimisation
  rather than a necessity.

## Cross-references

- Module: `lacuna/data/littles_cache.py`
- Builder script: `scripts/build_littles_cache.py`
- Cache consumer: `lacuna/data/semisynthetic.py` (`SemiSyntheticDataLoader`)
- Feature extractor: `lacuna/data/missingness_features.py`
  (`MissingnessFeatureExtractor.forward`)
- Tests: `tests/unit/data/test_littles_cache.py` (10 tests)
- Ablation evidence motivating the upgrade: ADR 0001, raw CSV at
  `/mnt/artifacts/project_lacuna/ablation/semisynth_5seed.csv`
