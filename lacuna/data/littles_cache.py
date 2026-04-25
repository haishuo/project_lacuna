"""
lacuna.data.littles_cache

Offline cache of Little's MCAR test statistics, keyed by (dataset, generator).

Why offline: `pystatistics.mvnmle.little_mcar_test` runs an EM-based MLE
and chi-squared computation that is too slow to call per training batch
(milliseconds to seconds per call; forward-pass-breaking at batch scale).
Caching once per (raw_dataset, generator_id) pair makes the per-batch cost
a dictionary lookup.

Each cache entry holds FIVE test results computed on the same (sampled,
masked) array, so the mcar-alternatives-bakeoff (see
docs/experiments/PLANNED.md §3) can compare every spec against the same
data draw without multiple cache builds:

    mle        — Little (1988) MLE plug-in (the reference path)
    mom        — method-of-moments pairwise-deletion plug-in (ADR 0003)
    propensity — RF/GBM propensity-AUC (lacuna.analysis.mcar)
    hsic       — Hilbert-Schmidt independence criterion (Gretton 2005/8)
    missmech   — Jamshidian-Jalal-style k-NN + permutation

The model reads whichever method the data loader is configured to emit.
See ADR 0003 (MoM addition) and the 2026-04-20 schema-v3 extension notes
in `.release/UNRELEASED.md` of lacuna.

See `docs/decisions/0001-remove-pointbiserial-distributional.md` for the
original ablation evidence and `docs/decisions/0002-real-littles-mcar-cached.md`
for the decision to move from heuristic → cached real Little's.

Contract
--------
Cache is a JSON file (human-readable, diffable) with schema:

    {
      "version": 3,
      "generator_registry": "lacuna_tabular_110",
      "sample_rows_per_evaluation": 1000,
      "seed_base": 20260418,
      "entries": [
        {"dataset": "iris", "generator_id": 0,
         "generator_name": "MCAR-Bernoulli-10",
         "n_used": 150,
         "mle_statistic": 12.34, "mle_p_value": 0.21,
         "mle_df": 8,  "mle_rejected": false,
         "mom_statistic": 14.02, "mom_p_value": 0.14,
         "mom_df": 8,  "mom_rejected": false,
         "propensity_statistic": 0.03, "propensity_p_value": 0.42,
         "hsic_statistic": 0.001, "hsic_p_value": 0.31,
         "missmech_statistic": 0.22, "missmech_p_value": 0.18}
      ]
    }

Schema v1 (MLE only) and v2 (MLE+MoM only) caches are REJECTED on load —
rebuild with `scripts/build_littles_cache.py`.

Lookup key is `(dataset_name, generator_id)`. Missing keys raise KeyError —
fail-loud per Coding Bible rule 1; silently substituting a default would let
a stale cache corrupt training.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from pystatistics.core.exceptions import PyStatisticsError
from pystatistics.mvnmle import little_mcar_test

from lacuna.analysis.mcar import (
    hsic_mcar_test,
    missmech_mcar_test,
    mom_mcar_test,
    propensity_mcar_test,
)
from lacuna.core.rng import RNGState
from lacuna.data.ingestion import RawDataset
from lacuna.generators.base import Generator


CACHE_SCHEMA_VERSION = 3

# Valid method selectors for `LittlesCache.get(...)`.
VALID_METHODS = ("mle", "mom", "propensity", "hsic", "missmech")


# =============================================================================
# Result container
# =============================================================================


@dataclass(frozen=True)
class LittlesCacheEntry:
    """One (dataset, generator) pair's cached MCAR-test results.

    Both MLE (Little 1988) and MoM (method-of-moments) statistics are
    computed on the same sample, so downstream comparisons are paired by
    construction.
    """
    dataset: str
    generator_id: int
    generator_name: str
    n_used: int
    # MLE — asymptotically efficient, slower, the reference.
    mle_statistic: float
    mle_p_value: float
    mle_df: int
    mle_rejected: bool
    # MoM — consistent under MCAR but not efficient; ~30-50× faster.
    mom_statistic: float
    mom_p_value: float
    mom_df: int
    mom_rejected: bool
    # Propensity (RF/GBM OOF-AUC, permutation p) — schema v3, 2026-04-20.
    # Distribution-free; uses stochastic imputation under the hood. df is
    # not meaningful for this test and is not stored.
    propensity_statistic: float
    propensity_p_value: float
    # HSIC (Gaussian RBF kernel, median-heuristic bandwidth, permutation p).
    hsic_statistic: float
    hsic_p_value: float
    # MissMech (Jamshidian-Jalal-style k-NN impute + permutation).
    missmech_statistic: float
    missmech_p_value: float


@dataclass(frozen=True)
class LittlesCache:
    """In-memory representation of the Little's MCAR cache.

    Access via `cache.get(dataset, generator_id) -> (statistic, p_value)`.
    """
    version: int
    generator_registry: str
    sample_rows_per_evaluation: int
    seed_base: int
    entries: Dict[Tuple[str, int], LittlesCacheEntry] = field(default_factory=dict)

    def get(
        self,
        dataset: str,
        generator_id: int,
        *,
        method: str = "mle",
    ) -> Tuple[float, float]:
        """Look up cached (statistic, p_value) for a (dataset, generator) pair.

        Args:
            dataset: Dataset name.
            generator_id: Generator ID.
            method: "mle" or "mom". Selects which cached scalar pair to
                return. Defaults to "mle" (the reference path).

        Raises:
            KeyError: If the pair is not in the cache.
            ValueError: If method is not in VALID_METHODS.
        """
        if method not in VALID_METHODS:
            raise ValueError(
                f"method must be one of {VALID_METHODS}; got {method!r}"
            )
        key = (dataset, generator_id)
        if key not in self.entries:
            raise KeyError(
                f"No cached MCAR result for dataset={dataset!r}, "
                f"generator_id={generator_id}. Rebuild the cache."
            )
        entry = self.entries[key]
        if method == "mle":
            return entry.mle_statistic, entry.mle_p_value
        if method == "mom":
            return entry.mom_statistic, entry.mom_p_value
        if method == "propensity":
            return entry.propensity_statistic, entry.propensity_p_value
        if method == "hsic":
            return entry.hsic_statistic, entry.hsic_p_value
        # method == "missmech"
        return entry.missmech_statistic, entry.missmech_p_value

    def __contains__(self, key: Tuple[str, int]) -> bool:
        return key in self.entries


# =============================================================================
# Cache I/O
# =============================================================================


def save_cache(cache: LittlesCache, path: Path) -> None:
    """Write the cache to disk as JSON. Overwrites existing file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": cache.version,
        "generator_registry": cache.generator_registry,
        "sample_rows_per_evaluation": cache.sample_rows_per_evaluation,
        "seed_base": cache.seed_base,
        "entries": [asdict(e) for e in cache.entries.values()],
    }
    path.write_text(json.dumps(payload, indent=2))


def load_cache(path: Path) -> LittlesCache:
    """Read a cache from disk.

    Raises:
        FileNotFoundError: If the cache file does not exist.
        ValueError: On version mismatch or schema corruption.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"MCAR cache not found: {path}")
    payload = json.loads(path.read_text())

    version = payload.get("version")
    if version != CACHE_SCHEMA_VERSION:
        if version == 1:
            raise ValueError(
                f"Cache {path} is schema v1 (MLE only); this Lacuna build "
                f"requires v{CACHE_SCHEMA_VERSION} (MLE + MoM + propensity "
                f"+ HSIC + MissMech). Rebuild with scripts/build_littles_cache.py."
            )
        if version == 2:
            raise ValueError(
                f"Cache {path} is schema v2 (MLE + MoM only); this Lacuna "
                f"build requires v{CACHE_SCHEMA_VERSION}, which adds "
                f"propensity / HSIC / MissMech entries for the "
                f"mcar-alternatives-bakeoff. Rebuild with "
                f"scripts/build_littles_cache.py."
            )
        raise ValueError(
            f"Cache {path} is version {version}; expected "
            f"{CACHE_SCHEMA_VERSION}. Rebuild the cache."
        )

    entries: Dict[Tuple[str, int], LittlesCacheEntry] = {}
    for raw in payload["entries"]:
        entry = LittlesCacheEntry(**raw)
        entries[(entry.dataset, entry.generator_id)] = entry

    return LittlesCache(
        version=version,
        generator_registry=payload["generator_registry"],
        sample_rows_per_evaluation=payload["sample_rows_per_evaluation"],
        seed_base=payload["seed_base"],
        entries=entries,
    )


# =============================================================================
# Compute primitive
# =============================================================================


def compute_entry(
    raw: RawDataset,
    generator: Generator,
    *,
    rng: RNGState,
    sample_rows: int,
    backend: str = "auto",
    # Cache-build tuning for the nonparametric methods (schema v3).
    # Propensity uses its analytical (Mann-Whitney-U) null by default —
    # zero refit cost per pair, so no permutation knob is exposed for
    # propensity here. HSIC and MissMech keep their permutation counts
    # because their analytical nulls are weaker (or not yet implemented).
    hsic_permutations: int = 199,
    missmech_permutations: int = 199,
    propensity_cv_folds: int = 3,
    propensity_model: str = "hgb",
    propensity_n_estimators: int = 50,
    propensity_n_jobs: int = -1,
    missmech_n_neighbors: int = 5,
    missmech_min_pattern_size: int = 6,
    nonparametric_seed: int = 0,
) -> LittlesCacheEntry:
    """Compute Little's MCAR test for one (dataset, generator) pair.

    Procedure:
        1. Subsample `sample_rows` rows from `raw` (if larger than raw.n,
           uses all available rows).
        2. Apply the generator to produce a missingness mask at (sampled_n,
           raw.d).
        3. Build an (sampled_n, d) numpy array with NaN for missing entries.
        4. Run `pystatistics.mvnmle.little_mcar_test` on the result.

    Args:
        raw: Real tabular dataset with complete X.
        generator: Missingness generator to apply.
        rng: RNG for reproducibility (subsampling + generator seed).
        sample_rows: Target sample size for Little's test. Clamped to raw.n.
        backend: pystatistics compute backend. "auto" uses GPU when available
            (CUDA or MPS), falling back to CPU. "cpu" and "gpu" force.
            GPU uses FP32 by default so stat/p-value may differ from CPU by
            ~1e-4; for cache construction this is well within the noise of
            subsample and generator randomness.

    Returns:
        LittlesCacheEntry with the test result.
    """
    n_total = raw.data.shape[0]
    n_use = min(sample_rows, n_total)

    # Sentinel helper: any "this (dataset, generator) pair cannot produce
    # a valid MCAR test result" goes through here. Covers: degenerate
    # masks, too-few-usable-rows, both tests failing numerically, AND
    # generator-vs-dataset shape incompatibility (e.g. MARManyPredictor
    # requiring d >= 5 when the dataset has d=4).
    def _sentinel(n_used: int) -> LittlesCacheEntry:
        return LittlesCacheEntry(
            dataset=raw.name,
            generator_id=generator.generator_id,
            generator_name=generator.name,
            n_used=n_used,
            mle_statistic=0.0, mle_p_value=1.0, mle_df=0, mle_rejected=False,
            mom_statistic=0.0, mom_p_value=1.0, mom_df=0, mom_rejected=False,
            propensity_statistic=0.0, propensity_p_value=1.0,
            hsic_statistic=0.0, hsic_p_value=1.0,
            missmech_statistic=0.0, missmech_p_value=1.0,
        )

    # Deterministic subsample.
    if n_use < n_total:
        idx = rng.choice(n_total, size=n_use, replace=False)
        x_sub = raw.data[idx]
    else:
        x_sub = raw.data

    x_tensor = torch.as_tensor(x_sub, dtype=torch.float32)
    # Generators sample on (n, d); we use the subsample's (n, d) directly.
    # Some generators validate (n, d) against their own parameter schema
    # and raise ValueError on incompatible shapes (e.g. MARManyPredictor
    # with p predictors requires d >= p+1). Incompatibility is a property
    # of the (dataset, generator) pair, not a bug — emit sentinel.
    try:
        _, r = generator.sample(rng, n_use, raw.d)
    except Exception as e:
        warnings.warn(
            f"Generator {generator.name} incompatible with dataset "
            f"{raw.name} (d={raw.d}, n={n_use}): {e.__class__.__name__}: "
            f"{e}. Emitting sentinel entry (stat=0, p=1) for both methods."
        )
        return _sentinel(n_use)

    if r.shape != (n_use, raw.d):
        raise RuntimeError(
            f"Generator {generator.name} returned mask shape {tuple(r.shape)}, "
            f"expected ({n_use}, {raw.d})"
        )

    # Assemble (n, d) numpy with NaN for missing.
    arr = x_tensor.detach().cpu().numpy().astype(np.float64, copy=True)
    mask = (~r).detach().cpu().numpy()
    arr[mask] = np.nan

    # Guard against fully-observed and fully-missing cases — both tests
    # are undefined. Emit a sentinel (stat=0, p=1) so downstream feature
    # consumers see a "no evidence against MCAR" signal rather than NaN.
    n_missing = int(np.isnan(arr).sum())
    if n_missing == 0 or n_missing == arr.size:
        warnings.warn(
            f"Degenerate mask from generator={generator.name} on "
            f"dataset={raw.name}: {n_missing}/{arr.size} missing; "
            f"returning sentinel (stat=0, p=1)."
        )
        return _sentinel(n_use)

    # pystatistics 2.1.0 handles all-missing rows internally
    # (drop_all_missing_rows=True), but we still need to know n_eff for
    # the entry's bookkeeping.
    row_has_any_obs = ~np.isnan(arr).all(axis=1)
    n_eff = int(row_has_any_obs.sum())
    if n_eff <= arr.shape[1]:
        warnings.warn(
            f"Too few usable rows from generator={generator.name} on "
            f"dataset={raw.name} (n_eff={n_eff} <= d={arr.shape[1]}); "
            f"returning sentinel (stat=0, p=1)."
        )
        return _sentinel(n_eff)

    # Short-circuit MLE when the missingness mask has so many unique
    # patterns that EM cannot converge in practice. The 2026-04-20 v3
    # build log showed `little_mcar_test` burning 1002 EM iterations
    # before raising on any pair where n_patterns ≈ n_rows (e.g.
    # `credit_card_default × MCAR-Cauchy` with 1000 patterns in 1000
    # rows). Each sentinel-bound MLE call costs ~100s. Detect the
    # condition up front via a cheap row-hash and emit the per-method
    # sentinel immediately — the other four methods still run.
    pattern_threshold = 0.8
    unique_patterns = len(set(tuple(row) for row in mask))
    mle_will_fail = unique_patterns >= pattern_threshold * n_eff
    if mle_will_fail:
        warnings.warn(
            f"Skipping little_mcar_test for generator={generator.name} on "
            f"dataset={raw.name}: {unique_patterns} unique missingness "
            f"patterns in {n_eff} rows (>= {pattern_threshold:.0%} "
            f"threshold); EM would not converge. Using MLE sentinel "
            f"(stat=0, p=1)."
        )

    # Run both MVN-based tests on the same array. MoM cost is negligible
    # vs MLE, so computing both gives a paired comparison at cache-build
    # time. If either raises, record a sentinel for THAT method only —
    # the other may still succeed.
    #
    # At this layer the semantic is "couldn't compute this method on this
    # (dataset, generator) pair — use sentinel and keep going"; any
    # exception from the MCAR-test entry point means that. Catching
    # Exception is appropriate because we want the build to proceed on
    # every pair regardless of which failure mode pystatistics surfaces
    # (PyStatisticsError for numerical issues, RuntimeError for EM
    # non-convergence, ValueError for fully-missing columns in MoM's
    # pairwise-deletion moments, etc.). KeyboardInterrupt /
    # BaseException are correctly NOT caught.
    def _run(fn) -> Optional[Tuple[float, float, int, bool]]:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = fn(arr, backend=backend)
            return (
                float(res.statistic),
                float(res.p_value),
                int(res.df),
                bool(res.rejected),
            )
        except Exception as e:
            warnings.warn(
                f"{fn.__name__} failed for generator={generator.name} on "
                f"dataset={raw.name}: {e.__class__.__name__}: {e}. "
                f"Using per-method sentinel (stat=0, p=1)."
            )
            return None

    mle = None if mle_will_fail else _run(little_mcar_test)
    mom = _run(mom_mcar_test)

    mle_stat, mle_p, mle_df, mle_rej = mle if mle is not None else (0.0, 1.0, 0, False)
    mom_stat, mom_p, mom_df, mom_rej = mom if mom is not None else (0.0, 1.0, 0, False)

    # Schema v3: three distribution-free tests. Each takes (stat, p_value)
    # from a NonparametricMCARResult — no df / rejected fields since those
    # aren't meaningful for permutation-based tests used as feature scalars.
    # Keep the same swallow-and-sentinel discipline as the MVN-based tests:
    # pystatistics may raise ValueError for degenerate inputs that slipped
    # past the earlier guards (e.g. missmech needing >=2 patterns at
    # min_pattern_size); we record (stat=0, p=1) for that method and move on.
    def _run_nonparam(name: str, fn, **kwargs) -> Tuple[float, float]:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = fn(arr, **kwargs)
            return float(res.statistic), float(res.p_value)
        except Exception as e:
            warnings.warn(
                f"{name} failed for generator={generator.name} on "
                f"dataset={raw.name}: {e.__class__.__name__}: {e}. "
                f"Using sentinel (stat=0, p=1) for {name}."
            )
            return 0.0, 1.0

    propensity_stat, propensity_p = _run_nonparam(
        "propensity_mcar_test", propensity_mcar_test,
        model=propensity_model,
        cv_folds=propensity_cv_folds,
        null="analytical",
        n_estimators=propensity_n_estimators,
        n_jobs=propensity_n_jobs,
        seed=nonparametric_seed,
    )
    hsic_stat, hsic_p = _run_nonparam(
        "hsic_mcar_test", hsic_mcar_test,
        n_permutations=hsic_permutations,
        seed=nonparametric_seed,
    )
    missmech_stat, missmech_p = _run_nonparam(
        "missmech_mcar_test", missmech_mcar_test,
        n_permutations=missmech_permutations,
        n_neighbors=missmech_n_neighbors,
        min_pattern_size=missmech_min_pattern_size,
        seed=nonparametric_seed,
    )

    return LittlesCacheEntry(
        dataset=raw.name,
        generator_id=generator.generator_id,
        generator_name=generator.name,
        n_used=n_eff,
        mle_statistic=mle_stat, mle_p_value=mle_p, mle_df=mle_df, mle_rejected=mle_rej,
        mom_statistic=mom_stat, mom_p_value=mom_p, mom_df=mom_df, mom_rejected=mom_rej,
        propensity_statistic=propensity_stat, propensity_p_value=propensity_p,
        hsic_statistic=hsic_stat, hsic_p_value=hsic_p,
        missmech_statistic=missmech_stat, missmech_p_value=missmech_p,
    )


# =============================================================================
# Cache builder
# =============================================================================


def _pair_seed(seed_base: int, ds_idx: int, generator_id: int) -> int:
    """Deterministic per-pair seed derivation — must match the sequential
    and parallel paths bit-for-bit."""
    return seed_base + ds_idx * 1_000_003 + generator_id


def _compute_pair_task(args):
    """Process-pool worker. Reconstructs the RNG and defers to
    ``compute_entry``. Kept top-level + argument-based so the whole call
    graph is picklable by multiprocessing.

    ``args`` is a tuple so we can pass it to ``ProcessPoolExecutor.map``
    without building a long keyword payload per pair.
    """
    (raw, gen, pair_seed, sample_rows, backend, kwargs) = args
    rng = RNGState(seed=pair_seed)
    return compute_entry(
        raw, gen, rng=rng, sample_rows=sample_rows, backend=backend, **kwargs,
    )


def _init_worker():
    """Runtime safety net for per-worker thread limiting.

    The *primary* mechanism is the parent process setting
    ``OMP_NUM_THREADS`` etc. *before* any import, which ``spawn``-ed
    children inherit via their environment — see the env-var block at
    the top of ``scripts/build_littles_cache.py``. Children resolve
    the initializer reference by importing this module, which
    transitively imports numpy / torch / sklearn / scipy at module
    level; by that point BLAS pools are already created, so setting
    env vars *inside* the initializer is too late.

    However, some libraries (notably sklearn via its Cython-compiled
    extensions) expose a runtime thread-pool control via
    ``threadpoolctl``. Calling ``threadpool_limits(1)`` here pins the
    live pools to 1 thread each, closing any gap left by env vars.
    Always safe to call; no-ops if threadpoolctl isn't installed.
    """
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except ImportError:
        pass


def build_cache(
    raw_datasets: List[RawDataset],
    generators: List[Generator],
    *,
    generator_registry_name: str,
    sample_rows: int = 1000,
    seed_base: int = 20260418,
    backend: str = "auto",
    jobs: int = 1,
    on_entry: Optional[callable] = None,
    # Schema-v3 nonparametric tuning — forwarded verbatim to compute_entry.
    hsic_permutations: int = 199,
    missmech_permutations: int = 199,
    propensity_cv_folds: int = 3,
    propensity_model: str = "hgb",
    propensity_n_estimators: int = 50,
    propensity_n_jobs: int = -1,
    missmech_n_neighbors: int = 5,
    missmech_min_pattern_size: int = 6,
    nonparametric_seed: int = 0,
) -> LittlesCache:
    """Compute every (dataset, generator) pair's MCAR-test result.

    Args:
        raw_datasets: Real tabular datasets to cache against.
        generators: Generator instances to apply.
        generator_registry_name: Label stored in cache metadata for audit.
        sample_rows: Rows to subsample per evaluation.
        seed_base: Base seed for deterministic subsampling + generator
            application. Each (dataset, generator) pair derives its own
            seed via ``_pair_seed``.
        backend: pystatistics backend for Little's MLE / MoM. ``"auto"``
            picks GPU when available. Works fine in parallel mode — the
            'spawn' start method gives each worker its own CUDA context;
            concurrent GPU submissions from multiple contexts are
            serialised by the GPU scheduler at the ~ms level, which is
            far below the per-pair cost. One-time cost: each worker
            spends ~3-5 s initialising CUDA, but that amortises over
            ~3000 pairs.
        jobs: Number of worker processes for the (dataset, generator)
            sweep. ``1`` runs in-process (legacy behaviour, easier to
            debug). ``N > 1`` uses ``ProcessPoolExecutor`` with
            ``N`` workers.
        on_entry: Optional callable invoked on the main process after
            each entry is computed — for live progress logging. In
            parallel mode this runs once per `future.result()`.

    Returns:
        A fully populated LittlesCache.
    """
    tasks: List[tuple] = []
    for ds_idx, raw in enumerate(raw_datasets):
        for gen in generators:
            pair_seed = _pair_seed(seed_base, ds_idx, gen.generator_id)
            kwargs = dict(
                hsic_permutations=hsic_permutations,
                missmech_permutations=missmech_permutations,
                propensity_cv_folds=propensity_cv_folds,
                propensity_model=propensity_model,
                propensity_n_estimators=propensity_n_estimators,
                propensity_n_jobs=propensity_n_jobs,
                missmech_n_neighbors=missmech_n_neighbors,
                missmech_min_pattern_size=missmech_min_pattern_size,
                nonparametric_seed=nonparametric_seed,
            )
            tasks.append((raw, gen, pair_seed, sample_rows, backend, kwargs))

    entries: Dict[Tuple[str, int], LittlesCacheEntry] = {}

    if jobs <= 1:
        # In-process path: unchanged semantics, just wrapped in the new
        # task-tuple contract so both paths are testable via the same
        # fixture.
        for task in tasks:
            entry = _compute_pair_task(task)
            entries[(entry.dataset, entry.generator_id)] = entry
            if on_entry is not None:
                on_entry(entry)
        return LittlesCache(
            version=CACHE_SCHEMA_VERSION,
            generator_registry=generator_registry_name,
            sample_rows_per_evaluation=sample_rows,
            seed_base=seed_base,
            entries=entries,
        )

    # Parallel path. Use the 'spawn' start method so each worker gets
    # a fresh Python interpreter — `fork` is broken when the parent has
    # already initialised a CUDA context (torch + CUDA forked children
    # hit "CUDA initialization error" and die silently). `spawn` is
    # slower to boot (~1-3 s/worker) but amortises over ~3000 pairs to
    # negligible overhead.
    #
    # Result collection uses ``submit`` + ``as_completed`` rather than
    # ``pool.map``. ``map`` buffers out-of-order completions until the
    # in-order result is available — with N workers of varying pair
    # cost, that means zero visible progress until the slowest pair of
    # the head-of-queue batch finishes. ``as_completed`` streams
    # entries as soon as any worker returns one. Log lines may now
    # appear out of submission order; the line itself identifies the
    # pair so that's a cosmetic change, not an ambiguity.
    import multiprocessing as _mp
    import time
    import threading
    from concurrent.futures import ProcessPoolExecutor, as_completed

    with ProcessPoolExecutor(
        max_workers=jobs, initializer=_init_worker,
        mp_context=_mp.get_context("spawn"),
    ) as pool:
        # Submit all tasks up front; collect as they complete.
        futures = [pool.submit(_compute_pair_task, task) for task in tasks]

        # Heartbeat thread: prints "waiting, N/total done" every 30 s so
        # a long first-pair warmup doesn't look like a hang. Stops as
        # soon as we've collected every future.
        total = len(futures)
        heartbeat_stop = threading.Event()
        done_counter = {"n": 0}
        heartbeat_lock = threading.Lock()

        def _heartbeat():
            t0 = time.time()
            while not heartbeat_stop.wait(30.0):
                with heartbeat_lock:
                    n = done_counter["n"]
                elapsed_min = (time.time() - t0) / 60.0
                print(
                    f"  [heartbeat] {n}/{total} pairs complete after "
                    f"{elapsed_min:.1f} min — workers running",
                    flush=True,
                )

        hb_thread = threading.Thread(target=_heartbeat, daemon=True)
        hb_thread.start()

        try:
            for future in as_completed(futures):
                entry = future.result()
                entries[(entry.dataset, entry.generator_id)] = entry
                with heartbeat_lock:
                    done_counter["n"] += 1
                if on_entry is not None:
                    on_entry(entry)
        finally:
            heartbeat_stop.set()
            hb_thread.join(timeout=1.0)

    return LittlesCache(
        version=CACHE_SCHEMA_VERSION,
        generator_registry=generator_registry_name,
        sample_rows_per_evaluation=sample_rows,
        seed_base=seed_base,
        entries=entries,
    )
