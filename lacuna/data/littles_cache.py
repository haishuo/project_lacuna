"""
lacuna.data.littles_cache

Offline cache of Little's MCAR test statistics, keyed by (dataset, generator).

Why offline: `pystatistics.mvnmle.little_mcar_test` runs an EM-based MLE
and chi-squared computation that is too slow to call per training batch
(milliseconds to seconds per call; forward-pass-breaking at batch scale).
Caching once per (raw_dataset, generator_id) pair makes the per-batch cost
a dictionary lookup.

Each cache entry holds BOTH an MLE result (Little 1988, asymptotically
efficient, slower) and a MoM result (method-of-moments, consistent under
MCAR but not asymptotically efficient, ~30-50× faster at typical sizes).
The model reads whichever method the data loader is configured to emit.
This lets the MLE-vs-MoM comparison happen at model-forward time against
a single cache build. See ADR 0003.

See `docs/decisions/0001-remove-pointbiserial-distributional.md` for the
original ablation evidence and `docs/decisions/0002-real-littles-mcar-cached.md`
for the decision to move from heuristic → cached real Little's.

Contract
--------
Cache is a JSON file (human-readable, diffable) with schema:

    {
      "version": 2,
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
         "mom_df": 8,  "mom_rejected": false}
      ]
    }

Version 1 caches (MLE only, no MoM fields) are REJECTED on load — rebuild.

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
from pystatistics.mvnmle import little_mcar_test, mom_mcar_test

from lacuna.core.rng import RNGState
from lacuna.data.ingestion import RawDataset
from lacuna.generators.base import Generator


CACHE_SCHEMA_VERSION = 2

# Valid method selectors for `LittlesCache.get(...)`.
VALID_METHODS = ("mle", "mom")


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
        return entry.mom_statistic, entry.mom_p_value

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
                f"requires v{CACHE_SCHEMA_VERSION} (MLE + MoM). Rebuild with "
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

    # Run both tests on the same array. MoM cost is negligible vs MLE, so
    # computing both gives us a paired comparison for free at cache-build
    # time. If either one raises, record a sentinel for THAT method only
    # — the other may still succeed.
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

    mle = _run(little_mcar_test)
    mom = _run(mom_mcar_test)

    mle_stat, mle_p, mle_df, mle_rej = mle if mle is not None else (0.0, 1.0, 0, False)
    mom_stat, mom_p, mom_df, mom_rej = mom if mom is not None else (0.0, 1.0, 0, False)

    return LittlesCacheEntry(
        dataset=raw.name,
        generator_id=generator.generator_id,
        generator_name=generator.name,
        n_used=n_eff,
        mle_statistic=mle_stat, mle_p_value=mle_p, mle_df=mle_df, mle_rejected=mle_rej,
        mom_statistic=mom_stat, mom_p_value=mom_p, mom_df=mom_df, mom_rejected=mom_rej,
    )


# =============================================================================
# Cache builder
# =============================================================================


def build_cache(
    raw_datasets: List[RawDataset],
    generators: List[Generator],
    *,
    generator_registry_name: str,
    sample_rows: int = 1000,
    seed_base: int = 20260418,
    backend: str = "auto",
    on_entry: Optional[callable] = None,
) -> LittlesCache:
    """Compute Little's MCAR for every (dataset, generator) pair.

    Args:
        raw_datasets: Real tabular datasets to cache against.
        generators: Generator instances to apply.
        generator_registry_name: Label stored in cache metadata for audit.
        sample_rows: Rows to subsample per evaluation. Larger → more
            statistical power but slower; 1000 is typically enough for
            stable Little's chi-squared on 5-20 column data.
        seed_base: Base seed for deterministic subsampling + generator
            application. Each (dataset, generator) pair derives its own
            seed from this base.
        on_entry: Optional callable invoked after each entry is computed
            (for live progress logging).

    Returns:
        A fully populated LittlesCache.
    """
    entries: Dict[Tuple[str, int], LittlesCacheEntry] = {}
    for ds_idx, raw in enumerate(raw_datasets):
        for gen in generators:
            pair_seed = seed_base + ds_idx * 1_000_003 + gen.generator_id
            rng = RNGState(seed=pair_seed)
            entry = compute_entry(
                raw, gen, rng=rng, sample_rows=sample_rows, backend=backend
            )
            entries[(raw.name, gen.generator_id)] = entry
            if on_entry is not None:
                on_entry(entry)

    return LittlesCache(
        version=CACHE_SCHEMA_VERSION,
        generator_registry=generator_registry_name,
        sample_rows_per_evaluation=sample_rows,
        seed_base=seed_base,
        entries=entries,
    )
