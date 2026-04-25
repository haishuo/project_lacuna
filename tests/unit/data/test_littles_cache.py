"""
Tests for lacuna.data.littles_cache

Covers:
    Normal cases:
        - compute_entry returns a populated LittlesCacheEntry.
        - build_cache covers every (dataset, generator) pair.
        - save / load round-trip preserves all fields.
        - cache.get returns the stored (statistic, p_value).
    Edge cases:
        - sample_rows > dataset size is clamped to dataset size.
        - A generator producing no missingness emits a degenerate entry
          (stat=0, p=1) rather than raising.
    Failure cases:
        - cache.get raises KeyError on unknown (dataset, generator).
        - load_cache raises FileNotFoundError on missing path.
        - load_cache raises ValueError on version mismatch.
"""

import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest
import torch

from lacuna.core.rng import RNGState
from lacuna.core.types import MCAR
from lacuna.data.ingestion import RawDataset
from lacuna.data.littles_cache import (
    CACHE_SCHEMA_VERSION,
    LittlesCache,
    LittlesCacheEntry,
    build_cache,
    compute_entry,
    load_cache,
    save_cache,
)
from lacuna.generators.base import Generator
from lacuna.generators.params import GeneratorParams


# =============================================================================
# Test fixtures and mock generators
# =============================================================================


class _BernoulliMCAR(Generator):
    """Plain 30%-Bernoulli missingness, for a predictable test signal."""

    def __init__(self, generator_id: int = 0, name: str = "mock_bern"):
        super().__init__(
            generator_id=generator_id,
            name=name,
            class_id=MCAR,
            params=GeneratorParams(miss_rate=0.3),
        )

    def sample(self, rng: RNGState, n: int, d: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = rng.randn(n, d)
        r = rng.rand(n, d) > 0.3
        return x, r


class _NoMissing(Generator):
    """Never drops a value. Produces a degenerate mask for Little's."""

    def __init__(self, generator_id: int = 1):
        super().__init__(
            generator_id=generator_id,
            name="no_missing",
            class_id=MCAR,
            params=GeneratorParams(miss_rate=0.0),
        )

    def sample(self, rng: RNGState, n: int, d: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = rng.randn(n, d)
        r = torch.ones(n, d, dtype=torch.bool)
        return x, r


@pytest.fixture
def small_raw():
    data = np.random.default_rng(0).standard_normal((100, 5)).astype(np.float32)
    return RawDataset(data, tuple(f"c{i}" for i in range(5)), name="small")


# =============================================================================
# Normal cases
# =============================================================================


def test_compute_entry_populates_fields(small_raw):
    gen = _BernoulliMCAR()
    entry = compute_entry(
        small_raw, gen, rng=RNGState(seed=1), sample_rows=100,
        # Small permutation counts keep the test fast. Propensity uses
        # its analytical null — no permutation knob needed.
        hsic_permutations=9, missmech_permutations=9,
    )
    assert isinstance(entry, LittlesCacheEntry)
    assert entry.dataset == "small"
    assert entry.generator_id == 0
    assert entry.generator_name == "mock_bern"
    assert entry.n_used == 100
    # MLE and MoM populated
    assert np.isfinite(entry.mle_statistic)
    assert 0.0 <= entry.mle_p_value <= 1.0
    assert np.isfinite(entry.mom_statistic)
    assert 0.0 <= entry.mom_p_value <= 1.0
    # Schema v3 nonparametric fields populated.
    assert np.isfinite(entry.propensity_statistic)
    assert 0.0 <= entry.propensity_p_value <= 1.0
    assert np.isfinite(entry.hsic_statistic)
    assert 0.0 <= entry.hsic_p_value <= 1.0
    assert np.isfinite(entry.missmech_statistic)
    assert 0.0 <= entry.missmech_p_value <= 1.0


def _fast_build_cache(raw_datasets, generators, name="mock"):
    """build_cache with minimal permutation counts for test speed."""
    return build_cache(
        raw_datasets=raw_datasets,
        generators=generators,
        generator_registry_name=name,
        sample_rows=100,
        hsic_permutations=9,
        missmech_permutations=9,
    )


def test_build_cache_covers_all_pairs(small_raw):
    gen_a = _BernoulliMCAR(generator_id=0, name="a")
    gen_b = _BernoulliMCAR(generator_id=1, name="b")
    cache = _fast_build_cache([small_raw], [gen_a, gen_b])
    assert len(cache.entries) == 2
    assert ("small", 0) in cache.entries
    assert ("small", 1) in cache.entries
    assert cache.version == CACHE_SCHEMA_VERSION


def test_cache_get_returns_pair_for_all_five_methods(small_raw):
    gen = _BernoulliMCAR()
    cache = _fast_build_cache([small_raw], [gen])
    entry = cache.entries[("small", 0)]
    for method, (want_stat, want_p) in [
        ("mle",        (entry.mle_statistic,        entry.mle_p_value)),
        ("mom",        (entry.mom_statistic,        entry.mom_p_value)),
        ("propensity", (entry.propensity_statistic, entry.propensity_p_value)),
        ("hsic",       (entry.hsic_statistic,       entry.hsic_p_value)),
        ("missmech",   (entry.missmech_statistic,   entry.missmech_p_value)),
    ]:
        stat, p = cache.get("small", 0, method=method)
        assert stat == want_stat, f"statistic mismatch for method={method}"
        assert p == want_p, f"p_value mismatch for method={method}"
    # Default method is MLE
    d_stat, d_p = cache.get("small", 0)
    assert d_stat == entry.mle_statistic


def test_cache_get_rejects_bad_method(small_raw):
    gen = _BernoulliMCAR()
    cache = _fast_build_cache([small_raw], [gen])
    with pytest.raises(ValueError, match="method"):
        cache.get("small", 0, method="nonsense")


@pytest.mark.parametrize("backend", ["cpu", "auto"])
def test_parallel_build_matches_sequential_bit_for_bit(small_raw, backend):
    """jobs=2 must produce the exact same entries as jobs=1 (bit-for-bit
    on the statistics) — the parallel path is a pure speed optimisation,
    not a different algorithm.

    Parametrised over ``backend``: validates the parallel path under
    both the CPU reference path and ``"auto"`` (which will use GPU if
    CUDA is available — each worker spawns its own CUDA context).

    Uses ``MCARBernoulli`` from ``lacuna.generators.families.mcar``
    rather than a local fixture because worker processes under the
    'spawn' start method need to unpickle the generator by its
    fully-qualified module path; a class defined inside a test file
    has a synthetic ``__module__`` that subprocesses can't resolve.
    """
    from lacuna.generators.families.mcar.bernoulli import MCARBernoulli
    gen_a = MCARBernoulli(
        generator_id=0, name="bern30",
        params=GeneratorParams(miss_rate=0.3),
    )
    gen_b = MCARBernoulli(
        generator_id=1, name="bern20",
        params=GeneratorParams(miss_rate=0.2),
    )
    rng = np.random.default_rng(7)
    other = RawDataset(
        rng.standard_normal((80, 4)).astype(np.float32),
        tuple(f"c{i}" for i in range(4)), name="other",
    )

    kwargs = dict(
        generator_registry_name="mock", sample_rows=80,
        backend=backend,
        hsic_permutations=9, missmech_permutations=9,
    )
    seq = build_cache(
        raw_datasets=[small_raw, other], generators=[gen_a, gen_b],
        jobs=1, **kwargs,
    )
    par = build_cache(
        raw_datasets=[small_raw, other], generators=[gen_a, gen_b],
        jobs=2, **kwargs,
    )
    assert set(seq.entries) == set(par.entries)
    for key, seq_entry in seq.entries.items():
        par_entry = par.entries[key]
        # Bit-for-bit equality on every stored scalar — the per-pair
        # seed is deterministic, and no stage draws from a process-local
        # RNG. This must hold on CPU *and* GPU backends: a parallel run
        # that diverges from sequential is a correctness bug, not a
        # rounding tolerance issue.
        assert seq_entry == par_entry, f"divergence at {key}"


def test_save_load_round_trip(small_raw, tmp_path: Path):
    gen = _BernoulliMCAR()
    cache = _fast_build_cache([small_raw], [gen])
    path = tmp_path / "cache.json"
    save_cache(cache, path)
    restored = load_cache(path)
    assert restored.version == cache.version
    assert restored.generator_registry == cache.generator_registry
    assert restored.sample_rows_per_evaluation == cache.sample_rows_per_evaluation
    assert restored.seed_base == cache.seed_base
    assert set(restored.entries) == set(cache.entries)
    for key, orig in cache.entries.items():
        assert restored.entries[key] == orig


# =============================================================================
# Edge cases
# =============================================================================


def test_sample_rows_clamped_to_dataset_size(small_raw):
    """Requesting more rows than the dataset has falls back to all rows."""
    gen = _BernoulliMCAR()
    entry = compute_entry(small_raw, gen, rng=RNGState(seed=2), sample_rows=10_000)
    assert entry.n_used == 100  # dataset has 100 rows


def test_no_missingness_produces_degenerate_entry(small_raw):
    """A generator that never drops values -> stat=0, p=1 sentinel for every
    method (MLE, MoM, propensity, HSIC, MissMech)."""
    gen = _NoMissing()
    with pytest.warns(UserWarning, match="Degenerate mask"):
        entry = compute_entry(small_raw, gen, rng=RNGState(seed=3), sample_rows=100)
    assert entry.mle_statistic == 0.0
    assert entry.mle_p_value == 1.0
    assert entry.mle_rejected is False
    assert entry.mom_statistic == 0.0
    assert entry.mom_p_value == 1.0
    assert entry.mom_rejected is False
    assert entry.propensity_statistic == 0.0
    assert entry.propensity_p_value == 1.0
    assert entry.hsic_statistic == 0.0
    assert entry.hsic_p_value == 1.0
    assert entry.missmech_statistic == 0.0
    assert entry.missmech_p_value == 1.0


# =============================================================================
# Failure cases
# =============================================================================


def test_get_unknown_key_raises():
    cache = LittlesCache(
        version=CACHE_SCHEMA_VERSION,
        generator_registry="mock",
        sample_rows_per_evaluation=100,
        seed_base=0,
        entries={},
    )
    with pytest.raises(KeyError, match="No cached MCAR"):
        cache.get("nonexistent", 0)


def test_load_v1_cache_raises_with_migration_guidance(tmp_path: Path):
    """A v1 cache file must be rejected with a clear rebuild message."""
    path = tmp_path / "v1.json"
    path.write_text(json.dumps({
        "version": 1,
        "generator_registry": "mock",
        "sample_rows_per_evaluation": 100,
        "seed_base": 0,
        "entries": [],
    }))
    with pytest.raises(ValueError, match="v1"):
        load_cache(path)


def test_load_v2_cache_raises_with_migration_guidance(tmp_path: Path):
    """A v2 cache file (MLE + MoM only) must be rejected with a
    schema-v3 rebuild message now that propensity/HSIC/MissMech are
    required."""
    path = tmp_path / "v2.json"
    path.write_text(json.dumps({
        "version": 2,
        "generator_registry": "mock",
        "sample_rows_per_evaluation": 100,
        "seed_base": 0,
        "entries": [],
    }))
    with pytest.raises(ValueError, match="v2"):
        load_cache(path)


def test_load_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_cache(tmp_path / "does_not_exist.json")


def test_load_version_mismatch_raises(tmp_path: Path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({
        "version": 9999,
        "generator_registry": "mock",
        "sample_rows_per_evaluation": 100,
        "seed_base": 0,
        "entries": [],
    }))
    with pytest.raises(ValueError, match="version"):
        load_cache(path)


# =============================================================================
# Integration with SemiSyntheticDataLoader
# =============================================================================


def test_loader_validates_cache_coverage(small_raw):
    """Missing cache keys fail loud at SemiSyntheticDataLoader construction."""
    from lacuna.data.semisynthetic import SemiSyntheticDataLoader
    from lacuna.generators.priors import GeneratorPrior
    from lacuna.generators.registry import GeneratorRegistry

    gen_a = _BernoulliMCAR(generator_id=0, name="a")
    gen_b = _BernoulliMCAR(generator_id=1, name="b")
    registry = GeneratorRegistry((gen_a, gen_b))
    # Cache covers only generator 0, not 1
    partial = _fast_build_cache([small_raw], [gen_a], name="partial")
    with pytest.raises(ValueError, match="missing"):
        SemiSyntheticDataLoader(
            raw_datasets=[small_raw],
            registry=registry,
            prior=GeneratorPrior.uniform(registry),
            max_rows=64,
            max_cols=16,
            batch_size=4,
            batches_per_epoch=1,
            seed=42,
            littles_cache=partial,
        )
