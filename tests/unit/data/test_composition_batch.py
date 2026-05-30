"""Tests for lacuna.data.composition_batch — Stage-C composition-labelled batches (ADR-0007)."""

import pytest
import torch

from lacuna.core.rng import RNGState
from lacuna.data.ingestion import RawDataset
from lacuna.data.composition_batch import (
    build_composition_batch, CompositionBatch, N_FOOTPRINT_FEATURES,
)

MAX_ROWS, MAX_COLS = 64, 16


def _raws(seed=0):
    """A few complete Gaussian RawDatasets of varying width (all 4 <= d <= MAX_COLS)."""
    rng = RNGState(seed=seed)
    out = []
    for i, d in enumerate((6, 10, 14)):
        data = rng.spawn().randn(300, d).numpy()
        out.append(RawDataset(data=data, feature_names=tuple(f"f{j}" for j in range(d)),
                              name=f"gauss{i}_d{d}"))
    return out


# ---------------------------------------------------------------------------
# Normal cases
# ---------------------------------------------------------------------------

def test_batch_shapes():
    mb = build_composition_batch(_raws(), RNGState(seed=1),
                                 max_rows=MAX_ROWS, max_cols=MAX_COLS, batch_size=8)
    assert isinstance(mb, CompositionBatch)
    assert mb.batch.tokens.dim() == 4
    assert mb.batch.tokens.shape[:3] == (8, MAX_ROWS, MAX_COLS)
    assert mb.composition.shape == (8, 3)
    assert mb.target_drawn.shape == (8, 3)
    assert mb.miss_rate.shape == (8,)
    assert len(mb.source_names) == 8


def test_composition_is_valid_simplex():
    mb = build_composition_batch(_raws(), RNGState(seed=2),
                                 max_rows=MAX_ROWS, max_cols=MAX_COLS, batch_size=12)
    assert torch.allclose(mb.composition.sum(-1), torch.ones(12), atol=1e-5)
    assert bool((mb.composition >= 0).all()) and bool((mb.composition <= 1).all())
    assert torch.allclose(mb.target_drawn.sum(-1), torch.ones(12), atol=1e-5)


def test_footprints_optional():
    """footprints is None by default, and a [B, 20] tensor when requested (deployable features)."""
    off = build_composition_batch(_raws(), RNGState(seed=4),
                                  max_rows=MAX_ROWS, max_cols=MAX_COLS, batch_size=6)
    assert off.footprints is None
    on = build_composition_batch(_raws(), RNGState(seed=4),
                                 max_rows=MAX_ROWS, max_cols=MAX_COLS, batch_size=6,
                                 with_footprints=True)
    assert on.footprints is not None
    assert on.footprints.shape == (6, N_FOOTPRINT_FEATURES)
    assert torch.isfinite(on.footprints).all()


def test_miss_rate_in_range():
    mb = build_composition_batch(_raws(), RNGState(seed=3),
                                 max_rows=MAX_ROWS, max_cols=MAX_COLS, batch_size=10,
                                 miss_rate_range=(0.1, 0.5))
    assert bool((mb.miss_rate > 0).all()) and bool((mb.miss_rate < 1).all())


def test_deterministic_under_same_seed():
    a = build_composition_batch(_raws(), RNGState(seed=5),
                                max_rows=MAX_ROWS, max_cols=MAX_COLS, batch_size=6)
    b = build_composition_batch(_raws(), RNGState(seed=5),
                                max_rows=MAX_ROWS, max_cols=MAX_COLS, batch_size=6)
    assert torch.equal(a.composition, b.composition)
    assert torch.equal(a.batch.tokens, b.batch.tokens)
    assert a.source_names == b.source_names


def test_realized_label_matches_sampler():
    """The batch label is the sampler's realised composition (the by-cell ground truth), not the
    drawn target — they generally differ (realisation noise)."""
    mb = build_composition_batch(_raws(), RNGState(seed=8),
                                 max_rows=MAX_ROWS, max_cols=MAX_COLS, batch_size=8)
    # at least one item's realised composition differs from its drawn target (it is not echoing it)
    assert not torch.allclose(mb.composition, mb.target_drawn, atol=1e-4)


# ---------------------------------------------------------------------------
# Failure cases
# ---------------------------------------------------------------------------

def test_empty_raws_raises():
    with pytest.raises(ValueError, match="non-empty"):
        build_composition_batch([], RNGState(seed=0), max_rows=MAX_ROWS, max_cols=MAX_COLS, batch_size=4)


def test_bad_batch_size_raises():
    with pytest.raises(ValueError, match="batch_size"):
        build_composition_batch(_raws(), RNGState(seed=0),
                                max_rows=MAX_ROWS, max_cols=MAX_COLS, batch_size=0)


def test_too_narrow_dataset_raises():
    bad = [RawDataset(data=RNGState(seed=0).randn(100, 3).numpy(),
                      feature_names=("a", "b", "c"), name="narrow")]
    with pytest.raises(ValueError, match="require 4 <= d"):
        build_composition_batch(bad, RNGState(seed=0),
                                max_rows=MAX_ROWS, max_cols=MAX_COLS, batch_size=4)


def test_too_wide_dataset_raises():
    wide = [RawDataset(data=RNGState(seed=0).randn(100, MAX_COLS + 2).numpy(),
                       feature_names=tuple(f"f{j}" for j in range(MAX_COLS + 2)), name="wide")]
    with pytest.raises(ValueError, match="require 4 <= d"):
        build_composition_batch(wide, RNGState(seed=0),
                                max_rows=MAX_ROWS, max_cols=MAX_COLS, batch_size=4)
