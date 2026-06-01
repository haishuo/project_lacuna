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
# fixed_composition + mnar_subtypes (Stage-F: hold the composition fixed, vary only the MNAR family)
# ---------------------------------------------------------------------------

_LOUD = ("threshold_left", "threshold_right", "threshold_two_sided", "soft_threshold",
         "col_specific_thresh", "detection_lower", "detection_upper", "detection_both")
_QUIET = ("self_censoring", "selfcensor_high", "selfcensor_low", "selfcensor_extreme",
          "selfcensor_weak", "selfcensor_strong")


def test_fixed_composition_realises_near_target():
    """fixed_composition forces every item toward the same by-cell mix (vs a fresh simplex draw);
    the realised composition (the supervised label) lands near it on average."""
    comp = (0.2, 0.3, 0.5)
    mb = build_composition_batch(_raws(), RNGState(seed=3), max_rows=MAX_ROWS, max_cols=MAX_COLS,
                                 batch_size=24, fixed_composition=comp, mnar_block_share=0.0)
    mean = mb.composition.mean(0)
    assert torch.allclose(mean, torch.tensor(comp), atol=0.08), f"mean realised {mean.tolist()}"


def test_fixed_composition_corpora_matched_across_subtypes():
    """THE Stage-F batch invariant: same rng + fixed composition, varying ONLY mnar_subtypes, gives
    identical realised miss rates (matched data/plan) while the realised composition stays ~fixed —
    so a downstream loud-vs-quiet model read differs only by the MNAR mechanism family."""
    kw = dict(max_rows=MAX_ROWS, max_cols=MAX_COLS, batch_size=16,
              fixed_composition=(0.2, 0.3, 0.5), mnar_block_share=0.0, with_footprints=True)
    loud = build_composition_batch(_raws(), RNGState(seed=5), mnar_subtypes=_LOUD, **kw)
    quiet = build_composition_batch(_raws(), RNGState(seed=5), mnar_subtypes=_QUIET, **kw)
    assert loud.source_names == quiet.source_names                       # same datasets, same order
    assert torch.allclose(loud.miss_rate, quiet.miss_rate, atol=0.05)    # matched overall miss level
    # The MNAR-shape footprint features (obs skew/kurt) should differ between families…
    assert not torch.allclose(loud.footprints, quiet.footprints)
    # …while the composition stays the held-fixed target for both.
    assert torch.allclose(loud.composition.mean(0), quiet.composition.mean(0), atol=0.06)


def test_fixed_composition_bad_length_raises():
    with pytest.raises(ValueError, match="3 entries"):
        build_composition_batch(_raws(), RNGState(seed=0), max_rows=MAX_ROWS, max_cols=MAX_COLS,
                                batch_size=4, fixed_composition=(0.5, 0.5))


def test_fixed_composition_bad_sum_raises():
    with pytest.raises(ValueError, match="sum to 1"):
        build_composition_batch(_raws(), RNGState(seed=0), max_rows=MAX_ROWS, max_cols=MAX_COLS,
                                batch_size=4, fixed_composition=(0.5, 0.3, 0.5))


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
