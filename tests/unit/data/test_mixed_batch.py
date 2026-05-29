"""Tests for lacuna.data.mixed_batch — per-column-labelled batch construction (Stage 1)."""

import pytest
import torch

from lacuna.core.rng import RNGState
from lacuna.core.types import MCAR, MAR, MNAR
from lacuna.data.ingestion import RawDataset
from lacuna.data.mixed_missingness import OBSERVED
from lacuna.data.mixed_batch import (
    sample_column_classes,
    build_mixed_batch,
    MixedBatch,
)

_MECHS = (MCAR, MAR, MNAR)


def _raws(n=100, d=6, k=2):
    out = []
    for i in range(k):
        data = RNGState(seed=i).randn(n, d).numpy()
        out.append(RawDataset(data=data, feature_names=tuple(f"c{j}" for j in range(d)),
                              name=f"synth{i}"))
    return out


# ---------------------------------------------------------------------------
# sample_column_classes
# ---------------------------------------------------------------------------

def test_sample_classes_length_and_values():
    cls = sample_column_classes(6, RNGState(seed=1))
    assert len(cls) == 6
    assert all(c in (OBSERVED, MCAR, MAR, MNAR) for c in cls)


def test_sample_classes_guarantees():
    # Over many seeds, every assignment has a clean predictor AND a supervised column.
    for s in range(50):
        cls = sample_column_classes(5, RNGState(seed=s))
        assert any(c in (OBSERVED, MCAR) for c in cls), "needs a clean predictor"
        assert any(c in _MECHS for c in cls), "needs a supervised column"


def test_sample_classes_deterministic():
    a = sample_column_classes(7, RNGState(seed=3))
    b = sample_column_classes(7, RNGState(seed=3))
    assert a == b


def test_sample_classes_small_d_no_mar_mnar():
    # d<2 cannot use logistic MAR/MNAR generators -> only OBSERVED/MCAR.
    for s in range(20):
        cls = sample_column_classes(1, RNGState(seed=s))
        assert all(c in (OBSERVED, MCAR) for c in cls)


def test_sample_classes_bad_p_observed():
    with pytest.raises(ValueError, match="p_observed"):
        sample_column_classes(5, RNGState(seed=0), p_observed=2.0)


# ---------------------------------------------------------------------------
# build_mixed_batch
# ---------------------------------------------------------------------------

def test_build_batch_shapes():
    mb = build_mixed_batch(_raws(d=6), RNGState(seed=1),
                           max_rows=64, max_cols=8, batch_size=4)
    assert isinstance(mb, MixedBatch)
    assert mb.batch.tokens.shape == (4, 64, 8, 4)
    assert mb.labels.shape == (4, 8)
    assert mb.supervision_mask.shape == (4, 8)
    assert mb.labels.dtype == torch.long
    assert mb.supervision_mask.dtype == torch.bool
    assert len(mb.compositions) == 4


def test_supervision_mask_matches_composition():
    mb = build_mixed_batch(_raws(d=6), RNGState(seed=2),
                           max_rows=64, max_cols=8, batch_size=4)
    for i, comp in enumerate(mb.compositions):
        d = len(comp)
        for j in range(8):
            expected = (j < d) and (comp[j] in _MECHS)
            assert bool(mb.supervision_mask[i, j]) == expected
            if expected:
                assert int(mb.labels[i, j]) == comp[j]


def test_padding_columns_unsupervised():
    # d=6, max_cols=8 -> columns 6,7 are padding and must never be supervised.
    mb = build_mixed_batch(_raws(d=6), RNGState(seed=5),
                           max_rows=64, max_cols=8, batch_size=3)
    assert not mb.supervision_mask[:, 6:].any()


def test_labels_valid_class_indices_where_supervised():
    mb = build_mixed_batch(_raws(d=6), RNGState(seed=7),
                           max_rows=64, max_cols=8, batch_size=4)
    sup = mb.supervision_mask
    assert torch.all((mb.labels[sup] >= 0) & (mb.labels[sup] <= 2))


def test_build_batch_deterministic():
    a = build_mixed_batch(_raws(d=6), RNGState(seed=11),
                          max_rows=64, max_cols=8, batch_size=3)
    b = build_mixed_batch(_raws(d=6), RNGState(seed=11),
                          max_rows=64, max_cols=8, batch_size=3)
    assert torch.equal(a.batch.tokens, b.batch.tokens)
    assert torch.equal(a.labels, b.labels)
    assert torch.equal(a.supervision_mask, b.supervision_mask)
    assert a.compositions == b.compositions


def test_at_least_one_supervised_column_per_item():
    mb = build_mixed_batch(_raws(d=6), RNGState(seed=13),
                           max_rows=64, max_cols=8, batch_size=6)
    assert torch.all(mb.supervision_mask.any(dim=1)), "every item must have ≥1 supervised col"


def test_empty_raws_raises():
    with pytest.raises(ValueError, match="non-empty"):
        build_mixed_batch([], RNGState(seed=0), max_rows=64, max_cols=8, batch_size=2)


def test_bad_batch_size_raises():
    with pytest.raises(ValueError, match="batch_size"):
        build_mixed_batch(_raws(), RNGState(seed=0), max_rows=64, max_cols=8, batch_size=0)
