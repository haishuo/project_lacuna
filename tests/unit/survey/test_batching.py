"""Tests for lacuna.survey.batching."""

import numpy as np
import pytest
import torch

from lacuna.core.rng import RNGState
from lacuna.data.catalog import create_default_catalog
from lacuna.data.ingestion import RawDataset
from lacuna.survey.batching import collate, make_example
from lacuna.survey.delta_bins import assign_delta_bin


def _raw(n=1000, d=5, seed=0):
    rng = np.random.default_rng(seed)
    return RawDataset(data=rng.standard_normal((n, d)).astype("float32"),
                      feature_names=tuple(f"f{c}" for c in range(d)), name="syn")


# ---------- normal ----------

def test_make_example_subsamples_rows():
    raw = _raw(n=1000)
    ex = make_example(raw, beta1=1.0, delta=0.5, target_rate=0.3,
                      rng=RNGState(seed=1), max_rows=128)
    assert ex.observed.n == 128
    assert ex.answer_sheet.delta_bin == assign_delta_bin(0.5)


def test_collate_shapes_and_out_of_band_labels():
    raw = _raw()
    rng = RNGState(seed=2)
    exs = [make_example(raw, beta1=1.0, delta=d, target_rate=0.3, rng=rng.spawn(), max_rows=64)
           for d in (0.0, 0.4, 2.5)]
    db = collate(exs, max_rows=64, max_cols=32)
    assert db.tokens.tokens.shape == (3, 64, 32, 4)
    assert db.delta_bin.tolist() == [0, 2, 6]
    assert db.delta.tolist() == pytest.approx([0.0, 0.4, 2.5])
    # 3-class fields must remain unused
    assert db.tokens.generator_ids is None
    assert db.tokens.class_ids is None
    assert db.tokens.variant_ids is None
    assert len(db.sheets) == 3


def test_real_survey_example():
    raw = create_default_catalog().load("survey_bfi")
    ex = make_example(raw, beta1=1.0, delta=1.0, target_rate=0.25,
                      rng=RNGState(seed=3), max_rows=200)
    assert ex.observed.n == 200
    assert ex.answer_sheet.source_name == "survey_bfi"


# ---------- determinism ----------

def test_collate_deterministic():
    raw = _raw()
    def build():
        rng = RNGState(seed=9)
        exs = [make_example(raw, beta1=1.0, delta=d, target_rate=0.3, rng=rng.spawn(), max_rows=64)
               for d in (0.0, 1.0, 2.0)]
        return collate(exs, max_rows=64, max_cols=32)
    a, b = build(), build()
    assert torch.equal(a.tokens.tokens, b.tokens.tokens)
    assert torch.equal(a.delta_bin, b.delta_bin)


# ---------- failure ----------

def test_collate_rejects_empty():
    with pytest.raises(ValueError):
        collate([], max_rows=64, max_cols=32)


def test_collate_rejects_too_wide():
    raw = _raw(d=40)
    ex = make_example(raw, beta1=1.0, delta=0.5, target_rate=0.3,
                      rng=RNGState(seed=1), max_rows=64)
    with pytest.raises(ValueError):
        collate([ex], max_rows=64, max_cols=32)
