"""Tests for lacuna.survey.lod_generator (LOD / top-coding step mechanism)."""

import numpy as np
import pytest
import torch

from lacuna.core.rng import RNGState
from lacuna.data.catalog import create_default_catalog
from lacuna.data.ingestion import RawDataset
from lacuna.data.semisynthetic import _zscore_columns
from lacuna.survey.answer_sheet import LOD_FAMILY, AnswerSheet
from lacuna.survey.delta_bins import assign_delta_bin
from lacuna.survey.lod_generator import apply_lod_censor, generate_lod_example


def _indep(n=4000, d=4, seed=0) -> RawDataset:
    rng = np.random.default_rng(seed)
    return RawDataset(data=rng.standard_normal((n, d)).astype("float32"),
                      feature_names=tuple(f"f{c}" for c in range(d)), name="indep")


def _pearson(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = (a.norm() * b.norm()).item()
    return 0.0 if d == 0 else float((a * b).sum().item() / d)


# ---------- rate matching + τ ----------

def test_rate_matched_any_delta():
    raw = _indep()
    for delta in (0.0, 0.5, 2.0, 3.0):
        res = generate_lod_example(raw, beta1=1.0, delta=delta, target_rate=0.3,
                                   tau_quantile=0.70, rng=RNGState(seed=1), target_idx=0)
        assert abs(res.answer_sheet.realized_rate - 0.3) < 0.04


def test_tau_split_matches_quantile():
    raw = _indep()
    res = generate_lod_example(raw, beta1=1.0, delta=1.0, target_rate=0.3,
                               tau_quantile=0.70, rng=RNGState(seed=1), target_idx=0)
    # ~30% of rows are above the 70th percentile
    assert abs(res.answer_sheet.frac_above_tau - 0.30) < 0.03


# ---------- δ=0 (MAR) vs δ>0 (value-localized) ----------

def test_delta_zero_independent_of_step():
    raw = _indep(seed=1)
    res = generate_lod_example(raw, beta1=2.0, delta=0.0, target_rate=0.3,
                               tau_quantile=0.70, rng=RNGState(seed=3), target_idx=0)
    z_t = _zscore_columns(torch.from_numpy(raw.data))[:, 0]
    step = (z_t > res.answer_sheet.tau).float()
    missing = (~res.mask[:, 0]).float()
    assert abs(_pearson(missing, step)) < 0.06  # MAR: independent of own-value region


def test_delta_positive_concentrates_above_tau():
    raw = _indep(seed=1)
    res = generate_lod_example(raw, beta1=0.0, delta=3.0, target_rate=0.3,
                               tau_quantile=0.70, rng=RNGState(seed=3), target_idx=0)
    z_t = _zscore_columns(torch.from_numpy(raw.data))[:, 0]
    step = (z_t > res.answer_sheet.tau).float()
    missing = (~res.mask[:, 0]).float()
    assert _pearson(missing, step) > 0.3  # missingness localized to the upper region


# ---------- answer sheet ----------

def test_answer_sheet_records_lod_fields():
    raw = _indep()
    res = generate_lod_example(raw, beta1=1.0, delta=0.8, target_rate=0.3,
                               tau_quantile=0.70, rng=RNGState(seed=5), target_idx=0)
    s = res.answer_sheet
    assert s.generator_family == LOD_FAMILY
    assert s.delta == 0.8 and s.delta_bin == assign_delta_bin(0.8)
    assert s.tau is not None and s.frac_above_tau is not None
    assert AnswerSheet.from_dict(s.to_dict()) == s  # round-trips with LOD family + τ


# ---------- determinism + real data ----------

def test_determinism():
    raw = _indep(seed=2)
    a = generate_lod_example(raw, beta1=1.0, delta=1.0, target_rate=0.3, tau_quantile=0.70,
                             rng=RNGState(seed=9))
    b = generate_lod_example(raw, beta1=1.0, delta=1.0, target_rate=0.3, tau_quantile=0.70,
                             rng=RNGState(seed=9))
    assert torch.equal(a.mask, b.mask) and a.answer_sheet == b.answer_sheet


def test_runs_on_real_survey():
    raw = create_default_catalog().load("survey_cps1985")
    res = generate_lod_example(raw, beta1=1.0, delta=2.0, target_rate=0.3, tau_quantile=0.70,
                               rng=RNGState(seed=13))
    assert res.answer_sheet.generator_family == LOD_FAMILY
    assert abs(res.answer_sheet.realized_rate - 0.3) < 0.05


# ---------- failures ----------

def test_rejects_bad_tau_quantile():
    raw = _indep(n=200)
    for bad in (0.0, 1.0, 1.5):
        with pytest.raises(ValueError):
            apply_lod_censor(torch.from_numpy(raw.data), 0, 1, beta1=1.0, delta=1.0,
                             tau_quantile=bad, target_rate=0.3, rng=RNGState(seed=1))


def test_rejects_negative_delta_or_beta1():
    raw = _indep(n=200)
    X = torch.from_numpy(raw.data)
    with pytest.raises(ValueError):
        apply_lod_censor(X, 0, 1, beta1=-1.0, delta=1.0, tau_quantile=0.7, target_rate=0.3,
                         rng=RNGState(seed=1))
    with pytest.raises(ValueError):
        apply_lod_censor(X, 0, 1, beta1=1.0, delta=-1.0, tau_quantile=0.7, target_rate=0.3,
                         rng=RNGState(seed=1))


def test_rejects_constant_target():
    data = np.random.default_rng(0).standard_normal((200, 3)).astype("float32")
    data[:, 2] = 5.0
    X = torch.from_numpy(data)
    with pytest.raises(ValueError):
        apply_lod_censor(X, 2, 0, beta1=1.0, delta=1.0, tau_quantile=0.7, target_rate=0.3,
                         rng=RNGState(seed=1))
