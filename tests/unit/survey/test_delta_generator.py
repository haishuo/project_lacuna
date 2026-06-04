"""Tests for lacuna.survey.delta_generator.

Run on a small synthetic-but-real-shaped RawDataset and on a real catalog dataset
(survey_bfi). Covers rate-match, MAR vs MNAR own-value correlation, no rate leakage,
determinism, answer-sheet round-trip, and the failure boundaries.
"""

import numpy as np
import pytest
import torch

from lacuna.core.rng import RNGState
from lacuna.data.catalog import create_default_catalog
from lacuna.data.ingestion import RawDataset
from lacuna.survey.answer_sheet import GENERATOR_FAMILY, AnswerSheet
from lacuna.survey.delta_generator import (
    check_realized_rate,
    generate_self_censor_example,
    rate_tolerance,
    select_target_predictor,
)


def _raw(n=4000, d=5, seed=0) -> RawDataset:
    """Synthetic-but-real-shaped dataset: column 1 is correlated with column 0."""
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((n, d))
    base[:, 1] = 0.7 * base[:, 0] + 0.3 * rng.standard_normal(n)  # predictor signal
    return RawDataset(
        data=base.astype("float32"),
        feature_names=tuple(f"f{c}" for c in range(d)),
        source="synthetic",
        name="synthetic_demo",
    )


def _raw_independent(n=4000, d=4, seed=0) -> RawDataset:
    """Dataset with mutually independent columns (no induced own-value correlation).

    Used for the MAR/MNAR own-value tests: with an independent predictor, a δ=0
    mechanism cannot induce own-value correlation through column coupling.
    """
    rng = np.random.default_rng(seed)
    return RawDataset(
        data=rng.standard_normal((n, d)).astype("float32"),
        feature_names=tuple(f"f{c}" for c in range(d)),
        source="synthetic",
        name="synthetic_indep",
    )


def _pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).item()
    return 0.0 if denom == 0 else float((a * b).sum().item() / denom)


# ---------- column selection ----------

def test_predictor_is_most_correlated_column():
    raw = _raw()
    X = torch.from_numpy(raw.data)
    t, p, corr = select_target_predictor(X, RNGState(seed=1), target_idx=0)
    assert t == 0
    assert p == 1  # column 1 is the constructed correlate of column 0
    assert abs(corr) > 0.5


def test_select_rejects_too_few_columns():
    X = torch.randn(50, 1)
    with pytest.raises(ValueError):
        select_target_predictor(X, RNGState(seed=1))


def test_select_rejects_constant_pinned_target():
    X = torch.randn(50, 3)
    X[:, 2] = 5.0
    with pytest.raises(ValueError):
        select_target_predictor(X, RNGState(seed=1), target_idx=2)


def test_select_rejects_when_fewer_than_two_nonconstant():
    X = torch.zeros(50, 4)
    X[:, 0] = torch.randn(50)  # only one non-constant column
    with pytest.raises(ValueError):
        select_target_predictor(X, RNGState(seed=1))


# ---------- normal cases ----------

def test_rate_matching_hits_target():
    raw = _raw()
    res = generate_self_censor_example(
        raw, beta1=1.0, delta=1.0, target_rate=0.3, rng=RNGState(seed=7), target_idx=0
    )
    assert check_realized_rate(res.answer_sheet)
    assert abs(res.answer_sheet.realized_rate - 0.3) < 0.03


def test_only_target_column_censored():
    raw = _raw()
    res = generate_self_censor_example(
        raw, beta1=1.0, delta=1.0, target_rate=0.3, rng=RNGState(seed=7), target_idx=0
    )
    t = res.answer_sheet.target_col_idx
    for c in range(res.answer_sheet.d):
        if c == t:
            assert not res.mask[:, c].all()
        else:
            assert res.mask[:, c].all()
    # observed X is original scale with missing cells zeroed
    assert torch.equal(res.x_observed, torch.from_numpy(raw.data) * res.mask.float())


def test_answer_sheet_fields_populated():
    raw = _raw()
    res = generate_self_censor_example(
        raw, beta1=0.5, delta=0.8, target_rate=0.3, rng=RNGState(seed=11), target_idx=0
    )
    s = res.answer_sheet
    assert s.source_name == "synthetic_demo"
    assert s.generator_family == GENERATOR_FAMILY
    assert s.target_col_name == "f0"
    assert s.predictor_col_name == raw.feature_names[s.predictor_col_idx]
    assert s.delta == 0.8 and s.delta_bin == 3
    assert s.seed == 11
    # round-trips losslessly
    assert AnswerSheet.from_dict(s.to_dict()) == s


def test_delta_zero_is_mar_independent_of_own_value():
    raw = _raw_independent(seed=1)
    res = generate_self_censor_example(
        raw, beta1=2.0, delta=0.0, target_rate=0.3, rng=RNGState(seed=3), target_idx=0
    )
    z_t = torch.from_numpy(raw.data)[:, 0]
    missing = (~res.mask[:, 0]).float()
    assert abs(_pearson(missing, z_t)) < 0.06  # ~0 corr with own value


def test_delta_positive_is_mnar_correlated_with_own_value():
    raw = _raw_independent(seed=1)
    res = generate_self_censor_example(
        raw, beta1=0.0, delta=2.0, target_rate=0.3, rng=RNGState(seed=3), target_idx=0
    )
    z_t = torch.from_numpy(raw.data)[:, 0]
    missing = (~res.mask[:, 0]).float()
    assert _pearson(missing, z_t) > 0.15  # clearly positive: self-censoring


def test_no_rate_leakage_across_delta_sweep():
    """Realized rates cluster around target across a δ sweep — no systematic δ→rate cue."""
    raw = _raw(n=6000, seed=5)
    grid = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
    realized = []
    for i, delta in enumerate(grid):
        res = generate_self_censor_example(
            raw, beta1=1.0, delta=delta, target_rate=0.3,
            rng=RNGState(seed=100 + i), target_idx=0,
        )
        realized.append(res.answer_sheet.realized_rate)
    realized_t = torch.tensor(realized)
    # every realized rate sits near the matched target
    assert torch.all((realized_t - 0.3).abs() < 0.03)
    # no systematic correlation between δ and realized rate
    delta_t = torch.tensor(grid)
    assert abs(_pearson(delta_t, realized_t)) < 0.5


def test_determinism_same_seed_same_mask_and_sheet():
    raw = _raw(seed=2)
    a = generate_self_censor_example(
        raw, beta1=1.0, delta=1.0, target_rate=0.3, rng=RNGState(seed=99)
    )
    b = generate_self_censor_example(
        raw, beta1=1.0, delta=1.0, target_rate=0.3, rng=RNGState(seed=99)
    )
    assert torch.equal(a.mask, b.mask)
    assert a.answer_sheet == b.answer_sheet


def test_sampled_target_is_deterministic_given_seed():
    raw = _raw(seed=2)
    a = generate_self_censor_example(
        raw, beta1=1.0, delta=1.0, target_rate=0.3, rng=RNGState(seed=42)
    )
    b = generate_self_censor_example(
        raw, beta1=1.0, delta=1.0, target_rate=0.3, rng=RNGState(seed=42)
    )
    assert a.answer_sheet.target_col_idx == b.answer_sheet.target_col_idx
    assert a.answer_sheet.predictor_col_idx == b.answer_sheet.predictor_col_idx


# ---------- failure cases ----------

def test_rejects_bad_target_rate():
    raw = _raw(n=200)
    for bad in (0.0, 1.0, -0.1, 1.2):
        with pytest.raises(ValueError):
            generate_self_censor_example(
                raw, beta1=1.0, delta=1.0, target_rate=bad,
                rng=RNGState(seed=1), target_idx=0,
            )


def test_rejects_negative_delta_or_beta1():
    raw = _raw(n=200)
    with pytest.raises(ValueError):
        generate_self_censor_example(
            raw, beta1=-1.0, delta=1.0, target_rate=0.3, rng=RNGState(seed=1), target_idx=0
        )
    with pytest.raises(ValueError):
        generate_self_censor_example(
            raw, beta1=1.0, delta=-0.5, target_rate=0.3, rng=RNGState(seed=1), target_idx=0
        )


def test_rejects_too_few_columns():
    raw = RawDataset(
        data=np.random.default_rng(0).standard_normal((100, 1)).astype("float32"),
        feature_names=("only",),
        name="one_col",
    )
    with pytest.raises(ValueError):
        generate_self_censor_example(
            raw, beta1=1.0, delta=1.0, target_rate=0.3, rng=RNGState(seed=1)
        )


def test_rejects_constant_target_column():
    data = np.random.default_rng(0).standard_normal((200, 3)).astype("float32")
    data[:, 2] = 7.0
    raw = RawDataset(data=data, feature_names=("a", "b", "c"), name="const_target")
    with pytest.raises(ValueError):
        generate_self_censor_example(
            raw, beta1=1.0, delta=1.0, target_rate=0.3, rng=RNGState(seed=1), target_idx=2
        )


def test_rate_tolerance_failures():
    with pytest.raises(ValueError):
        rate_tolerance(0, 0.3)
    with pytest.raises(ValueError):
        rate_tolerance(100, 1.5)
    with pytest.raises(ValueError):
        rate_tolerance(100, 0.3, n_sigma=0.0)


# ---------- real catalog dataset ----------

def test_runs_on_real_survey_bfi():
    raw = create_default_catalog().load("survey_bfi")
    res = generate_self_censor_example(
        raw, beta1=1.0, delta=1.0, target_rate=0.25, rng=RNGState(seed=13)
    )
    s = res.answer_sheet
    assert s.source_name == "survey_bfi"
    assert 0 <= s.target_col_idx < s.d
    assert s.predictor_col_idx != s.target_col_idx
    assert s.target_col_name in raw.feature_names
    # small-n survey: realized rate within the n-aware band
    assert check_realized_rate(s)
    assert res.mask.shape == (raw.n, raw.d)
