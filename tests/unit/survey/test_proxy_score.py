"""Tests for lacuna.survey.proxy_score."""

import numpy as np
import pytest
import torch

from lacuna.data.catalog import create_default_catalog
from lacuna.data.ingestion import RawDataset
from lacuna.survey.proxy_score import (
    column_r2,
    proxy_score,
    r2_distribution,
    target_r2_table,
)


def _full(x):
    return x, torch.ones(x.shape, dtype=torch.bool)


# ---------- normal cases ----------

def test_perfectly_predictable_target_high_r2():
    g = np.random.default_rng(0)
    a = g.standard_normal(2000)
    b = g.standard_normal(2000)
    target = 2.0 * a - 3.0 * b  # exact linear combo
    x = torch.tensor(np.stack([a, b, target], axis=1), dtype=torch.float32)
    r2 = column_r2(*_full(x), target_idx=2)
    assert r2 > 0.98


def test_independent_target_low_r2():
    g = np.random.default_rng(1)
    x = torch.tensor(g.standard_normal((4000, 3)), dtype=torch.float32)
    r2 = column_r2(*_full(x), target_idx=2)
    assert r2 < 0.05


def test_proxy_score_is_one_minus_r2():
    g = np.random.default_rng(2)
    a = g.standard_normal(1500)
    target = a + 0.01 * g.standard_normal(1500)
    x = torch.tensor(np.stack([a, target], axis=1), dtype=torch.float32)
    r2 = column_r2(*_full(x), target_idx=1)
    assert proxy_score(*_full(x), target_idx=1) == pytest.approx(1.0 - r2)


def test_uses_only_observed_rows():
    g = np.random.default_rng(3)
    a = g.standard_normal(2000)
    target = a.copy()
    x = torch.tensor(np.stack([a, target], axis=1), dtype=torch.float32)
    r = torch.ones(2000, 2, dtype=torch.bool)
    r[:1000, 1] = False  # censor half the target
    r2 = column_r2(x, r, target_idx=1)
    assert r2 > 0.98  # still recoverable from the observed half


# ---------- table + distribution ----------

def test_target_r2_table_and_distribution_real():
    raw = create_default_catalog().load("survey_bfi")
    table = target_r2_table([raw])
    assert len(table) >= 2
    assert all(0.0 <= row["r2"] <= 1.0 for row in table)
    dist = r2_distribution(table)
    assert dist["min"] <= dist["median"] <= dist["max"]
    assert dist["n_targets"] == len(table)


# ---------- failure cases ----------

def test_constant_target_raises():
    x = torch.randn(200, 3)
    x[:, 2] = 4.0
    with pytest.raises(ValueError):
        column_r2(*_full(x), target_idx=2)


def test_bad_target_idx_raises():
    x = torch.randn(200, 3)
    with pytest.raises(ValueError):
        column_r2(*_full(x), target_idx=9)


def test_too_few_observed_rows_raises():
    x = torch.randn(200, 5)
    r = torch.ones(200, 5, dtype=torch.bool)
    r[3:, 0] = False  # only 3 observed target rows, < predictors+2
    with pytest.raises(ValueError):
        column_r2(x, r, target_idx=0)


def test_empty_distribution_raises():
    with pytest.raises(ValueError):
        r2_distribution([])
