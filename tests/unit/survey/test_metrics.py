"""Tests for lacuna.survey.metrics."""

import pytest
import torch

from lacuna.survey.delta_bins import NUM_BINS
from lacuna.survey import metrics as M

K = NUM_BINS


def _onehot(labels, k=K):
    p = torch.zeros(len(labels), k)
    p[torch.arange(len(labels)), torch.tensor(labels)] = 1.0
    return p


# ---------- bin centers / E[delta] ----------

def test_bin_centers_shape_and_zero_bin():
    centers = M.bin_centers()
    assert centers.shape == (K,)
    assert centers[0].item() == 0.0
    assert torch.all(centers[1:] > 0)


def test_expected_delta_onehot_recovers_center():
    probs = _onehot([0, 3, 6])
    centers = M.bin_centers()
    ed = M.expected_delta(probs)
    assert torch.allclose(ed, centers[torch.tensor([0, 3, 6])], atol=1e-6)


def test_e_delta_error_zero_when_perfect():
    # true δ at the bin centers => zero error for a one-hot predictor at those bins
    centers = M.bin_centers()
    probs = _onehot([1, 4])
    true_delta = centers[torch.tensor([1, 4])]
    err = M.e_delta_error(probs, true_delta)
    assert err["mae"] < 1e-6 and err["rmse"] < 1e-6


# ---------- accuracy ----------

def test_bin_and_adjacent_accuracy():
    probs = _onehot([2, 2, 2])
    labels = torch.tensor([2, 3, 5])
    assert M.bin_accuracy(probs, labels) == pytest.approx(1 / 3)
    assert M.adjacent_accuracy(probs, labels) == pytest.approx(2 / 3)  # bins 2,3 within ±1


def test_p_delta_zero():
    probs = torch.tensor([[1.0, 0, 0, 0, 0, 0, 0], [0.0, 0, 0, 0, 0, 0, 1.0]])
    assert M.p_delta_zero(probs) == pytest.approx(0.5)


# ---------- coverage ----------

def test_coverage_full_interval_is_one():
    probs = torch.softmax(torch.randn(20, K), dim=-1)
    labels = torch.randint(0, K, (20,))
    # a 99.9% interval should essentially always contain the truth
    assert M.interval_coverage(probs, labels, 0.999) == pytest.approx(1.0)


def test_confident_correct_has_high_coverage():
    labels = torch.tensor([0, 3, 6, 2])
    probs = _onehot(labels.tolist())
    assert M.interval_coverage(probs, labels, 0.8) == pytest.approx(1.0)


def test_coverage_table_keys():
    probs = torch.softmax(torch.randn(10, K), dim=-1)
    labels = torch.randint(0, K, (10,))
    tab = M.coverage_table(probs, labels)
    assert set(tab.keys()) == {"50", "80", "90"}


# ---------- ece wiring ----------

def test_ece_runs():
    probs = torch.softmax(torch.randn(30, K), dim=-1)
    labels = torch.randint(0, K, (30,))
    out = M.ece(probs, labels)
    assert "ece" in out and 0.0 <= out["ece"] <= 1.0


# ---------- failure cases ----------

def test_expected_delta_rejects_wrong_k():
    with pytest.raises(ValueError):
        M.expected_delta(torch.rand(3, K + 1))


def test_coverage_rejects_bad_nominal():
    with pytest.raises(ValueError):
        M.interval_coverage(torch.rand(3, K), torch.zeros(3, dtype=torch.long), 1.5)
