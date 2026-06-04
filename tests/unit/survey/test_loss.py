"""Tests for lacuna.survey.loss — RPS, log-score, temperature fit."""

import math

import pytest
import torch

from lacuna.survey.loss import fit_temperature, log_score, rps_loss, uniform_rps

K = 7


def _onehot_logits(labels, k=K, scale=30.0):
    """Near-one-hot logits putting almost all mass on each label's bin."""
    logits = torch.zeros(len(labels), k)
    for i, b in enumerate(labels):
        logits[i, b] = scale
    return logits


# ---------- normal cases ----------

def test_perfect_prediction_zero_rps():
    labels = torch.tensor([0, 3, 6])
    loss = rps_loss(_onehot_logits(labels), labels, reduction="mean")
    assert loss.item() < 1e-5


def test_uniform_rps_matches_helper():
    labels = torch.arange(K)
    logits = torch.zeros(K, K)
    got = rps_loss(logits, labels, normalize=True).item()
    assert abs(got - uniform_rps(K)) < 1e-9


# ---------- the defining property: order-awareness ----------

def test_rps_penalizes_far_misses_more():
    true_bin = 5
    labels = torch.tensor([true_bin])
    near = rps_loss(_onehot_logits([4]), labels).item()
    mid = rps_loss(_onehot_logits([3]), labels).item()
    far = rps_loss(_onehot_logits([0]), labels).item()
    assert near < mid < far


def test_rps_is_symmetric_in_distance():
    labels = torch.tensor([3])
    below = rps_loss(_onehot_logits([1]), labels).item()
    above = rps_loss(_onehot_logits([5]), labels).item()
    assert abs(below - above) < 1e-5  # |3-1| == |5-3|


def test_properness_minimized_at_truth():
    """Expected RPS over a fixed true-distribution is minimized by predicting it."""
    true_p = torch.tensor([0.1, 0.0, 0.2, 0.4, 0.1, 0.1, 0.1])
    # predicting the true distribution
    pred_logits = torch.log(true_p.clamp(min=1e-9)).unsqueeze(0)
    # sample expected RPS analytically: average per-bin RPS weighted by true_p
    exp_at_truth = sum(
        true_p[b].item() * rps_loss(pred_logits, torch.tensor([b])).item() for b in range(K)
    )
    # a wrong (shifted) prediction
    wrong_logits = torch.log(true_p.flip(0).clamp(min=1e-9)).unsqueeze(0)
    exp_at_wrong = sum(
        true_p[b].item() * rps_loss(wrong_logits, torch.tensor([b])).item() for b in range(K)
    )
    assert exp_at_truth < exp_at_wrong


# ---------- gradient ----------

def test_rps_backprops_finite():
    logits = torch.randn(8, K, requires_grad=True)
    labels = torch.randint(0, K, (8,))
    loss = rps_loss(logits, labels)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


# ---------- edge ----------

def test_reduction_none_shape():
    labels = torch.tensor([0, 6])
    per = rps_loss(_onehot_logits(labels), labels, reduction="none")
    assert per.shape == (2,)


def test_boundary_bins_no_error():
    for b in (0, K - 1):
        labels = torch.tensor([b])
        rps_loss(torch.zeros(1, K), labels)  # must not raise


# ---------- log-score / temperature ----------

def test_log_score_uniform_is_log_k():
    labels = torch.tensor([0, 3])
    ls = log_score(torch.zeros(2, K), labels).item()
    assert abs(ls - math.log(K)) < 1e-5


def test_fit_temperature_inflates_overconfident_logits():
    # Overconfident but correct logits -> NLL minimized by T>1 only if miscalibrated.
    labels = torch.tensor([0, 1, 2, 3, 4, 5, 6])
    # half-right overconfident predictions create miscalibration
    logits = _onehot_logits([0, 1, 2, 3, 0, 1, 2], scale=10.0)  # 4/7 wrong, very confident
    t = fit_temperature(logits, labels)
    assert t > 1.0


# ---------- failure cases ----------

def test_rejects_non_2d_logits():
    with pytest.raises(ValueError):
        rps_loss(torch.zeros(K), torch.tensor([0]))


def test_rejects_label_out_of_range():
    with pytest.raises(ValueError):
        rps_loss(torch.zeros(1, K), torch.tensor([K]))


def test_rejects_float_labels():
    with pytest.raises(ValueError):
        rps_loss(torch.zeros(1, K), torch.tensor([0.0]))


def test_rejects_k_lt_2():
    with pytest.raises(ValueError):
        rps_loss(torch.zeros(1, 1), torch.tensor([0]))


def test_rejects_bad_reduction():
    with pytest.raises(ValueError):
        rps_loss(torch.zeros(1, K), torch.tensor([0]), reduction="sum")
