"""Tests for lacuna.training.composition_loss — the Stage-C evidential Dirichlet loss (ADR-0007)."""

import pytest
import torch

from lacuna.models.composition_head import composition_mean, cant_tell_mass
from lacuna.training.composition_loss import (
    kl_dirichlet_uniform, expected_cross_entropy, dirichlet_edl_loss,
)


# ---------------------------------------------------------------------------
# KL to the uniform Dirichlet
# ---------------------------------------------------------------------------

def test_kl_zero_at_uniform():
    assert torch.allclose(kl_dirichlet_uniform(torch.ones(4, 3)), torch.zeros(4), atol=1e-5)


def test_kl_positive_when_concentrated():
    assert float(kl_dirichlet_uniform(torch.tensor([[20.0, 20.0, 20.0]]))) > 0.0
    # more concentration => larger KL from uniform
    a = kl_dirichlet_uniform(torch.tensor([[5.0, 5.0, 5.0]]))
    b = kl_dirichlet_uniform(torch.tensor([[50.0, 50.0, 50.0]]))
    assert float(b) > float(a)


# ---------------------------------------------------------------------------
# Expected cross-entropy data term
# ---------------------------------------------------------------------------

def test_expected_ce_lower_when_mean_matches_target():
    target = torch.tensor([[0.2, 0.3, 0.5]])
    aligned = torch.tensor([[20.0, 30.0, 50.0]])      # mean == target, concentrated
    misaligned = torch.tensor([[50.0, 30.0, 20.0]])   # mean wrong
    assert float(expected_cross_entropy(aligned, target)) < float(expected_cross_entropy(misaligned, target))


def test_loss_decreases_under_optimization():
    """Functional check: minimising the EDL loss drives the Dirichlet mean to the target and lowers
    the loss (the head learns the composition)."""
    target = torch.tensor([[0.15, 0.25, 0.60]])
    raw = torch.zeros(1, 3, requires_grad=True)        # alpha = softplus(raw)+1 = uniform-ish start
    opt = torch.optim.Adam([raw], lr=0.05)
    def alpha():
        return torch.nn.functional.softplus(raw) + 1.0
    with torch.no_grad():
        start = float(dirichlet_edl_loss(alpha(), target))
    for _ in range(400):
        opt.zero_grad()
        loss = dirichlet_edl_loss(alpha(), target, kl_weight=0.0)
        loss.backward(); opt.step()
    with torch.no_grad():
        end = float(dirichlet_edl_loss(alpha(), target))
    assert end < start
    # the mean moves decisively from the uniform start (L1 ~0.53 to this target) toward it
    end_l1 = float((composition_mean(alpha()).detach() - target).abs().sum())
    assert end_l1 < 0.10, f"mean did not converge to target (L1 {end_l1:.3f})"


def test_kl_term_keeps_uncertainty_higher():
    """With the KL regulariser on, the fitted head retains more vacuity (less overconfidence) than
    with it off — the mechanism behind calibration."""
    target = torch.tensor([[0.15, 0.25, 0.60]])
    def fit(kl_weight):
        raw = torch.zeros(1, 3, requires_grad=True)
        opt = torch.optim.Adam([raw], lr=0.05)
        for _ in range(400):
            opt.zero_grad()
            a = torch.nn.functional.softplus(raw) + 1.0
            dirichlet_edl_loss(a, target, kl_weight=kl_weight).backward(); opt.step()
        return torch.nn.functional.softplus(raw).detach() + 1.0
    ct_off = float(cant_tell_mass(fit(0.0)))
    ct_on = float(cant_tell_mass(fit(0.5)))
    assert ct_on > ct_off


def test_gradients_finite():
    head_alpha = (torch.rand(8, 3) + 1.0).requires_grad_(True)   # leaf tensor
    target = torch.softmax(torch.randn(8, 3), dim=-1)
    dirichlet_edl_loss(head_alpha, target, kl_weight=0.3).backward()
    assert head_alpha.grad is not None and torch.isfinite(head_alpha.grad).all()


# ---------------------------------------------------------------------------
# Failure cases
# ---------------------------------------------------------------------------

def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="equal"):
        dirichlet_edl_loss(torch.ones(4, 3), torch.ones(4, 2) / 2)


def test_nonpositive_alpha_raises():
    t = torch.tensor([[0.5, 0.5, 0.0]])
    with pytest.raises(ValueError, match="positive"):
        dirichlet_edl_loss(torch.tensor([[1.0, 0.0, 1.0]]), t)


def test_target_not_simplex_raises():
    with pytest.raises(ValueError, match="sum to 1"):
        dirichlet_edl_loss(torch.ones(1, 3), torch.tensor([[0.5, 0.5, 0.5]]))


def test_negative_kl_weight_raises():
    t = torch.tensor([[0.34, 0.33, 0.33]])
    with pytest.raises(ValueError, match="kl_weight"):
        dirichlet_edl_loss(torch.ones(1, 3), t, kl_weight=-0.1)
