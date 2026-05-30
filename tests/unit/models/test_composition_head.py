"""Tests for lacuna.models.composition_head — the Stage-C Dirichlet composition head (ADR-0007)."""

import numpy as np
import pytest
import torch

from lacuna.core.rng import RNGState
from lacuna.models.composition_head import (
    CompositionHead, composition_mean, cant_tell_mass, ensemble_alpha, prob_region,
)


# ---------------------------------------------------------------------------
# Head forward
# ---------------------------------------------------------------------------

def test_forward_shape_and_positive_concentration():
    head = CompositionHead(evidence_dim=64)
    alpha = head(torch.randn(8, 64))
    assert alpha.shape == (8, 3)
    assert bool((alpha >= 1.0).all())            # softplus + 1 => alpha_k >= 1
    assert torch.isfinite(alpha).all()


def test_direct_linear_variant():
    head = CompositionHead(evidence_dim=32, hidden_dim=None)
    alpha = head(torch.randn(4, 32))
    assert alpha.shape == (4, 3) and bool((alpha >= 1.0).all())


def test_head_is_differentiable():
    head = CompositionHead(evidence_dim=16)
    ev = torch.randn(5, 16, requires_grad=True)
    head(ev).sum().backward()
    assert ev.grad is not None and torch.isfinite(ev.grad).all()


# ---------------------------------------------------------------------------
# Queries on the Dirichlet
# ---------------------------------------------------------------------------

def test_composition_mean_on_simplex():
    alpha = torch.tensor([[2.0, 3.0, 5.0], [1.0, 1.0, 1.0]])
    m = composition_mean(alpha)
    assert torch.allclose(m.sum(-1), torch.ones(2), atol=1e-6)
    assert torch.allclose(m[0], torch.tensor([0.2, 0.3, 0.5]), atol=1e-6)


def test_cant_tell_mass_range_and_monotonicity():
    """Vacuity = K/alpha0: 1.0 at the uniform prior (no evidence), shrinks as evidence grows."""
    uniform = torch.tensor([[1.0, 1.0, 1.0]])
    confident = torch.tensor([[50.0, 50.0, 50.0]])
    assert torch.allclose(cant_tell_mass(uniform), torch.tensor([1.0]), atol=1e-6)
    assert float(cant_tell_mass(confident)) < 0.05
    # monotone: more total evidence => less can't-tell
    a = torch.tensor([[2.0, 2.0, 2.0], [10.0, 10.0, 10.0], [1.0, 1.0, 1.0]])
    ct = cant_tell_mass(a)
    assert ct[2] > ct[0] > ct[1]


def test_ensemble_alpha_sums_evidence():
    alphas = torch.stack([torch.tensor([[2.0, 3.0, 4.0]]), torch.tensor([[1.5, 1.0, 2.0]])])  # [2,1,3]
    ens = ensemble_alpha(alphas)
    # 1 + (1 + 2 + 3) per class evidence... evidence = alpha-1: [1,2,3]+[0.5,0,1] = [1.5,2,4]; +1
    assert torch.allclose(ens, torch.tensor([[2.5, 3.0, 5.0]]), atol=1e-6)
    # pooling reduces vacuity vs either member (more total evidence)
    assert float(cant_tell_mass(ens)) < float(cant_tell_mass(alphas[0]))


def test_prob_region_uniform_known_value():
    """Under the uniform Dirichlet(1,1,1), P(coord > 0.5) = (1-0.5)^2 = 0.25 (closed form)."""
    alpha = torch.tensor([[1.0, 1.0, 1.0]])
    p = prob_region(alpha, lambda s: s[:, 2] > 0.5, RNGState(seed=0), n_samples=20000)
    assert abs(float(p) - 0.25) < 0.02


def test_prob_region_confident_mass():
    """A Dirichlet concentrated on MNAR puts ~all mass on {f_MNAR >= 0.7}."""
    alpha = torch.tensor([[1.0, 1.0, 60.0]])
    p = prob_region(alpha, lambda s: s[:, 2] >= 0.7, RNGState(seed=1), n_samples=8000)
    assert float(p) > 0.95


def test_prob_region_deterministic():
    alpha = torch.tensor([[3.0, 4.0, 5.0]])
    p1 = prob_region(alpha, lambda s: s[:, 0] > 0.3, RNGState(seed=7), n_samples=3000)
    p2 = prob_region(alpha, lambda s: s[:, 0] > 0.3, RNGState(seed=7), n_samples=3000)
    assert torch.equal(p1, p2)


# ---------------------------------------------------------------------------
# Failure cases
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [torch.tensor([1.0, 2.0, 3.0]), torch.zeros(2, 1)])
def test_check_alpha_rejects_bad_shape(bad):
    with pytest.raises(ValueError, match="alpha must be"):
        composition_mean(bad)


def test_check_alpha_rejects_nonpositive():
    with pytest.raises(ValueError, match="positive"):
        cant_tell_mass(torch.tensor([[1.0, 0.0, 2.0]]))


def test_ensemble_alpha_rejects_bad_shape():
    with pytest.raises(ValueError, match=r"\[M, B, K\]"):
        ensemble_alpha(torch.ones(2, 3))


def test_prob_region_rejects_bad_predicate():
    alpha = torch.tensor([[2.0, 2.0, 2.0]])
    with pytest.raises(ValueError, match="predicate must return"):
        prob_region(alpha, lambda s: np.array([True, False]), RNGState(seed=0), n_samples=100)
