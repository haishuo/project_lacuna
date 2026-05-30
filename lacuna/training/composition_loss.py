"""
lacuna.training.composition_loss

Evidential (Dirichlet) loss for the Stage-C composition head (ADR-0007).

Trains `CompositionHead`'s Dirichlet `Dir(alpha)` against a dataset's REALISED by-cell composition
`p*` (a soft point on the simplex, from the Stage-B generator's per-cell tags). Two terms, the
standard evidential-deep-learning recipe (Sensoy et al. 2018) generalised to a soft target:

  1. **Expected cross-entropy (Bayes risk).** `L_ce = Σ_k p*_k (ψ(alpha0) − ψ(alpha_k))` =
     `E_{p~Dir(alpha)}[ CE(p*, p) ]`. Minimising it pulls the Dirichlet MEAN toward `p*` and raises
     the concentration (confidence) — point accuracy.
  2. **KL regulariser toward the uniform Dirichlet.** `kl_weight · KL(Dir(alpha_tilde) || Dir(1))`
     with `alpha_tilde = 1 + (1 − p*) ⊙ (alpha − 1)` — the "misleading evidence" (concentration on
     classes the target does NOT support). Driving it toward the uniform prior makes the head retain
     uncertainty (vacuity) exactly when the observable footprint cannot pin the composition — the
     non-identifiable MAR↔MNAR part of the honest seam. This term is what produces CALIBRATION
     (the Stage-D headline); the caller anneals `kl_weight` from 0 so the head first learns to fit,
     then learns to be honest about what it cannot.

Both terms are scale-free in the composition denominator: the target is the by-cell FRACTION `p*`, not
the cell COUNTS — so confidence reflects the model's epistemic certainty given the footprint, not the
(huge) multinomial sample size of the cells. This keeps the head's uncertainty meaningful.

Deterministic (Coding Bible Rule 6) — pure tensor math, no RNG. Fails loud (Rule 1) on shape/contract
violations.
"""

import torch


def _validate(alpha: torch.Tensor, target: torch.Tensor) -> None:
    if alpha.shape != target.shape or alpha.dim() != 2:
        raise ValueError(f"alpha and target must be equal [B, K] 2-D tensors, "
                         f"got {tuple(alpha.shape)} and {tuple(target.shape)}")
    if not torch.isfinite(alpha).all() or bool((alpha <= 0).any()):
        raise ValueError("alpha must be finite and strictly positive")
    s = target.sum(dim=-1)
    if bool((target < -1e-6).any()) or not torch.allclose(s, torch.ones_like(s), atol=1e-3):
        raise ValueError("target rows must be non-negative and sum to 1 (a composition on the simplex)")


def kl_dirichlet_uniform(alpha: torch.Tensor) -> torch.Tensor:
    """KL( Dir(alpha) || Dir(1) ) per row [B], closed form. Dir(1) = uniform over the simplex."""
    k = alpha.shape[-1]
    a0 = alpha.sum(dim=-1)
    term = (torch.lgamma(a0) - torch.lgamma(alpha).sum(dim=-1)
            - torch.lgamma(torch.tensor(float(k), device=alpha.device, dtype=alpha.dtype)))
    digamma_diff = torch.digamma(alpha) - torch.digamma(a0).unsqueeze(-1)
    return term + ((alpha - 1.0) * digamma_diff).sum(dim=-1)


def expected_cross_entropy(alpha: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Bayes-risk expected CE per row [B]: Σ_k target_k (ψ(alpha0) − ψ(alpha_k))."""
    a0 = alpha.sum(dim=-1, keepdim=True)
    return (target * (torch.digamma(a0) - torch.digamma(alpha))).sum(dim=-1)


def dirichlet_edl_loss(
    alpha: torch.Tensor,
    target: torch.Tensor,
    *,
    kl_weight: float = 0.0,
) -> torch.Tensor:
    """Evidential composition loss: expected-CE Bayes risk + annealed uniform-KL regulariser.

    Args:
        alpha: [B, K] Dirichlet concentrations from the head (alpha_k > 0).
        target: [B, K] realised composition (non-negative, rows sum to 1).
        kl_weight: weight on the misleading-evidence KL term (anneal from 0; >= 0).

    Returns:
        Scalar loss (mean over the batch).

    Raises:
        ValueError: on shape / contract violations or kl_weight < 0.
    """
    _validate(alpha, target)
    if kl_weight < 0.0:
        raise ValueError(f"kl_weight must be >= 0, got {kl_weight}")

    ce = expected_cross_entropy(alpha, target).mean()
    if kl_weight == 0.0:
        return ce
    alpha_tilde = 1.0 + (1.0 - target) * (alpha - 1.0)   # evidence the target does not support
    kl = kl_dirichlet_uniform(alpha_tilde).mean()
    return ce + kl_weight * kl
