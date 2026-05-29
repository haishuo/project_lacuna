"""
lacuna.models.column_recon_features

Per-column reconstruction-error features for the Stage 2 column head (ADR-0006).

Stage 1 found that the per-column head, reading only encoder representations, recovers MAR but
not MNAR — because the self-censoring (MNAR) signal lives in the reconstruction-error pathway,
which the dataset-level gate consumes but the column head did not. This module computes that
signal PER COLUMN: for each reconstruction head, the mean squared error on a column's naturally
missing cells (vs the batch's `original_values`, exactly as the dataset gate's natural-error
signal is defined), → [B, C, n_heads]. log1p is applied for scale stability so the downstream
linear layer can form log-ratios (the discriminative quantity the dataset gate uses).

This mirrors `ReconstructionHeads.get_natural_error_tensor` ([B, n_heads], per-sample) but keeps
the column axis. Deterministic (no randomness).
"""

from typing import Dict

import torch

from lacuna.data.tokenization import IDX_OBSERVED
from lacuna.models.reconstruction.heads_container import ReconstructionHeads
from lacuna.models.reconstruction.base import ReconstructionConfig


def per_column_natural_error(
    recon_heads: ReconstructionHeads,
    token_repr: torch.Tensor,       # [B, R, C, H]
    tokens: torch.Tensor,           # [B, R, C, TOKEN_DIM]
    row_mask: torch.Tensor,         # [B, R] bool
    col_mask: torch.Tensor,         # [B, C] bool
    original_values: torch.Tensor,  # [B, R, C] reconstruction targets
) -> torch.Tensor:
    """Per-column, per-head MSE on each column's naturally-missing cells → [B, C, n_heads].

    A column with no naturally-missing cells (fully observed / padding) gets 0 for every head.
    """
    B, R, C, _ = token_repr.shape
    is_observed = tokens[..., IDX_OBSERVED] > 0.5                  # [B, R, C]
    valid = row_mask.unsqueeze(-1) & col_mask.unsqueeze(1)         # [B, R, C]
    natural_missing = (valid & ~is_observed).to(token_repr.dtype)  # [B, R, C]
    count = natural_missing.sum(dim=1).clamp(min=1.0)             # [B, C] (rows per column)

    # Public API: run all heads, pull their per-cell predictions.
    results = recon_heads.forward(
        token_repr, tokens, row_mask, col_mask,
        original_values=original_values, reconstruction_mask=None,
        compute_natural_errors=False,
    )
    preds: Dict[str, torch.Tensor] = recon_heads.get_predictions_dict(results)

    errs = []
    for name in recon_heads.head_names:
        se = ((preds[name] - original_values) ** 2) * natural_missing  # [B, R, C]
        errs.append(se.sum(dim=1) / count)                             # [B, C]
    return torch.stack(errs, dim=-1)                                   # [B, C, n_heads]


def per_column_recon_features(
    recon_heads: ReconstructionHeads,
    token_repr: torch.Tensor,
    tokens: torch.Tensor,
    row_mask: torch.Tensor,
    col_mask: torch.Tensor,
    original_values: torch.Tensor,
) -> torch.Tensor:
    """log1p of the per-column natural error per head → [B, C, n_heads]. Stage 2 head input."""
    return torch.log1p(
        per_column_natural_error(recon_heads, token_repr, tokens, row_mask, col_mask, original_values)
    )


def load_reconstruction_heads_from_baseline(
    checkpoint: str,
    *,
    hidden_dim: int,
    head_hidden_dim: int = 64,
    n_head_layers: int = 2,
    dropout: float = 0.1,
    mnar_variants=None,
    device: str = "cpu",
) -> ReconstructionHeads:
    """Build a ReconstructionHeads container and load the baseline's `reconstruction.*` weights."""
    if mnar_variants is None:
        mnar_variants = ["self_censoring"]
    sd = torch.load(checkpoint, map_location="cpu", weights_only=False)["model_state"]
    keys = {k[len("reconstruction."):]: v for k, v in sd.items() if k.startswith("reconstruction.")}
    if not keys:
        raise RuntimeError(f"No 'reconstruction.*' weights in {checkpoint}")
    config = ReconstructionConfig(
        hidden_dim=hidden_dim, head_hidden_dim=head_hidden_dim,
        n_head_layers=n_head_layers, dropout=dropout, mnar_variants=mnar_variants,
    )
    rh = ReconstructionHeads(config)
    rh.load_state_dict(keys, strict=True)  # fail loud on mismatch
    return rh.to(device)
