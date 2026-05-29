"""
lacuna.models.column_head

Per-column mechanism readout head (Stage 1, ADR-0006).

Where the dataset-level model pools the encoder's representations all the way down to one
evidence vector and emits one posterior, this head reads out a SEPARATE MCAR/MAR/MNAR
posterior for EACH column. It consumes the encoder's per-token representations
(`LacunaEncoder.get_token_representations` → [B, R, C, H]), which have already passed
through the row-wise transformer — so each column's representation carries cross-column
context. That matters: a column cannot be classified in isolation (MAR is defined by
dependence on OTHER columns), and the team rejected per-column *encoding* for exactly this
reason (docs/notebooklm/02). Here the encoder stays cross-column; only the READOUT is
per-column.

Pooling over rows is a masked MEAN (deterministic, Coding Bible Rule 6). Attention pooling
is a possible later upgrade; mean is the simplest first choice for the Stage 1 "can it read
out per-column at all?" question.

Contract:
    forward(token_repr [B,R,C,H], row_mask [B,R] bool, col_mask [B,C] bool) -> logits [B,C,3]
Logits for padding columns are produced but meaningless; the caller masks them out of the
loss/metrics via the supervision mask (non-OBSERVED, non-padding columns).
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ColumnReadoutHead(nn.Module):
    """Read a per-column MCAR/MAR/MNAR posterior off the encoder's token representations."""

    def __init__(
        self,
        hidden_dim: int,
        n_classes: int = 3,
        head_hidden: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if n_classes < 2:
            raise ValueError(f"n_classes must be >= 2, got {n_classes}")

        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, n_classes),
        )

    def forward(
        self,
        token_repr: torch.Tensor,  # [B, R, C, H]
        row_mask: torch.Tensor,    # [B, R] bool (True = real row)
        col_mask: torch.Tensor,    # [B, C] bool (True = real column)
    ) -> torch.Tensor:
        """Return per-column logits [B, C, n_classes].

        Pooling is a row-masked mean over valid rows, computed per column. Padding columns
        are zeroed (their logits are not meaningful; mask them downstream).
        """
        if token_repr.dim() != 4:
            raise ValueError(f"token_repr must be [B,R,C,H], got shape {tuple(token_repr.shape)}")
        B, R, C, H = token_repr.shape
        if H != self.hidden_dim:
            raise ValueError(f"token_repr hidden dim {H} != head hidden_dim {self.hidden_dim}")
        if row_mask.shape != (B, R):
            raise ValueError(f"row_mask shape {tuple(row_mask.shape)} != ({B}, {R})")
        if col_mask.shape != (B, C):
            raise ValueError(f"col_mask shape {tuple(col_mask.shape)} != ({B}, {C})")

        # Masked mean over valid rows, per (batch, column).
        rm = row_mask.to(token_repr.dtype).unsqueeze(-1).unsqueeze(-1)  # [B, R, 1, 1]
        summed = (token_repr * rm).sum(dim=1)                          # [B, C, H]
        count = row_mask.to(token_repr.dtype).sum(dim=1).clamp(min=1.0).view(B, 1, 1)
        pooled = summed / count                                        # [B, C, H]

        logits = self.mlp(pooled)                                      # [B, C, n_classes]

        # Zero out padding columns so they cannot leak into any unmasked downstream reduction.
        logits = logits * col_mask.to(logits.dtype).unsqueeze(-1)
        return logits


def per_column_posterior(logits: torch.Tensor) -> torch.Tensor:
    """Softmax over the class dimension of per-column logits [B, C, n_classes]."""
    if logits.dim() != 3:
        raise ValueError(f"logits must be [B, C, n_classes], got shape {tuple(logits.shape)}")
    return F.softmax(logits, dim=-1)


def masked_per_column_ce(
    logits: torch.Tensor,            # [B, C, n_classes]
    labels: torch.Tensor,            # [B, C] long in [0, n_classes)
    supervision_mask: torch.Tensor,  # [B, C] bool (True = column has a known mechanism)
) -> torch.Tensor:
    """Mean cross-entropy over supervised columns only.

    Columns that are padding or fully observed (no mechanism to predict) are excluded via
    `supervision_mask`. Fails loud if no column is supervised in the batch (Rule 1) — a silent
    zero loss would mask a broken data path.
    """
    if logits.dim() != 3:
        raise ValueError(f"logits must be [B, C, n_classes], got {tuple(logits.shape)}")
    B, C, K = logits.shape
    if labels.shape != (B, C) or supervision_mask.shape != (B, C):
        raise ValueError(
            f"labels/supervision_mask must be [B={B}, C={C}]; got "
            f"{tuple(labels.shape)} / {tuple(supervision_mask.shape)}"
        )
    flat_logits = logits.reshape(B * C, K)
    flat_labels = labels.reshape(B * C)
    flat_mask = supervision_mask.reshape(B * C)

    if not bool(flat_mask.any()):
        raise ValueError("No supervised columns in batch — check the mixed-label data path.")

    # Per-element CE, then average over supervised positions only.
    ce = F.cross_entropy(flat_logits[flat_mask], flat_labels[flat_mask], reduction="mean")
    return ce
