"""
lacuna.models.reconstruction

Reconstruction heads for mechanism classification.

CRITICAL FIX (2026-01-10):
--------------------------
The original implementation computed reconstruction errors on ARTIFICIALLY
masked cells (for self-supervised pretraining). This does NOT distinguish
MAR from MNAR because both mechanisms look similar under artificial masking.

The discriminative signal comes from errors on NATURALLY MISSING cells:
    - MAR: MARHead predicts well (cross-attention to observed values helps)
    - MNAR: MARHead predicts poorly (the missing value itself is informative)

This refactored module computes BOTH:
    1. `artificial_errors`: For self-supervised learning (training signal)
    2. `natural_errors`: For mechanism discrimination (classification signal)

The natural_errors are fed to the MoE gating network as the discriminative
feature that separates MAR from MNAR.

Usage:
    from lacuna.models.reconstruction import (
        ReconstructionHeads,
        create_reconstruction_heads,
    )

    heads = create_reconstruction_heads(hidden_dim=128)
    results = heads(token_repr, tokens, row_mask, col_mask, original_values,
                   reconstruction_mask, compute_natural_errors=True)

    # For self-supervised loss
    artificial_errors = heads.get_error_tensor(results)  # [B, n_heads]

    # For mechanism discrimination (feed to MoE)
    natural_errors = heads.get_natural_error_tensor(results)  # [B, n_heads]
"""

from lacuna.models.reconstruction.base import (
    BaseReconstructionHead,
    ReconstructionConfig,
)
from lacuna.models.reconstruction.heads import (
    MCARHead,
    MARHead,
    MNARSelfCensoringHead,
    HEAD_REGISTRY,
    create_head,
)
from lacuna.models.reconstruction.heads_container import (
    ExtendedReconstructionResult,
    ReconstructionHeads,
    create_reconstruction_heads,
)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Config
    "ReconstructionConfig",
    # Base class
    "BaseReconstructionHead",
    # Head implementations
    "MCARHead",
    "MARHead",
    "MNARSelfCensoringHead",
    # Registry
    "HEAD_REGISTRY",
    "create_head",
    # Container
    "ReconstructionHeads",
    "ExtendedReconstructionResult",
    "create_reconstruction_heads",
]
