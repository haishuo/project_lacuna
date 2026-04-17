"""
lacuna.data.batching

Batch collation for Lacuna's semi-synthetic data pipeline.

Lacuna trains on real tabular X with synthetic missingness mechanisms —
see `lacuna.data.semisynthetic.SemiSyntheticDataLoader`. Pure-synthetic X
was removed deliberately: per Molenberghs, real missingness is
unidentifiable, so the mechanism must be synthetic; but the underlying X
has to be real for any downstream accuracy claim to mean anything. This
module used to host the pure-synthetic loader and its variants; those
have been deleted.

This file now contains only `collate_fn`, shared by all DataLoader users.
"""

import torch
from typing import List

from lacuna.core.types import TokenBatch


def collate_fn(batches: List[TokenBatch]) -> TokenBatch:
    """
    Collate multiple TokenBatch objects into a single batch.

    For use with PyTorch DataLoader when datasets are pre-tokenized.

    Args:
        batches: List of TokenBatch objects (typically batch_size=1 each).

    Returns:
        Combined TokenBatch with all samples.

    Example:
        >>> loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)
    """
    tokens = torch.cat([b.tokens for b in batches], dim=0)
    row_mask = torch.cat([b.row_mask for b in batches], dim=0)
    col_mask = torch.cat([b.col_mask for b in batches], dim=0)

    gen_ids = None
    if all(b.generator_ids is not None for b in batches):
        gen_ids = torch.cat([b.generator_ids for b in batches], dim=0)

    class_ids = None
    if all(b.class_ids is not None for b in batches):
        class_ids = torch.cat([b.class_ids for b in batches], dim=0)

    variant_ids = None
    if all(b.variant_ids is not None for b in batches):
        variant_ids = torch.cat([b.variant_ids for b in batches], dim=0)

    original_values = None
    if all(b.original_values is not None for b in batches):
        original_values = torch.cat([b.original_values for b in batches], dim=0)

    reconstruction_mask = None
    if all(b.reconstruction_mask is not None for b in batches):
        reconstruction_mask = torch.cat([b.reconstruction_mask for b in batches], dim=0)

    return TokenBatch(
        tokens=tokens,
        row_mask=row_mask,
        col_mask=col_mask,
        generator_ids=gen_ids,
        class_ids=class_ids,
        variant_ids=variant_ids,
        original_values=original_values,
        reconstruction_mask=reconstruction_mask,
    )
