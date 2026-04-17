"""
Tests for lacuna.data.batching

Only `collate_fn` remains in this module — the synthetic-X data loaders
were removed. Tests cover:
    Normal cases:
        - Concatenation across batches for all tensor fields.
        - Optional-field propagation (all present → stacked; any missing → None).
    Edge cases:
        - Single-batch input returns equivalent shapes.
        - Heterogeneous optional presence drops the field cleanly.
"""

import pytest
import torch

from lacuna.core.types import TokenBatch
from lacuna.data.batching import collate_fn
from lacuna.data.tokenization import TOKEN_DIM


def _make_batch(
    *,
    batch_size: int = 1,
    max_rows: int = 4,
    max_cols: int = 3,
    with_generator_ids: bool = True,
    with_class_ids: bool = True,
    with_variant_ids: bool = False,
    with_original_values: bool = False,
    with_reconstruction_mask: bool = False,
    seed: int = 0,
) -> TokenBatch:
    """Build a minimal valid TokenBatch for testing."""
    g = torch.Generator().manual_seed(seed)
    tokens = torch.randn(batch_size, max_rows, max_cols, TOKEN_DIM, generator=g)
    row_mask = torch.ones(batch_size, max_rows, dtype=torch.bool)
    col_mask = torch.ones(batch_size, max_cols, dtype=torch.bool)
    return TokenBatch(
        tokens=tokens,
        row_mask=row_mask,
        col_mask=col_mask,
        generator_ids=torch.zeros(batch_size, dtype=torch.long) if with_generator_ids else None,
        class_ids=torch.zeros(batch_size, dtype=torch.long) if with_class_ids else None,
        variant_ids=torch.zeros(batch_size, dtype=torch.long) if with_variant_ids else None,
        original_values=torch.zeros(batch_size, max_rows, max_cols) if with_original_values else None,
        reconstruction_mask=torch.zeros(batch_size, max_rows, max_cols, dtype=torch.bool)
            if with_reconstruction_mask else None,
    )


def test_collate_concatenates_core_tensors():
    b1 = _make_batch(batch_size=2, seed=1)
    b2 = _make_batch(batch_size=3, seed=2)
    b3 = _make_batch(batch_size=1, seed=3)
    out = collate_fn([b1, b2, b3])

    assert out.tokens.shape[0] == 6
    assert out.row_mask.shape[0] == 6
    assert out.col_mask.shape[0] == 6
    # Contents preserved in order.
    assert torch.equal(out.tokens[:2], b1.tokens)
    assert torch.equal(out.tokens[2:5], b2.tokens)
    assert torch.equal(out.tokens[5:], b3.tokens)


def test_collate_propagates_generator_and_class_ids_when_all_present():
    b1 = _make_batch(batch_size=2, with_generator_ids=True, with_class_ids=True)
    b2 = _make_batch(batch_size=1, with_generator_ids=True, with_class_ids=True)
    out = collate_fn([b1, b2])
    assert out.generator_ids is not None and out.generator_ids.shape[0] == 3
    assert out.class_ids is not None and out.class_ids.shape[0] == 3


def test_collate_drops_optional_field_if_any_batch_missing_it():
    b1 = _make_batch(with_generator_ids=True)
    b2 = _make_batch(with_generator_ids=False)
    out = collate_fn([b1, b2])
    assert out.generator_ids is None


def test_collate_optional_original_values_and_reconstruction_mask():
    b1 = _make_batch(with_original_values=True, with_reconstruction_mask=True)
    b2 = _make_batch(with_original_values=True, with_reconstruction_mask=True)
    out = collate_fn([b1, b2])
    assert out.original_values is not None
    assert out.reconstruction_mask is not None

    b3 = _make_batch(with_original_values=False, with_reconstruction_mask=True)
    out2 = collate_fn([b1, b3])
    assert out2.original_values is None
    assert out2.reconstruction_mask is not None


def test_collate_single_batch_input_is_passthrough_shape():
    b1 = _make_batch(batch_size=2)
    out = collate_fn([b1])
    assert out.tokens.shape == b1.tokens.shape
    assert torch.equal(out.tokens, b1.tokens)


def test_collate_variant_ids_round_trip():
    b1 = _make_batch(batch_size=2, with_variant_ids=True)
    b2 = _make_batch(batch_size=1, with_variant_ids=True)
    out = collate_fn([b1, b2])
    assert out.variant_ids is not None
    assert out.variant_ids.shape[0] == 3
