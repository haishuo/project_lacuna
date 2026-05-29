"""Tests for lacuna.models.column_head — per-column readout head (Stage 1)."""

import pytest
import torch

from lacuna.models.column_head import (
    ColumnReadoutHead,
    per_column_posterior,
    masked_per_column_ce,
)


def _head(hidden_dim=32):
    torch.manual_seed(0)
    h = ColumnReadoutHead(hidden_dim=hidden_dim, n_classes=3, head_hidden=16, dropout=0.0)
    return h.eval()  # eval -> no dropout, deterministic forward


def _inputs(B=2, R=5, C=4, H=32, seed=1):
    g = torch.Generator().manual_seed(seed)
    token_repr = torch.randn(B, R, C, H, generator=g)
    row_mask = torch.ones(B, R, dtype=torch.bool)
    col_mask = torch.ones(B, C, dtype=torch.bool)
    return token_repr, row_mask, col_mask


# ---------------------------------------------------------------------------
# Normal cases
# ---------------------------------------------------------------------------

def test_forward_shape():
    head = _head()
    tr, rm, cm = _inputs()
    logits = head(tr, rm, cm)
    assert logits.shape == (2, 4, 3)


def test_deterministic_in_eval():
    head = _head()
    tr, rm, cm = _inputs()
    a = head(tr, rm, cm)
    b = head(tr, rm, cm)
    assert torch.equal(a, b)


def test_padding_rows_excluded_from_pool():
    """Logits for valid columns must not change when padded rows hold arbitrary garbage."""
    head = _head()
    tr, rm, cm = _inputs(B=1, R=6, C=3)
    rm[:, 4:] = False  # last two rows are padding
    base = head(tr, rm, cm)
    tr2 = tr.clone()
    tr2[:, 4:] = 1e3  # poison the padded rows
    poisoned = head(tr2, rm, cm)
    assert torch.allclose(base, poisoned, atol=1e-5)


def test_per_column_independence():
    """Changing one column's representations must not change another column's logits."""
    head = _head()
    tr, rm, cm = _inputs(B=1, R=5, C=4)
    base = head(tr, rm, cm)
    tr2 = tr.clone()
    tr2[:, :, 1, :] += 5.0  # perturb only column 1
    perturbed = head(tr2, rm, cm)
    # columns 0, 2, 3 unchanged; column 1 changed
    for c in (0, 2, 3):
        assert torch.allclose(base[:, c], perturbed[:, c], atol=1e-5)
    assert not torch.allclose(base[:, 1], perturbed[:, 1], atol=1e-4)


def test_padding_columns_zeroed():
    head = _head()
    tr, rm, cm = _inputs(B=1, R=5, C=4)
    cm[:, 3:] = False  # column 3 is padding
    logits = head(tr, rm, cm)
    assert torch.all(logits[:, 3] == 0.0)


def test_per_column_posterior_sums_to_one():
    head = _head()
    tr, rm, cm = _inputs()
    probs = per_column_posterior(head(tr, rm, cm))
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2, 4), atol=1e-5)


def test_masked_ce_ignores_unsupervised():
    """CE must depend only on supervised columns."""
    torch.manual_seed(0)
    logits = torch.randn(1, 4, 3)
    labels = torch.tensor([[0, 1, 2, 0]])
    mask = torch.tensor([[True, True, False, False]])
    ce = masked_per_column_ce(logits, labels, mask)
    # Changing labels/logits on an UNsupervised column leaves CE unchanged.
    logits2 = logits.clone(); logits2[:, 2:] += 9.0
    labels2 = labels.clone(); labels2[:, 2:] = 1
    ce2 = masked_per_column_ce(logits2, labels2, mask)
    assert torch.allclose(ce, ce2)


def test_gradients_flow():
    head = ColumnReadoutHead(hidden_dim=16, head_hidden=8, dropout=0.0)
    tr = torch.randn(2, 4, 3, 16, requires_grad=True)
    rm = torch.ones(2, 4, dtype=torch.bool)
    cm = torch.ones(2, 3, dtype=torch.bool)
    logits = head(tr, rm, cm)
    loss = masked_per_column_ce(
        logits, torch.zeros(2, 3, dtype=torch.long), torch.ones(2, 3, dtype=torch.bool)
    )
    loss.backward()
    assert tr.grad is not None and torch.isfinite(tr.grad).all()


# ---------------------------------------------------------------------------
# Failure cases (fail loud — Coding Bible Rule 1)
# ---------------------------------------------------------------------------

def test_wrong_token_repr_rank_raises():
    head = _head()
    with pytest.raises(ValueError, match="token_repr must be"):
        head(torch.randn(2, 4, 32), torch.ones(2, 4, dtype=torch.bool), torch.ones(2, 4, dtype=torch.bool))


def test_hidden_dim_mismatch_raises():
    head = _head(hidden_dim=32)
    tr, rm, cm = _inputs(H=16)  # wrong H
    with pytest.raises(ValueError, match="hidden dim"):
        head(tr, rm, cm)


def test_ce_no_supervised_columns_raises():
    logits = torch.randn(1, 3, 3)
    labels = torch.zeros(1, 3, dtype=torch.long)
    mask = torch.zeros(1, 3, dtype=torch.bool)
    with pytest.raises(ValueError, match="No supervised columns"):
        masked_per_column_ce(logits, labels, mask)


def test_bad_n_classes_raises():
    with pytest.raises(ValueError, match="n_classes"):
        ColumnReadoutHead(hidden_dim=16, n_classes=1)


# ---------------------------------------------------------------------------
# Extra per-column features (Stage 2)
# ---------------------------------------------------------------------------

def _head_extra(hidden_dim=32, n_extra=2):
    torch.manual_seed(0)
    return ColumnReadoutHead(hidden_dim=hidden_dim, n_extra_features=n_extra,
                             head_hidden=16, dropout=0.0).eval()


def test_extra_features_shape():
    head = _head_extra(n_extra=2)
    tr, rm, cm = _inputs(B=2, C=4, H=32)
    extra = torch.randn(2, 4, 2)
    assert head(tr, rm, cm, extra).shape == (2, 4, 3)


def test_extra_features_change_logits():
    head = _head_extra(n_extra=2)
    tr, rm, cm = _inputs(B=2, C=4, H=32)
    e1 = torch.zeros(2, 4, 2)
    e2 = torch.ones(2, 4, 2) * 3.0
    a = head(tr, rm, cm, e1)
    b = head(tr, rm, cm, e2)
    # valid columns' logits must respond to the extra features
    assert not torch.allclose(a[cm], b[cm], atol=1e-4)


def test_missing_extra_raises():
    head = _head_extra(n_extra=2)
    tr, rm, cm = _inputs(B=2, C=4, H=32)
    with pytest.raises(ValueError, match="extra_features was not provided"):
        head(tr, rm, cm)


def test_extra_wrong_shape_raises():
    head = _head_extra(n_extra=2)
    tr, rm, cm = _inputs(B=2, C=4, H=32)
    with pytest.raises(ValueError, match="extra_features shape"):
        head(tr, rm, cm, torch.randn(2, 4, 3))  # n_extra mismatch


def test_extra_provided_when_zero_raises():
    head = _head()  # n_extra_features=0
    tr, rm, cm = _inputs(B=2, C=4, H=32)
    with pytest.raises(ValueError, match="n_extra_features=0"):
        head(tr, rm, cm, torch.randn(2, 4, 2))


def test_bad_n_extra_raises():
    with pytest.raises(ValueError, match="n_extra_features"):
        ColumnReadoutHead(hidden_dim=16, n_extra_features=-1)
