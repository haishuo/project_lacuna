"""Tests for lacuna.survey.distributional_stream — learned ECDF / order-statistic pooling.

Covers: masked-quantile correctness vs torch.quantile, masking, single-row degeneracy, ordering,
max, gradient flow (differentiability), determinism, fail-loud contracts, and the RepECDFPooling
module's dims/backprop. These are the §7 distributional-stream pre-run checks (contract + determinism
+ differentiability + edge), separate from the corpus leakage gate.
"""

import pytest
import torch

from lacuna.core.rng import RNGState
from lacuna.survey.distributional_stream import (
    N_QUANTILES,
    QUANTILE_LEVELS,
    RepECDFPooling,
    masked_quantile_pool,
)
from lacuna.survey.delta_head import init_parameters_


# ---- masked_quantile_pool ----

def test_matches_torch_quantile_full_mask():
    """With every row valid, the pooled quantiles match torch.quantile (linear interp)."""
    torch.manual_seed(0)
    values = torch.randn(5, 64, 3)
    mask = torch.ones(5, 64, dtype=torch.bool)
    out = masked_quantile_pool(values, mask)  # [5, 3, Q+1]
    assert out.shape == (5, 3, N_QUANTILES + 1)
    for li, q in enumerate(QUANTILE_LEVELS):
        ref = torch.quantile(values, q, dim=1)  # [5, 3]
        assert torch.allclose(out[:, :, li], ref, atol=1e-5), f"quantile {q} mismatch"
    # last stat is the max
    assert torch.allclose(out[:, :, -1], values.max(dim=1).values, atol=1e-6)


def test_masking_ignores_invalid_rows():
    """Pooling over a masked batch equals pooling over the valid subset."""
    torch.manual_seed(1)
    full = torch.randn(1, 40, 2)
    mask = torch.zeros(1, 40, dtype=torch.bool)
    mask[0, :25] = True  # only first 25 rows valid
    out = masked_quantile_pool(full, mask)
    ref = masked_quantile_pool(full[:, :25], torch.ones(1, 25, dtype=torch.bool))
    assert torch.allclose(out, ref, atol=1e-6)


def test_single_valid_row_is_constant():
    """One valid row ⇒ every quantile and the max equal that row's value."""
    values = torch.tensor([[[3.0, -1.0], [9.0, 9.0], [9.0, 9.0]]])  # [1, 3, 2]
    mask = torch.tensor([[True, False, False]])
    out = masked_quantile_pool(values, mask)  # [1, 2, Q+1]
    assert torch.allclose(out[0, 0], torch.full((N_QUANTILES + 1,), 3.0))
    assert torch.allclose(out[0, 1], torch.full((N_QUANTILES + 1,), -1.0))


def test_quantiles_are_nondecreasing_and_max_is_largest():
    torch.manual_seed(2)
    values = torch.randn(4, 50, 3)
    out = masked_quantile_pool(values, torch.ones(4, 50, dtype=torch.bool))
    diffs = out[..., 1:] - out[..., :-1]
    assert (diffs >= -1e-5).all(), "quantile grid (incl. max) must be non-decreasing"


def test_gradient_flows_through_pooling():
    values = torch.randn(2, 30, 2, requires_grad=True)
    out = masked_quantile_pool(values, torch.ones(2, 30, dtype=torch.bool))
    out.sum().backward()
    assert values.grad is not None and torch.isfinite(values.grad).all()
    assert float(values.grad.abs().sum()) > 0.0


def test_determinism():
    torch.manual_seed(3)
    values = torch.randn(3, 20, 4)
    mask = torch.ones(3, 20, dtype=torch.bool)
    assert torch.equal(masked_quantile_pool(values, mask), masked_quantile_pool(values, mask))


def test_zero_valid_rows_fails_loud():
    values = torch.randn(2, 10, 2)
    mask = torch.ones(2, 10, dtype=torch.bool)
    mask[1] = False  # example 1 has no valid rows
    with pytest.raises(ValueError):
        masked_quantile_pool(values, mask)


def test_bad_shapes_fail_loud():
    with pytest.raises(ValueError):
        masked_quantile_pool(torch.randn(2, 10), torch.ones(2, 10, dtype=torch.bool))
    with pytest.raises(ValueError):
        masked_quantile_pool(torch.randn(2, 10, 2), torch.ones(3, 10, dtype=torch.bool))


# ---- RepECDFPooling module ----

def test_module_out_dim_and_shape():
    m = RepECDFPooling(hidden_dim=16, n_probes=4)
    assert m.out_dim == 4 * (N_QUANTILES + 1)
    reps = torch.randn(6, 32, 16)
    mask = torch.ones(6, 32, dtype=torch.bool)
    out = m(reps, mask)
    assert out.shape == (6, m.out_dim)
    assert torch.isfinite(out).all()


def test_module_gradient_reaches_proj_and_input():
    m = RepECDFPooling(hidden_dim=8, n_probes=3)
    reps = torch.randn(2, 24, 8, requires_grad=True)
    out = m(reps, torch.ones(2, 24, dtype=torch.bool))
    out.sum().backward()
    assert m.proj.weight.grad is not None and float(m.proj.weight.grad.abs().sum()) > 0
    assert reps.grad is not None and float(reps.grad.abs().sum()) > 0


def test_module_init_determinism():
    a = RepECDFPooling(8, n_probes=2)
    b = RepECDFPooling(8, n_probes=2)
    init_parameters_(a, RNGState(seed=9))
    init_parameters_(b, RNGState(seed=9))
    assert torch.allclose(a.proj.weight, b.proj.weight)


def test_module_rejects_bad_dims():
    with pytest.raises(ValueError):
        RepECDFPooling(hidden_dim=0)
    with pytest.raises(ValueError):
        RepECDFPooling(hidden_dim=8, n_probes=0)
    m = RepECDFPooling(hidden_dim=8)
    with pytest.raises(ValueError):
        m(torch.randn(2, 10, 4), torch.ones(2, 10, dtype=torch.bool))  # wrong hidden_dim
