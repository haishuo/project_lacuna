"""Tests for the T2 mask-topology stream and the T2 model."""

import pytest
import torch

from lacuna.core.rng import RNGState
from lacuna.survey.mask_topology_stream import MaskTopologyStream, _masked_mean
from lacuna.survey.skip_logic_generator import FAMILIES
from lacuna.survey.t2_model import T2Model, T2ModelConfig, create_t2_model


def _inputs(b=4, n=12, d=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    Xv = torch.randn(b, n, d, generator=g)
    R = (torch.rand(b, n, d, generator=g) > 0.3).float()
    row_mask = torch.ones(b, n, dtype=torch.bool)
    row_mask[:, -1] = False                                  # one pad row
    pcol_mask = torch.ones(b, d, dtype=torch.bool)
    pcol_mask[:, 0] = False                                  # target column excluded
    tij = (torch.rand(b, n, generator=g) > 0.3).float()
    tval = torch.randn(b, n, generator=g) * tij
    Xv = Xv * R
    return Xv, R, row_mask, pcol_mask, tij, tval


# ---- mask-topology stream ----
def test_stream_shape_and_finite():
    Xv, R, rm, pm, tij, _ = _inputs()
    out = MaskTopologyStream()(Xv, R, rm, pm, tij)
    assert out.shape == (4, 48) and torch.isfinite(out).all()


def test_stream_variable_width():
    for d in (3, 6, 9):
        Xv, R, rm, pm, tij, _ = _inputs(d=d)
        out = MaskTopologyStream(out_dim=32)(Xv, R, rm, pm, tij)
        assert out.shape == (4, 32) and torch.isfinite(out).all()


def test_stream_gradients_flow():
    Xv, R, rm, pm, tij, _ = _inputs()
    MaskTopologyStream()(Xv, R, rm, pm, tij).sum().backward()


def test_masked_mean_empty_group_zero():
    tokens = torch.randn(2, 6, 3, 4)
    mask = torch.zeros(2, 6, 3, dtype=torch.bool)
    out = _masked_mean(tokens, mask)
    assert out.shape == (2, 3, 4) and torch.count_nonzero(out) == 0


def test_stream_no_valid_predictor_raises():
    Xv, R, rm, pm, tij, _ = _inputs()
    pm[1] = False
    with pytest.raises(ValueError, match="valid predictor"):
        MaskTopologyStream()(Xv, R, rm, pm, tij)


def test_stream_bad_R_shape_raises():
    Xv, R, rm, pm, tij, _ = _inputs()
    with pytest.raises(ValueError, match="R must match"):
        MaskTopologyStream()(Xv, R[:, :-1], rm, pm, tij)


# ---- T2 model ----
def test_model_output_shapes():
    Xv, R, rm, pm, tij, tval = _inputs()
    fam, dlt = T2Model()(Xv, R, rm, pm, tij, tval)
    assert fam.shape == (4, len(FAMILIES))
    assert dlt.shape == (4,)
    assert torch.isfinite(fam).all() and torch.isfinite(dlt).all()


def test_model_deterministic_init_reproducible():
    m1 = create_t2_model(rng=RNGState(seed=7))
    m2 = create_t2_model(rng=RNGState(seed=7))
    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        assert torch.equal(p1, p2)


def test_model_gradients_reach_phi_and_topo():
    Xv, R, rm, pm, tij, tval = _inputs()
    model = T2Model()
    fam, dlt = model(Xv, R, rm, pm, tij, tval)
    (fam.sum() + dlt.sum()).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for n, p in model.named_parameters() if n.startswith("phi"))
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for n, p in model.named_parameters() if n.startswith("topo"))


def test_model_custom_config():
    cfg = T2ModelConfig(e_col=16, topo_out=24, trunk_hidden=48)
    Xv, R, rm, pm, tij, tval = _inputs()
    fam, dlt = T2Model(cfg)(Xv, R, rm, pm, tij, tval)
    assert fam.shape == (4, len(FAMILIES)) and dlt.shape == (4,)
