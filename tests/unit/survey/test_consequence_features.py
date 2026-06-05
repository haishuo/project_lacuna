"""Tests for lacuna.survey.consequence_features + the conditioned-model wiring."""

import numpy as np
import pytest
import torch

from lacuna.core.rng import RNGState
from lacuna.data.catalog import create_default_catalog
from lacuna.survey.batching import collate, make_example, make_lod_example
from lacuna.survey.conditioned_head import create_target_conditioned_model
from lacuna.survey.consequence_features import (
    FEATURE_NAMES,
    N_FEATURES,
    compute_consequence_features,
)
from lacuna.survey.delta_bins import NUM_BINS


def _full(x):
    return x, torch.ones(x.shape, dtype=torch.bool)


# ---------- shape / dtype / schema ----------

def test_shape_and_dtype():
    x = torch.randn(500, 3)
    f = compute_consequence_features(*_full(x), target_idx=1)
    assert f.shape == (N_FEATURES,) and f.dtype == torch.float32
    assert N_FEATURES == len(FEATURE_NAMES) == 17


# ---------- determinism ----------

def test_determinism():
    x = torch.randn(400, 4)
    r = torch.rand(400, 4) > 0.3
    a = compute_consequence_features(x, r, 2)
    b = compute_consequence_features(x, r, 2)
    assert torch.equal(a, b)


# ---------- scale invariance ----------

def test_scale_and_shift_invariant():
    g = np.random.default_rng(0)
    x = torch.tensor(g.standard_normal((2000, 2)), dtype=torch.float32)
    r = torch.ones(2000, 2, dtype=torch.bool)
    base = compute_consequence_features(x, r, 0)
    x2 = x.clone(); x2[:, 0] = x[:, 0] * 7.5 + 3.0  # affine transform of the target column
    scaled = compute_consequence_features(x2, r, 0)
    # missing_rate identical; all z-scored shape features invariant to affine scaling
    assert torch.allclose(base, scaled, atol=1e-4)


# ---------- missing handling ----------

def test_missing_rate_recorded_and_censored_excluded():
    x = torch.randn(1000, 2)
    r = torch.ones(1000, 2, dtype=torch.bool)
    r[:300, 0] = False  # 30% of target censored
    f = compute_consequence_features(x, r, 0)
    assert abs(float(f[0]) - 0.30) < 1e-6  # missing_rate feature


def test_few_observed_safe():
    x = torch.randn(100, 3)
    r = torch.ones(100, 3, dtype=torch.bool)
    r[5:, 1] = False  # only 5 observed (< MIN_OBS=8)
    f = compute_consequence_features(x, r, 1)
    assert f.shape == (N_FEATURES,)
    assert torch.isfinite(f).all()
    assert float(f[0]) == pytest.approx(0.95)  # missing_rate still set; shape features 0


# ---------- discriminative sanity (NOT a training target) ----------

def test_top_coding_lowers_upper_tail_features():
    g = np.random.default_rng(1)
    base = torch.tensor(g.standard_normal(4000), dtype=torch.float32)
    x = torch.stack([base, torch.randn(4000)], dim=1)
    r = torch.ones(4000, 2, dtype=torch.bool)
    full = compute_consequence_features(x, r, 0)
    # top-code: censor the upper ~30% of the target
    thr = torch.quantile(base, 0.70)
    r_tc = r.clone(); r_tc[:, 0] = base <= thr
    tc = compute_consequence_features(x, r_tc, 0)
    zmax_i = FEATURE_NAMES.index("z_max")
    q95_i = FEATURE_NAMES.index("zq95")
    fgt15_i = FEATURE_NAMES.index("frac_gt_1.5")
    assert tc[zmax_i] < full[zmax_i]
    assert tc[q95_i] < full[q95_i]
    assert tc[fgt15_i] < full[fgt15_i]


# ---------- failures ----------

def test_bad_target_idx():
    with pytest.raises(ValueError):
        compute_consequence_features(*_full(torch.randn(50, 3)), target_idx=9)


# ---------- model wiring ----------

def _batch(seed=7, lod=False, max_rows=96):
    raw = create_default_catalog().load("survey_cps1985")
    rng = RNGState(seed=seed)
    mk = make_lod_example if lod else make_example
    kw = {"tau_quantile": 0.70} if lod else {}
    exs = [mk(raw, beta1=1.0, delta=d, target_rate=0.3, rng=rng.spawn(), max_rows=max_rows, **kw)
           for d in (0.0, 0.5, 1.5, 2.5)]
    return collate(exs, max_rows=max_rows, max_cols=8)


def test_collate_attaches_consequence():
    db = _batch()
    assert db.consequence.shape == (4, N_FEATURES)


def test_conditioned_model_with_consequence_forward_and_head_dim():
    m = create_target_conditioned_model(hidden_dim=64, evidence_dim=32, n_layers=1, n_heads=2,
                                        max_cols=8, num_bins=NUM_BINS, n_consequence_features=N_FEATURES,
                                        rng=RNGState(seed=1))
    db = _batch(lod=True)
    logits = m(db.tokens, db.target_idx, db.consequence)
    assert logits.shape == (4, NUM_BINS)
    # head input dim = evidence + hidden + N_FEATURES
    assert m.head.net[0].in_features == 32 + 64 + N_FEATURES
    # still no v1.0 heads
    assert {n for n, _ in m.named_children()} == {"encoder", "consequence_norm", "head"}


def test_conditioned_model_requires_consequence_when_enabled():
    m = create_target_conditioned_model(hidden_dim=64, evidence_dim=32, n_layers=1, n_heads=2,
                                        max_cols=8, num_bins=NUM_BINS, n_consequence_features=N_FEATURES,
                                        rng=RNGState(seed=1))
    db = _batch()
    with pytest.raises(ValueError):
        m(db.tokens, db.target_idx, None)
