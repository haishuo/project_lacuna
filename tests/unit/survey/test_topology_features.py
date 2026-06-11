"""Tests for lacuna.survey.topology_features (T1 frozen feature arm)."""

import numpy as np
import pytest
import torch

from lacuna.core.rng import RNGState
from lacuna.survey import topology_generators as TG
from lacuna.survey.topology_features import N_TOPO_FEATURES, compute_topology_features


def _example(name, seed=0, n=400, d=6):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(n, 1))
    X = torch.from_numpy((0.5 * base + 0.87 * rng.normal(size=(n, d))).astype(np.float32))
    R, cls, _ = TG.generate(name, X, RNGState(seed=50 + seed))
    return X * R.float(), R, cls


def test_shape_finite_deterministic():
    x, R, _ = _example("mar_module_skip")
    a = compute_topology_features(x, R)
    b = compute_topology_features(x.clone(), R.clone())
    assert a.shape == (N_TOPO_FEATURES,) and torch.isfinite(a).all() and torch.equal(a, b)


def test_signatures_separate_mechanisms():
    # block score (idx 9) high for the row-aligned module pair, low for uniform MCAR
    def feat(name, idx, seed=1):
        x, R, _ = _example(name, seed=seed)
        return float(compute_topology_features(x, R)[idx])
    BLOCK = 9
    assert feat("mar_module_skip", BLOCK) > 0.9
    assert feat("mnar_module_refusal", BLOCK) > 0.9
    assert feat("mcar_uniform", BLOCK) < 0.6
    # gate AUC (idx 12) separates skip (observed driver) from refusal (latent driver)
    GATE = 12
    assert feat("mar_module_skip", GATE) > 0.9
    assert feat("mar_module_skip", GATE) - feat("mnar_module_refusal", GATE) > 0.15


def test_all_generators_produce_features():
    for name in TG.GENERATORS:
        x, R, _ = _example(name, seed=2)
        v = compute_topology_features(x, R)
        assert torch.isfinite(v).all(), name


def test_fail_loud():
    x, R, _ = _example("mcar_uniform")
    with pytest.raises(ValueError):
        compute_topology_features(x[:5], R[:5])           # n too small
    with pytest.raises(ValueError):
        compute_topology_features(x, R.float())           # mask not bool
    with pytest.raises(ValueError):
        compute_topology_features(x[:, :1], R[:, :1])     # d too small
