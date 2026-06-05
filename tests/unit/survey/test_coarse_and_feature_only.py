"""Tests for coarse_bins, feature_only_head, and the coarse train seam."""

import pytest
import torch

from lacuna.core.rng import RNGState
from lacuna.data.catalog import create_default_catalog
from lacuna.survey.coarse_bins import (
    assign_bins,
    scheme_centers,
    scheme_num_bins,
    scheme_spec,
)
from lacuna.survey.consequence_features import N_FEATURES
from lacuna.survey.example_source import LODSurveyExampleSource, SurveyExampleSource
from lacuna.survey.feature_only_head import (
    FeatureOnlyDeltaModel,
    create_feature_only_model,
)
from lacuna.survey.run_manifest import validate_manifest
from lacuna.survey.train import TrainConfig, train_delta_prior


# ---------- coarse_bins ----------

def test_scheme_num_bins():
    assert scheme_num_bins("none") == 7
    assert scheme_num_bins("binary") == 2
    assert scheme_num_bins("coarse3") == 3
    with pytest.raises(ValueError):
        scheme_num_bins("nope")


def test_assign_binary():
    d = torch.tensor([0.0, 0.1, 1.0, 2.5])
    assert assign_bins("binary", d).tolist() == [0, 1, 1, 1]


def test_assign_coarse3_tau1():
    d = torch.tensor([0.0, 0.4, 1.0, 1.0001, 2.5])
    assert assign_bins("coarse3", d).tolist() == [0, 1, 1, 2, 2]  # τ=1.0 inclusive in weak


def test_assign_rejects_negative():
    with pytest.raises(ValueError):
        assign_bins("binary", torch.tensor([-0.1]))


def test_centers_lengths_match():
    for s in ("none", "binary", "coarse3"):
        assert scheme_centers(s).shape[0] == scheme_num_bins(s)


def test_scheme_spec():
    assert scheme_spec("coarse3")["tau"] == 1.0
    assert scheme_spec("binary")["kind"] == "diagnostic"


# ---------- feature_only_head ----------

def test_feature_only_forward_and_shape():
    m = create_feature_only_model(n_features=N_FEATURES, num_bins=3, rng=RNGState(seed=1))
    c = torch.randn(5, N_FEATURES)
    logits = m(consequence=c)
    assert logits.shape == (5, 3)
    assert {n for n, _ in m.named_children()} == {"norm", "head"}  # no encoder, no v1.0 heads


def test_feature_only_requires_consequence():
    m = create_feature_only_model(n_features=N_FEATURES, num_bins=2, rng=RNGState(seed=1))
    with pytest.raises(ValueError):
        m(consequence=None)


def test_feature_only_determinism():
    a = create_feature_only_model(n_features=N_FEATURES, num_bins=3, rng=RNGState(seed=5))
    b = create_feature_only_model(n_features=N_FEATURES, num_bins=3, rng=RNGState(seed=5))
    a.eval(); b.eval()
    c = torch.randn(4, N_FEATURES)
    assert torch.allclose(a(consequence=c), b(consequence=c), atol=1e-6)


# ---------- coarse train seam (smoke) ----------

def _tiny_cfg(scheme, model_kind):
    return TrainConfig(
        delta_grid=[0.0, 0.5, 1.0, 2.5], beta1_range=(0.0, 1.0), target_rate=0.3,
        max_rows=64, max_cols=8, batch_size=4, train_batches_per_epoch=3, max_epochs=1, patience=1,
        val_size=8, test_size=8, hidden_dim=32, evidence_dim=16, n_layers=1, n_heads=2,
        target_conditioned=True, consequence_features=True,
        coarse_scheme=scheme, model_kind=model_kind,
    )


def test_coarse3_features_only_trains_and_manifest():
    cat = create_default_catalog()
    pool = [cat.load(n) for n in ("survey_cps1985", "survey_computers")]
    src = lambda: LODSurveyExampleSource(pool, tau_quantile=0.70)
    out = train_delta_prior(src(), src(), src(), _tiny_cfg("coarse3", "features_only"),
                            RNGState(seed=0), kind="smoke", run_id="q", git_commit="x", timestamp="t")
    validate_manifest(out["manifest"])
    m = out["manifest"]
    assert m["num_bins"] == 3
    assert m["model_arch"]["model_kind"] == "features_only"
    assert m["model_arch"]["coarse_scheme"]["scheme"] == "coarse3"


def test_binary_encoder_features_diagnostic_trains():
    cat = create_default_catalog()
    pool = [cat.load(n) for n in ("survey_cps1985", "survey_computers")]
    src = lambda: SurveyExampleSource(pool)
    out = train_delta_prior(src(), src(), src(), _tiny_cfg("binary", "auto"),
                            RNGState(seed=0), kind="smoke", run_id="q", git_commit="x", timestamp="t")
    assert out["manifest"]["num_bins"] == 2
