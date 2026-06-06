"""Tests for the Level-1 φ-spine (Stage-1): survey_catalog, column_batching, column_phi, level1_model.

Covers contracts (Rule 1/2), determinism (Rule 6), and the architecture guards (no BERT encoder, no
v1.0 heads) — the Stage-1 build's pre-run checks.
"""

import pytest
import torch

from lacuna.core.rng import RNGState
from lacuna.core.types import ObservedDataset
from lacuna.data.catalog import create_default_catalog
from lacuna.survey.answer_sheet import AnswerSheet, GENERATOR_FAMILY
from lacuna.survey.batching import DeltaExample, make_example, make_lod_example
from lacuna.survey.column_batching import ColumnBatch, collate_columns
from lacuna.survey.column_phi import ColumnPhi, N_QUANTILES, create_column_phi
from lacuna.survey.delta_bins import NUM_BINS
from lacuna.survey.level1_model import Level1Model, create_level1_model
from lacuna.survey import survey_catalog as SC


# ---------- survey_catalog ----------

def test_genuine_list_excludes_contaminants():
    assert len(SC.GENUINE_SURVEYS) == 9
    assert SC.CONTAMINANTS == {"survey_cars93", "survey_computers", "survey_survey"}
    assert not (set(SC.GENUINE_SURVEYS) & SC.CONTAMINANTS)


def test_assert_genuine_guards():
    assert SC.assert_genuine("survey_cps1988") == "survey_cps1988"
    for bad in ("survey_cars93", "survey_computers", "survey_survey"):
        with pytest.raises(ValueError):
            SC.assert_genuine(bad)
    with pytest.raises(ValueError):
        SC.assert_genuine("abalone")


def test_domain_of():
    assert SC.domain_of("survey_cps1985") == "labor"
    assert SC.domain_of("survey_yrbss") == "health"
    assert SC.domain_of("survey_bfi") == "psychology"


# ---------- column_batching ----------

def _examples(n=4, family="lod", max_rows=128, seed=7):
    raw = create_default_catalog().load("survey_cps1985")
    rng = RNGState(seed=seed)
    out = []
    for d in (0.0, 0.4, 1.25, 2.5)[:n]:
        if family == "lod":
            out.append(make_lod_example(raw, beta1=1.0, delta=d, target_rate=0.3,
                                        tau_quantile=0.70, rng=rng.spawn(), max_rows=max_rows))
        else:
            out.append(make_example(raw, beta1=1.0, delta=d, target_rate=0.3,
                                    rng=rng.spawn(), max_rows=max_rows))
    return out


def test_collate_columns_shapes_and_standardization():
    ex = _examples(max_rows=128)
    cb = collate_columns(ex, max_rows=128)
    assert isinstance(cb, ColumnBatch)
    assert cb.target_values.shape == (4, 128, 1)
    assert cb.value_mask.shape == (4, 128) and cb.value_mask.dtype == torch.bool
    assert cb.delta_bin.shape == (4,) and cb.delta.shape == (4,)
    assert cb.target_idx.tolist() == [s.target_col_idx for s in cb.sheets]
    # standardized within observed: mean ~0 over the valid entries of each row-set
    for i in range(4):
        v = cb.target_values[i, cb.value_mask[i], 0]
        assert abs(float(v.mean())) < 1e-4


def test_collate_empty_fails():
    with pytest.raises(ValueError):
        collate_columns([], max_rows=64)


def test_collate_all_missing_target_fails():
    x = torch.randn(10, 2)
    r = torch.ones(10, 2, dtype=torch.bool)
    r[:, 1] = False  # target col fully missing
    obs = ObservedDataset(x=x * r.float(), r=r, n=10, d=2,
                          feature_names=("p", "t"), dataset_id="synthA", meta={})
    sheet = AnswerSheet(source_name="synthA", n=10, d=2, target_col_idx=1, target_col_name="t",
                        predictor_col_idx=0, predictor_col_name="p", beta0=0.0, beta1=1.0,
                        delta=0.0, delta_bin=0, generator_family=GENERATOR_FAMILY,
                        target_rate=0.3, realized_rate=1.0, corr_target_predictor=0.0, seed=1)
    with pytest.raises(ValueError):
        collate_columns([DeltaExample(observed=obs, answer_sheet=sheet)], max_rows=16)


# ---------- column_phi ----------

def test_phi_output_shape_and_grad():
    phi = create_column_phi(m=16, e_col=32, rng=RNGState(seed=1))
    V = torch.randn(5, 64, 1, requires_grad=True)
    MK = torch.ones(5, 64, dtype=torch.bool)
    e = phi(V, MK)
    assert e.shape == (5, 32)
    e.sum().backward()
    assert V.grad is not None and float(V.grad.abs().sum()) > 0
    assert float(phi.h[0].weight.grad.abs().sum()) > 0


def test_phi_schema_and_quantiles():
    phi = ColumnPhi(m=16, e_col=32)
    assert N_QUANTILES == 12
    s = phi.schema()
    assert s["kind"] == "column_phi" and s["e_col"] == 32 and len(s["quantile_levels"]) == 12


def test_phi_init_determinism():
    a = create_column_phi(rng=RNGState(seed=9))
    b = create_column_phi(rng=RNGState(seed=9))
    assert torch.allclose(a.h[0].weight, b.h[0].weight)
    assert torch.allclose(a.rho[0].weight, b.rho[0].weight)


def test_phi_bad_dims():
    with pytest.raises(ValueError):
        ColumnPhi(m=0)
    with pytest.raises(ValueError):
        ColumnPhi(e_col=0)
    phi = create_column_phi(rng=RNGState(seed=1))
    with pytest.raises(ValueError):
        phi(torch.randn(2, 10), torch.ones(2, 10, dtype=torch.bool))  # not [B,R,1]


# ---------- level1_model ----------

def test_level1_forward_and_backprop():
    model = create_level1_model(m=16, e_col=32, rng=RNGState(seed=2))
    model.train()
    cb = collate_columns(_examples(max_rows=128), max_rows=128)
    logits = model(cb)
    assert logits.shape == (4, NUM_BINS)
    torch.nn.functional.cross_entropy(logits, cb.delta_bin).backward()
    phi_grad = sum(p.grad.abs().sum().item() for p in model.phi.parameters() if p.grad is not None)
    assert phi_grad > 0


def test_level1_no_encoder_no_v1_heads():
    model = create_level1_model(rng=RNGState(seed=2))
    for h in ("encoder", "moe", "reconstruction", "decision_rule", "missingness_extractor"):
        assert not hasattr(model, h)
    assert {n for n, _ in model.named_children()} == {"phi", "head"}


def test_level1_predict_proba_and_temperature():
    model = create_level1_model(rng=RNGState(seed=2))
    model.eval()
    cb = collate_columns(_examples(max_rows=96), max_rows=96)
    p = model.predict_proba(cb)
    assert torch.allclose(p.sum(-1), torch.ones(4), atol=1e-5)
    with pytest.raises(ValueError):
        model.set_temperature(-1.0)


def test_level1_init_determinism():
    a = create_level1_model(rng=RNGState(seed=5))
    b = create_level1_model(rng=RNGState(seed=5))
    a.eval(); b.eval()
    cb = collate_columns(_examples(max_rows=96), max_rows=96)
    assert torch.allclose(a(cb), b(cb), atol=1e-6)


# ---------- named_prior manifest block ----------

def _good_named_prior():
    return {
        "dataset_catalog": {"datasets": ["survey_cps1988"]},
        "contaminants_excluded": ["survey_cars93", "survey_computers", "survey_survey"],
        "idiom_vocabulary": ["top_coding"], "delta_grid": [0.0, 2.5],
        "delta_grid_weights": [0.5, 0.5], "prior_marginal": {"0": 0.5, "1": 0.5},
        "rate_regime": {"target_rate": 0.3, "matched": True},
        "phi_config": {"kind": "column_phi", "e_col": 32}, "data_role": "B_complete_projection",
    }


def test_validate_named_prior_ok():
    from lacuna.survey.run_manifest import validate_named_prior
    validate_named_prior(_good_named_prior())  # no raise


def test_validate_named_prior_missing_key():
    from lacuna.survey.run_manifest import validate_named_prior
    np_ = _good_named_prior(); del np_["prior_marginal"]
    with pytest.raises(ValueError):
        validate_named_prior(np_)


def test_validate_named_prior_rejects_non_role_b():
    from lacuna.survey.run_manifest import validate_named_prior
    np_ = _good_named_prior(); np_["data_role"] = "A_natural"
    with pytest.raises(ValueError):
        validate_named_prior(np_)  # never train on natural missingness
