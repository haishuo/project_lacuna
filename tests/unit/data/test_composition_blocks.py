"""Tests for lacuna.data.composition_blocks — joint/block mechanisms at a controlled rate (ADR-0007)."""

import numpy as np
import pytest
import torch

from lacuna.core.rng import RNGState
from lacuna.core.types import MAR, MNAR
from lacuna.data.semisynthetic import _zscore_columns
from lacuna.data.missingness_footprint import missingness_footprint
from lacuna.data.composition_blocks import (
    MNAR_BLOCK_KINDS, MAR_BLOCK_KINDS, block_class, apply_block,
)

_ALL = MNAR_BLOCK_KINDS + MAR_BLOCK_KINDS


def _zblock(n, w, seed):
    return _zscore_columns(RNGState(seed=seed).randn(n, w))


# ---------------------------------------------------------------------------
# Normal cases
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", _ALL)
def test_returns_block_shaped_bool_mask(kind):
    z = _zblock(800, 5, 1)
    r = apply_block(kind, z, 0.3, RNGState(seed=2))
    assert r.shape == (800, 5)
    assert r.dtype == torch.bool


@pytest.mark.parametrize("kind", _ALL)
def test_block_partially_censors(kind):
    """A block deletes some — but not all — of its cells (no degenerate all-missing/all-observed)."""
    z = _zblock(1000, 6, 1)
    r = apply_block(kind, z, 0.3, RNGState(seed=2))
    frac_obs = r.float().mean().item()
    assert 0.0 < frac_obs < 1.0


def test_block_class_mapping():
    for k in MNAR_BLOCK_KINDS:
        assert block_class(k) == MNAR
    for k in MAR_BLOCK_KINDS:
        assert block_class(k) == MAR


@pytest.mark.parametrize("kind", _ALL)
@pytest.mark.parametrize("target", [0.15, 0.3, 0.45])
def test_realized_rate_tracks_target_across_seeds(kind, target):
    """Confound control: realised average per-column rate stays near the requested target across
    data seeds and block widths (mechanism TYPE varies, missing QUANTITY ~controlled — the same
    discipline as the column pools' miss-rate guard)."""
    rates = []
    for sd in range(5):
        w = 3 + sd  # widths 3..7
        z = _zblock(1200, w, 50 + sd)
        r = apply_block(kind, z, target, RNGState(seed=10 + sd))
        rates.append(1.0 - r.float().mean().item())
    rates = np.array(rates)
    assert (rates >= 0.5 * target).all() and (rates <= 1.6 * target).all(), \
        f"{kind} target {target}: per-seed rates {rates.round(3)} out of band"
    assert 0.7 * target <= rates.mean() <= 1.3 * target, \
        f"{kind} target {target}: mean rate {rates.mean():.3f} off target"


def test_refusal_and_skip_produce_sharp_blocks():
    """refusal/skip = all-or-nothing co-missingness: high pairwise corr, very few distinct patterns."""
    for kind in ("refusal", "skip"):
        z = _zblock(1500, 6, 7)
        r = apply_block(kind, z, 0.3, RNGState(seed=3))
        fp = missingness_footprint(z.numpy(), r.numpy())
        assert fp["miss_corr_mean_abs"] > 0.8, f"{kind} corr {fp['miss_corr_mean_abs']}"
        assert fp["distinct_pattern_ratio"] < 0.05, f"{kind} dpr {fp['distinct_pattern_ratio']}"


def test_attrition_is_monotone():
    z = _zblock(1500, 6, 7)
    r = apply_block("attrition", z, 0.3, RNGState(seed=3))
    fp = missingness_footprint(z.numpy(), r.numpy())
    assert fp["monotone_row_frac"] > 0.9


def test_latent_is_graded_comissingness():
    """latent = a graded middle ground: real co-missingness, but more distinct patterns than the
    all-or-nothing refusal block (the intermediate regime real survey data shows)."""
    z = _zblock(1500, 6, 7)
    r = apply_block("latent", z, 0.3, RNGState(seed=3))
    fp = missingness_footprint(z.numpy(), r.numpy())
    assert fp["miss_corr_mean_abs"] > 0.1                  # genuine co-missingness
    assert fp["distinct_pattern_ratio"] > 0.01             # but not the ~0 of a sharp refusal block


def test_deterministic_under_same_seed():
    z = _zblock(600, 5, 1)
    r1 = apply_block("refusal", z, 0.3, RNGState(seed=8))
    r2 = apply_block("refusal", z, 0.3, RNGState(seed=8))
    assert torch.equal(r1, r2)


# ---------------------------------------------------------------------------
# Failure cases
# ---------------------------------------------------------------------------

def test_unknown_kind_raises():
    z = _zblock(100, 4, 1)
    with pytest.raises(ValueError, match="unknown block kind"):
        apply_block("nonsense", z, 0.3, RNGState(seed=0))


def test_unknown_kind_in_block_class_raises():
    with pytest.raises(ValueError, match="unknown block kind"):
        block_class("nonsense")


@pytest.mark.parametrize("kind", ["refusal", "skip"])
def test_too_narrow_block_raises(kind):
    z = _zblock(100, 1, 1)
    with pytest.raises(ValueError, match="needs w >= 2"):
        apply_block(kind, z, 0.3, RNGState(seed=0))


@pytest.mark.parametrize("bad_rate", [0.0, 1.0, -0.1, 2.0])
def test_bad_target_rate_raises(bad_rate):
    z = _zblock(100, 4, 1)
    with pytest.raises(ValueError, match="target_rate"):
        apply_block("refusal", z, bad_rate, RNGState(seed=0))


def test_non_2d_block_raises():
    with pytest.raises(ValueError, match="2-D"):
        apply_block("refusal", torch.zeros(10), 0.3, RNGState(seed=0))
