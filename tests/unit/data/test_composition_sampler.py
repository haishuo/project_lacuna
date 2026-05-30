"""Tests for lacuna.data.composition_sampler — the Stage-B composition-controlled generator (ADR-0007).

The HEADLINE test here is GATE-1: the REALISED by-cell composition tracks the drawn target (the
shared-denominator discipline — target, per-cell tags, and realised composition all use the missing
cell as the denominator). This is the Stage-B generalisation of the confound test that Stage 5
established (cf. the column pools' miss-rate guard in test_mnar_column_pool.py).
"""

import numpy as np
import pytest
import torch

from lacuna.core.rng import RNGState
from lacuna.core.types import MCAR, MAR, MNAR
from lacuna.data.ingestion import RawDataset
from lacuna.data.missingness_footprint import missingness_footprint
from lacuna.data.composition_target import CompositionTarget, sample_composition_target
from lacuna.data.composition_sampler import compose_composition_missingness, NOT_MISSING


def _complete(n, d, seed):
    """A complete (no-missing) Gaussian RawDataset — data-agnostic substrate for the invariant."""
    return RawDataset(data=RNGState(seed=seed).randn(n, d).numpy(),
                      feature_names=tuple(f"f{i}" for i in range(d)), name="gauss")


def _l1(a, b):
    return float(np.abs(np.array(a) - np.array(b)).sum())


# ---------------------------------------------------------------------------
# GATE-1 — realised composition tracks the target (the shared-denominator invariant)
# ---------------------------------------------------------------------------

_FIXED_TARGETS = [
    (0.34, 0.33, 0.33),   # balanced
    (0.60, 0.20, 0.20),   # MCAR-heavy
    (0.20, 0.60, 0.20),   # MAR-heavy
    (0.20, 0.20, 0.60),   # MNAR-heavy
    (0.45, 0.35, 0.20),   # mixed
]


@pytest.mark.parametrize("frac", _FIXED_TARGETS)
def test_realized_composition_matches_target_fixed(frac):
    """For each representative target, the realised by-cell composition lands near it across data
    seeds (reported as a distribution, never a single seed — the rigor standard of this project)."""
    target = CompositionTarget(*frac, miss_rate=0.3)
    rng = RNGState(seed=0)
    l1s = []
    for s in range(5):
        res = compose_composition_missingness(_complete(1500, 18, 200 + s), target, rng.spawn())
        l1s.append(_l1(res.realized_composition, frac))
    l1s = np.array(l1s)
    assert l1s.mean() < 0.10, f"target {frac}: mean realised L1 {l1s.mean():.3f}"
    assert l1s.max() < 0.20, f"target {frac}: worst realised L1 {l1s.max():.3f}"


def test_realized_composition_tracks_random_targets():
    """Over the broad simplex prior, realised tracks target (distribution-level: mean L1)."""
    rng = RNGState(seed=1)
    l1s, mr = [], []
    for _ in range(40):
        t = sample_composition_target(rng.spawn())
        res = compose_composition_missingness(_complete(1500, 18, 99), t, rng.spawn())
        l1s.append(_l1(res.realized_composition, t.as_fractions()))
        mr.append(abs(res.realized_miss_rate - t.miss_rate))
    assert np.mean(l1s) < 0.10, f"mean realised composition L1 {np.mean(l1s):.3f}"
    assert np.mean(mr) < 0.06, f"mean realised miss-rate error {np.mean(mr):.3f}"


def test_realized_composition_equals_tag_counts():
    """Realised composition is exactly the per-cell tag counts over the missing-cell denominator."""
    t = CompositionTarget(0.3, 0.3, 0.4, miss_rate=0.3)
    res = compose_composition_missingness(_complete(1000, 16, 5), t, RNGState(seed=2))
    tags = res.cell_tags
    n_missing = int((tags != NOT_MISSING).sum().item())
    assert n_missing == res.n_missing_cells
    for cls in (MCAR, MAR, MNAR):
        assert abs(res.realized_composition[cls] - (tags == cls).sum().item() / n_missing) < 1e-9


# ---------------------------------------------------------------------------
# Per-cell tags — the by-cell ground truth (exact, unambiguous)
# ---------------------------------------------------------------------------

def test_tags_align_with_mask_exactly():
    """Every missing cell carries a class tag; every observed cell carries NOT_MISSING."""
    t = CompositionTarget(0.34, 0.33, 0.33, miss_rate=0.3)
    res = compose_composition_missingness(_complete(800, 14, 6), t, RNGState(seed=3))
    missing = ~res.observed.r
    tagged = res.cell_tags != NOT_MISSING
    assert torch.equal(missing, tagged), "missing mask and tagged mask disagree"
    # every tag is a valid class
    valid = (res.cell_tags == NOT_MISSING) | (res.cell_tags == MCAR) | \
            (res.cell_tags == MAR) | (res.cell_tags == MNAR)
    assert bool(valid.all())


def test_missing_cells_are_zeroed():
    t = CompositionTarget(0.34, 0.33, 0.33, miss_rate=0.3)
    res = compose_composition_missingness(_complete(600, 12, 7), t, RNGState(seed=4))
    assert torch.equal(res.observed.x[~res.observed.r], torch.zeros_like(res.observed.x[~res.observed.r]))


def test_reserved_observed_columns_are_fully_observed():
    t = CompositionTarget(0.34, 0.33, 0.33, miss_rate=0.3)
    res = compose_composition_missingness(_complete(600, 16, 8), t, RNGState(seed=5))
    for c in res.plan.observed_cols:
        assert bool(res.observed.r[:, c].all()), f"reserved observed column {c} has missing cells"


def test_cell_tags_shape_and_dtype():
    t = CompositionTarget(0.34, 0.33, 0.33, miss_rate=0.3)
    res = compose_composition_missingness(_complete(500, 10, 9), t, RNGState(seed=6))
    assert res.cell_tags.shape == (500, 10)
    assert res.cell_tags.dtype == torch.long


# ---------------------------------------------------------------------------
# Realism direction-of-travel (the AUC magnitude is measured by scripts/stagea_realism_gap.py)
# ---------------------------------------------------------------------------

def test_produces_comissingness_unlike_per_column_independence():
    """The whole point of Stage B: blocks give co-missingness and fewer patterns than the old
    per-column-independent generator (Stage A: frac_pairs_coupled .002, distinct_pattern_ratio .42).
    A structured-heavy draw must move both clearly in the real direction."""
    rng = RNGState(seed=0)
    coupled, dpr = [], []
    for s in range(5):
        t = CompositionTarget(0.15, 0.40, 0.45, miss_rate=0.35)
        res = compose_composition_missingness(_complete(1500, 20, 40 + s), t, rng.spawn())
        fp = missingness_footprint(res.observed.x, res.observed.r)
        coupled.append(fp["frac_pairs_coupled"])
        dpr.append(fp["distinct_pattern_ratio"])
    assert np.mean(coupled) > 0.01, f"frac_pairs_coupled {np.mean(coupled):.3f} not above per-column floor"
    assert np.mean(dpr) < 0.42, f"distinct_pattern_ratio {np.mean(dpr):.3f} no better than per-column"


# ---------------------------------------------------------------------------
# Determinism + failure cases
# ---------------------------------------------------------------------------

def test_deterministic_under_same_seed():
    t = CompositionTarget(0.3, 0.3, 0.4, miss_rate=0.3)
    raw = _complete(500, 14, 10)
    a = compose_composition_missingness(raw, t, RNGState(seed=11))
    b = compose_composition_missingness(raw, t, RNGState(seed=11))
    assert torch.equal(a.cell_tags, b.cell_tags)
    assert a.realized_composition == b.realized_composition


def test_incomplete_input_raises():
    data = RNGState(seed=0).randn(100, 8).numpy()
    data[3, 2] = np.nan
    raw = RawDataset(data=data, feature_names=tuple(f"f{i}" for i in range(8)), name="bad")
    with pytest.raises(ValueError, match="non-finite"):
        compose_composition_missingness(raw, CompositionTarget(0.34, 0.33, 0.33, miss_rate=0.3),
                                        RNGState(seed=0))
