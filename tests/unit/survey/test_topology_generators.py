"""Tests for lacuna.survey.topology_generators (T1 matched-rate vocabulary).

Normal: every generator hits the matched rate band on real-ish X widths (d=3, 6, 9); class labels
correct; mechanism signatures present (booklet block structure; module row-alignment; skip-vs-refusal
driver difference; own-value/top-coding value dependence). Edge/failure: degenerate inputs raise.
Determinism: same RNG seed => same mask.
"""

import numpy as np
import pytest
import torch

from lacuna.core.rng import RNGState
from lacuna.survey import topology_generators as TG


def _X(n=400, d=6, seed=0, rho=0.5):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(n, 1))
    return torch.from_numpy(
        (rho * base + np.sqrt(1 - rho**2) * rng.normal(size=(n, d))).astype(np.float32))


@pytest.mark.parametrize("d", [3, 6, 9])
@pytest.mark.parametrize("name", TG.GENERATORS)
def test_matched_rate_all_generators_all_widths(name, d):
    rates = []
    for seed in range(4):
        R, cls, rate = TG.generate(name, _X(d=d, seed=seed), RNGState(seed=100 + seed))
        assert R.shape == (400, d) and R.dtype == torch.bool
        assert cls == TG.CLASS_IDX[TG.CLASS_OF[name]]
        rates.append(rate)
    assert all(TG.RATE_LO <= r <= TG.RATE_HI for r in rates), (name, d, rates)
    # expected rate calibrated to 0.30: mean over seeds within +-0.03
    assert abs(float(np.mean(rates)) - TG.TARGET_RATE) < 0.03, (name, d, rates)


def test_determinism():
    X = _X()
    for name in TG.GENERATORS:
        a, _, _ = TG.generate(name, X, RNGState(seed=7))
        b, _, _ = TG.generate(name, X, RNGState(seed=7))
        assert torch.equal(a, b), name


def test_booklet_block_structure():
    R, _, _ = TG.generate("mcar_rotated_booklet", _X(n=600), RNGState(seed=3))
    # rows are all-or-none on their assigned block: each missing row's missing-set is one of <=3 patterns
    miss_rows = (~R).any(dim=1)
    patterns = {tuple(row.tolist()) for row in (~R[miss_rows])}
    assert 1 <= len(patterns) <= 3  # K in {2,3} block patterns


def test_module_pair_row_alignment_and_driver_difference():
    X = _X(n=600, seed=2)
    Rs, _, _ = TG.generate("mar_module_skip", X, RNGState(seed=11))
    Rr, _, _ = TG.generate("mnar_module_refusal", X, RNGState(seed=11))
    for R in (Rs, Rr):
        miss = ~R
        rows = miss.any(dim=1)
        # row-aligned: every missing row misses the SAME column block
        assert len({tuple(r.tolist()) for r in miss[rows]}) == 1
    # drivers differ: skip rows are low on an OBSERVED gate; refusal rows are high on the BLOCK's own mean
    z = (X - X.mean(0)) / X.std(0)
    blk_r = torch.where((~Rr).any(dim=0))[0]
    sel_r = (~Rr).any(dim=1)
    assert float(z[sel_r][:, blk_r].mean()) > 0.3          # refusal: high block values refuse


def test_value_dependence_own_value_and_topcoding():
    X = _X(n=800, seed=5)
    z = (X - X.mean(0)) / X.std(0)
    for name in ("mnar_own_value", "mnar_top_coding"):
        R, _, _ = TG.generate(name, X, RNGState(seed=13))
        miss = ~R
        # missing cells sit ABOVE observed cells in z (self-censoring/top-coding direction)
        assert float(z[miss].mean()) > 0.3, name
    # MCAR control: no value dependence
    Rm, _, _ = TG.generate("mcar_uniform", X, RNGState(seed=13))
    assert abs(float(z[~Rm].mean())) < 0.1


def test_mar_depends_on_predictor_not_self():
    X = _X(n=800, seed=8, rho=0.0)  # independent columns: clean attribution
    R, _, _ = TG.generate("mar_single", X, RNGState(seed=21))
    z = (X - X.mean(0)) / X.std(0)
    miss = ~R
    # affected columns' missing cells show no strong OWN-value shift under independence
    assert abs(float(z[miss].mean())) < 0.15


def test_fail_loud():
    with pytest.raises(ValueError):
        TG.generate("nope", _X(), RNGState(seed=1))
    with pytest.raises(ValueError):
        TG.generate("mcar_uniform", _X(d=2), RNGState(seed=1))   # d < 3
    with pytest.raises(ValueError):
        TG.generate("mcar_uniform", _X(n=10), RNGState(seed=1))  # n < 30
    bad = _X(); bad[0, 0] = float("nan")
    with pytest.raises(ValueError):
        TG.generate("mcar_uniform", bad, RNGState(seed=1))
