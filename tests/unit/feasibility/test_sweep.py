"""Tests for lacuna.feasibility.sweep (oracle orchestration; no training)."""

from dataclasses import dataclass

from lacuna.core.rng import RNGState
from lacuna.feasibility.oracle import OracleCell
from lacuna.feasibility.sweep import (
    boundary_refinement_deltas,
    coarse_oracle_surface,
    compute_oracle_cell,
)


def test_compute_cell_delta_zero_is_chance():
    cell = compute_oracle_cell(
        delta=0.0, beta1=0.0, target_rate=0.3, n=128, rng=RNGState(seed=0),
        rho=0.0, n_quad=24, n_mc=400, n_pop_sample=4000, n_kl_sample=4000,
    )
    assert abs(cell.bayes_error - 0.5) < 1e-6
    assert cell.rho == 0.0


def test_compute_cell_strong_delta_has_signal():
    cell = compute_oracle_cell(
        delta=2.0, beta1=0.0, target_rate=0.3, n=256, rng=RNGState(seed=1),
        rho=0.0, n_quad=24, n_mc=600, n_pop_sample=4000, n_kl_sample=4000,
    )
    assert cell.bayes_error < 0.47
    assert cell.kl_10 > 0 and cell.kl_01 > 0


def test_coarse_surface_cardinality():
    cells = coarse_oracle_surface(
        deltas=[0.0, 1.0], beta1s=[0.0], rates=[0.3], rhos=[0.0, 0.6], ns=[128],
        rng=RNGState(seed=2), n_quad=16, n_mc=200, n_pop_sample=3000, n_kl_sample=3000,
    )
    assert len(cells) == 2 * 1 * 1 * 2 * 1


def _cell(delta, be, rho=0.0):
    return OracleCell(delta=delta, beta1=0.0, target_rate=0.3, n=128, rho=rho,
                      beta0_h0=0.0, beta0_h1=0.0, bayes_error=be, err_h0=be, err_h1=be,
                      kl_10=0.0, kl_01=0.0, xmodel={})


def test_boundary_refinement_targets_steep_region():
    # flat 0.50 -> 0.49 (no refine), then a steep 0.49 -> 0.20 jump (refine midpoint)
    cells = [_cell(0.0, 0.50), _cell(0.5, 0.49), _cell(1.0, 0.20), _cell(1.5, 0.18)]
    proposals = boundary_refinement_deltas(cells, be_gap=0.05)
    mids = proposals[(0.0, 0.3, 0.0, 128)]
    assert 0.75 in mids          # midpoint of the steep 0.5->1.0 jump
    assert 0.25 not in mids      # flat region not refined
    assert 1.25 not in mids      # shallow tail not refined
