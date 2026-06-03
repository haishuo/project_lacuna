"""
lacuna.feasibility.sweep

Two-stage oracle-surface orchestration (PROPOSAL P1, ruling #1):
  Stage 1 — coarse oracle surface over (δ, β1, rate, ρ, n).
  Stage 2 — refine δ around the distinguishability boundary (where Bayes error moves
            from near-chance to distinguishable), NOT over trivially-easy/-impossible
            regions. The interesting object is the boundary.

This module computes ONLY the oracle (the ceiling). It trains nothing and loads no
checkpoint. The model arm is a separate, gated step. Importing/running this module
performs no I/O; the calling script handles artifacts + manifest.
"""

from typing import Dict, List, Optional

from lacuna.core.rng import RNGState
from .oracle import (
    OracleCell,
    bayes_error_nsample,
    gauss_hermite,
    per_row_kl,
    solve_beta0_population,
)
from .xmodel import ConditionalGaussian, XModel


def compute_oracle_cell(
    delta: float,
    beta1: float,
    target_rate: float,
    n: int,
    rng: RNGState,
    *,
    rho: Optional[float] = None,
    xmodel: Optional[XModel] = None,
    n_quad: int = 64,
    n_mc: int = 4000,
    n_pop_sample: int = 20000,
    n_kl_sample: int = 50000,
) -> OracleCell:
    """Compute one oracle cell. Synthetic-X if `xmodel` is None (uses ConditionalGaussian.synthetic(rho))."""
    if xmodel is None:
        if rho is None:
            raise ValueError("provide either an xmodel (real-X) or rho (synthetic-X)")
        xmodel = ConditionalGaussian.synthetic(rho)
    nodes, weights = gauss_hermite(n_quad)

    # Each hypothesis independently rate-matched to target_rate under the X-model.
    params_h0 = solve_beta0_population(
        xmodel, beta1, 0.0, target_rate, rng.spawn(), nodes, weights, n_sample=n_pop_sample
    )
    params_h1 = solve_beta0_population(
        xmodel, beta1, delta, target_rate, rng.spawn(), nodes, weights, n_sample=n_pop_sample
    )

    be = bayes_error_nsample(params_h0, params_h1, xmodel, n, rng.spawn(), nodes, weights, n_mc=n_mc)
    kl_10, kl_01 = per_row_kl(params_h0, params_h1, xmodel, rng.spawn(), nodes, weights, n_sample=n_kl_sample)

    return OracleCell(
        delta=delta,
        beta1=beta1,
        target_rate=target_rate,
        n=n,
        rho=rho,
        beta0_h0=params_h0.beta0,
        beta0_h1=params_h1.beta0,
        bayes_error=be["bayes_error"],
        err_h0=be["err_h0"],
        err_h1=be["err_h1"],
        kl_10=kl_10,
        kl_01=kl_01,
        xmodel=xmodel.descriptor,
    )


def coarse_oracle_surface(
    deltas: List[float],
    beta1s: List[float],
    rates: List[float],
    rhos: List[float],
    ns: List[int],
    rng: RNGState,
    **cell_kwargs,
) -> List[OracleCell]:
    """Stage 1: synthetic-X oracle over the full product grid. δ list should include 0.0
    (the null self-check, where Bayes error must be ≈ 0.5)."""
    cells: List[OracleCell] = []
    for rho in rhos:
        for rate in rates:
            for beta1 in beta1s:
                for n in ns:
                    for delta in deltas:
                        cells.append(
                            compute_oracle_cell(
                                delta, beta1, rate, n, rng.spawn(), rho=rho, **cell_kwargs
                            )
                        )
    return cells


def _group_key(c: OracleCell):
    return (c.beta1, c.target_rate, c.rho, c.n)


def boundary_refinement_deltas(
    cells: List[OracleCell], be_gap: float = 0.05
) -> Dict[tuple, List[float]]:
    """For each (β1, rate, ρ, n) group, propose midpoint δ's between adjacent δ whose
    Bayes error differs by more than `be_gap` — i.e. the steep (boundary) region."""
    groups: Dict[tuple, List[OracleCell]] = {}
    for c in cells:
        groups.setdefault(_group_key(c), []).append(c)

    proposals: Dict[tuple, List[float]] = {}
    for key, group in groups.items():
        ordered = sorted(group, key=lambda c: c.delta)
        mids: List[float] = []
        for a, b in zip(ordered, ordered[1:]):
            if abs(a.bayes_error - b.bayes_error) > be_gap:
                mids.append(0.5 * (a.delta + b.delta))
        if mids:
            proposals[key] = mids
    return proposals


def refine_oracle_surface(
    cells: List[OracleCell], rng: RNGState, be_gap: float = 0.05, **cell_kwargs
) -> List[OracleCell]:
    """Stage 2: compute extra oracle cells at the boundary δ's proposed from `cells`."""
    proposals = boundary_refinement_deltas(cells, be_gap=be_gap)
    refined: List[OracleCell] = []
    for (beta1, rate, rho, n), mids in proposals.items():
        for delta in mids:
            refined.append(
                compute_oracle_cell(delta, beta1, rate, n, rng.spawn(), rho=rho, **cell_kwargs)
            )
    return refined


def run_oracle_two_stage(
    deltas: List[float],
    beta1s: List[float],
    rates: List[float],
    rhos: List[float],
    ns: List[int],
    rng: RNGState,
    be_gap: float = 0.05,
    **cell_kwargs,
) -> List[OracleCell]:
    """Full synthetic-X oracle: coarse surface + boundary refinement. Returns all cells."""
    coarse = coarse_oracle_surface(deltas, beta1s, rates, rhos, ns, rng.spawn(), **cell_kwargs)
    refined = refine_oracle_surface(coarse, rng.spawn(), be_gap=be_gap, **cell_kwargs)
    return coarse + refined


def cells_to_records(cells: List[OracleCell]) -> List[Dict]:
    """Flatten cells to JSON/CSV-ready records."""
    return [c.to_dict() for c in cells]
