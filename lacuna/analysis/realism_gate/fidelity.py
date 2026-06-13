"""
lacuna.analysis.realism_gate.fidelity

The fidelity panel (Dankar et al. 2022 taxonomy) + authenticity guard (Alaa et al.
2021), reported alongside the C2ST.

One job: descriptive realism metrics comparing a generated mask sample to a real
mask sample on the same columns —

- attribute  : per-column missing rates match (Dankar attribute fidelity).
- bivariate  : pairwise co-missingness correlations match (Dankar bivariate).
- population : run-length and per-row-count distributions match, via total-
               variation distance (Dankar population fidelity).
- authenticity: nearest-neighbour Hamming distance from generated rows to real
               rows, guarding against a generator merely reproducing real masks.

Note on authenticity for plasmode generators: the current generators compute
masks from real X and NEVER see real masks, so memorisation is structurally
impossible — authenticity here is a sanity panel that becomes load-bearing only
for the future *learned* generator (Part B), which will see real masks. With a
small column count the binary pattern space is tiny, so exact-match overlap is
expected and is reported, not interpreted as memorisation. See the field note in
``authenticity``.

Deterministic given ``seed`` (only the authenticity subsample is stochastic).
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.spatial.distance import cdist

from .mask_stats import (
    check_mask,
    column_rates,
    comissingness_vector,
    overall_rate,
    row_count_hist,
    run_length_hist,
)


@dataclass(frozen=True)
class FidelityResult:
    """Descriptive realism panel for one (generated, real) mask comparison."""

    overall_rate_real: float
    overall_rate_gen: float
    attr_max_gap: float        # max_j |rate_gen_j - rate_real_j|
    attr_mean_gap: float       # mean_j |rate_gen_j - rate_real_j|
    bivar_mean_gap: float      # mean pairwise co-missingness corr abs diff
    bivar_max_gap: float
    runlen_tv: float           # total-variation dist of run-length hists
    rowcount_tv: float         # total-variation dist of per-row missing-count hists
    auth_median_nn: float      # median gen->real nearest-neighbour Hamming distance
    auth_exact_dup_frac: float # fraction of gen rows exactly matching a real row


def _tv(p: np.ndarray, q: np.ndarray) -> float:
    """Total-variation distance between two normalised histograms."""
    return float(0.5 * np.abs(p - q).sum())


def _authenticity(
    M_real: np.ndarray, M_gen: np.ndarray, *, seed: int, cap: int
) -> Tuple[float, float]:
    """Median gen→real NN Hamming distance and exact-duplicate fraction.

    Subsamples both sides to ``cap`` rows for tractability (Hamming on small d).
    """
    rng = np.random.default_rng(seed)

    def _sub(M: np.ndarray) -> np.ndarray:
        if M.shape[0] <= cap:
            return M
        idx = rng.choice(M.shape[0], size=cap, replace=False)
        return M[idx]

    A = _sub(M_gen).astype(np.float64)
    B = _sub(M_real).astype(np.float64)
    # cdist 'hamming' returns fraction of differing entries; * d -> count.
    D = cdist(A, B, metric="hamming") * M_real.shape[1]
    nn = D.min(axis=1)
    return float(np.median(nn)), float((nn == 0).mean())


def run_fidelity(
    M_real: np.ndarray,
    M_gen: np.ndarray,
    *,
    seed: int,
    run_max_len: int = 8,
    count_bins: int = 10,
    auth_cap: int = 2000,
) -> FidelityResult:
    """Compute the full fidelity + authenticity panel.

    Args:
        M_real / M_gen: [n, d] uint8 masks on the same d columns.
        seed: RNG seed for the authenticity subsample.
        run_max_len / count_bins: histogram resolutions for population fidelity.
        auth_cap: per-side row cap for the authenticity NN computation.

    Raises:
        ValueError: on column mismatch or degenerate masks.
    """
    M_real = check_mask(M_real, name="M_real")
    M_gen = check_mask(M_gen, name="M_gen")
    if M_real.shape[1] != M_gen.shape[1]:
        raise ValueError(f"column mismatch: real d={M_real.shape[1]} gen d={M_gen.shape[1]}")

    rr, rg = column_rates(M_real), column_rates(M_gen)
    attr_abs = np.abs(rg - rr)

    cr, cg = comissingness_vector(M_real), comissingness_vector(M_gen)
    bivar_abs = np.abs(cg - cr)

    runlen_tv = _tv(
        run_length_hist(M_real, max_len=run_max_len),
        run_length_hist(M_gen, max_len=run_max_len),
    )
    rowcount_tv = _tv(
        row_count_hist(M_real, n_bins=count_bins),
        row_count_hist(M_gen, n_bins=count_bins),
    )
    auth_median_nn, auth_exact_dup = _authenticity(M_real, M_gen, seed=seed, cap=auth_cap)

    return FidelityResult(
        overall_rate_real=overall_rate(M_real),
        overall_rate_gen=overall_rate(M_gen),
        attr_max_gap=float(attr_abs.max()),
        attr_mean_gap=float(attr_abs.mean()),
        bivar_mean_gap=float(bivar_abs.mean()),
        bivar_max_gap=float(bivar_abs.max()),
        runlen_tv=runlen_tv,
        rowcount_tv=rowcount_tv,
        auth_median_nn=auth_median_nn,
        auth_exact_dup_frac=auth_exact_dup,
    )
