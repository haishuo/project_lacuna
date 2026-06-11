"""
lacuna.survey.topology_generators

T1 matched-rate multi-column mechanism vocabulary (PREREGISTRATION-network-load-bearing-review §2,
commit 7714f0c). EIGHT mechanisms over a complete real-X matrix, every one calibrated to the SAME
overall missingness rate (the v1.0 rate-confound guard): the class signal must live in the mask
TOPOLOGY and value-dependence, never in the rate.

Mechanisms (class → generators):
  MCAR : mcar_uniform · mcar_rotated_booklet  (ported semantics: families/mcar/blocks.MCARRotatedBooklet)
  MAR  : mar_single · mar_multi · mar_module_skip  (ported semantics: families/mar/survey.MARSkipLogic)
  MNAR : mnar_own_value · mnar_top_coding · mnar_module_refusal
         (module_refusal = the PHQ-9 row-aligned pattern: identical topology to module_skip but the
          block's OWN values drive the refusal — the literature-grounded hard pair)

Interface: each generator is `fn(X [n,d] float tensor, rng: RNGState) -> R [n,d] bool (True=observed)`.
Calibration: continuous-knob mechanisms solve their intercept by bisection so the EXPECTED overall
missing fraction = TARGET_RATE; deterministic block mechanisms solve their block/quantile algebra
exactly. Every mask is verified: realized rate within [0.25, 0.35] else fail loud (Rule 1); realized
rate is returned to the caller for the auditable rate feature. Deterministic via injected RNGState.
"""

import math
from typing import Callable, Dict, List, Tuple

import torch

from lacuna.core.rng import RNGState
from lacuna.feasibility.delta_generator import _bisect_increasing

TARGET_RATE = 0.30
RATE_LO, RATE_HI = 0.25, 0.35
OWN_VALUE_DELTA = 2.5
TOPCODE_TAU_Q = 0.60

CLASS_OF = {
    "mcar_uniform": "MCAR", "mcar_rotated_booklet": "MCAR",
    "mar_single": "MAR", "mar_multi": "MAR", "mar_module_skip": "MAR",
    "mnar_own_value": "MNAR", "mnar_top_coding": "MNAR", "mnar_module_refusal": "MNAR",
}
GENERATORS: Tuple[str, ...] = tuple(CLASS_OF)
CLASS_IDX = {"MCAR": 0, "MAR": 1, "MNAR": 2}


def _z(X: torch.Tensor) -> torch.Tensor:
    mu, sd = X.mean(dim=0), X.std(dim=0)
    sd = torch.where(sd > 0, sd, torch.ones_like(sd))
    return (X - mu) / sd


def _check(X: torch.Tensor, name: str) -> Tuple[int, int]:
    if X.ndim != 2:
        raise ValueError(f"{name}: X must be [n, d], got {tuple(X.shape)}")
    n, d = X.shape
    if n < 30 or d < 3:
        raise ValueError(f"{name}: need n>=30, d>=3 (got {n}x{d}) for topology mechanisms")
    if not torch.isfinite(X).all():
        raise ValueError(f"{name}: X must be complete/finite")
    return n, d


def _verify(R: torch.Tensor, name: str) -> torch.Tensor:
    rate = float((~R).float().mean())
    if not (RATE_LO <= rate <= RATE_HI):
        raise ValueError(f"{name}: realized rate {rate:.3f} outside [{RATE_LO}, {RATE_HI}]")
    return R


def _solve_intercept(eta_core: torch.Tensor, cells: int) -> float:
    """Solve a in mean(sigmoid(a + eta_core)) * (numel/cells_total) == TARGET_RATE over given cells."""
    target = TARGET_RATE * cells / eta_core.numel() if eta_core.numel() != cells else TARGET_RATE
    return _bisect_increasing(
        lambda a: float(torch.sigmoid(a + eta_core).mean()) - target, -30.0, 30.0)


# ---------------------------------------------------------------- MCAR
def mcar_uniform(X: torch.Tensor, rng: RNGState) -> torch.Tensor:
    n, d = _check(X, "mcar_uniform")
    R = rng.rand(n, d) >= TARGET_RATE
    return _verify(R, "mcar_uniform")


def mcar_rotated_booklet(X: torch.Tensor, rng: RNGState) -> torch.Tensor:
    """K blocks; each booklet-row misses exactly one block; participation prob p sets the rate exactly.

    rate = p * n_rot / (K * d); n_rot = smallest multiple of K with n_rot >= TARGET_RATE*K*d gives
    p = TARGET_RATE*K*d/n_rot in (0, 1]. Assignment independent of values => MCAR-by-design.
    """
    n, d = _check(X, "mcar_rotated_booklet")
    K = int(rng.randint(2, 4, (1,)).item())            # K in {2, 3}
    n_rot = K * math.ceil(TARGET_RATE * d)             # multiple of K, >= 0.3*K*d
    if n_rot > d:
        n_rot = d - (d % K)
    p = TARGET_RATE * K * d / n_rot
    if not (0.0 < p <= 1.0) or n_rot < K:
        raise ValueError(f"mcar_rotated_booklet infeasible: d={d}, K={K}, n_rot={n_rot}, p={p:.3f}")
    cols = [int(i) for i in rng.choice(d, n_rot, replace=False)]
    blocks = [cols[i::K] for i in range(K)]
    assign = rng.randint(0, K, (n,))
    participate = rng.rand(n) < p
    R = torch.ones(n, d, dtype=torch.bool)
    for k, blk in enumerate(blocks):
        rows = torch.where((assign == k) & participate)[0]
        R[rows.unsqueeze(1), torch.as_tensor(blk).unsqueeze(0)] = False
    return _verify(R, "mcar_rotated_booklet")


# ---------------------------------------------------------------- MAR
def _mar_logistic(X, rng, n_pred: int, name: str) -> torch.Tensor:
    n, d = _check(X, name)
    Z = _z(X)
    # keep affected >= ceil(d/2) so the per-affected-cell target rate stays <= 0.6 (feasible)
    n_pred_eff = max(1, min(n_pred, d - math.ceil(d / 2)))
    preds = [int(i) for i in rng.choice(d, n_pred_eff, replace=False)]
    affected = [j for j in range(d) if j not in preds]
    w = torch.where(rng.rand(len(preds)) > 0.5, 1.0, -1.0) * (1.0 + rng.rand(len(preds)))
    eta = (Z[:, preds] * w).sum(dim=1, keepdim=True).expand(n, len(affected))
    a = _solve_intercept(eta, cells=n * d)
    miss = rng.rand(n, len(affected)) < torch.sigmoid(a + eta)
    R = torch.ones(n, d, dtype=torch.bool)
    R[:, affected] = ~miss
    return _verify(R, name)


def mar_single(X: torch.Tensor, rng: RNGState) -> torch.Tensor:
    return _mar_logistic(X, rng, 1, "mar_single")


def mar_multi(X: torch.Tensor, rng: RNGState) -> torch.Tensor:
    return _mar_logistic(X, rng, 3, "mar_multi")


def _block_rows(n: int, d: int, rng) -> Tuple[List[int], int, float]:
    """Shared block algebra for the module pair: block columns + #refusing rows for exact rate."""
    b = max(2, min(d - 1, round(d * 0.5)))
    cols = [int(i) for i in rng.choice(d, b, replace=False)]
    q = TARGET_RATE * d / b                      # fraction of rows hitting the block
    if not (0.05 <= q <= 0.95):
        raise ValueError(f"module block infeasible: q={q:.3f} for d={d}, b={b}")
    return cols, b, q


def mar_module_skip(X: torch.Tensor, rng: RNGState) -> torch.Tensor:
    """Observed GATE column drives a row-aligned block (ported MARSkipLogic semantics)."""
    n, d = _check(X, "mar_module_skip")
    Z = _z(X)
    cols, b, q = _block_rows(n, d, rng)
    gate_choices = [j for j in range(d) if j not in cols]
    gate = gate_choices[int(rng.randint(0, len(gate_choices), (1,)).item())]
    k = round(q * n)
    rows = torch.argsort(Z[:, gate])[:k]         # lowest-gate rows skip the block
    R = torch.ones(n, d, dtype=torch.bool)
    R[rows.unsqueeze(1), torch.as_tensor(cols).unsqueeze(0)] = False
    return _verify(R, "mar_module_skip")


# ---------------------------------------------------------------- MNAR
def mnar_module_refusal(X: torch.Tensor, rng: RNGState) -> torch.Tensor:
    """Row-aligned block driven by the BLOCK'S OWN values (PHQ-9 pattern; topology == module_skip)."""
    n, d = _check(X, "mnar_module_refusal")
    Z = _z(X)
    cols, b, q = _block_rows(n, d, rng)
    k = round(q * n)
    score = Z[:, cols].mean(dim=1)               # the block's own (latent) severity
    rows = torch.argsort(score, descending=True)[:k]   # highest-severity rows refuse the block
    R = torch.ones(n, d, dtype=torch.bool)
    R[rows.unsqueeze(1), torch.as_tensor(cols).unsqueeze(0)] = False
    return _verify(R, "mnar_module_refusal")


def mnar_own_value(X: torch.Tensor, rng: RNGState) -> torch.Tensor:
    """Every affected column self-censors on its OWN z-value at fixed delta."""
    n, d = _check(X, "mnar_own_value")
    Z = _z(X)
    keep = int(rng.randint(0, d, (1,)).item())   # one always-observed predictor column
    affected = [j for j in range(d) if j != keep]
    eta = OWN_VALUE_DELTA * Z[:, affected]
    a = _solve_intercept(eta, cells=n * d)
    miss = rng.rand(n, len(affected)) < torch.sigmoid(a + eta)
    R = torch.ones(n, d, dtype=torch.bool)
    R[:, affected] = ~miss
    return _verify(R, "mnar_own_value")


def mnar_top_coding(X: torch.Tensor, rng: RNGState) -> torch.Tensor:
    """Cells above each affected column's tau-quantile censored with calibrated probability."""
    n, d = _check(X, "mnar_top_coding")
    Z = _z(X)
    keep = int(rng.randint(0, d, (1,)).item())
    affected = [j for j in range(d) if j != keep]
    Za = Z[:, affected]
    # adapt tau to width AND to ties (discrete real columns can leave < (1-tau) mass above the
    # quantile): lower tau until the censorable mass suffices for p_hi <= 1.
    p_lo = 0.02
    tau = min(TOPCODE_TAU_Q, 1.0 - (TARGET_RATE / 0.8) * d / (d - 1))
    p_hi, above = 2.0, None
    while tau >= 0.05:
        above = Za > torch.quantile(Za, tau, dim=0, keepdim=True)
        frac_above = float(above.float().mean()) * len(affected) / d
        p_hi = (TARGET_RATE - p_lo * (len(affected) / d - frac_above)) / max(frac_above, 1e-6)
        if 0.0 < p_hi <= 1.0:
            break
        tau -= 0.05
    if not (0.0 < p_hi <= 1.0):
        raise ValueError(f"mnar_top_coding infeasible even at tau=0.05: p_hi={p_hi:.3f}")
    p = torch.where(above, p_hi, p_lo)
    miss = rng.rand(n, len(affected)) < p
    R = torch.ones(n, d, dtype=torch.bool)
    R[:, affected] = ~miss
    return _verify(R, "mnar_top_coding")


GENERATOR_FNS: Dict[str, Callable] = {
    "mcar_uniform": mcar_uniform, "mcar_rotated_booklet": mcar_rotated_booklet,
    "mar_single": mar_single, "mar_multi": mar_multi, "mar_module_skip": mar_module_skip,
    "mnar_own_value": mnar_own_value, "mnar_top_coding": mnar_top_coding,
    "mnar_module_refusal": mnar_module_refusal,
}


def generate(name: str, X: torch.Tensor, rng: RNGState) -> Tuple[torch.Tensor, int, float]:
    """Apply mechanism `name`; return (R, class_idx, realized_rate). Fail loud on unknown name."""
    if name not in GENERATOR_FNS:
        raise ValueError(f"unknown T1 generator {name!r}; known: {GENERATORS}")
    R = GENERATOR_FNS[name](X, rng)
    return R, CLASS_IDX[CLASS_OF[name]], float((~R).float().mean())
