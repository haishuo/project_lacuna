"""
lacuna/survey/skip_logic_generator.py

T2 per-column mechanism vocabulary + table MIXTURES (PREREGISTRATION-network-load-bearing-review §3,
commit 7714f0c). Vocabulary-3 + a matched-rate null, applied per COLUMN so each flagged column carries
a (family, δ-bin) label; a table draws 1–2 mechanisms on DIFFERENT columns (the mixture step that
breaks single-mechanism enumeration).

Family ← mechanism (the four frozen family labels):
  null            : matched-rate MCAR on the column (ignorable; δ-bin always 0).
  predictor_driven: MAR — logistic on ANOTHER (observed) column. δ-bin 0 = zero coupling
                    (= the matched-rate MAR null); δ-bin 1 = real coupling.
  self_driven     : MNAR on the column's OWN value — own_value self-censoring OR top_coding tail
                    censoring (the vocabulary's two self-driven idioms). δ-bin 0 = zero strength;
                    δ-bin 1 = active (δ=2.5 / tail).
  skip            : skip-logic — an observed GATE column drives the column's missingness. δ-bin 0 =
                    zero gate coupling; δ-bin 1 = active gate.

δ-bin mirrors G1 exactly: 0 = the matched-rate null slice (no value/predictor dependence), 1 = active.
At δ-bin 0 the four families are deliberately near-indistinguishable (all ≈ matched-rate masking) —
that unidentifiability is real and is faced identically by both review arms.

Every flagged column is calibrated to per-column missing rate ≈ TARGET_RATE (the rate-confound guard):
the family signal must live in topology/value-dependence, never in the rate. Predictors/gates stay
complete. Deterministic via injected RNGState; fail loud on infeasible draws (Rule 1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch

from lacuna.core.rng import RNGState
from lacuna.feasibility.delta_generator import _bisect_increasing

TARGET_RATE = 0.30
RATE_LO, RATE_HI = 0.25, 0.35
OWN_VALUE_DELTA = 2.5
TOPCODE_TAU_Q = 0.60
FAMILIES = ("null", "predictor_driven", "self_driven", "skip")
FAMILY_IDX = {f: i for i, f in enumerate(FAMILIES)}


@dataclass(frozen=True)
class FlaggedColumn:
    col: int
    family: str
    dbin: int
    detail: str       # the concrete idiom used (e.g. "own_value", "top_coding", "mcar")
    rate: float


def _z(col: torch.Tensor) -> torch.Tensor:
    sd = col.std()
    return (col - col.mean()) / (sd if sd > 0 else 1.0)


def _solve_rate(eta: torch.Tensor) -> float:
    """Intercept a with mean(sigmoid(a + eta)) == TARGET_RATE."""
    return _bisect_increasing(lambda a: float(torch.sigmoid(a + eta).mean()) - TARGET_RATE,
                              -30.0, 30.0)


def _mcar_col(n: int, rng: RNGState) -> torch.Tensor:
    return rng.rand(n) >= TARGET_RATE                       # True = observed


def _predictor_col(X: torch.Tensor, j: int, pred: int, active: bool, rng: RNGState) -> torch.Tensor:
    z = _z(X[:, pred])
    coef = (1.0 + float(rng.rand(1).item())) if active else 0.0    # 0 ⇒ matched-rate null
    eta = coef * z * (1.0 if rng.rand(1).item() > 0.5 else -1.0)
    a = _solve_rate(eta)
    return ~(rng.rand(X.shape[0]) < torch.sigmoid(a + eta))


def _self_col(X: torch.Tensor, j: int, idiom: str, active: bool, rng: RNGState) -> torch.Tensor:
    z = _z(X[:, j])
    if idiom == "own_value":
        eta = (OWN_VALUE_DELTA if active else 0.0) * z
        a = _solve_rate(eta)
        return ~(rng.rand(X.shape[0]) < torch.sigmoid(a + eta))
    # top_coding: censor the upper tail with high prob, the rest with a low floor
    if not active:
        return _mcar_col(X.shape[0], rng)                  # null slice = matched-rate masking
    above = z > torch.quantile(z, TOPCODE_TAU_Q)
    frac = float(above.float().mean())
    p_lo = 0.02
    p_hi = (TARGET_RATE - p_lo * (1 - frac)) / max(frac, 1e-6)
    if not (0.0 < p_hi <= 1.0):
        raise ValueError(f"top_coding infeasible on col {j}: p_hi={p_hi:.3f}")
    p = torch.where(above, torch.tensor(p_hi), torch.tensor(p_lo))
    return ~(rng.rand(X.shape[0]) < p)


def _skip_col(X: torch.Tensor, j: int, gate: int, active: bool, rng: RNGState) -> torch.Tensor:
    n = X.shape[0]
    if not active:
        return _mcar_col(n, rng)                           # null slice
    k = round(TARGET_RATE * n)
    order = torch.argsort(_z(X[:, gate]))                  # lowest-gate rows skip the column
    R = torch.ones(n, dtype=torch.bool)
    R[order[:k]] = False
    return R


def _apply_one(X: torch.Tensor, j: int, family: str, active: bool,
               rng: RNGState) -> Tuple[torch.Tensor, str]:
    """Return (observed-mask for column j, idiom detail). Predictors/gates are other columns."""
    d = X.shape[1]
    others = [c for c in range(d) if c != j]
    if family == "null":
        return _mcar_col(X.shape[0], rng), "mcar"
    if family == "predictor_driven":
        pred = others[int(rng.randint(0, len(others), (1,)).item())]
        return _predictor_col(X, j, pred, active, rng), f"mar_pred[{pred}]"
    if family == "self_driven":
        idiom = "own_value" if rng.rand(1).item() > 0.5 else "top_coding"
        return _self_col(X, j, idiom, active, rng), idiom
    if family == "skip":
        gate = others[int(rng.randint(0, len(others), (1,)).item())]
        return _skip_col(X, j, gate, active, rng), f"skip[{gate}]"
    raise ValueError(f"unknown family {family!r}")


def generate_table(X: torch.Tensor, rng: RNGState,
                   max_mech: int = 2) -> Tuple[torch.Tensor, List[FlaggedColumn]]:
    """Apply 1–2 per-column mechanisms on distinct columns of a complete X.

    Args:
        X: [n, d] complete, finite real matrix (n>=30, d>=3).
        rng: injected RNGState.
        max_mech: at most this many flagged columns (mixture size drawn in [1, max_mech]).

    Returns:
        (R, flagged) — R [n, d] bool observed-mask (unflagged columns fully observed); `flagged`
        one FlaggedColumn per imposed mechanism. Fail loud on degenerate X or infeasible rate.
    """
    if X.ndim != 2:
        raise ValueError(f"X must be [n, d], got {tuple(X.shape)}")
    n, d = X.shape
    if n < 30 or d < 3:
        raise ValueError(f"need n>=30, d>=3, got {n}x{d}")
    if not torch.isfinite(X).all():
        raise ValueError("X must be complete/finite")

    n_mech = 1 + int(rng.randint(0, max_mech, (1,)).item())          # 1..max_mech
    cols = [int(c) for c in rng.choice(d, min(n_mech, d), replace=False)]
    R = torch.ones(n, d, dtype=torch.bool)
    flagged: List[FlaggedColumn] = []
    for j in cols:
        fam = FAMILIES[int(rng.randint(0, len(FAMILIES), (1,)).item())]
        active = False if fam == "null" else bool(rng.rand(1).item() > 0.5)
        col_mask, detail = _apply_one(X, j, fam, active, rng.spawn())
        rate = float((~col_mask).float().mean())
        if not (RATE_LO <= rate <= RATE_HI):
            raise ValueError(f"{fam} on col {j}: realized rate {rate:.3f} outside band")
        R[:, j] = col_mask
        flagged.append(FlaggedColumn(col=j, family=fam, dbin=int(active), detail=detail, rate=rate))
    return R, flagged
