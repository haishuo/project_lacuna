"""
lacuna.feasibility.delta_generator

Own-value self-censoring δ-generator for the P1 feasibility probe.

ONE mechanism family, ONE censored column. The missingness of a single target
column depends on its OWN (z-scored) value — the MNAR signature — plus an observed
predictor column (the MAR nuisance):

    P(missing_i) = sigmoid(beta0 + beta1 * Z[i, predictor] + beta2 * Z[i, target])

    delta == beta2     departure from MAR toward MNAR; MAR  <=>  beta2 == 0
    beta1              observed-coupling nuisance (swept, not the inference target)
    beta0              solved so the EXPECTED marginal missing rate hits `target_rate`

Only `positive` δ is exercised in P1: the self-censoring story is directional
(missingness rises with the latent value, e.g. high earners decline). Negative δ
(missingness falls with the value) is a valid but DISTINCT directional mechanism and
is intentionally EXCLUDED from this probe to keep the axis a single directional family;
it is a documented future extension, not an oversight.

Contract (Coding Bible Rules 1, 2, 6):
  Input : real complete X [n, d] (no missing), target/predictor indices, beta1>=0,
          delta>=0, target_rate in (0,1), injected RNGState.
  Output: SelfCensorResult(mask [n,d] bool True=observed; only `target` censored).
  Fails loud on invalid input; never silently compensates.

ISOLATION: reuses only `_zscore_columns` (the predictor view) so saturation behaviour
matches the v1.0 training path. Does not touch the production generator registry.
"""

from dataclasses import dataclass
from typing import Callable

import torch

from lacuna.core.rng import RNGState
from lacuna.data.semisynthetic import _zscore_columns


@dataclass(frozen=True)
class SelfCensorParams:
    """Frozen parameters of one self-censoring mechanism instance."""

    beta0: float
    beta1: float
    beta2: float  # == delta

    @property
    def delta(self) -> float:
        return self.beta2


@dataclass(frozen=True)
class SelfCensorResult:
    """Result of applying one self-censoring mechanism to a real X."""

    mask: torch.Tensor  # [n, d] bool, True = observed (only `target_idx` censored)
    params: SelfCensorParams
    target_idx: int
    predictor_idx: int
    target_rate: float  # requested expected rate
    realized_rate: float  # actually-sampled fraction missing in column target_idx


def _logits(
    z_p: torch.Tensor, z_t: torch.Tensor, beta0: float, beta1: float, beta2: float
) -> torch.Tensor:
    return beta0 + beta1 * z_p + beta2 * z_t


def expected_missing_rate(
    z_p: torch.Tensor, z_t: torch.Tensor, beta0: float, beta1: float, beta2: float
) -> float:
    """Population missing rate over the given sample: mean sigmoid(logits).

    Deterministic given (z_p, z_t) — no RNG. Monotone increasing in beta0.
    """
    logits = _logits(z_p, z_t, beta0, beta1, beta2)
    return float(torch.sigmoid(logits).mean().item())


def _bisect_increasing(
    f: Callable[[float], float], lo: float, hi: float, tol: float = 1e-9, max_iter: int = 200
) -> float:
    """Find root of a strictly increasing f on [lo, hi]. Fails loud if not bracketed."""
    flo, fhi = f(lo), f(hi)
    if flo > 0.0 or fhi < 0.0:
        raise ValueError(
            f"root not bracketed in [{lo}, {hi}] (f(lo)={flo:.4g}, f(hi)={fhi:.4g}); "
            f"target rate likely unreachable for these (beta1, beta2)"
        )
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        fmid = f(mid)
        if abs(fmid) < tol or (hi - lo) < tol:
            return mid
        if fmid < 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def solve_beta0_for_rate(
    z_p: torch.Tensor,
    z_t: torch.Tensor,
    beta1: float,
    beta2: float,
    target_rate: float,
    bracket: tuple = (-50.0, 50.0),
) -> float:
    """Solve beta0 so the expected marginal missing rate over (z_p, z_t) equals target_rate.

    Deterministic; defuses the rate confound by construction (every (beta1, beta2) is
    matched to the same rate). Raises if target_rate not in (0,1) or not bracketable.
    """
    if not (0.0 < target_rate < 1.0):
        raise ValueError(f"target_rate must be in (0, 1), got {target_rate}")
    return _bisect_increasing(
        lambda b0: expected_missing_rate(z_p, z_t, b0, beta1, beta2) - target_rate,
        bracket[0],
        bracket[1],
    )


def apply_self_censor(
    X_complete: torch.Tensor,
    target_idx: int,
    predictor_idx: int,
    beta1: float,
    delta: float,
    target_rate: float,
    rng: RNGState,
) -> SelfCensorResult:
    """Apply a single-column own-value self-censoring mechanism to real complete X.

    Args:
        X_complete: [n, d] real data, no missing values.
        target_idx: column to censor (missingness depends on its own z-scored value).
        predictor_idx: observed predictor column (the MAR nuisance), != target_idx.
        beta1: observed-coupling nuisance coefficient (>= 0).
        delta: == beta2, the MAR->MNAR departure (>= 0 in P1; see module docstring).
        target_rate: desired EXPECTED marginal missing rate for the target column, in (0,1).
        rng: injected RNG (Rule 6).

    Returns:
        SelfCensorResult with the full mask (only target_idx censored), solved beta0,
        and the realized (sampled) missing fraction.
    """
    if X_complete.dim() != 2:
        raise ValueError(f"X_complete must be 2D [n, d], got shape {tuple(X_complete.shape)}")
    n, d = X_complete.shape
    if d < 2:
        raise ValueError(f"self-censoring probe requires d >= 2 (predictor + target), got d={d}")
    if not (0 <= target_idx < d):
        raise ValueError(f"target_idx {target_idx} out of range [0, {d})")
    if not (0 <= predictor_idx < d):
        raise ValueError(f"predictor_idx {predictor_idx} out of range [0, {d})")
    if target_idx == predictor_idx:
        raise ValueError("target_idx and predictor_idx must differ")
    if beta1 < 0.0 or delta < 0.0:
        raise ValueError(f"P1 probe expects beta1>=0 and delta>=0, got beta1={beta1}, delta={delta}")

    Z = _zscore_columns(X_complete)
    z_t = Z[:, target_idx]
    z_p = Z[:, predictor_idx]
    if float(z_t.std(unbiased=False).item()) == 0.0:
        raise ValueError(
            f"target column {target_idx} is constant; own-value self-censoring is undefined"
        )

    beta0 = solve_beta0_for_rate(z_p, z_t, beta1, delta, target_rate)
    params = SelfCensorParams(beta0=beta0, beta1=beta1, beta2=delta)

    p_missing = torch.sigmoid(_logits(z_p, z_t, beta0, beta1, delta))
    u = rng.rand(n)
    missing = u < p_missing  # True where the cell is dropped

    R = torch.ones(n, d, dtype=torch.bool)
    R[:, target_idx] = ~missing
    realized_rate = float(missing.float().mean().item())

    return SelfCensorResult(
        mask=R,
        params=params,
        target_idx=target_idx,
        predictor_idx=predictor_idx,
        target_rate=target_rate,
        realized_rate=realized_rate,
    )
