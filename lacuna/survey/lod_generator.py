"""
lacuna.survey.lod_generator

LOD / top-coding idiom: SHARP, value-localized self-censoring on real survey X (PROPOSAL-P2.2c).

ONE job: produce one semi-synthetic example whose target column is censored by a STEP mechanism —
missingness jumps at a fixed threshold τ on the column's own (z-scored) value:

    P(missing_target_i) = sigmoid( β₀ + β₁·z_pred_i + δ · 1[ z_target_i > τ ] )

    δ ≡ the log-odds JUMP at the threshold (a logit coefficient, same units as the own-value β₂ —
        so the existing 7 δ-bins apply unchanged). MAR ⇔ δ == 0.
    τ   = a fixed quantile of z_target (e.g. 0.70), held CONSTANT across δ so the value-axis split
        is never a δ cue; only the missingness CONCENTRATION above τ varies with δ.
    β₀  solved for the matched marginal rate (ANY δ) — the amount missing is never a cue; only the
        value-localization (the hard truncation edge in the observed target marginal) carries δ.

This is the hypothesized DETECTABLE end of the spectrum (NORTH-STAR §2): a hard truncation edge is a
low-entropy observable footprint that survives matched rate, unlike smooth own-value self-censoring.

Contract (Rules 1, 2, 6): real complete X [n, d>=2], β1>=0, δ>=0, target_rate & τ_quantile in (0,1);
fails loud on a constant target / no valid predictor / bad params. Deterministic given injected RNG.
Reuses `_zscore_columns`, `_bisect_increasing`, and `select_target_predictor` (no duplication).
"""

from dataclasses import dataclass

import torch

from lacuna.core.rng import RNGState
from lacuna.data.ingestion import RawDataset
from lacuna.data.semisynthetic import _zscore_columns
from lacuna.feasibility.delta_generator import _bisect_increasing

from .answer_sheet import LOD_FAMILY, AnswerSheet
from .delta_bins import assign_delta_bin
from .delta_generator import SurveyCensorResult, select_target_predictor


@dataclass(frozen=True)
class LODParams:
    """Frozen parameters of one LOD/top-coding mechanism instance."""

    beta0: float
    beta1: float
    delta: float  # log-odds jump at the threshold
    tau: float  # z-scored threshold value


def _expected_rate(z_p, step, beta0, beta1, delta) -> float:
    logit = beta0 + beta1 * z_p + delta * step
    return float(torch.sigmoid(logit).mean().item())


def solve_beta0_lod(z_p, step, beta1, delta, target_rate, bracket=(-50.0, 50.0)) -> float:
    """Solve β₀ so the expected marginal missing rate equals target_rate (monotone in β₀)."""
    if not (0.0 < target_rate < 1.0):
        raise ValueError(f"target_rate must be in (0, 1), got {target_rate}")
    return _bisect_increasing(
        lambda b0: _expected_rate(z_p, step, b0, beta1, delta) - target_rate, bracket[0], bracket[1]
    )


def apply_lod_censor(
    X_complete: torch.Tensor,
    target_idx: int,
    predictor_idx: int,
    beta1: float,
    delta: float,
    tau_quantile: float,
    target_rate: float,
    rng: RNGState,
):
    """Apply the LOD/top-coding step mechanism to the target column of real complete X.

    Returns (mask [n,d] bool True=observed, LODParams, realized_rate, frac_above_tau).
    """
    if X_complete.dim() != 2:
        raise ValueError(f"X_complete must be 2D [n, d], got {tuple(X_complete.shape)}")
    n, d = X_complete.shape
    if d < 2:
        raise ValueError(f"LOD requires d >= 2 (predictor + target), got d={d}")
    if not (0 <= target_idx < d) or not (0 <= predictor_idx < d):
        raise ValueError("target/predictor index out of range")
    if target_idx == predictor_idx:
        raise ValueError("target_idx and predictor_idx must differ")
    if beta1 < 0.0 or delta < 0.0:
        raise ValueError(f"require beta1>=0 and delta>=0, got beta1={beta1}, delta={delta}")
    if not (0.0 < tau_quantile < 1.0):
        raise ValueError(f"tau_quantile must be in (0, 1), got {tau_quantile}")

    Z = _zscore_columns(X_complete)
    z_t = Z[:, target_idx]
    z_p = Z[:, predictor_idx]
    if float(z_t.std(unbiased=False).item()) == 0.0:
        raise ValueError(f"target column {target_idx} is constant; LOD censoring is undefined")

    tau = float(torch.quantile(z_t.to(torch.float64), tau_quantile).item())
    step = (z_t > tau).to(z_t.dtype)
    frac_above = float(step.mean().item())

    beta0 = solve_beta0_lod(z_p, step, beta1, delta, target_rate)
    p_missing = torch.sigmoid(beta0 + beta1 * z_p + delta * step)
    missing = rng.rand(n) < p_missing

    R = torch.ones(n, d, dtype=torch.bool)
    R[:, target_idx] = ~missing
    realized = float(missing.float().mean().item())
    return R, LODParams(beta0=beta0, beta1=beta1, delta=delta, tau=tau), realized, frac_above


def generate_lod_example(
    raw: RawDataset,
    *,
    beta1: float,
    delta: float,
    target_rate: float,
    tau_quantile: float,
    rng: RNGState,
    target_idx: int = None,
) -> SurveyCensorResult:
    """Generate one semi-synthetic LOD/top-coding example from a real survey dataset.

    Same column-selection rule and result/answer-sheet shape as the own-value generator; the
    answer sheet records `generator_family=lod_top_coding`, τ, and the realized fraction above τ.
    """
    X = torch.from_numpy(raw.data.astype("float32"))
    sel_rng = rng.spawn()
    t_idx, p_idx, corr = select_target_predictor(X, sel_rng, target_idx=target_idx)

    R, params, realized, frac_above = apply_lod_censor(
        X, t_idx, p_idx, beta1=beta1, delta=delta, tau_quantile=tau_quantile,
        target_rate=target_rate, rng=rng.spawn(),
    )
    x_observed = X * R.float()

    sheet = AnswerSheet(
        source_name=raw.name, n=int(X.shape[0]), d=int(X.shape[1]),
        target_col_idx=int(t_idx), target_col_name=str(raw.feature_names[t_idx]),
        predictor_col_idx=int(p_idx), predictor_col_name=str(raw.feature_names[p_idx]),
        beta0=float(params.beta0), beta1=float(beta1), delta=float(delta),
        delta_bin=assign_delta_bin(delta), generator_family=LOD_FAMILY,
        target_rate=float(target_rate), realized_rate=float(realized),
        corr_target_predictor=float(corr), seed=int(rng.seed),
        tau=float(params.tau), frac_above_tau=float(frac_above),
    )
    return SurveyCensorResult(mask=R, x_observed=x_observed, answer_sheet=sheet)
