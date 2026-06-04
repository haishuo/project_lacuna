"""
lacuna.survey.delta_generator

δ-parameterized own-value self-censoring on REAL multi-column survey X (PROPOSAL-P2 §3).

ONE job: produce ONE semi-synthetic training example — real survey X with a synthetic,
δ-parameterized self-censoring mechanism on ONE chosen target column — plus its known-δ
answer sheet. It generalizes the P1-validated 2-column seed
(`lacuna.feasibility.delta_generator`) to real multi-column survey X by adding a
deterministic target/predictor column-selection rule; the censoring formula, the matched
β₀ rate-solve, and the bisection are REUSED, not duplicated:

    P(missing_target_i) = sigmoid(beta0 + beta1 * Z[i, predictor] + delta * Z[i, target])

    delta == beta2      own-value dependence; MAR <=> delta == 0
    beta1               observed-coupling (MAR) nuisance, swept, >= 0
    beta0               solved so E[marginal target missing rate] == target_rate, ANY delta
                        => no rate cue between MAR and MNAR examples (charter §4.7)

Only the target column is censored; all other columns are fully observed. The observed
values are handed back on the ORIGINAL scale (`X * mask`), exactly like
`semisynthetic.apply_missingness`.

Column-selection rule (deterministic given the injected RNG):
  - target   : sampled uniformly from the NON-CONSTANT columns (or pinned via target_idx).
  - predictor: the non-constant column (!= target) with the LARGEST |Pearson corr| to the
               target on the observed X — the strongest available MAR nuisance.

Contract (Coding Bible Rules 1, 2, 6): real complete X [n, d>=2], beta1 >= 0, delta >= 0,
target_rate in (0,1); fails loud on a constant target, no valid predictor, or bad params.
Deterministic: same seed => identical mask AND answer sheet.
"""

from dataclasses import dataclass

import torch

from lacuna.core.rng import RNGState
from lacuna.data.ingestion import RawDataset
from lacuna.data.semisynthetic import _zscore_columns
from lacuna.feasibility.delta_generator import apply_self_censor

from .answer_sheet import AnswerSheet, GENERATOR_FAMILY
from .delta_bins import assign_delta_bin


@dataclass(frozen=True)
class SurveyCensorResult:
    """One generated example: the mask, the original-scale observed X, and the answer sheet."""

    mask: torch.Tensor  # [n, d] bool, True = observed (only the target column censored)
    x_observed: torch.Tensor  # [n, d] float, original scale, missing cells zeroed
    answer_sheet: AnswerSheet


def _pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson correlation between two 1-D tensors; 0.0 if either is constant."""
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).item()
    if denom == 0.0:
        return 0.0
    return float((a * b).sum().item() / denom)


def _nonconstant_columns(X: torch.Tensor) -> list:
    """Indices of columns with non-zero (population) standard deviation."""
    std = X.std(dim=0, unbiased=False)
    return [c for c in range(X.shape[1]) if float(std[c].item()) > 0.0]


def select_target_predictor(
    X: torch.Tensor, rng: RNGState, target_idx: int = None
) -> tuple:
    """Choose (target_idx, predictor_idx, corr) deterministically.

    Target is sampled uniformly from the non-constant columns unless pinned via
    `target_idx`. Predictor is the non-constant column (!= target) with the largest
    |Pearson corr| to the target on X. `corr` is the SIGNED correlation of the chosen pair.

    Raises:
        ValueError: d < 2; pinned target out of range / constant; or no valid predictor.
    """
    if X.dim() != 2:
        raise ValueError(f"X must be 2D [n, d], got shape {tuple(X.shape)}")
    n, d = X.shape
    if d < 2:
        raise ValueError(f"self-censoring requires d >= 2 (predictor + target), got d={d}")

    nonconst = _nonconstant_columns(X)
    if len(nonconst) < 2:
        raise ValueError(
            f"need >= 2 non-constant columns for (target, predictor), "
            f"found {len(nonconst)} in a d={d} dataset"
        )

    if target_idx is None:
        pick = rng.randint(0, len(nonconst), (1,)).item()
        target_idx = nonconst[pick]
    else:
        if not (0 <= target_idx < d):
            raise ValueError(f"target_idx {target_idx} out of range [0, {d})")
        if target_idx not in nonconst:
            raise ValueError(
                f"target column {target_idx} is constant; own-value self-censoring is undefined"
            )

    z_t = X[:, target_idx]
    best_idx = None
    best_abs = -1.0
    best_signed = 0.0
    for c in nonconst:
        if c == target_idx:
            continue
        corr = _pearson(z_t, X[:, c])
        if abs(corr) > best_abs:
            best_abs = abs(corr)
            best_idx = c
            best_signed = corr
    if best_idx is None:
        raise ValueError(
            f"no valid predictor column (!= target {target_idx}, non-constant) available"
        )
    return target_idx, best_idx, best_signed


def generate_self_censor_example(
    raw: RawDataset,
    *,
    beta1: float,
    delta: float,
    target_rate: float,
    rng: RNGState,
    target_idx: int = None,
) -> SurveyCensorResult:
    """Generate one semi-synthetic self-censoring example from a real survey dataset.

    Args:
        raw: complete real survey dataset (no missing values).
        beta1: observed-coupling (MAR) nuisance coefficient, >= 0.
        delta: own-value dependence (== β₂), >= 0. MAR <=> delta == 0.
        target_rate: desired expected marginal missing rate of the target column, in (0,1).
        rng: injected RNG (Rule 6); its seed is recorded in the answer sheet.
        target_idx: optional pinned target column; otherwise sampled (non-constant only).

    Returns:
        SurveyCensorResult (mask, original-scale observed X, AnswerSheet).

    Raises:
        ValueError: on any boundary violation (delegated to selection + apply_self_censor).
    """
    X = torch.from_numpy(raw.data.astype("float32"))

    # Column selection consumes its own RNG stream so the censoring draw is unaffected
    # by whether the target was pinned or sampled (stable, labeled streams).
    sel_rng = rng.spawn()
    t_idx, p_idx, corr = select_target_predictor(X, sel_rng, target_idx=target_idx)

    censor = apply_self_censor(
        X_complete=X,
        target_idx=t_idx,
        predictor_idx=p_idx,
        beta1=beta1,
        delta=delta,
        target_rate=target_rate,
        rng=rng.spawn(),
    )

    x_observed = X * censor.mask.float()

    sheet = AnswerSheet(
        source_name=raw.name,
        n=int(X.shape[0]),
        d=int(X.shape[1]),
        target_col_idx=int(t_idx),
        target_col_name=str(raw.feature_names[t_idx]),
        predictor_col_idx=int(p_idx),
        predictor_col_name=str(raw.feature_names[p_idx]),
        beta0=float(censor.params.beta0),
        beta1=float(beta1),
        delta=float(delta),
        delta_bin=assign_delta_bin(delta),
        generator_family=GENERATOR_FAMILY,
        target_rate=float(target_rate),
        realized_rate=float(censor.realized_rate),
        corr_target_predictor=float(corr),
        seed=int(rng.seed),
    )

    return SurveyCensorResult(mask=censor.mask, x_observed=x_observed, answer_sheet=sheet)


def rate_tolerance(n: int, target_rate: float, n_sigma: float = 4.0) -> float:
    """A loose, n-aware tolerance for the realized-vs-target rate check.

    The realized rate is a Bernoulli mean of n draws, so its sampling std is
    sqrt(p(1-p)/n). Small survey datasets have a genuinely wide band; this returns
    `n_sigma` standard deviations (floored to absorb solver/Monte-Carlo slack).
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if not (0.0 < target_rate < 1.0):
        raise ValueError(f"target_rate must be in (0, 1), got {target_rate}")
    if n_sigma <= 0.0:
        raise ValueError(f"n_sigma must be positive, got {n_sigma}")
    std = (target_rate * (1.0 - target_rate) / n) ** 0.5
    return max(n_sigma * std, 0.02)


def check_realized_rate(sheet: AnswerSheet, n_sigma: float = 4.0) -> bool:
    """True iff the sheet's realized rate is within an n-aware band of its target rate."""
    tol = rate_tolerance(sheet.n, sheet.target_rate, n_sigma=n_sigma)
    return abs(sheet.realized_rate - sheet.target_rate) <= tol
