"""
lacuna.survey.transfer_features

Transfer-robust consequence features for the LOD/top-coding idiom (PROPOSAL-P2.2c iteration).

ONE job: express the truncation consequence of the supplied target column RELATIVE to a per-column,
mechanism-untruncated reference (the fully-observed predictors / the predictor-conditional
expectation) — so that the δ→feature DIRECTION is the same across columns and the MAR (δ=0) baseline
is ≈ column-agnostic. This is the bounded hypothesis that the previous within-column features
(`consequence_features`) failed to TRANSFER leave-datasets-out (OOF AUC 0.43, direction flipped)
because their δ-direction was column-specific.

Feature groups (FROZEN; see FEATURE_NAMES):
  A — predictor-conditional residual asymmetry (ridge-fit target~others; residual e):
      skew, upper/lower spread ratio, signed reach deficit (≈0 under symmetric MAR), tail asymmetry.
  B — predictor-as-ruler (target's upper reach vs the untruncated most-correlated predictor's).
  C — anchors: matched missing_rate (no δ cue), overall corr(target,predictor).

Deterministic (quantiles/moments/ridge LS; no RNG). Scale-free by construction (ratios, z-diffs,
IQR-standardized residuals, correlations). Fail loud on bad index/shape.
"""

import torch

FEATURE_NAMES = [
    "missing_rate",            # C1 anchor (matched across δ ⇒ no δ cue)
    "corr_tp",                 # C2 anchor: overall corr(target, predictor)
    "res_skew",                # A1: residual skew (upper truncation ⇒ negative)
    "res_spread_ratio",        # A2: (q95−q50)/(q50−q05) of residual; <1 under upper cut
    "res_reach_deficit",       # A3: (q975−q50)−(q50−q025); ≈0 under symmetric MAR, <0 under top-code
    "res_tail_asym",           # A4: frac(e*>+1) − frac(e*<−1); ≈0 under MAR
    "ruler_q95_diff",          # B1: zq95(target) − zq95(predictor)
    "ruler_max_diff",          # B2: zmax(target) − zmax(predictor)
    "ruler_uppspread_diff",    # B3: (zq99−zq50)(target) − (zq99−zq50)(predictor)
    "corr_upper_minus_lower",  # B4: corr(t,p | upper half of p) − corr(t,p | lower half of p)
]
N_TRANSFER_FEATURES = len(FEATURE_NAMES)

_EPS = 1e-8
_MIN_OBS = 12
_Q = torch.tensor([0.025, 0.05, 0.25, 0.50, 0.75, 0.95, 0.975, 0.99], dtype=torch.float64)


def _corr(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a - a.mean(); b = b - b.mean()
    denom = (a.norm() * b.norm()).item()
    return 0.0 if denom == 0 else float((a * b).sum().item() / denom)


def compute_transfer_features(
    x: torch.Tensor, r: torch.Tensor, target_idx: int, *, ridge: float = 1e-3
) -> torch.Tensor:
    """Fixed [N_TRANSFER_FEATURES] transfer-robust consequence vector for column `target_idx`.

    Uses observed target rows; predictors = the other non-constant columns (fully observed).
    Returns float32. Degenerate cases (few observed / no predictor / constant) return a safe vector
    with `missing_rate` set and the rest 0.
    """
    if x.dim() != 2:
        raise ValueError(f"x must be 2D [n, d], got {tuple(x.shape)}")
    n, d = x.shape
    if not (0 <= target_idx < d):
        raise ValueError(f"target_idx {target_idx} out of range [0, {d})")
    obs = r[:, target_idx].bool()
    n_obs = int(obs.sum().item())
    f = torch.zeros(N_TRANSFER_FEATURES, dtype=torch.float64)
    f[0] = 1.0 - n_obs / n
    if n_obs < _MIN_OBS:
        return f.to(torch.float32)

    t = x[obs, target_idx].to(torch.float64)
    if float(t.std(unbiased=False)) == 0.0:
        return f.to(torch.float32)
    col_std = x.std(dim=0, unbiased=False)
    cols = [c for c in range(d) if c != target_idx and float(col_std[c]) > 0.0]
    if len(cols) == 0:
        return f.to(torch.float32)
    P = x[obs][:, cols].to(torch.float64)  # [n_obs, k] predictors at observed rows

    # --- Group A: predictor-conditional residual asymmetry ---
    A = torch.cat([P, torch.ones(P.shape[0], 1, dtype=torch.float64)], dim=1)
    beta = torch.linalg.solve(A.T @ A + ridge * torch.eye(A.shape[1], dtype=torch.float64), A.T @ t)
    e = t - A @ beta
    if float(e.std(unbiased=False)) > 0.0:
        z = (e - e.mean()) / (e.std(unbiased=False) + _EPS)
        f[2] = (z ** 3).mean()  # res_skew
        q = torch.quantile(e, _Q)  # robust quantiles of the residual
        med = q[3]
        iqr = (q[4] - q[2]).clamp_min(_EPS)
        es = (e - med) / iqr  # IQR-standardized residual
        f[3] = (q[5] - med) / ((med - q[1]).abs() + _EPS)        # A2 upper/lower spread ratio
        f[4] = ((q[6] - med) - (med - q[0])) / iqr               # A3 signed reach deficit (IQR units)
        f[5] = (es > 1.0).double().mean() - (es < -1.0).double().mean()  # A4 tail-count asymmetry

    # --- Group B: predictor-as-ruler (most-correlated other column) ---
    best_p, best_abs = None, -1.0
    for j in range(P.shape[1]):
        c = abs(_corr(t, P[:, j]))
        if c > best_abs:
            best_abs, best_p = c, j
    p = P[:, best_p]
    f[1] = _corr(t, p)  # C2 overall corr
    if float(p.std(unbiased=False)) > 0.0:
        zt = (t - t.mean()) / (t.std(unbiased=False) + _EPS)
        zp = (p - p.mean()) / (p.std(unbiased=False) + _EPS)
        qt = torch.quantile(zt, _Q)
        qp = torch.quantile(zp, _Q)
        f[6] = qt[5] - qp[5]                       # B1 zq95 diff
        f[7] = zt.max() - zp.max()                # B2 zmax diff
        f[8] = (qt[7] - qt[3]) - (qp[7] - qp[3])  # B3 upper-spread diff (q99-q50)
        pmed = p.median()
        up = p >= pmed
        lo = ~up
        if int(up.sum()) > 2 and int(lo.sum()) > 2:
            f[9] = _corr(t[up], p[up]) - _corr(t[lo], p[lo])  # B4 upper vs lower corr

    return f.to(torch.float32)


def schema() -> dict:
    """Manifest-ready description of the frozen transfer-feature set."""
    return {"n_features": N_TRANSFER_FEATURES, "feature_names": list(FEATURE_NAMES),
            "kind": "transfer_robust", "reference": "predictor_conditional_and_predictor_ruler"}
