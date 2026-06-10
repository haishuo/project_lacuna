"""
lacuna.survey.imputation_channel

MAR-imputation counterfactual channel — EVALUATION INSTRUMENTATION ONLY
(`PROPOSAL-MAR-imputation-counterfactual-channel.md`; G1 protocol locked in
`PREREGISTRATION-G1-imputation-channel.md`, commit 6f1bff9). Not architecture; no runtime pathway.

ONE job: given a complete target column, a mechanism mask, and a paired matched-rate MCAR mask, fit
fixed-config MAR-style imputers on observed rows ONLY and summarize how their imputations fail against
the KNOWN truth at the punched cells — as PAIRED DIFFERENCES (mechanism − MCAR), which cancel intrinsic
unpredictability and imputer weakness (the proposal §6 confound-killer).

Imputer classes (fixed, auditable, NO tuning — pre-registered): linear / mice_lite / rf / gbm / nn.
Every imputer returns a Gaussian predictive form (mu, sigma) per punched cell; rf widens sigma with the
per-tree spread. Statistics (per mask): B (signed mean bias), PIT_loc (mean PIT − 0.5), PIT_tail
(frac PIT > 0.9), B_top (signed bias over the top truth-quintile cells), W1 (imputed vs truth), cov80
(central-80% interval coverage). Channel feature vector = the 6 paired differences.

Deterministic given the injected seed (Rule 6); fail loud on degenerate input (Rule 1). Truth values are
OUR punched cells from complete role-B data — never natural missingness, never multiply-imputed sources.
"""

import math
from typing import Dict, Tuple

import numpy as np

IMPUTERS = ("linear", "mice_lite", "rf", "gbm", "nn")
STAT_NAMES = ("B", "PIT_loc", "PIT_tail", "B_top", "W1", "cov80")
FEATURE_NAMES = tuple(f"d_{s}" for s in STAT_NAMES)
_MIN_OBS = 20
_MIN_MIS = 5
_SIGMA_FLOOR = 1e-6
_Z80 = 1.2816  # central-80% Gaussian half-width


def _validate(X: np.ndarray, target_idx: int, mask_obs: np.ndarray) -> None:
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D [n, d], got shape {X.shape}")
    n, d = X.shape
    if d < 2:
        raise ValueError(f"imputation needs >=1 predictor column, got d={d}")
    if not (0 <= target_idx < d):
        raise ValueError(f"target_idx {target_idx} out of range [0, {d})")
    if mask_obs.shape != (n,) or mask_obs.dtype != bool:
        raise ValueError(f"mask_obs must be bool [n={n}], got {mask_obs.shape} {mask_obs.dtype}")
    if int(mask_obs.sum()) < _MIN_OBS:
        raise ValueError(f"need >= {_MIN_OBS} observed target rows, got {int(mask_obs.sum())}")
    if int((~mask_obs).sum()) < _MIN_MIS:
        raise ValueError(f"need >= {_MIN_MIS} punched target rows, got {int((~mask_obs).sum())}")
    if not np.isfinite(X).all():
        raise ValueError("X must be finite (complete role-B data)")


def _zscore_cols(M: np.ndarray) -> np.ndarray:
    mu, sd = M.mean(axis=0), M.std(axis=0)
    sd[sd == 0] = 1.0
    return (M - mu) / sd


def _ols(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    A = np.column_stack([np.ones(len(P)), P])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coef


def _ols_predict(coef: np.ndarray, P: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(P)), P]) @ coef


def _fit_predict(name: str, P_obs, y_obs, P_mis, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Fit one fixed-config imputer on observed rows; return (mu, sigma) at punched rows."""
    if name == "linear":
        coef = _ols(P_obs, y_obs)
        resid = y_obs - _ols_predict(coef, P_obs)
        return _ols_predict(coef, P_mis), np.full(len(P_mis), max(float(resid.std()), _SIGMA_FLOOR))
    if name == "mice_lite":
        # 2 chained rounds: refine each predictor by regressing on the others (all complete here, so
        # rounds are a fixed smoothing pass), then final target regression. Auditable, no iteration on y.
        P_o, P_m = P_obs.copy(), P_mis.copy()
        for _ in range(2):
            for j in range(P_o.shape[1]):
                others = [k for k in range(P_o.shape[1]) if k != j]
                if not others:
                    continue
                cj = _ols(P_o[:, others], P_o[:, j])
                P_o[:, j] = 0.5 * P_o[:, j] + 0.5 * _ols_predict(cj, P_o[:, others])
                P_m[:, j] = 0.5 * P_m[:, j] + 0.5 * _ols_predict(cj, P_m[:, others])
        coef = _ols(P_o, y_obs)
        resid = y_obs - _ols_predict(coef, P_o)
        return _ols_predict(coef, P_m), np.full(len(P_m), max(float(resid.std()), _SIGMA_FLOOR))
    if name == "rf":
        from sklearn.ensemble import RandomForestRegressor
        rf = RandomForestRegressor(n_estimators=50, random_state=seed, n_jobs=1).fit(P_obs, y_obs)
        mu = rf.predict(P_mis)
        tree_preds = np.stack([t.predict(P_mis) for t in rf.estimators_])
        resid_var = float(np.var(y_obs - rf.predict(P_obs)))
        sigma = np.sqrt(tree_preds.var(axis=0) + max(resid_var, _SIGMA_FLOOR ** 2))
        return mu, np.maximum(sigma, _SIGMA_FLOOR)
    if name == "gbm":
        from sklearn.ensemble import GradientBoostingRegressor
        gbm = GradientBoostingRegressor(n_estimators=100, random_state=seed).fit(P_obs, y_obs)
        resid = y_obs - gbm.predict(P_obs)
        return gbm.predict(P_mis), np.full(len(P_mis), max(float(resid.std()), _SIGMA_FLOOR))
    if name == "nn":
        import torch
        torch.manual_seed(seed)
        net = torch.nn.Sequential(torch.nn.Linear(P_obs.shape[1], 32), torch.nn.ReLU(),
                                  torch.nn.Linear(32, 1))
        Xo = torch.from_numpy(P_obs).float()
        yo = torch.from_numpy(y_obs).float().unsqueeze(1)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        for _ in range(200):
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(net(Xo), yo)
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            mu = net(torch.from_numpy(P_mis).float()).numpy().ravel()
            resid = y_obs - net(Xo).numpy().ravel()
        return mu, np.full(len(P_mis), max(float(resid.std()), _SIGMA_FLOOR))
    raise ValueError(f"unknown imputer {name!r}; known: {IMPUTERS}")


def _norm_cdf(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))


def mask_stats(y_true_mis: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> Dict[str, float]:
    """The 6 per-mask statistics (proposal §1), on within-observed-standardized values."""
    if not (len(y_true_mis) == len(mu) == len(sigma)) or len(mu) < _MIN_MIS:
        raise ValueError(f"stat inputs must align with >= {_MIN_MIS} cells, got {len(mu)}")
    err = mu - y_true_mis
    u = _norm_cdf((y_true_mis - mu) / np.maximum(sigma, _SIGMA_FLOOR))
    k = max(1, len(y_true_mis) // 5)
    top = np.argsort(y_true_mis)[-k:]
    w1 = float(np.mean(np.abs(np.sort(mu) - np.sort(y_true_mis))))
    return {
        "B": float(err.mean()),
        "PIT_loc": float(u.mean() - 0.5),
        "PIT_tail": float((u > 0.9).mean()),
        "B_top": float(err[top].mean()),
        "W1": w1,
        "cov80": float((np.abs(err) <= _Z80 * sigma).mean()),
    }


def imputer_channel_features(
    X: np.ndarray, target_idx: int, mask_mech: np.ndarray, mask_mcar: np.ndarray,
    *, imputer: str, seed: int,
) -> Dict[str, float]:
    """The 6 paired-difference channel features (mech − MCAR) for one example and one imputer.

    `X` is the COMPLETE [n, d] matrix (we punched the holes; truth is known). `mask_*` are bool [n]
    observed-indicators for the TARGET column under each mask. Target and predictors are standardized
    (target within the MECH-observed cells — the φ convention; predictors over all rows).
    """
    _validate(X, target_idx, mask_mech)
    _validate(X, target_idx, mask_mcar)
    y = X[:, target_idx].astype(np.float64)
    mu_o, sd_o = y[mask_mech].mean(), y[mask_mech].std()
    if sd_o == 0:
        raise ValueError("target is constant on mech-observed rows; channel undefined")
    yz = (y - mu_o) / sd_o
    P = _zscore_cols(np.delete(X, target_idx, axis=1).astype(np.float64))
    out = {}
    for tag, m in (("mech", mask_mech), ("mcar", mask_mcar)):
        mu, sigma = _fit_predict(imputer, P[m], yz[m], P[~m], seed)
        out[tag] = mask_stats(yz[~m], mu, sigma)
    return {f"d_{s}": out["mech"][s] - out["mcar"][s] for s in STAT_NAMES}


def mcar_pair_mask(mask_mech: np.ndarray, rng) -> np.ndarray:
    """Matched-rate MCAR observed-mask: same #punched cells, positions drawn uniformly (injected RNG)."""
    n = len(mask_mech)
    n_mis = int((~mask_mech).sum())
    if n_mis < _MIN_MIS or n - n_mis < _MIN_OBS:
        raise ValueError(f"mech mask degenerate for pairing: n={n}, punched={n_mis}")
    idx = rng.choice(n, n_mis, replace=False)
    m = np.ones(n, dtype=bool)
    m[np.asarray(idx)] = False
    return m
