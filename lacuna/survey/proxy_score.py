"""
lacuna.survey.proxy_score

Operational proxy-strength estimator — R²(target | observed predictors) (PROPOSAL-P2 §6).

ONE job: from observed data alone, estimate how well the OTHER observed columns linearly predict a
target column's observed values. proxy-score ≈ 1 − R². Small proxy-score ⇒ the target is nearly
reconstructible from the observed columns ⇒ own-value MNAR ≈ MAR-on-proxy (the P1R-C regime).

**Caveat (recorded, per §6):** under MNAR the observed target rows are TRUNCATED (high/low values
preferentially missing), so an R² fit on observed rows is BIASED — this is a *conservative
operational* signal, not the true Var(z_t | z_t-unobserved). A debiased estimator is a later
refinement; here we use it as the governance/abstention variable with this caveat documented.

Deterministic (Rule 6): pure function of its tensor inputs (ridge least squares). Fail loud
(Rule 1) on a constant observed target, too few observed rows, or a bad target index.
"""

from typing import List

import torch

from lacuna.data.ingestion import RawDataset


def column_r2(
    x: torch.Tensor, r: torch.Tensor, target_idx: int, *, ridge: float = 1e-6
) -> float:
    """In-sample R² of the target column's OBSERVED values on the other (non-constant) columns.

    Args:
        x: [n, d] data (missing target cells may be zeroed — only observed rows are used).
        r: [n, d] bool observed mask (True = observed).
        target_idx: column whose observed values are regressed on the others.
        ridge: tiny L2 on the normal equations for numerical stability.

    Returns:
        R² clamped to [0, 1]. (1 ⇒ target fully reconstructible from observed columns.)

    Raises:
        ValueError: bad index; fewer observed rows than predictors+2; constant observed target.
    """
    if x.dim() != 2:
        raise ValueError(f"x must be 2D [n, d], got {tuple(x.shape)}")
    n, d = x.shape
    if not (0 <= target_idx < d):
        raise ValueError(f"target_idx {target_idx} out of range [0, {d})")
    obs = r[:, target_idx].bool()
    y = x[obs, target_idx].float()
    cols = [c for c in range(d) if c != target_idx]
    if len(cols) == 0:
        raise ValueError("need >= 1 predictor column for R²")
    Xp = x[obs][:, cols].float()
    # drop constant predictors on the observed subset to avoid singularity
    std = Xp.std(dim=0, unbiased=False)
    keep = std > 0
    Xp = Xp[:, keep]
    if Xp.shape[1] == 0:
        return 0.0  # no informative predictor ⇒ target not linearly reconstructible
    if y.numel() < Xp.shape[1] + 2:
        raise ValueError(
            f"too few observed target rows ({y.numel()}) for {Xp.shape[1]} predictors"
        )
    yc = y - y.mean()
    ss_tot = float((yc * yc).sum().item())
    if ss_tot == 0.0:
        raise ValueError(f"observed target column {target_idx} is constant; R² undefined")

    A = torch.cat([Xp, torch.ones(Xp.shape[0], 1)], dim=1)  # + intercept
    AtA = A.T @ A
    lam = ridge * torch.eye(A.shape[1], dtype=A.dtype)
    beta = torch.linalg.solve(AtA + lam, A.T @ y)
    resid = y - A @ beta
    ss_res = float((resid * resid).sum().item())
    r2 = 1.0 - ss_res / ss_tot
    return max(0.0, min(1.0, r2))


def proxy_score(x: torch.Tensor, r: torch.Tensor, target_idx: int) -> float:
    """Operational proxy-score = 1 − R²(target | observed). Small ⇒ strong proxy (absorption)."""
    return 1.0 - column_r2(x, r, target_idx)


def target_r2_table(raw_datasets: List[RawDataset]) -> List[dict]:
    """Full-data (no censoring) R² for every non-constant candidate target of every dataset.

    Used to CHARACTERIZE the empirical R² distribution before stratification (the safeguard that
    'low' and 'high' R² strata are genuinely different). Each row:
    {dataset, target_idx, target_name, r2, n, d}.
    """
    table: List[dict] = []
    for raw in raw_datasets:
        x = torch.from_numpy(raw.data.astype("float32"))
        n, d = x.shape
        if d < 2:
            continue
        r = torch.ones(n, d, dtype=torch.bool)
        std = x.std(dim=0, unbiased=False)
        nonconst = [c for c in range(d) if float(std[c].item()) > 0.0]
        if len(nonconst) < 2:
            continue
        for t in nonconst:
            r2 = column_r2(x, r, t)
            table.append({
                "dataset": raw.name, "target_idx": int(t),
                "target_name": str(raw.feature_names[t]), "r2": r2, "n": int(n), "d": int(d),
            })
    return table


def r2_distribution(table: List[dict]) -> dict:
    """min / q25 / median / q75 / max of the R² values in a target_r2_table (the safeguard report)."""
    if len(table) == 0:
        raise ValueError("empty R² table")
    vals = torch.tensor([row["r2"] for row in table], dtype=torch.float64)
    q = torch.quantile(vals, torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float64))
    return {
        "n_targets": len(table),
        "min": float(q[0]), "q25": float(q[1]), "median": float(q[2]),
        "q75": float(q[3]), "max": float(q[4]),
    }
