"""
lacuna.survey.topology_features

T1 FROZEN feature arm (PREREGISTRATION-network-load-bearing-review §2, commit 7714f0c). The strongest
enumerable mask-topology baseline we can construct: 37 statistics of one (X_obs, R) example, locked
before any network result exists (anti-handicap guard). Pure torch/numpy; deterministic; no RNG;
fail loud on degenerate input. The shallow arm fits LR/GBM on exactly this vector.

Groups (37 total):
  rates(4)      per-column missing-rate mean / var / range / max
  xcorr(3)      cross-column missingness |corr| mean / max / frac > 0.5
  eig(1)        top-eigenvalue share of the missingness-correlation matrix
  fano(1)       row missing-count dispersion vs binomial (Fano factor)
  block(1)      max over detected column blocks of frac rows all-or-none
  bimod(2)      column-rate variance · frac columns with rate in {<0.02, >0.98}
  gate(3)       max AUC(observed col -> row-has-missing) · point-biserial |corr| max / mean
  vshift(2)     observed-vs-missing-row mean shift on other columns, max / mean (z-units)
  conseq(17)    the frozen consequence features of the highest-missing-rate column
  anchors(3)    overall rate · log10(n) · d
"""

from typing import Dict, List

import torch

from .consequence_features import N_FEATURES as N_CONSEQ
from .consequence_features import compute_consequence_features

N_TOPO_FEATURES = 37
_EPS = 1e-9


def _rank_auc(score: torch.Tensor, label: torch.Tensor) -> float:
    """Mann-Whitney AUC of score predicting bool label (0.5 on degenerate labels)."""
    pos, neg = score[label], score[~label]
    if pos.numel() == 0 or neg.numel() == 0:
        return 0.5
    ranks = torch.argsort(torch.argsort(torch.cat([pos, neg]))).float() + 1.0
    auc = (ranks[: pos.numel()].sum() - pos.numel() * (pos.numel() + 1) / 2) / (pos.numel() * neg.numel())
    return float(auc)


def _corr(a: torch.Tensor, b: torch.Tensor) -> float:
    sa, sb = a.std(), b.std()
    if sa < _EPS or sb < _EPS:
        return 0.0
    return float(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb))


def compute_topology_features(x_obs: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """The frozen 37-dim topology vector for one example (x_obs zeroed at missing; R True=observed)."""
    if x_obs.ndim != 2 or x_obs.shape != R.shape or R.dtype != torch.bool:
        raise ValueError(f"need x_obs [n,d] + bool R same shape, got {x_obs.shape}/{R.shape} {R.dtype}")
    n, d = x_obs.shape
    if n < 10 or d < 2:
        raise ValueError(f"need n>=10, d>=2, got {n}x{d}")
    M = (~R).float()                                      # 1 = missing
    rates = M.mean(dim=0)                                 # [d]

    f: List[float] = [float(rates.mean()), float(rates.var(unbiased=False)),
                      float(rates.max() - rates.min()), float(rates.max())]

    # cross-column missingness correlation over non-degenerate columns
    nz = [j for j in range(d) if 0.0 < float(rates[j]) < 1.0]
    pair_cor = []
    for i in range(len(nz)):
        for j in range(i + 1, len(nz)):
            pair_cor.append(abs(_corr(M[:, nz[i]], M[:, nz[j]])))
    pc = torch.tensor(pair_cor) if pair_cor else torch.zeros(1)
    f += [float(pc.mean()), float(pc.max()), float((pc > 0.5).float().mean())]

    # top-eigenvalue share of the missingness correlation matrix (non-degenerate cols)
    if len(nz) >= 2:
        Z = (M[:, nz] - M[:, nz].mean(0)) / (M[:, nz].std(0) + _EPS)
        C = (Z.T @ Z) / n
        ev = torch.linalg.eigvalsh(C)
        f.append(float(ev.max() / ev.clamp_min(0).sum().clamp_min(_EPS)))
    else:
        f.append(1.0)

    # Fano factor of row missing counts (binomial baseline ~ 1 - mean_rate)
    rc = M.sum(dim=1)
    f.append(float(rc.var(unbiased=False) / rc.mean().clamp_min(_EPS)))

    # detected blocks: column groups with pairwise miss-corr > 0.5; max frac rows all-or-none
    groups: List[List[int]] = []
    used = set()
    for i in nz:
        if i in used:
            continue
        g = [i] + [j for j in nz if j != i and j not in used and abs(_corr(M[:, i], M[:, j])) > 0.5]
        if len(g) >= 2:
            groups.append(g)
            used.update(g)
    block_score = 0.0
    for g in groups:
        sub = M[:, g]
        allnone = ((sub.sum(1) == 0) | (sub.sum(1) == len(g))).float().mean()
        block_score = max(block_score, float(allnone))
    f.append(block_score)

    # column-rate bimodality
    f += [float(rates.var(unbiased=False)), float(((rates < 0.02) | (rates > 0.98)).float().mean())]

    # gate signatures
    row_has_missing = M.sum(1) > 0
    gate_aucs = [0.5]
    for j in range(d):
        if float(rates[j]) < 0.05 and 0 < int(row_has_missing.sum()) < n:
            a = _rank_auc(x_obs[:, j], row_has_missing)
            gate_aucs.append(max(a, 1.0 - a))
    f.append(max(gate_aucs))
    pb = [0.0]
    for j in nz:
        for k in range(d):
            if k != j:
                obs_k = R[:, k]
                if int(obs_k.sum()) > 10 and 0 < float(M[obs_k, j].mean()) < 1:
                    pb.append(abs(_corr(M[obs_k, j], x_obs[obs_k, k])))
    pbt = torch.tensor(pb)
    f += [float(pbt.max()), float(pbt.mean())]

    # value-conditional shift: mean(X_k | R_j missing) - mean(X_k | R_j observed), z-units
    shifts = [0.0]
    for j in nz:
        mj = ~R[:, j]
        for k in range(d):
            if k == j:
                continue
            both = R[:, k]
            a, b = both & mj, both & ~mj
            if int(a.sum()) > 5 and int(b.sum()) > 5:
                sd = x_obs[both, k].std().clamp_min(_EPS)
                shifts.append(abs(float((x_obs[a, k].mean() - x_obs[b, k].mean()) / sd)))
    st = torch.tensor(shifts)
    f += [float(st.max()), float(st.mean())]

    # consequence features of the highest-missing-rate column
    target = int(torch.argmax(rates))
    f += compute_consequence_features(x_obs, R, target).tolist()

    # anchors
    f += [float(M.mean()), float(torch.log10(torch.tensor(float(n)))), float(d)]

    out = torch.tensor(f, dtype=torch.float32)
    if out.numel() != N_TOPO_FEATURES:
        raise RuntimeError(f"feature count drifted: {out.numel()} != {N_TOPO_FEATURES}")
    if not torch.isfinite(out).all():
        raise ValueError("non-finite topology feature produced")
    return out


def schema() -> Dict:
    return {"n_features": N_TOPO_FEATURES, "consequence_block": N_CONSEQ,
            "frozen_by": "PREREGISTRATION-network-load-bearing-review (7714f0c)"}
