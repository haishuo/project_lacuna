#!/usr/bin/env python3
"""
Stage Q — richer-feature probe: can richer per-column distributional features lift the loud-vs-reject
deployable ceiling above the 5-feature baseline (~0.82 AUC)?

The Stage-Q detector reads the loud MNAR fingerprints (threshold/detection) vs the reject region
(quiet self-censoring MNAR + MAR + MCAR) off the 5 deployable features
(`column_deployable_features`: miss_rate, robust_skew, excess_kurtosis, signed_skew, smd_to_others),
ceiling AUC ~0.82. The open lead (Stage Q doc) was that RICHER per-column distributional features might
lift it. This probe measures the RF ceiling (one variable = the feature set) over >=5 seeds for:

  - base5                       : the current deployable features (control).
  - +shape9 (own-value)         : robust quantile-edge + L-moment shape features, targeting the SHARP
                                  cutoff/pileup signature that should separate loud (abrupt) from quiet
                                  self-censoring (graded). All scale-free.
  - +mar4 (cross-column)        : richer MAR-axis coupling beyond the single smd-to-others
                                  (max / top-3 / fraction-coupled / sd of per-other-column SMD) — the
                                  Stage-5 "richer MAR-axis features" lead; targets loud-vs-MAR.
  - +disc1 (density jump)       : max adjacent-bin density jump of the observed histogram — a targeted
                                  density-discontinuity feature for loud-vs-self-censoring.
  - +all                        : everything.

Matched rate, full diversity (the Stage-5 cue-free regime). Deterministic via explicit seeds; the
features are all scale-free (computed from observed values only — deployable, no oracle).

Usage:
    python scripts/stageQ_richer_features_probe.py --seeds 5
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

from lacuna.core.rng import RNGState
from lacuna.config import load_config
from lacuna.models.column_deployable_features import per_column_deployable_features, _MIN_GROUP
from lacuna.data.catalog import create_default_catalog
from lacuna.data.mixed_batch import build_mixed_batch
from lacuna.data.subtype_targets import subtype_targets
from lacuna.data.tokenization import IDX_VALUE, IDX_OBSERVED
from lacuna.priors.subtype_ontology import LIKE_INDETERMINATE

BASELINE = "/mnt/artifacts/project_lacuna/runs/stage0_general_baseline"
MIXTURE_KWARGS = dict(p_observed=0.3, target_miss_rate=0.25,
                      mnar_diverse=True, mar_diverse=True, compensate_rate=True)
_EPS = 1e-6


def load_raws(cat, names, max_cols):
    out = []
    for n in names:
        try:
            raw = cat.load(n)
        except Exception:  # noqa: BLE001
            continue
        if raw.d <= max_cols:
            out.append(raw)
    return out


def shape9(v):
    """Own-value robust shape features [9]: quantile-edge ratios + L-moment ratios (scale-free)."""
    n = len(v)
    if n < 10:
        return np.zeros(9)
    q = np.quantile(v, [0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0])
    iqr, outer = q[5] - q[3], q[7] - q[1]
    bowley = ((q[5] - q[4]) - (q[4] - q[3])) / (iqr + _EPS)
    tail_asym = ((q[7] - q[4]) - (q[4] - q[1])) / (outer + _EPS)
    moors = outer / (iqr + _EPS)
    lower_edge = (q[1] - q[0]) / (outer + _EPS)
    upper_edge = (q[8] - q[7]) / (outer + _EPS)
    low_pileup = float(np.mean(v <= q[0] + 0.1 * outer))
    high_pileup = float(np.mean(v >= q[8] - 0.1 * outer))
    vs = np.sort(v); i = np.arange(1, n + 1)
    b0 = vs.mean()
    b1 = np.sum((i - 1) / (n - 1) * vs) / n
    b2 = np.sum((i - 1) * (i - 2) / ((n - 1) * (n - 2)) * vs) / n
    b3 = np.sum((i - 1) * (i - 2) * (i - 3) / ((n - 1) * (n - 2) * (n - 3)) * vs) / n
    l2, l3, l4 = 2 * b1 - b0, 6 * b2 - 6 * b1 + b0, 20 * b3 - 30 * b2 + 12 * b1 - b0
    return np.clip(np.array([bowley, tail_asym, moors, lower_edge, upper_edge, low_pileup, high_pileup,
                             l3 / (l2 + _EPS), l4 / (l2 + _EPS)]), -10, 10)


def disc1(v):
    """Max adjacent-bin density jump of the observed histogram (a density-discontinuity signature)."""
    if len(v) < 20:
        return 0.0
    lo, hi = np.quantile(v, [0.01, 0.99])
    if hi - lo < _EPS:
        return 0.0
    h, _ = np.histogram(np.clip(v, lo, hi), bins=20)
    h = h / max(h.sum(), 1)
    return float(np.abs(np.diff(h)).max())


def mar4(b):
    """Richer MAR-axis coupling [B,C,4]: max / top-3-mean / fraction-coupled / sd of per-other-column SMD."""
    t = b.tokens
    vals = t[..., IDX_VALUE]; is_obs = t[..., IDX_OBSERVED] > 0.5
    valid = b.row_mask.unsqueeze(-1) & b.col_mask.unsqueeze(1)
    obs = (is_obs & valid).float(); miss = (valid & ~is_obs).float()
    _B, _R, C = vals.shape
    n_obs = obs.sum(1).clamp(min=1.0); mean = (vals * obs).sum(1) / n_obs
    std = ((((vals - mean.unsqueeze(1)) * obs) ** 2).sum(1) / n_obs).sqrt().clamp(min=_EPS)
    w = vals * obs
    nm = torch.einsum("brj,brk->bjk", miss, w); dm = torch.einsum("brj,brk->bjk", miss, obs)
    no = torch.einsum("brj,brk->bjk", obs, w); do = torch.einsum("brj,brk->bjk", obs, obs)
    smd = ((nm / dm.clamp(min=1.0)) - (no / do.clamp(min=1.0))).abs() / std.unsqueeze(1).clamp(min=_EPS)
    eye = torch.eye(C, dtype=torch.bool).unsqueeze(0)
    vp = (dm >= _MIN_GROUP) & (do >= _MIN_GROUP) & ~eye & b.col_mask.unsqueeze(1) & b.col_mask.unsqueeze(2)
    smd = torch.where(vp & ~(torch.isnan(smd) | torch.isinf(smd)), smd, torch.zeros_like(smd)).clamp(0, 10)
    npair = vp.float().sum(2).clamp(min=1.0)
    mx = smd.max(2).values
    top3 = smd.topk(min(3, C), dim=2).values.sum(2) / torch.clamp(npair, max=3.0)
    frac = ((smd > 0.25) & vp).float().sum(2) / npair
    sm = smd.sum(2) / npair; sd = ((smd ** 2).sum(2) / npair - sm ** 2).clamp(min=0).sqrt()
    out = torch.stack([mx, top3, frac, sd], -1)
    return (out * b.col_mask.unsqueeze(-1).float()).numpy()


def collect(raws, seed, n_batches, max_rows, max_cols):
    rng = RNGState(seed=seed)
    B5, S9, M4, D1, L = [], [], [], [], []
    for _ in range(n_batches):
        mb = build_mixed_batch(raws, rng.spawn(), max_rows=max_rows, max_cols=max_cols,
                               batch_size=16, **MIXTURE_KWARGS)
        b = mb.batch
        f5 = per_column_deployable_features(b.tokens, b.row_mask, b.col_mask).numpy()
        m4 = mar4(b)
        vals = b.tokens[..., IDX_VALUE].numpy(); isobs = b.tokens[..., IDX_OBSERVED].numpy() > 0.5
        rm = b.row_mask.numpy()
        like, _onto, sup = subtype_targets(mb)
        Bn, C = sup.shape
        for i in range(Bn):
            for j in range(C):
                if not bool(sup[i, j]):
                    continue
                v = vals[i, rm[i] & isobs[i, :, j], j]
                B5.append(f5[i, j]); S9.append(shape9(v)); M4.append(m4[i, j]); D1.append([disc1(v)])
                L.append(int(like[i, j]))
    return dict(b5=np.array(B5), s9=np.array(S9), m4=np.array(M4), d1=np.array(D1), like=np.array(L))


def loud_auc(Xtr, ytr_like, Xte, yte_like, seed):
    clf = RandomForestClassifier(n_estimators=400, class_weight="balanced", random_state=seed, n_jobs=1)
    clf.fit(Xtr, ytr_like)
    P = clf.predict_proba(Xte); cls = list(clf.classes_)
    p_loud = sum(P[:, cls.index(k)] for k in (0, 1) if k in cls)
    yl = (yte_like != LIKE_INDETERMINATE).astype(int)
    return float(roc_auc_score(yl, p_loud)), float(average_precision_score(yl, p_loud))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline-config", default=f"{BASELINE}/config.yaml")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--train-batches", type=int, default=120)
    ap.add_argument("--eval-batches", type=int, default=50)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()
    cfg = load_config(args.baseline_config)
    cat = create_default_catalog()
    train = load_raws(cat, cfg.data.train_datasets, cfg.data.max_cols)
    val = load_raws(cat, cfg.data.val_datasets, cfg.data.max_cols)
    print(f"Stage Q richer-feature probe | train {len(train)} / val {len(val)} datasets | {args.seeds} seeds")

    groups = {"base5": ["b5"], "+shape9": ["b5", "s9"], "+mar4": ["b5", "m4"],
              "+disc1": ["b5", "d1"], "+all": ["b5", "s9", "m4", "d1"]}
    res = {g: [] for g in groups}
    for s in range(args.seeds):
        seed = 20260601 + 101 * s
        tr = collect(train, seed, args.train_batches, cfg.data.max_rows, cfg.data.max_cols)
        te = collect(val, seed + 7, args.eval_batches, cfg.data.max_rows, cfg.data.max_cols)
        line = [f"seed{s+1}"]
        for g, parts in groups.items():
            Xtr = np.concatenate([tr[p] for p in parts], 1)
            Xte = np.concatenate([te[p] for p in parts], 1)
            res[g].append(loud_auc(Xtr, tr["like"], Xte, te["like"], seed))
            line.append(f"{g} {res[g][-1][0]:.3f}")
        print("  " + " | ".join(line), flush=True)

    summary = {}
    print("\n" + "=" * 64)
    print(f"loud-vs-reject ceiling (RF, {args.seeds} seeds, mean+/-sd)")
    print("=" * 64)
    for g in groups:
        a = np.array(res[g])
        summary[g] = dict(auc=[round(float(a[:, 0].mean()), 4), round(float(a[:, 0].std()), 4)],
                          ap=[round(float(a[:, 1].mean()), 4), round(float(a[:, 1].std()), 4)])
        print(f"  {g:9s} AUC {summary[g]['auc']} | AP {summary[g]['ap']}")
    print("=" * 64)
    out = args.output or Path(f"{BASELINE}/stageQ_richer_features_probe.json")
    out.write_text(json.dumps({"summary_mean_sd": summary, "per_seed": res}, indent=2))
    print(f"Wrote -> {out}")


if __name__ == "__main__":
    main()
