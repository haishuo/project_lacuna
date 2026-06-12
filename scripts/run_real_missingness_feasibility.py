"""
scripts/run_real_missingness_feasibility.py

REAL-MISSINGNESS feasibility (Stage 1 of docs/PROPOSAL-real-missingness-showdown-and-learned-
generator.md): on REAL documented ESS missingness (names stripped), can the observed data tell
apart the documented missingness MECHANISM of a cell — refusal vs don't-know vs skip(NAP)?

This is the UPPER BOUND (in-distribution, held-out respondents; strongest engineered features +
LR/GBM). Logic: if even here the classes are near-chance, no network can do better and the matrix
channel is confirmed exhausted on REAL labels. If signal exists, escalate to the network-vs-feature
showdown + leave-instrument-out (Stage 2).

Curation (documented, codebook-free, CONSERVATIVE — handles the age-77 trap):
  - ESS standardized missing families by field width: REF={7,77,777,7777} DK={8,88,888,8888}
    NA={9,99,999,9999} NAP/skip={6,66,666,6666}.
  - A column is admitted ONLY if it is a bounded integer item (valid block 0..vmax, vmax<=30) AND
    the sentinel family is strictly OUT OF RANGE (gap above vmax) — so sentinels cannot collide
    with legitimate values (excludes counts/codes/continuous like emplno/isco08p).

Determinism: fixed seed; no tuning. Fail-loud on missing data / empty classes.
Run: /home/haishuo/miniconda3/envs/lacuna/bin/python -u scripts/run_real_missingness_feasibility.py
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ESS = Path("/mnt/data/lacuna/rejected/ESS11e04_1.csv")
OUT = Path("runs/real_missingness_feasibility.json")
SEED = 2026
N_PER_CLASS = 20000  # balanced cap per documented class
REF = {7, 77, 777, 7777}
DK = {8, 88, 888, 8888}
NA = {9, 99, 999, 9999}
NAP = {6, 66, 666, 6666}
SENT = REF | DK | NA | NAP
LABELS = {"refusal": 0, "dont_know": 1, "skip": 2}


def admit_column(s):
    """Return (valid_mask_value_set, vmax) if the column is a clean bounded item, else None."""
    v = s.dropna().values
    if len(v) == 0:
        return None
    valid = v[~np.isin(v, list(SENT))]
    if len(valid) == 0:
        return None
    if not np.all(valid == valid.astype(int)):
        return None
    vmin, vmax = valid.min(), valid.max()
    if vmin < 0 or vmax > 30:
        return None
    # sentinel family must be strictly above vmax (out-of-range, no collision)
    for fams in [(6, 7, 8, 9), (66, 77, 88, 99), (666, 777, 888, 999)]:
        if any(x in s.values for x in fams) and min(fams) > vmax:
            return vmax
    return None


def cell_label(x):
    if x in REF:
        return 0
    if x in DK:
        return 1
    if x in NAP:
        return 2
    return -1  # observed value, plain-NA, or NaN — not a target


def main():
    rng = np.random.default_rng(SEED)
    print("=" * 96)
    print("REAL-MISSINGNESS FEASIBILITY — ESS documented refusal/DK/skip, names stripped (Stage 1)")
    print("=" * 96)
    ess = pd.read_csv(ESS, low_memory=False)
    num = ess.select_dtypes(include=[np.number])
    cols = [c for c in num.columns if admit_column(num[c]) is not None]
    if not cols:
        raise RuntimeError("no admissible ESS columns — curation filter rejected all")
    X = num[cols].to_numpy(np.float64)
    n, d = X.shape
    print(f"[{time.strftime('%H:%M:%S')}] admitted {d} clean bounded ESS columns, {n} respondents")

    # ---- per-cell documented labels + observed/value matrices (names stripped) ----
    lab = np.vectorize(cell_label)(X).astype(int)          # 0/1/2 target, -1 otherwise
    is_sent = np.isin(X, list(SENT))
    observed = (~np.isnan(X)) & (~is_sent)                  # genuine answered value
    Vraw = np.where(observed, X, np.nan)
    # standardize observed values per column (by observed cells only)
    mu = np.nanmean(np.where(observed, X, np.nan), axis=0)
    sd = np.nanstd(np.where(observed, X, np.nan), axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    Vz = np.where(observed, (X - mu) / sd, 0.0)

    # ---- respondent-level stats (over each row, leave-one-out applied at feature time) ----
    row_obs = observed.sum(1).astype(float)
    row_ref = (lab == 0).sum(1).astype(float)
    row_dk = (lab == 1).sum(1).astype(float)
    row_skip = (lab == 2).sum(1).astype(float)
    row_vmean = np.where(row_obs > 0, np.nansum(Vz * observed, 1) / np.maximum(row_obs, 1), 0.0)
    row_vstd = np.sqrt(np.maximum(
        np.where(row_obs > 0, np.nansum((Vz ** 2) * observed, 1) / np.maximum(row_obs, 1), 0.0)
        - row_vmean ** 2, 0.0))

    # ---- split respondents (in-distribution upper bound) ----
    perm = rng.permutation(n)
    cut = int(0.7 * n)
    tr_rows, te_rows = set(perm[:cut].tolist()), set(perm[cut:].tolist())

    # ---- column-level footprint stats computed on TRAIN rows only (no leakage) ----
    trm = np.zeros(n, bool); trm[list(tr_rows)] = True
    col_ref = (lab[trm] == 0).mean(0); col_dk = (lab[trm] == 1).mean(0)
    col_skip = (lab[trm] == 2).mean(0); col_obs = observed[trm].mean(0)
    # value entropy over observed (coarse 8-bin) + range
    col_ent = np.zeros(d); col_rng = np.zeros(d)
    for j in range(d):
        vals = X[trm, j][observed[trm, j]]
        if len(vals):
            h, _ = np.histogram(vals, bins=8); p = h / h.sum(); p = p[p > 0]
            col_ent[j] = float(-(p * np.log(p)).sum()); col_rng[j] = float(vals.max() - vals.min())
    # gate signature: max |corr| of this column's missing-indicator with other cols' observed-indicator (train rows)
    miss_ind = (lab >= 0).astype(float)  # any documented-missing
    obs_ind = observed.astype(float)
    Mtr = miss_ind[trm] - miss_ind[trm].mean(0); Otr = obs_ind[trm] - obs_ind[trm].mean(0)
    Mn = Mtr / (np.linalg.norm(Mtr, axis=0) + 1e-9); On = Otr / (np.linalg.norm(Otr, axis=0) + 1e-9)
    gate = np.abs(Mn.T @ On)                       # d x d
    np.fill_diagonal(gate, 0.0)
    col_gate = gate.max(1)

    # ---- build labeled examples (balanced cap per class) ----
    feats, ys = [], []
    counts = {0: 0, 1: 0, 2: 0}
    ridx, cidx = np.where(lab >= 0)
    order = rng.permutation(len(ridx))
    for k in order:
        i, j = int(ridx[k]), int(cidx[k])
        y = int(lab[i, j])
        if counts[y] >= N_PER_CLASS:
            continue
        # respondent stats leave-one-out on the target class count
        ro = row_obs[i]; rr = row_ref[i] - (y == 0); rd = row_dk[i] - (y == 1); rs = row_skip[i] - (y == 2)
        tot = max(ro + rr + rd + rs, 1)
        f = [rr / tot, rd / tot, rs / tot, ro / tot, row_vmean[i], row_vstd[i], np.log1p(ro),
             col_ref[j], col_dk[j], col_skip[j], col_obs[j], col_ent[j], col_rng[j], col_gate[j]]
        feats.append(f); ys.append(y); counts[y] += 1
        if all(counts[c] >= N_PER_CLASS for c in (0, 1, 2)):
            break
    F = np.asarray(feats); Y = np.asarray(ys)
    inrow = np.array([int(ridx[order[m]]) for m in range(len(ys))]) if False else None
    print(f"[{time.strftime('%H:%M:%S')}] examples: refusal={counts[0]} dont_know={counts[1]} skip={counts[2]}")
    if min(counts.values()) < 1000:
        raise RuntimeError(f"a class is too rare for a credible test: {counts}")

    # train/test by the SAME respondent split (rebuild membership for the sampled cells)
    # recover the row of each example
    rows_of = []
    cc = {0: 0, 1: 0, 2: 0}
    for k in order:
        i, j = int(ridx[k]), int(cidx[k]); y = int(lab[i, j])
        if cc[y] >= N_PER_CLASS:
            continue
        rows_of.append(i); cc[y] += 1
        if all(cc[c] >= N_PER_CLASS for c in (0, 1, 2)):
            break
    rows_of = np.asarray(rows_of)
    te = np.array([r in te_rows for r in rows_of])
    tr = ~te
    if tr.sum() == 0 or te.sum() == 0:
        raise RuntimeError("empty train or test split")

    def macro3(P, y):
        return float(roc_auc_score(y, P, multi_class="ovr", average="macro", labels=[0, 1, 2]))

    def pair_auc(P, y, a, b):
        m = np.isin(y, [a, b])
        if m.sum() < 50 or len(set(y[m])) < 2:
            return float("nan")
        return float(roc_auc_score((y[m] == b).astype(int), P[m][:, b]))

    fnames = ["row_ref", "row_dk", "row_skip", "row_obs", "row_vmean", "row_vstd", "row_lognobs",
              "col_ref", "col_dk", "col_skip", "col_obs", "col_ent", "col_rng", "col_gate"]
    # ablation: FULL features vs NO column-base-rate (drop col_ref/col_dk/col_skip — the shortcut
    # that lets a model classify a cell by recognizing its column's base rate rather than inferring
    # the mechanism). If refusal-vs-DK survives the ablation, the signal is genuine respondent/value
    # structure; if it collapses, the in-distribution number was column recognition.
    base_rate_idx = [fnames.index(x) for x in ("col_ref", "col_dk", "col_skip")]
    keep_ablate = [i for i in range(len(fnames)) if i not in base_rate_idx]
    res = {"seed": SEED, "n_cols": d, "counts": counts}
    for fset, idx in [("full", list(range(len(fnames)))), ("no_col_baserate", keep_ablate)]:
        Ft = F[:, idx]
        sc = StandardScaler().fit(Ft[tr])
        lr = LogisticRegression(max_iter=3000).fit(sc.transform(Ft[tr]), Y[tr])
        gb = HistGradientBoostingClassifier(random_state=0).fit(Ft[tr], Y[tr])
        print(f"-- feature set: {fset} ({len(idx)} feats) --")
        for name, P in [("lr", lr.predict_proba(sc.transform(Ft[te]))), ("gbm", gb.predict_proba(Ft[te]))]:
            m = macro3(P, Y[te])
            ref_dk = pair_auc(P, Y[te], 0, 1)
            skip_vs = pair_auc(P, Y[te], 1, 2)
            ref_skip = pair_auc(P, Y[te], 0, 2)
            res[f"{fset}.{name}"] = {"macro_auc": m, "refusal_vs_dk": ref_dk,
                                     "dk_vs_skip": skip_vs, "refusal_vs_skip": ref_skip}
            print(f"  {name}: macro {m:.3f} | refusal-vs-DK {ref_dk:.3f} | DK-vs-skip {skip_vs:.3f} | refusal-vs-skip {ref_skip:.3f}")
    lr = LogisticRegression(max_iter=3000).fit(StandardScaler().fit_transform(F[tr]), Y[tr])

    res["lr_coef_absmean"] = {fnames[i]: float(np.abs(lr.coef_[:, i]).mean()) for i in range(len(fnames))}
    print("\nINTERPRETATION GUIDE:")
    print("  refusal-vs-DK near 0.5 => MNAR-shadow indistinguishable from data alone (synthetic finding replicates)")
    print("  DK/refusal-vs-skip high => skip is the detectable gate idiom on real data (feature-capturable)")
    OUT.write_text(json.dumps(res, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
