"""
scripts/run_real_missingness_transfer.py

Transfer probe for the ESS refusal-vs-DK signal (Stage 1b). The in-distribution feasibility run
(run_real_missingness_feasibility.py) found refusal-vs-DK ~0.74 (robust to a column-base-rate
ablation). The decisive question — does it TRANSFER, or is it sample-specific (the P2.2c
in-dist-0.94/OOF-0.43 failure mode)? Block-aware LEAVE-COUNTRY-OUT (5 folds over 30 ESS countries),
binary refusal-vs-DK, GBM, genuine-signal features (NO column-base-rate shortcut, computed on train
countries only). Pooled OOF AUC. Deterministic; fail-loud.

Run: /home/haishuo/miniconda3/envs/lacuna/bin/python -u scripts/run_real_missingness_transfer.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ESS = Path("/mnt/data/lacuna/rejected/ESS11e04_1.csv")
OUT = Path("runs/real_missingness_transfer.json")
SEED = 2026
REF = {7, 77, 777, 7777}; DK = {8, 88, 888, 8888}; NA = {9, 99, 999, 9999}; NAP = {6, 66, 666, 6666}
SENT = REF | DK | NA | NAP


def admit_column(s):
    v = s.dropna().values
    if len(v) == 0:
        return False
    valid = v[~np.isin(v, list(SENT))]
    if len(valid) == 0 or not np.all(valid == valid.astype(int)):
        return False
    if valid.min() < 0 or valid.max() > 30:
        return False
    for fams in [(6, 7, 8, 9), (66, 77, 88, 99), (666, 777, 888, 999)]:
        if any(x in s.values for x in fams) and min(fams) > valid.max():
            return True
    return False


def main():
    rng = np.random.default_rng(SEED)
    ess = pd.read_csv(ESS, low_memory=False)
    country = ess["cntry"].to_numpy()
    num = ess.select_dtypes(include=[np.number])
    cols = [c for c in num.columns if admit_column(num[c])]
    X = num[cols].to_numpy(np.float64)
    n, d = X.shape
    print(f"admitted {d} cols, {n} respondents, {len(set(country))} countries")

    lab = np.where(np.isin(X, list(REF)), 0, np.where(np.isin(X, list(DK)), 1,
                  np.where(np.isin(X, list(NAP)), 2, -1))).astype(int)
    observed = (~np.isnan(X)) & (~np.isin(X, list(SENT)))
    mu = np.nanmean(np.where(observed, X, np.nan), 0); sd = np.nanstd(np.where(observed, X, np.nan), 0)
    sd = np.where(sd > 0, sd, 1.0); Vz = np.where(observed, (X - mu) / sd, 0.0)
    row_obs = observed.sum(1).astype(float)
    row_ref = (lab == 0).sum(1).astype(float); row_dk = (lab == 1).sum(1).astype(float)
    row_skip = (lab == 2).sum(1).astype(float)
    row_vmean = np.where(row_obs > 0, (Vz * observed).sum(1) / np.maximum(row_obs, 1), 0.0)
    miss_ind = (lab >= 0).astype(float); obs_ind = observed.astype(float)

    # examples = refusal/DK cells only (binary)
    ridx, cidx = np.where((lab == 0) | (lab == 1))
    keep = rng.permutation(len(ridx))[:120000]
    ridx, cidx = ridx[keep], cidx[keep]
    countries = np.array(sorted(set(country)))
    folds = {c: int(i % 5) for i, c in enumerate(rng.permutation(countries))}
    fold_of = np.array([folds[country[i]] for i in ridx])

    def colstats(train_rows):
        cr = (lab[train_rows] == 0).mean(0); cd = (lab[train_rows] == 1).mean(0)
        cs = (lab[train_rows] == 2).mean(0); co = observed[train_rows].mean(0)
        ent = np.zeros(d); rng_ = np.zeros(d)
        for j in range(d):
            vv = X[train_rows, j][observed[train_rows, j]]
            if len(vv):
                h, _ = np.histogram(vv, bins=8); p = h / h.sum(); p = p[p > 0]
                ent[j] = -(p * np.log(p)).sum(); rng_[j] = vv.max() - vv.min()
        M = miss_ind[train_rows] - miss_ind[train_rows].mean(0)
        O = obs_ind[train_rows] - obs_ind[train_rows].mean(0)
        Mn = M / (np.linalg.norm(M, axis=0) + 1e-9); On = O / (np.linalg.norm(O, axis=0) + 1e-9)
        g = np.abs(Mn.T @ On); np.fill_diagonal(g, 0.0)
        return co, ent, rng_, g.max(1)

    def feats(ex_r, ex_c, ex_y, co, ent, rng_, gate):
        out = []
        for k in range(len(ex_r)):
            i, j, y = ex_r[k], ex_c[k], ex_y[k]
            ro = row_obs[i]; rr = row_ref[i] - (y == 0); rd = row_dk[i] - (y == 1); rs = row_skip[i]
            tot = max(ro + rr + rd + rs, 1)
            # NO column base-rate (col_ref/col_dk/col_skip) — genuine-signal features only
            out.append([rr / tot, rd / tot, rs / tot, ro / tot, row_vmean[i], np.log1p(ro),
                        co[j], ent[j], rng_[j], gate[j]])
        return np.asarray(out)

    P_oof = np.full(len(ridx), np.nan); Y = lab[ridx, cidx]
    for f in range(5):
        te = fold_of == f; tr = ~te
        train_rows = np.where(np.isin(country, countries[[fc == f0 for fc, f0 in
                              zip([folds[c] for c in country], [None])]]) == False)[0] if False else \
                     np.array([r for r in range(n) if folds[country[r]] != f])
        cs = colstats(train_rows)
        Ftr = feats(ridx[tr], cidx[tr], Y[tr], *cs)
        Fte = feats(ridx[te], cidx[te], Y[te], *cs)
        gb = HistGradientBoostingClassifier(random_state=0).fit(Ftr, Y[tr])
        P_oof[te] = gb.predict_proba(Fte)[:, 1]
        auc_f = roc_auc_score(Y[te], P_oof[te])
        print(f"  fold {f}: test countries {sorted([c for c in countries if folds[c]==f])[:4]}... OOF AUC {auc_f:.3f}")
    pooled = float(roc_auc_score(Y, P_oof))
    print(f"\nPOOLED leave-country-out refusal-vs-DK OOF AUC: {pooled:.3f}")
    print("  (in-distribution was ~0.74; compare — collapse toward 0.5 => sample-specific, "
          "holds => transferable real signal)")
    OUT.write_text(json.dumps({"pooled_oof_auc": pooled, "n_examples": int(len(ridx))}, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
