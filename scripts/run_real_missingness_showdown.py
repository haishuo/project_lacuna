"""
scripts/run_real_missingness_showdown.py

Real-missingness Stage 2 — THE SHOWDOWN. On real ESS documented refusal-vs-DK (names stripped),
does a trained NETWORK beat the frozen-feature null, block-aware leave-country-out, 5 seeds?

- Feature null (frozen): GBM on the 10 genuine-signal features (no column-base-rate shortcut) —
  Stage-1b pooled OOF 0.767.
- Network: a permutation-invariant DeepSets row encoder. Each column is a token
  [value_z, status onehot {observed,refusal,DK,skip,plainmissing}, is_target, col population stats
  {obs_rate, entropy, range, gate}]. The TARGET cell's status is hidden (is_target=1, status zeroed);
  other cells' statuses ARE visible (the respondent's disposition — the same info the feature arm's
  row_ref/row_dk used). Context = masked mean over non-target tokens; head([context; target_token]).
  Names stripped ⇒ set encoder (no column identity) is the right, fair inductive bias.
- Both arms consume identical examples per fold. Bar (prereg discipline): network (mean−SE over 5
  seeds) − GBM ≥ +0.05 pooled leave-country-out OOF AUC.

Deterministic per seed; degenerate seeds reported. Run:
  /home/haishuo/miniconda3/envs/lacuna/bin/python -u scripts/run_real_missingness_showdown.py
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ESS = Path("/mnt/data/lacuna/rejected/ESS11e04_1.csv")
OUT = Path("runs/real_missingness_showdown.json")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = [2026, 7, 99, 13, 41]
N_CAP = 60000          # total refusal+DK examples (balanced)
BS, EPOCHS, LR = 512, 10, 2e-3
REF = {7, 77, 777, 7777}; DK = {8, 88, 888, 8888}; NA = {9, 99, 999, 9999}; NAP = {6, 66, 666, 6666}
SENT = REF | DK | NA | NAP


def admit(s):
    v = s.dropna().values
    if len(v) == 0:
        return False
    valid = v[~np.isin(v, list(SENT))]
    if len(valid) == 0 or not np.all(valid == valid.astype(int)) or valid.min() < 0 or valid.max() > 30:
        return False
    for fams in [(6, 7, 8, 9), (66, 77, 88, 99), (666, 777, 888, 999)]:
        if any(x in s.values for x in fams) and min(fams) > valid.max():
            return True
    return False


class DeepSets(nn.Module):
    def __init__(self, tok_dim, h=64):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(tok_dim, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(2 * h, h), nn.ReLU(), nn.Linear(h, 2))

    def forward(self, tokens, tgt_idx):
        # tokens [B, C, D]; tgt_idx [B]
        e = self.phi(tokens)                                   # [B, C, h]
        B, C, _ = e.shape
        tgt_oh = torch.zeros(B, C, device=e.device, dtype=torch.bool)
        tgt_oh[torch.arange(B), tgt_idx] = True
        ctx = (e * (~tgt_oh).unsqueeze(-1)).sum(1) / (C - 1)   # mean over non-target tokens
        tgt = e[torch.arange(B), tgt_idx]                      # [B, h]
        return self.head(torch.cat([ctx, tgt], -1))


def main():
    print("=" * 96)
    print(f"REAL-MISSINGNESS SHOWDOWN — network vs feature null, leave-country-out (device={DEVICE})")
    print("=" * 96)
    ess = pd.read_csv(ESS, low_memory=False)
    country = ess["cntry"].to_numpy()
    num = ess.select_dtypes(include=[np.number])
    cols = [c for c in num.columns if admit(num[c])]
    X = num[cols].to_numpy(np.float64)
    n, C = X.shape
    lab = np.where(np.isin(X, list(REF)), 0, np.where(np.isin(X, list(DK)), 1,
                  np.where(np.isin(X, list(NAP)), 2, -1))).astype(int)
    observed = (~np.isnan(X)) & (~np.isin(X, list(SENT)))
    plainmiss = np.isnan(X)
    mu = np.nanmean(np.where(observed, X, np.nan), 0); sd = np.nanstd(np.where(observed, X, np.nan), 0)
    sd = np.where(sd > 0, sd, 1.0); Vz = np.where(observed, (X - mu) / sd, 0.0).astype(np.float32)
    # fold-independent per-cell status onehot (observed, refusal, dk, skip, plainmiss)
    status = np.stack([observed, lab == 0, lab == 1, lab == 2, plainmiss], -1).astype(np.float32)  # [n,C,5]
    rng = np.random.default_rng(2026)
    countries = np.array(sorted(set(country)))
    fold_of_country = {c: int(i % 5) for i, c in enumerate(rng.permutation(countries))}
    cfold = np.array([fold_of_country[c] for c in country])

    # examples: refusal/DK cells, balanced cap
    ri, ci = np.where((lab == 0) | (lab == 1))
    y_all = lab[ri, ci]
    idx0 = np.where(y_all == 0)[0]; idx1 = np.where(y_all == 1)[0]
    per = N_CAP // 2
    sel = np.concatenate([rng.permutation(idx0)[:per], rng.permutation(idx1)[:per]])
    sel = rng.permutation(sel)
    ex_r, ex_c, ex_y = ri[sel], ci[sel], y_all[sel]
    ex_fold = cfold[ex_r]
    print(f"[{time.strftime('%H:%M:%S')}] {C} cols, {n} resp; examples {len(ex_r)} "
          f"(refusal {int((ex_y==0).sum())} / DK {int((ex_y==1).sum())})")

    row_obs = observed.sum(1).astype(float); row_ref = (lab == 0).sum(1).astype(float)
    row_dk = (lab == 1).sum(1).astype(float); row_skip = (lab == 2).sum(1).astype(float)
    row_vmean = np.where(row_obs > 0, (Vz * observed).sum(1) / np.maximum(row_obs, 1), 0.0)
    miss_ind = (lab >= 0).astype(float); obs_ind = observed.astype(float)

    def fold_colstats(train_rows):
        co = observed[train_rows].mean(0)
        ent = np.zeros(C); rg = np.zeros(C)
        for j in range(C):
            vv = X[train_rows, j][observed[train_rows, j]]
            if len(vv):
                h, _ = np.histogram(vv, bins=8); p = h / h.sum(); p = p[p > 0]
                ent[j] = -(p * np.log(p)).sum(); rg[j] = vv.max() - vv.min()
        M = miss_ind[train_rows] - miss_ind[train_rows].mean(0); O = obs_ind[train_rows] - obs_ind[train_rows].mean(0)
        Mn = M / (np.linalg.norm(M, axis=0) + 1e-9); On = O / (np.linalg.norm(O, axis=0) + 1e-9)
        g = np.abs(Mn.T @ On); np.fill_diagonal(g, 0.0)
        return co.astype(np.float32), ent.astype(np.float32), rg.astype(np.float32), g.max(1).astype(np.float32)

    def eng_feats(rs, cs, ys, co, ent, rg, gate):
        out = []
        for k in range(len(rs)):
            i, j, y = rs[k], cs[k], ys[k]
            ro = row_obs[i]; rr = row_ref[i] - (y == 0); rd = row_dk[i] - (y == 1); rsk = row_skip[i]
            tot = max(ro + rr + rd + rsk, 1)
            out.append([rr / tot, rd / tot, rsk / tot, ro / tot, row_vmean[i], np.log1p(ro),
                        co[j], ent[j], rg[j], gate[j]])
        return np.asarray(out, np.float32)

    def build_tokens(rs, cs, co, ent, rg, gate):
        # [B, C, 11]: value_z, status(5), is_target(1), col stats(4); target status zeroed + is_target=1
        B = len(rs)
        tok = np.zeros((B, C, 11), np.float32)
        tok[:, :, 0] = Vz[rs]
        tok[:, :, 1:6] = status[rs]
        colblk = np.stack([co, ent, rg, gate], -1)            # [C,4]
        tok[:, :, 7:11] = colblk[None]
        for b in range(B):
            j = cs[b]
            tok[b, j, 1:6] = 0.0     # hide target status
            tok[b, j, 6] = 1.0       # is_target
        return tok

    # ---- run folds ----
    gbm_oof = np.full(len(ex_r), np.nan)
    net_oof = {s: np.full(len(ex_r), np.nan) for s in SEEDS}
    for f in range(5):
        te = ex_fold == f; tr = ~te
        train_rows = np.where(cfold != f)[0]
        cs_stats = fold_colstats(train_rows)
        # feature null
        Ftr = eng_feats(ex_r[tr], ex_c[tr], ex_y[tr], *cs_stats)
        Fte = eng_feats(ex_r[te], ex_c[te], ex_y[te], *cs_stats)
        gb = HistGradientBoostingClassifier(random_state=0).fit(Ftr, ex_y[tr])
        gbm_oof[te] = gb.predict_proba(Fte)[:, 1]
        # network tokens
        Ttr = torch.from_numpy(build_tokens(ex_r[tr], ex_c[tr], *cs_stats))
        Tte = torch.from_numpy(build_tokens(ex_r[te], ex_c[te], *cs_stats)).to(DEVICE)
        jtr = torch.from_numpy(ex_c[tr].astype(np.int64)); jte = torch.from_numpy(ex_c[te].astype(np.int64)).to(DEVICE)
        ytr = torch.from_numpy(ex_y[tr].astype(np.int64))
        for s in SEEDS:
            torch.manual_seed(s)
            net = DeepSets(11).to(DEVICE)
            opt = torch.optim.Adam(net.parameters(), lr=LR)
            lossf = nn.CrossEntropyLoss()
            perm_rng = np.random.default_rng(s)
            for ep in range(EPOCHS):
                net.train()
                order = perm_rng.permutation(len(ytr))
                for b in range(0, len(order), BS):
                    bi = order[b:b + BS]
                    tb = Ttr[bi].to(DEVICE); jb = jtr[bi].to(DEVICE); yb = ytr[bi].to(DEVICE)
                    opt.zero_grad()
                    loss = lossf(net(tb, jb), yb)
                    loss.backward(); opt.step()
            net.eval()
            with torch.no_grad():
                P = []
                for b in range(0, Tte.shape[0], 2048):
                    P.append(torch.softmax(net(Tte[b:b + 2048], jte[b:b + 2048]), -1)[:, 1].cpu().numpy())
                net_oof[s][te] = np.concatenate(P)
        print(f"[{time.strftime('%H:%M:%S')}] fold {f} done (gbm + 5 seeds)")

    gbm_auc = float(roc_auc_score(ex_y, gbm_oof))
    seed_aucs = [float(roc_auc_score(ex_y, net_oof[s])) for s in SEEDS]
    arr = np.array(seed_aucs)
    mean, se = float(arr.mean()), float(arr.std(ddof=1) / np.sqrt(len(arr)))
    margin = (mean - se) - gbm_auc
    res = {"device": DEVICE, "gbm_oof_auc": gbm_auc, "net_seed_aucs": dict(zip(map(str, SEEDS), seed_aucs)),
           "net_mean": mean, "net_se": se, "net_mean_minus_se": mean - se, "margin_vs_gbm": margin,
           "bar": 0.05, "network_load_bearing": bool(margin >= 0.05)}
    print("\n" + "=" * 96)
    print(f"GBM null pooled OOF: {gbm_auc:.3f}")
    print(f"network per-seed: {[round(a,3) for a in seed_aucs]}")
    print(f"network mean {mean:.3f} SE {se:.3f} -> mean-SE {mean-se:.3f}")
    print(f"margin (net mean-SE) - GBM = {margin:+.3f} (bar +0.05) -> "
          f"{'NETWORK LOAD-BEARING' if margin>=0.05 else 'NOT load-bearing (parity/loss)'}")
    OUT.write_text(json.dumps(res, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
