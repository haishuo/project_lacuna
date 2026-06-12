"""
scripts/run_t2_vocabulary_break.py

T2 — the vocabulary break (PREREGISTRATION-network-load-bearing-review §3, commit 7714f0c). Per-column
JOINT mechanism-family + δ-bin at vocabulary-3 with MIXTURES (1–2 mechanisms per table), block-aware
leave-one-domain-out. Does the union enumerable-feature ceiling crack when skip-logic + mixtures are
added — i.e. does the design-of-record network (φ-spine + Stage-2 mask-topology stream) beat it?

LOCKED pre-results decisions (within the prereg's stated freedoms; fixed BEFORE any T2 result):
  corpus    d>=3 continuous role-B bases from the 4 domains; per table generate_table draws 1–2
            per-column mechanisms (skip_logic_generator); matched per-column rate 0.3; deterministic
            per-cell seeds (62000+i); <=10 retried spawns on infeasible draws (logged). ~480 flagged
            columns per domain. BOTH arms consume the SAME generated (X, R, flagged) objects.
  metric    pooled OOF primary = mean(family macro-OvR AUC over the 4 families, δ-bin AUC). Reported
            components: family macro-AUC, δ-bin AUC. Two slices: skip/mixture (family==skip OR a
            2-mechanism table) and no-mixture (single-mechanism tables).
  features  FROZEN union per flagged column = 17 consequence (raw R) + 16 H/S/R no-truth (imputer=gbm;
            non-target missing cells mean-filled so predictors are complete) + 37 topology (raw R).
            LR (standardized) and HistGBM; the network must beat the BETTER of the two.
  network   T2Model (one fixed config): φ per-column + mask-topology stream + fusion + family/δ heads.
            Column-major standardized inputs (observed-cell stats); missing→0 in values, mask carried.
            Per held-out domain: train on the other 3 (10% val for early-stop), predict held-out;
            pooled OOF. Adam lr 2e-3, bs 64, <=60 epochs, patience 8 on val primary. 5 seeds.
  bar       PASS iff network (mean - SE) - better_shallow >= +0.05 pooled OOF primary AND the margin
            is concentrated on the skip/mixture slice (the no-mixture slice may be parity).

Run: /home/haishuo/miniconda3/envs/lacuna/bin/python -u scripts/run_t2_vocabulary_break.py
"""

import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from lacuna.core.rng import RNGState
from lacuna.survey.consequence_features import compute_consequence_features
from lacuna.survey.delta_head import init_parameters_
from lacuna.survey.imputation_channel import NO_TRUTH_FEATURE_NAMES, no_truth_features
from lacuna.survey.skip_logic_generator import FAMILY_IDX, generate_table
from lacuna.survey.t2_model import T2Model, T2ModelConfig
from lacuna.survey.topology_features import compute_topology_features
from scripts.run_g1_imputation_channel import DOMAINS, cont_base
from lacuna.data.semisynthetic import subsample_raw

OUT = Path("runs/t2_vocabulary_break.json")
MAX_ROWS, MIN_D = 384, 3
PER_DOMAIN = 480
IMPUTER = "gbm"
SEEDS = [2026, 7, 99, 13, 41]
LR_RATE, BS, MAX_EP, PATIENCE = 2e-3, 64, 60, 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def d3_bases():
    out = {}
    for dom in sorted(DOMAINS):
        names = []
        for name in DOMAINS[dom]:
            try:
                if cont_base(name).data.shape[1] >= MIN_D:
                    names.append(name)
            except ValueError:
                continue
        if names:
            out[dom] = names
    return out


def _standardize(X, R):
    """Column-major standardize by observed-cell stats; return (Xz with missing→0, mu, sd)."""
    n, d = X.shape
    Xz = np.zeros_like(X, np.float32)
    for j in range(d):
        obs = R[:, j]
        col = X[obs, j]
        mu, sd = (col.mean(), col.std()) if obs.sum() > 1 else (0.0, 1.0)
        sd = sd if sd > 0 else 1.0
        Xz[:, j] = np.where(R[:, j], (X[:, j] - mu) / sd, 0.0)
    return Xz


def _features(X, R, j, is_mix_others):
    """70-d union feature vector for flagged column j. Non-target missing cells mean-filled
    (predictors complete) for the no-truth channel; consequence/topology use the raw mask."""
    Xt = torch.from_numpy(X.astype(np.float32))
    Rt = torch.from_numpy(R)
    cons = compute_consequence_features(Xt * Rt.float(), Rt, j).numpy()
    topo = compute_topology_features(Xt * Rt.float(), Rt).numpy()
    view = X.astype(np.float64).copy()
    for k in range(X.shape[1]):                       # mean-fill OTHER missing columns
        if k == j:
            continue
        miss = ~R[:, k]
        if miss.any():
            view[miss, k] = X[R[:, k], k].mean()
    view[~R[:, j], j] = np.nan                        # punch the target
    hr = RNGState(seed=90000 + (j + 1) * 101 + is_mix_others)
    nt = no_truth_features(view, j, imputer=IMPUTER, seed=1234, holdout_rng=hr.spawn())
    return np.concatenate([cons, [nt[k] for k in NO_TRUTH_FEATURE_NAMES], topo]).astype(np.float32)


def build_examples():
    bases = d3_bases()
    examples, cell = [], 0
    for dom in sorted(bases):
        names = bases[dom]
        n_need = PER_DOMAIN
        produced = 0
        bi = 0
        while produced < n_need:
            name = names[bi % len(names)]
            bi += 1
            base = cont_base(name)
            rng = RNGState(seed=62000 + cell)
            cell += 1
            sub = subsample_raw(base, max_rows=MAX_ROWS, rng=rng.spawn())
            X = np.asarray(sub.data, np.float32)
            R = None
            for attempt in range(10):
                try:
                    Rt, flagged = generate_table(torch.from_numpy(X), rng.spawn())
                    R = Rt.numpy()
                    break
                except ValueError:
                    if attempt == 9:
                        R = None
            if R is None:
                continue
            is_mix = int(len(flagged) == 2)
            Xz = _standardize(X, R)
            for f in flagged:
                try:
                    feats = _features(X, R, f.col, is_mix)
                except ValueError:
                    continue                          # degenerate no-truth view (rare); skip column
                examples.append({
                    "domain": dom, "Xz": Xz, "R": R.astype(np.float32), "j": f.col,
                    "family": FAMILY_IDX[f.family], "dbin": f.dbin, "is_mix": is_mix,
                    "is_skip_or_mix": int(f.family == "skip" or is_mix), "feats": feats})
                produced += 1
        print(f"[{time.strftime('%H:%M:%S')}] domain {dom}: {produced} flagged examples")
    return examples


# ---------------------------------------------------------------- metric
def primary_metric(fam_prob, dbin_prob, fam_y, dbin_y):
    fam_y, dbin_y = np.asarray(fam_y), np.asarray(dbin_y)
    fam = roc_auc_score(fam_y, fam_prob, multi_class="ovr", average="macro", labels=[0, 1, 2, 3])
    dbin = roc_auc_score(dbin_y, dbin_prob)
    return 0.5 * (fam + dbin), float(fam), float(dbin)


def sliced(fam_prob, dbin_prob, fam_y, dbin_y, mask):
    """Primary metric restricted to a boolean slice (needs both δ classes + >=2 families)."""
    mask = np.asarray(mask, bool)
    if mask.sum() < 20 or len(set(np.asarray(dbin_y)[mask])) < 2 \
            or len(set(np.asarray(fam_y)[mask])) < 2:
        return None
    fam = roc_auc_score(np.asarray(fam_y)[mask], np.asarray(fam_prob)[mask],
                        multi_class="ovr", average="macro", labels=[0, 1, 2, 3]) \
        if len(set(np.asarray(fam_y)[mask])) == 4 else float("nan")
    dbin = roc_auc_score(np.asarray(dbin_y)[mask], np.asarray(dbin_prob)[mask])
    return float(np.nanmean([fam, dbin]))


# ---------------------------------------------------------------- feature arm
def feature_arm(examples, doms):
    res = {}
    for arm in ("lr", "gbm"):
        fam_P, dbin_P, fam_Y, dbin_Y, skipmix, nomix = [], [], [], [], [], []
        for held in doms:
            tr = [e for e in examples if e["domain"] != held]
            te = [e for e in examples if e["domain"] == held]
            Xtr = np.stack([e["feats"] for e in tr]); Xte = np.stack([e["feats"] for e in te])
            fam_tr = [e["family"] for e in tr]; dbin_tr = [e["dbin"] for e in tr]
            if arm == "lr":
                sc = StandardScaler().fit(Xtr)
                fam_clf = LogisticRegression(max_iter=3000).fit(sc.transform(Xtr), fam_tr)
                dbin_clf = LogisticRegression(max_iter=3000).fit(sc.transform(Xtr), dbin_tr)
                Xte_use = sc.transform(Xte)
            else:
                fam_clf = HistGradientBoostingClassifier(random_state=0).fit(Xtr, fam_tr)
                dbin_clf = HistGradientBoostingClassifier(random_state=0).fit(Xtr, dbin_tr)
                Xte_use = Xte
            fam_P.append(_proba4(fam_clf, Xte_use)); dbin_P.extend(dbin_clf.predict_proba(Xte_use)[:, 1])
            fam_Y.extend(e["family"] for e in te); dbin_Y.extend(e["dbin"] for e in te)
            skipmix.extend(e["is_skip_or_mix"] for e in te); nomix.extend(1 - e["is_mix"] for e in te)
        fam_P = np.vstack(fam_P)
        prim, fam, dbin = primary_metric(fam_P, dbin_P, fam_Y, dbin_Y)
        res[arm] = {"primary": prim, "family_auc": fam, "dbin_auc": dbin,
                    "skipmix_primary": sliced(fam_P, dbin_P, fam_Y, dbin_Y, skipmix),
                    "nomix_primary": sliced(fam_P, dbin_P, fam_Y, dbin_Y, nomix)}
        print(f"  shallow[{arm}]: primary {prim:.3f} (fam {fam:.3f} | δ {dbin:.3f}) "
              f"skip/mix {res[arm]['skipmix_primary']}")
    return res


def _proba4(clf, X):
    """predict_proba aligned to family classes [0,1,2,3] even if a class is absent in train."""
    p = np.zeros((X.shape[0], 4))
    for col, cls in enumerate(clf.classes_):
        p[:, int(cls)] = clf.predict_proba(X)[:, col]
    return p


# ---------------------------------------------------------------- network arm
def collate(batch, d_max, device):
    b = len(batch)
    r_max = max(e["R"].shape[0] for e in batch)
    Xv = torch.zeros(b, r_max, d_max); R = torch.zeros(b, r_max, d_max)
    rowm = torch.zeros(b, r_max, dtype=torch.bool); pcolm = torch.zeros(b, d_max, dtype=torch.bool)
    tij = torch.zeros(b, r_max); tval = torch.zeros(b, r_max)
    fam = torch.zeros(b, dtype=torch.long); dbin = torch.zeros(b)
    for i, e in enumerate(batch):
        n, d = e["R"].shape; j = e["j"]
        Xv[i, :n, :d] = torch.from_numpy(e["Xz"]); R[i, :n, :d] = torch.from_numpy(e["R"])
        rowm[i, :n] = True
        pcolm[i, :d] = True; pcolm[i, j] = False
        tij[i, :n] = torch.from_numpy(e["R"][:, j]); tval[i, :n] = torch.from_numpy(e["Xz"][:, j])
        fam[i] = e["family"]; dbin[i] = e["dbin"]
    return (Xv.to(device), R.to(device), rowm.to(device), pcolm.to(device), tij.to(device),
            tval.to(device), fam.to(device), dbin.to(device))


@torch.no_grad()
def net_predict(model, examples, d_max):
    model.eval()
    fam_P, dbin_P = [], []
    for s in range(0, len(examples), BS):
        Xv, R, rm, pm, tij, tval, _, _ = collate(examples[s:s + BS], d_max, DEVICE)
        fl, dl = model(Xv, R, rm, pm, tij, tval)
        fam_P.append(torch.softmax(fl, -1).cpu().numpy()); dbin_P.append(torch.sigmoid(dl).cpu().numpy())
    return np.vstack(fam_P), np.concatenate(dbin_P)


def train_net(tr, val, d_max, seed):
    model = T2Model(T2ModelConfig()).to(DEVICE)
    init_parameters_(model, RNGState(seed=seed))
    opt = torch.optim.Adam(model.parameters(), lr=LR_RATE)
    ce, bce = torch.nn.CrossEntropyLoss(), torch.nn.BCEWithLogitsLoss()
    vy_f = [e["family"] for e in val]; vy_d = [e["dbin"] for e in val]
    best, best_state, since = -1.0, None, 0
    for ep in range(MAX_EP):
        model.train()
        perm = np.random.default_rng(seed + ep).permutation(len(tr))
        for s in range(0, len(tr), BS):
            ex = [tr[i] for i in perm[s:s + BS]]
            Xv, R, rm, pm, tij, tval, fam, dbin = collate(ex, d_max, DEVICE)
            fl, dl = model(Xv, R, rm, pm, tij, tval)
            loss = ce(fl, fam) + bce(dl, dbin)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        fP, dP = net_predict(model, val, d_max)
        va = 0.5 * (roc_auc_score(vy_f, fP, multi_class="ovr", average="macro", labels=[0, 1, 2, 3])
                    + roc_auc_score(vy_d, dP)) if len(set(vy_d)) > 1 else 0.5
        if va > best + 1e-4:
            best, since = va, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            since += 1
            if since >= PATIENCE:
                break
    if best_state:
        model.load_state_dict(best_state)
    return model


def network_arm(examples, doms, d_max):
    per_seed = []
    for seed in SEEDS:
        fam_P, dbin_P, fam_Y, dbin_Y, skipmix, nomix = [], [], [], [], [], []
        for held in doms:
            pool = [e for e in examples if e["domain"] != held]
            te = [e for e in examples if e["domain"] == held]
            vperm = np.random.default_rng(seed * 17 + doms.index(held)).permutation(len(pool))
            nval = max(BS, len(pool) // 10)
            val = [pool[i] for i in vperm[:nval]]; tr = [pool[i] for i in vperm[nval:]]
            model = train_net(tr, val, d_max, seed)
            fP, dP = net_predict(model, te, d_max)
            fam_P.append(fP); dbin_P.extend(dP)
            fam_Y.extend(e["family"] for e in te); dbin_Y.extend(e["dbin"] for e in te)
            skipmix.extend(e["is_skip_or_mix"] for e in te); nomix.extend(1 - e["is_mix"] for e in te)
            del model
            torch.cuda.empty_cache() if DEVICE == "cuda" else None
        fam_P = np.vstack(fam_P)
        prim, fam, dbin = primary_metric(fam_P, dbin_P, fam_Y, dbin_Y)
        per_seed.append({"seed": seed, "primary": prim, "family_auc": fam, "dbin_auc": dbin,
                         "skipmix_primary": sliced(fam_P, dbin_P, fam_Y, dbin_Y, skipmix),
                         "nomix_primary": sliced(fam_P, dbin_P, fam_Y, dbin_Y, nomix)})
        print(f"[{time.strftime('%H:%M:%S')}] network seed {seed}: primary {prim:.3f} "
              f"(fam {fam:.3f} | δ {dbin:.3f}) skip/mix {per_seed[-1]['skipmix_primary']}")
    return per_seed


def main():
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    print("=" * 100)
    print(f"T2 — VOCABULARY BREAK (skip-logic + mixtures; prereg 7714f0c; device={DEVICE})")
    print("=" * 100)
    examples = build_examples()
    d_max = max(e["R"].shape[1] for e in examples)
    doms = sorted({e["domain"] for e in examples})
    print(f"total flagged examples: {len(examples)} | domains {doms} | d_max {d_max}")

    res = {"git": git, "device": DEVICE, "n_examples": len(examples), "domains": doms}
    res["shallow"] = feature_arm(examples, doms)
    best_shallow = max(res["shallow"]["lr"]["primary"], res["shallow"]["gbm"]["primary"])
    best_arm = "lr" if res["shallow"]["lr"]["primary"] >= res["shallow"]["gbm"]["primary"] else "gbm"

    per_seed = network_arm(examples, doms, d_max)
    arr = np.array([r["primary"] for r in per_seed])
    mean, se = float(arr.mean()), float(arr.std(ddof=1) / np.sqrt(len(arr)))
    res["network"] = {"per_seed": per_seed, "primary_mean": mean, "primary_se": se,
                      "mean_minus_se": mean - se}

    margin = (mean - se) - best_shallow
    # skip/mixture concentration: network advantage must live on the skip/mixture slice
    net_sm = np.nanmean([r["skipmix_primary"] for r in per_seed if r["skipmix_primary"] is not None])
    sh_sm = res["shallow"][best_arm]["skipmix_primary"]
    skipmix_margin = float(net_sm - sh_sm) if sh_sm is not None else float("nan")
    T2 = (margin >= 0.05) and (skipmix_margin >= 0.05)
    res["criteria"] = {"best_shallow": best_shallow, "best_arm": best_arm,
                       "network_mean_minus_se": mean - se, "margin": margin,
                       "net_skipmix": float(net_sm), "shallow_skipmix": sh_sm,
                       "skipmix_margin": skipmix_margin, "T2_pass": bool(T2)}
    print("\n" + "=" * 100)
    print(f"best shallow {best_shallow:.3f} ({best_arm}) | network mean-SE {mean-se:.3f} | "
          f"margin {margin:+.3f} (need >=+0.05)")
    print(f"skip/mix: network {net_sm:.3f} - shallow {sh_sm} = {skipmix_margin:+.3f} (need >=+0.05)")
    print(f"T2 VERDICT: {'PASS — vocabulary-scale network advantage on skip/mixtures' if T2 else 'FAIL'}")
    OUT.write_text(json.dumps(res, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
