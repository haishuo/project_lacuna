"""
scripts/run_t1_showdown.py

T1 — the mask-topology showdown (PREREGISTRATION-network-load-bearing-review §2, commit 7714f0c).
At MATCHED overall rate 0.3: does the v1.0-family network beat the FROZEN 37-statistic feature arm
(LR + HistGBM, better-of) for 3-class mechanism discrimination, block-aware leave-one-domain-out?

LOCKED pre-results implementation decisions (within the prereg's stated freedoms):
  corpus    bases with d>=3: labor={psid1976,cps1985,workinghours} nhanes={rb_nhanes_weight}
            hmda={survey_hmda} wealth={rb_scf2022_wealth_cont}; examples = 128-row subsamples x all
            columns; per LODO split: train 2400 / val 480 (train domains), test 480 (held-out domain),
            balanced over the 8 generators; deterministic per-cell seeds (71000+i); generators
            retried <=10 spawns on infeasible draws (logged).
  features  the frozen 37 topology statistics; LR(multinomial, standardized) + HistGradientBoosting
            (defaults); pooled OOF macro one-vs-rest AUC.
  network   LacunaModel, v1.0 dims (hidden 128, evidence 64, 4 layers, 4 heads, max_cols 48),
            use_missingness_features=False (no hand-fed stats — that IS the question); loss =
            class_cross_entropy(p_class) + 1.0 * multi_head_reconstruction_loss (BERT-style 15%
            artificial masking of observed cells, per-epoch deterministic); Adam lr 3e-4, bs 16,
            <=40 epochs, patience 6 on val macro-AUC; 5 seeds x 4 splits; GPU if available.
  bar       network (mean-SE over seeds) - max(LR, GBM) >= +0.05 pooled OOF macro-AUC, AND network
            MCAR-vs-not >= shallow - 0.02. Secondaries reported: MCAR-vs-not, skip-vs-refusal.
  both arms consume IDENTICAL example objects (bit-identity by construction).

Run: /home/haishuo/miniconda3/envs/lacuna/bin/python -u scripts/run_t1_showdown.py
"""

import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from lacuna.core.rng import RNGState
from lacuna.data.catalog import create_default_catalog
from lacuna.data.tokenization import MaskingConfig, apply_artificial_masking, tokenize_and_batch
from lacuna.models.assembly import LacunaModel, LacunaModelConfig
from lacuna.survey.topology_features import compute_topology_features
from lacuna.survey.topology_generators import CLASS_IDX, GENERATORS, generate
from lacuna.training.loss import class_cross_entropy, multi_head_reconstruction_loss

RB = Path("/mnt/data/lacuna/role_b")
OUT = Path("runs/t1_showdown.json")
DOMAINS = {"labor": ["survey_psid1976", "survey_cps1985", "survey_workinghours"],
           "nhanes": ["rb_nhanes_weight"], "hmda": ["survey_hmda"],
           "wealth": ["rb_scf2022_wealth_cont"]}
N_TRAIN, N_VAL, N_TEST = 2400, 480, 480
MAX_ROWS, MAX_COLS, BS, LR_RATE, MAX_EP, PATIENCE = 128, 48, 16, 3e-4, 40, 6
SEEDS = [2026, 7, 99, 13, 41]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
cat = create_default_catalog()


def load_base(name):
    if name.startswith("rb_"):
        df = pd.read_csv(RB / f"{name}.csv")
        return torch.from_numpy(df.to_numpy(np.float32))
    raw = cat.load(name)
    return torch.from_numpy(np.asarray(raw.data, np.float32))


def make_examples(domains, n_total, seed_base):
    """Balanced (generator x dataset) examples; deterministic; infeasible draws retried (logged)."""
    bases = {n: load_base(n) for dom in domains for n in DOMAINS[dom]}
    names = sorted(bases)
    per = -(-n_total // (len(GENERATORS) * len(names)))
    out, cell = [], 0
    for g in GENERATORS:
        for nm in names:
            for e in range(per):
                rng = RNGState(seed=seed_base + cell)
                cell += 1
                X = bases[nm]
                for attempt in range(10):
                    r = rng.spawn()
                    idx = r.choice(X.shape[0], min(MAX_ROWS, X.shape[0]), replace=False)
                    Xs = X[torch.as_tensor(idx)]
                    try:
                        R, cls, rate = generate(g, Xs, r.spawn())
                        break
                    except ValueError as err:
                        if attempt == 9:
                            raise RuntimeError(f"{g} on {nm}: 10 infeasible draws ({err})")
                # RIG-REPAIR (post-diagnostic, documented): standardize each column by OBSERVED-cell
                # mean/std (runtime-computable; v1.0 always trained on standardized inputs — raw
                # dollar-scale values froze the encoder and made recon loss ~1e7, drowning the CE
                # gradient; see runs/diag_t1_fit.log). Scale-free feature arm is unaffected.
                mu = torch.stack([Xs[R[:, j], j].mean() for j in range(Xs.shape[1])])
                sd = torch.stack([Xs[R[:, j], j].std() for j in range(Xs.shape[1])])
                sd = torch.where(sd > 0, sd, torch.ones_like(sd))
                Xz = (Xs - mu) / sd
                out.append({"x": Xz * R.float(), "R": R, "y": cls, "gen": g, "ds": nm, "rate": rate})
    rngp = np.random.default_rng(seed_base)
    return [out[i] for i in rngp.permutation(len(out))[:n_total]]


def feats(examples):
    return np.stack([compute_topology_features(e["x"], e["R"]).numpy() for e in examples])


# ---------------------------------------------------------------- network arm
class _DS:
    """Minimal ObservedDataset shim for tokenize_and_batch."""
    def __init__(self, e):
        self.x, self.r = e["x"], e["R"]
        self.n, self.d = e["x"].shape
        self.feature_names = tuple(f"c{j}" for j in range(self.d))
        self.dataset_id, self.meta = e["ds"], {}


def to_batch(examples, mask_rng):
    arts = []
    for e in examples:
        Xn = e["x"].numpy().astype(np.float64).copy()
        Xn[~e["R"].numpy()] = np.nan
        _, _, art = apply_artificial_masking(Xn, e["R"].numpy(), MaskingConfig(), rng=mask_rng)
        arts.append(art)
    return tokenize_and_batch([_DS(e) for e in examples], max_rows=MAX_ROWS, max_cols=MAX_COLS,
                              artificial_masks=arts)


def _to(batch, device):
    import dataclasses
    moved = {f.name: (getattr(batch, f.name).to(device)
                      if torch.is_tensor(getattr(batch, f.name)) else getattr(batch, f.name))
             for f in dataclasses.fields(batch)}
    return dataclasses.replace(batch, **moved)


@torch.no_grad()
def net_probs(model, examples, mask_rng):
    model.eval()
    P = []
    for s in range(0, len(examples), BS):
        b = _to(to_batch(examples[s:s + BS], mask_rng), DEVICE)
        out = model(b, compute_reconstruction=True, compute_decision=False)
        P.append(out.posterior.p_class.detach().cpu())
    return torch.cat(P).numpy()


def macro_auc(P, y):
    return float(roc_auc_score(y, P, multi_class="ovr", average="macro", labels=[0, 1, 2]))


def train_network(train_ex, val_ex, seed):
    torch.manual_seed(seed)
    cfg = LacunaModelConfig(hidden_dim=128, evidence_dim=64, n_layers=4, n_heads=4,
                            max_cols=MAX_COLS, use_missingness_features=False)
    model = LacunaModel(cfg).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR_RATE)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 400))
    yv = np.array([e["y"] for e in val_ex])
    best_auc, best_state, since = -1.0, None, 0
    for ep in range(MAX_EP):
        model.train()
        mrng = np.random.default_rng(seed * 1000 + ep)
        perm = np.random.default_rng(seed + ep).permutation(len(train_ex))
        for s in range(0, len(train_ex), BS):
            ex = [train_ex[i] for i in perm[s:s + BS]]
            b = _to(to_batch(ex, mrng), DEVICE)
            out = model(b, compute_reconstruction=True, compute_decision=False)
            y = torch.tensor([e["y"] for e in ex], device=DEVICE)
            loss = class_cross_entropy(out.posterior.p_class, y)
            rl, _ = multi_head_reconstruction_loss(out.reconstruction, b.original_values,
                                                   b.reconstruction_mask, b.row_mask, b.col_mask)
            (loss + rl).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
        va = macro_auc(net_probs(model, val_ex, np.random.default_rng(777)), yv)
        if va > best_auc + 1e-4:
            best_auc, since = va, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            since += 1
            if since >= PATIENCE:
                break
    if best_state:
        model.load_state_dict(best_state)
    return model


def secondaries(P, y, gens):
    mcar = float(roc_auc_score((np.array(y) == 0).astype(int), P[:, 0]))
    sel = [i for i, g in enumerate(gens) if g in ("mar_module_skip", "mnar_module_refusal")]
    sk = float(roc_auc_score([int(gens[i] == "mnar_module_refusal") for i in sel],
                             P[sel][:, 2])) if sel else float("nan")
    return mcar, sk


def main():
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    print("=" * 100)
    print(f"T1 — MASK-TOPOLOGY SHOWDOWN (prereg 7714f0c; device={DEVICE})")
    print("=" * 100)
    doms = sorted(DOMAINS)
    splits = {}
    for i, held in enumerate(doms):
        tr_dom = [d for d in doms if d != held]
        splits[held] = {"train": make_examples(tr_dom, N_TRAIN, 71000 + 10000 * i),
                        "val": make_examples(tr_dom, N_VAL, 75000 + 10000 * i),
                        "test": make_examples([held], N_TEST, 79000 + 10000 * i)}
        rates = [e["rate"] for e in splits[held]["test"]]
        print(f"[{time.strftime('%H:%M:%S')}] split {held}: examples ready "
              f"(test rate mean {np.mean(rates):.3f} sd {np.std(rates):.3f})")

    # ---- feature arm ----
    res = {"git": git, "device": DEVICE}
    pooled = {"lr": ([], []), "gbm": ([], [])}
    gens_all, y_all = [], []
    for held in doms:
        sp = splits[held]
        Xtr = feats(sp["train"] + sp["val"]); ytr = [e["y"] for e in sp["train"] + sp["val"]]
        Xte = feats(sp["test"]); yte = [e["y"] for e in sp["test"]]
        sc = StandardScaler().fit(Xtr)
        lr = LogisticRegression(max_iter=3000).fit(sc.transform(Xtr), ytr)
        gb = HistGradientBoostingClassifier(random_state=0).fit(Xtr, ytr)
        pooled["lr"][0].append(lr.predict_proba(sc.transform(Xte))); pooled["lr"][1].extend(yte)
        pooled["gbm"][0].append(gb.predict_proba(Xte)); pooled["gbm"][1].extend(yte)
        gens_all.extend(e["gen"] for e in sp["test"]); y_all.extend(yte)
        print(f"[{time.strftime('%H:%M:%S')}] features done for split {held}")
    shallow = {}
    for arm in ("lr", "gbm"):
        P = np.vstack(pooled[arm][0]); y = pooled[arm][1]
        m = macro_auc(P, y); mc, sk = secondaries(P, y, gens_all)
        shallow[arm] = {"macro_auc": m, "mcar_vs_not": mc, "skip_vs_refusal": sk}
        print(f"  shallow[{arm}]: macro-AUC {m:.3f} | MCAR-vs-not {mc:.3f} | skip-vs-refusal {sk:.3f}")
    best_shallow = max(shallow["lr"]["macro_auc"], shallow["gbm"]["macro_auc"])
    res["shallow"] = shallow

    # ---- network arm: 5 seeds x 4 splits ----
    per_seed = []
    for seed in SEEDS:
        Ps, ys, gs = [], [], []
        for held in doms:
            sp = splits[held]
            model = train_network(sp["train"], sp["val"], seed)
            P = net_probs(model, sp["test"], np.random.default_rng(777))
            Ps.append(P); ys.extend(e["y"] for e in sp["test"]); gs.extend(e["gen"] for e in sp["test"])
            del model; torch.cuda.empty_cache() if DEVICE == "cuda" else None
        P = np.vstack(Ps)
        m = macro_auc(P, ys); mc, sk = secondaries(P, ys, gs)
        per_seed.append({"seed": seed, "macro_auc": m, "mcar_vs_not": mc, "skip_vs_refusal": sk})
        print(f"[{time.strftime('%H:%M:%S')}] network seed {seed}: macro {m:.3f} | MCAR {mc:.3f} | s/r {sk:.3f}")
    arr = np.array([r["macro_auc"] for r in per_seed])
    mean, se = float(arr.mean()), float(arr.std(ddof=1) / np.sqrt(len(arr)))
    mc_mean = float(np.mean([r["mcar_vs_not"] for r in per_seed]))
    res["network"] = {"per_seed": per_seed, "macro_mean": mean, "macro_se": se, "mcar_mean": mc_mean}

    margin = (mean - se) - best_shallow
    mcar_ok = mc_mean >= max(shallow["lr"]["mcar_vs_not"], shallow["gbm"]["mcar_vs_not"]) - 0.02
    T1 = (margin >= 0.05) and mcar_ok
    res["criteria"] = {"best_shallow": best_shallow, "network_mean_minus_se": mean - se,
                       "margin": margin, "mcar_ok": bool(mcar_ok), "T1_pass": bool(T1)}
    print("\n" + "=" * 100)
    print(f"best shallow {best_shallow:.3f} | network mean-SE {mean - se:.3f} | margin {margin:+.3f} "
          f"(need >= +0.05) | MCAR-vs-not ok: {mcar_ok}")
    print(f"T1 VERDICT: {'PASS — network load-bearing on the mask-topology axis' if T1 else 'FAIL'}")
    OUT.write_text(json.dumps(res, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
