"""
scripts/run_t3_amortization.py

T3 — amortization / the PFN role (PREREGISTRATION-network-load-bearing-review §4, commit 7714f0c).
Can ONE forward pass of a conditional-φ reproduce, OOF, what the explicit multi-imputer pipeline
(5 imputers x 3 fits + H/S/R statistics + LR — the empty-cell `no_truth_features` channel) achieves
per column? Burden is on PARITY (the network must match its explicit teacher).

LOCKED pre-results decisions (within the prereg's stated freedoms; fixed BEFORE any T3 result):
  examples  BIT-IDENTICAL to G1 / the empty cell: G1's enumeration verbatim (4 domains, all
            continuous targets, idioms {own_value, top_coding}, delta {0.0, 2.5}, 24 ex/cell,
            max_rows 384, matched rate 0.3, per-cell seed 31000+i), with each example's frozen
            base17 recomputed and asserted equal to the stored G1 row before it is used (same
            guard the empty cell uses). The network consumes the SAME no-truth view (target cells
            punched; predictors complete).
  network   ConditionalPhi (one fixed config: pred_hidden 32, hidden 48, head_hidden 128,
            quantiles .25/.50/.75, 3 row-groups all/observed/missing). Column-major standardized
            inputs (observed-cell stats). Per (idiom, held-out domain, seed): train on the other 3
            domains (10% of train held out for early-stop + temperature), predict the held-out
            domain; pooled OOF across the 4 held-out domains. Adam lr 3e-3, bs 64, <=80 epochs,
            patience 10 on val AUC; temperature fit on the val split. 5 seeds; mean - SE gated.
  teacher   explicit-pipeline anchors from runs/conditional_without_truth_cell.json (per imputer,
            per idiom). Representative pipeline = MEAN over the 5 imputers (locked aggregation).
            Per-imputer best/worst reported for full transparency.
  bar       PASS iff network pooled OOF AUC (mean - SE over 5 seeds) >= teacher_mean - 0.02 for
            BOTH idioms, with single-forward-pass inference (wall-clock per column reported vs the
            ~15-fit explicit pipeline). ECE reported alongside AUC (prereg shared protocol).

Run: /home/haishuo/miniconda3/envs/lacuna/bin/python -u scripts/run_t3_amortization.py
"""

import importlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from lacuna.core.rng import RNGState
from lacuna.survey.conditional_phi import ConditionalPhi, ConditionalPhiConfig
from lacuna.survey.consequence_features import compute_consequence_features

G1 = importlib.import_module("scripts.run_g1_imputation_channel")
G1_ROWS = json.loads(Path("runs/g1_imputation_channel.json").read_text())["rows"]
EMPTY = json.loads(Path("runs/conditional_without_truth_cell.json").read_text())
OUT = Path("runs/t3_amortization.json")

MAX_ROWS = G1.MAX_ROWS
IDIOMS = G1.IDIOMS  # ("top_coding", "own_value") — MUST match G1's enumeration order (per-cell seeds)
SEEDS = [2026, 7, 99, 13, 41]
LR_RATE, BS, MAX_EP, PATIENCE = 3e-3, 64, 80, 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------- example corpus
def build_examples():
    """G1's enumeration verbatim; base17 identity-checked against the stored G1 rows (bit-identity)."""
    examples, cell_i, g1_ptr, checked = [], 0, 0, 0
    for dom in sorted(G1.DOMAINS):
        for name in G1.DOMAINS[dom]:
            base = G1.cont_base(name)
            for t_idx, t_name in enumerate(base.feature_names):
                for idiom in IDIOMS:
                    for delta in G1.DELTAS:
                        rng = RNGState(seed=31000 + cell_i)
                        cell_i += 1
                        for e in range(G1.N_EX):
                            sub = G1.subsample_raw(base, max_rows=MAX_ROWS, rng=rng.spawn())
                            X_t = torch.from_numpy(np.asarray(sub.data, np.float32))
                            try:
                                m_mech = G1.mech_mask(X_t, t_idx, idiom, delta, rng)
                                # consume G1's per-example mcar spawn (UNUSED here) to keep the
                                # shared cell RNG bit-identical across the 24 examples (same guard
                                # the empty cell uses; the next example's draws depend on it).
                                _ = G1.mcar_pair_mask(m_mech, rng.spawn())
                            except ValueError as err:
                                print(f"  skip {name}.{t_name} {idiom} d={delta} ex{e}: {err}")
                                continue
                            R = torch.ones_like(X_t, dtype=torch.bool)
                            R[:, t_idx] = torch.from_numpy(m_mech)
                            b17 = compute_consequence_features(X_t * R.float(), R, t_idx).numpy()
                            g1r = G1_ROWS[g1_ptr]
                            if (g1r["dataset"], g1r["target"], g1r["idiom"], g1r["delta"]) != \
                                    (name, t_name, idiom, delta):
                                raise RuntimeError(f"enumeration drift at G1 row {g1_ptr}")
                            if not np.allclose(b17, np.array(g1r["base17"]), atol=1e-5):
                                raise RuntimeError(f"base17 mismatch at G1 row {g1_ptr} — examples NOT identical")
                            g1_ptr += 1
                            checked += 1
                            examples.append(_encode(X_t, t_idx, m_mech, dom, idiom, int(delta > 0)))
        print(f"[{time.strftime('%H:%M:%S')}] domain {dom} done ({len(examples)} ex; {checked} identity-checked)")
    if g1_ptr != len(G1_ROWS):
        raise RuntimeError(f"row count mismatch: consumed {g1_ptr} of {len(G1_ROWS)} G1 rows")
    return examples


def _encode(X_t, t_idx, m_mech, domain, idiom, y):
    """Column-major standardized no-truth view → predictor / target tensors for ConditionalPhi."""
    X = X_t.numpy().astype(np.float64)
    n, d = X.shape
    obs_t = m_mech.astype(bool)                       # target observed mask
    pred_idx = [j for j in range(d) if j != t_idx]
    # standardize predictors by their (complete) column stats; target by OBSERVED-cell stats
    P = np.zeros((n, len(pred_idx)), np.float32)
    for k, j in enumerate(pred_idx):
        col = X[:, j]
        mu, sd = col.mean(), col.std()
        P[:, k] = (col - mu) / (sd if sd > 0 else 1.0)
    tcol = X[:, t_idx]
    om, osd = tcol[obs_t].mean(), tcol[obs_t].std()
    tval = np.where(obs_t, (tcol - om) / (osd if osd > 0 else 1.0), 0.0).astype(np.float32)
    return {"P": P, "obs": obs_t.astype(np.float32), "tval": tval,
            "domain": domain, "idiom": idiom, "y": y, "n": n, "dp": len(pred_idx)}


# ---------------------------------------------------------------- batching / model
def collate(batch, dp_max, device):
    b = len(batch)
    r_max = max(e["n"] for e in batch)
    P = torch.zeros(b, r_max, dp_max)
    pcol = torch.zeros(b, dp_max, dtype=torch.bool)
    obs = torch.zeros(b, r_max)
    tval = torch.zeros(b, r_max)
    rmask = torch.zeros(b, r_max, dtype=torch.bool)
    for i, e in enumerate(batch):
        n, dp = e["n"], e["dp"]
        P[i, :n, :dp] = torch.from_numpy(e["P"])
        pcol[i, :dp] = True
        obs[i, :n] = torch.from_numpy(e["obs"])
        tval[i, :n] = torch.from_numpy(e["tval"])
        rmask[i, :n] = True
    y = torch.tensor([e["y"] for e in batch], dtype=torch.float32)
    return (P.to(device), pcol.to(device), obs.to(device), tval.to(device),
            rmask.to(device), y.to(device))


@torch.no_grad()
def predict_logits(model, examples, dp_max):
    model.eval()
    out = []
    for s in range(0, len(examples), BS):
        P, pcol, obs, tval, rmask, _ = collate(examples[s:s + BS], dp_max, DEVICE)
        out.append(model(P, pcol, obs, tval, rmask).cpu())
    return torch.cat(out) if out else torch.zeros(0)


def fit_temperature(logits, y):
    """Single-parameter temperature scaling on held-out logits (calibration)."""
    t = torch.zeros(1, requires_grad=True)            # log-temperature
    opt = torch.optim.LBFGS([t], lr=0.1, max_iter=50)
    bce = torch.nn.BCEWithLogitsLoss()

    def closure():
        opt.zero_grad()
        loss = bce(logits / t.exp(), y)
        loss.backward()
        return loss
    opt.step(closure)
    return float(t.exp().detach())


def ece(p, y, bins=10):
    p, y = np.asarray(p), np.asarray(y)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if m.any():
            e += abs(p[m].mean() - y[m].mean()) * m.mean()
    return float(e)


def train_one(train_ex, val_ex, dp_max, seed):
    torch.manual_seed(seed)
    model = ConditionalPhi(ConditionalPhiConfig()).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR_RATE)
    bce = torch.nn.BCEWithLogitsLoss()
    yv = np.array([e["y"] for e in val_ex])
    best_auc, best_state, since = -1.0, None, 0
    for ep in range(MAX_EP):
        model.train()
        perm = np.random.default_rng(seed + ep).permutation(len(train_ex))
        for s in range(0, len(train_ex), BS):
            ex = [train_ex[i] for i in perm[s:s + BS]]
            P, pcol, obs, tval, rmask, y = collate(ex, dp_max, DEVICE)
            loss = bce(model(P, pcol, obs, tval, rmask), y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        vl = predict_logits(model, val_ex, dp_max).numpy()
        va = roc_auc_score(yv, vl) if len(set(yv)) > 1 else 0.5
        if va > best_auc + 1e-4:
            best_auc, since = va, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            since += 1
            if since >= PATIENCE:
                break
    if best_state:
        model.load_state_dict(best_state)
    temp = fit_temperature(predict_logits(model, val_ex, dp_max), torch.tensor(yv, dtype=torch.float32))
    return model, temp


def main():
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    print("=" * 100)
    print(f"T3 — AMORTIZATION / conditional-φ (prereg 7714f0c; device={DEVICE})")
    print("=" * 100)
    examples = build_examples()
    dp_max = max(e["dp"] for e in examples)
    print(f"total examples: {len(examples)} | dp_max {dp_max}")

    teacher = {}
    for idiom in IDIOMS:
        aucs = [EMPTY["per_idiom"][idiom][imp]["auc_cell"] for imp in G1.IMPUTERS]
        teacher[idiom] = {"per_imputer": dict(zip(G1.IMPUTERS, aucs)),
                          "mean": float(np.mean(aucs)), "best": float(np.max(aucs)),
                          "worst": float(np.min(aucs))}
        print(f"teacher[{idiom}] mean {teacher[idiom]['mean']:.3f} "
              f"(worst {teacher[idiom]['worst']:.3f} .. best {teacher[idiom]['best']:.3f})")

    doms = sorted(G1.DOMAINS)
    res = {"git": git, "device": DEVICE, "n_examples": len(examples), "teacher": teacher,
           "per_idiom": {}}
    wall_train_total, n_cols_trained = 0.0, 0
    for idiom in IDIOMS:
        idiom_ex = [e for e in examples if e["idiom"] == idiom]
        per_seed = []
        for seed in SEEDS:
            p_all, y_all, t_used = [], [], []
            for held in doms:
                tr = [e for e in idiom_ex if e["domain"] != held]
                te = [e for e in idiom_ex if e["domain"] == held]
                if not te or len({e["y"] for e in tr}) < 2:
                    continue
                vperm = np.random.default_rng(seed * 31 + doms.index(held)).permutation(len(tr))
                nval = max(BS, len(tr) // 10)
                val_ex = [tr[i] for i in vperm[:nval]]
                tr_ex = [tr[i] for i in vperm[nval:]]
                t0 = time.time()
                model, temp = train_one(tr_ex, val_ex, dp_max, seed)
                wall_train_total += time.time() - t0
                n_cols_trained += 1
                logits = predict_logits(model, te, dp_max).numpy()
                p_all.extend(1.0 / (1.0 + np.exp(-logits / temp)))
                y_all.extend(e["y"] for e in te)
                t_used.append(temp)
                del model
                torch.cuda.empty_cache() if DEVICE == "cuda" else None
            auc = float(roc_auc_score(y_all, p_all))
            per_seed.append({"seed": seed, "auc": auc, "ece": ece(p_all, y_all),
                             "temp_mean": float(np.mean(t_used))})
            print(f"[{time.strftime('%H:%M:%S')}] {idiom} seed {seed}: OOF AUC {auc:.3f} "
                  f"| ECE {per_seed[-1]['ece']:.3f}")
        arr = np.array([r["auc"] for r in per_seed])
        mean, se = float(arr.mean()), float(arr.std(ddof=1) / np.sqrt(len(arr)))
        bar = teacher[idiom]["mean"] - 0.02
        res["per_idiom"][idiom] = {
            "per_seed": per_seed, "auc_mean": mean, "auc_se": se,
            "mean_minus_se": mean - se, "bar": bar, "pass": bool((mean - se) >= bar),
            "ece_mean": float(np.mean([r["ece"] for r in per_seed]))}
        print(f"  {idiom}: mean {mean:.3f} SE {se:.3f} | mean-SE {mean-se:.3f} "
              f"| bar {bar:.3f} | {'PASS' if (mean-se) >= bar else 'FAIL'}")

    # single-forward-pass inference timing vs the ~15-fit explicit pipeline (reported, not gated)
    sample = examples[:BS]
    t0 = time.time()
    for _ in range(20):
        _ = predict_logits(ConditionalPhi().to(DEVICE).eval(), sample, dp_max)
    fwd_ms = (time.time() - t0) / 20 / len(sample) * 1000
    res["inference"] = {"forward_ms_per_column": fwd_ms,
                        "explicit_pipeline_fits_per_column": 15,
                        "train_wall_s_total": wall_train_total, "n_columns_trained": n_cols_trained}

    both_pass = all(res["per_idiom"][i]["pass"] for i in IDIOMS)
    res["T3_pass"] = bool(both_pass)
    print("\n" + "=" * 100)
    for idiom in IDIOMS:
        d = res["per_idiom"][idiom]
        print(f"{idiom:10}: mean-SE {d['mean_minus_se']:.3f} vs bar {d['bar']:.3f} "
              f"-> {'PASS' if d['pass'] else 'FAIL'}")
    print(f"single-pass inference {fwd_ms:.3f} ms/column (explicit pipeline = ~15 fits/column)")
    print(f"T3 VERDICT: {'PASS — conditional-φ reproduces the explicit pipeline OOF' if both_pass else 'FAIL'}")
    OUT.write_text(json.dumps(res, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
