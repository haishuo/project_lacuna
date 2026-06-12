"""
scripts/run_semantic_arm3.py

ARM 3 — frozen vs TRAINED text encoder (the load-bearing question of the semantic channel):
is an off-the-shelf embedding sufficient to predict real item-nonresponse behavior from question
text, or does training the encoder on behavior labels transfer better to UNSEEN instruments?

Items: REGISTER-MATCHED full question wording only (ESS source-questionnaire matches + NHANES
codebook text + GSS codebook-index wording) — the register confound is excluded by construction.
Targets: per-item refusal_rate + dk_rate, rank-z-scored WITHIN instrument (scale-free; rates are
not comparable across instruments). Eval: leave-one-instrument-out triangle; within-held-out
Spearman per target.

Arms (identical items, identical targets, identical splits):
  A2 frozen: all-MiniLM-L6-v2 embeddings (no gradient) + ridge.
  A3 trained: SAME architecture fine-tuned end-to-end (mean-pool + 2-target linear head, MSE),
     AdamW 2e-5, 8 epochs, bs 16, 5 seeds (mean-SE discipline; degenerate seeds reported).

The verdict semantics (recorded BEFORE running): A3 (mean-SE) > A2 per-leg/pooled by a clear
margin => training is load-bearing; A3 <= A2 => off-the-shelf suffices (the network we train adds
nothing -- the honest negative). Prior recorded: ~50/50.

Run: /home/haishuo/miniconda3/envs/lacuna/bin/python -u scripts/run_semantic_arm3.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import norm, spearmanr
from sklearn.linear_model import Ridge
from transformers import AutoModel, AutoTokenizer

ESS = Path("/mnt/data/lacuna/role_b/ess_text_corpus.csv")
NH = Path("/mnt/data/lacuna/role_b/nhanes_text_corpus.csv")
GSS = Path("/mnt/data/lacuna/role_b/gss_text_corpus.csv")
OUT = Path("runs/semantic_arm3.json")
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = [2026, 7, 99, 13, 41]
EPOCHS, BS, LR = 8, 16, 2e-5


def load_items():
    ess = pd.read_csv(ESS)
    ess = ess[ess["full_text"].fillna("") != ""].assign(instrument="ESS11")
    ess["item_text"] = ess["full_text"]
    nh = pd.read_csv(NH).assign(instrument="NHANES")
    nh["item_text"] = nh["text"]  # codebook English Text = full register already
    gss = pd.read_csv(GSS)
    gss = gss[gss["full_text"].fillna("") != ""].assign(instrument="GSS")
    gss["item_text"] = gss["full_text"]
    cols = ["instrument", "var", "item_text", "refusal_rate", "dk_rate"]
    df = pd.concat([ess[cols], nh[cols], gss[cols]], ignore_index=True)
    df = df[df["item_text"].str.len() > 15]
    df = df.drop_duplicates(subset=["instrument", "item_text"]).reset_index(drop=True)
    return df


def rank_z(y):
    r = pd.Series(y).rank(method="average").to_numpy()
    return norm.ppf(r / (len(y) + 1.0))


def make_targets(df):
    """Rank-z within instrument for both targets."""
    Z = np.zeros((len(df), 2))
    for inst in df.instrument.unique():
        m = (df.instrument == inst).to_numpy()
        Z[m, 0] = rank_z(df.loc[m, "refusal_rate"].to_numpy())
        Z[m, 1] = rank_z(df.loc[m, "dk_rate"].to_numpy())
    return Z


class TrainedEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = AutoModel.from_pretrained(MODEL)
        self.head = nn.Linear(self.enc.config.hidden_size, 2)

    def forward(self, ids, mask):
        h = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state
        pooled = (h * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        return self.head(pooled)


def tokenize_all(texts, tok):
    return tok(list(texts), padding=True, truncation=True, max_length=96, return_tensors="pt")


@torch.no_grad()
def frozen_embed(texts, tok):
    enc = AutoModel.from_pretrained(MODEL).to(DEVICE).eval()
    out = []
    for s in range(0, len(texts), 128):
        b = tokenize_all(texts[s:s + 128], tok)
        ids, mask = b["input_ids"].to(DEVICE), b["attention_mask"].to(DEVICE)
        h = enc(input_ids=ids, attention_mask=mask).last_hidden_state
        out.append(((h * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)).cpu().numpy())
    return np.vstack(out)


def main():
    df = load_items()
    print(f"register-matched items: {len(df)} ({df.instrument.value_counts().to_dict()})")
    Z = make_targets(df)
    tok = AutoTokenizer.from_pretrained(MODEL)
    E = frozen_embed(df["item_text"].tolist(), tok)
    results = {}
    for held in ("ESS11", "NHANES", "GSS"):
        tr = np.where(df.instrument != held)[0]
        te = np.where(df.instrument == held)[0]
        y_ref = df.loc[te, "refusal_rate"].to_numpy()
        y_dk = df.loc[te, "dk_rate"].to_numpy()

        # ---- A2 frozen + ridge ----
        a2 = {}
        for j, (name, yt) in enumerate([("refusal", y_ref), ("dk", y_dk)]):
            pred = Ridge(alpha=1.0).fit(E[tr], Z[tr, j]).predict(E[te])
            a2[name] = float(spearmanr(yt, pred).statistic)

        # ---- A3 fine-tuned, 5 seeds ----
        batch_all = tokenize_all(df["item_text"].tolist(), tok)
        ids_all, mask_all = batch_all["input_ids"], batch_all["attention_mask"]
        ztr = torch.from_numpy(Z[tr]).float()
        a3_seeds = {"refusal": [], "dk": []}
        for seed in SEEDS:
            torch.manual_seed(seed)
            net = TrainedEncoder().to(DEVICE)
            opt = torch.optim.AdamW(net.parameters(), lr=LR)
            perm_rng = np.random.default_rng(seed)
            net.train()
            for ep in range(EPOCHS):
                order = perm_rng.permutation(len(tr))
                for s in range(0, len(order), BS):
                    bi = tr[order[s:s + BS]]
                    out = net(ids_all[bi].to(DEVICE), mask_all[bi].to(DEVICE))
                    loss = nn.functional.mse_loss(out, ztr[torch.from_numpy(
                        np.searchsorted(tr, bi))].to(DEVICE))
                    opt.zero_grad(); loss.backward(); opt.step()
            net.eval()
            with torch.no_grad():
                P = []
                for s in range(0, len(te), 128):
                    bi = te[s:s + 128]
                    P.append(net(ids_all[bi].to(DEVICE), mask_all[bi].to(DEVICE)).cpu().numpy())
                P = np.vstack(P)
            a3_seeds["refusal"].append(float(spearmanr(y_ref, P[:, 0]).statistic))
            a3_seeds["dk"].append(float(spearmanr(y_dk, P[:, 1]).statistic))
            del net
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
        a3 = {}
        for name in ("refusal", "dk"):
            arr = np.array(a3_seeds[name])
            a3[name] = {"seeds": a3_seeds[name], "mean": float(arr.mean()),
                        "se": float(arr.std(ddof=1) / np.sqrt(len(arr))),
                        "mean_minus_se": float(arr.mean() - arr.std(ddof=1) / np.sqrt(len(arr)))}
        results[held] = {"n_test": int(len(te)), "frozen": a2, "trained": a3}
        print(f"\nhold out {held} (n={len(te)}):")
        for name in ("refusal", "dk"):
            t = a3[name]
            print(f"  {name:8s}: frozen {a2[name]:+.3f} | trained mean {t['mean']:+.3f} "
                  f"(SE {t['se']:.3f}, mean-SE {t['mean_minus_se']:+.3f}) "
                  f"seeds {[round(x,3) for x in t['seeds']]}")
    # pooled verdict per target: mean over legs
    print("\n" + "=" * 90)
    for name in ("refusal", "dk"):
        fz = float(np.mean([results[h]["frozen"][name] for h in results]))
        td = float(np.mean([results[h]["trained"][name]["mean"] for h in results]))
        print(f"POOLED {name}: frozen {fz:+.3f} vs trained {td:+.3f} -> "
              f"{'TRAINING ADDS' if td > fz else 'frozen sufficient (training adds nothing)'}")
    OUT.write_text(json.dumps(results, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
