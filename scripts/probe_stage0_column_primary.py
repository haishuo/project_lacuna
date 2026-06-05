"""
scripts/probe_stage0_column_primary.py

STAGE 0 — column-primary representation probe (PI-approved; analysis only, NOT a rewrite, NOT a patch
of the BERT backbone, NOT a new mechanism, NO metadata channel). Earns (or refuses) the Option-C
rewrite by testing whether a DISTRIBUTION-NATIVE target-column representation recovers the raw-ECDF
LOD signal that the old row-primary backbone loses (probe-encoder-representation-findings.md: old reps
~0.52 vs raw-ECDF ~0.72 OOF).

Three-way comparison on ONE shared corpus, binary LOD δ0-vs-δ2.5, same held-out leave-datasets-out
split, same LR-on-raw-ECDF reference, same leakage discipline, own-value negative control:
  1. raw-ECDF LR                  — fixed column-primary baseline (the reference signal).
  2. old encoder reps linear probe — δ-trained-then-frozen BERT backbone (row-primary; the thing we suspect is lossy).
  3. column-primary φ (LEARNED)   — DeepSets over OBSERVED TARGET VALUES + quantile pooling + small MLP.
       φ: standardize observed target values within observed -> per-value MLP -> masked-quantile pool
          (reuses the tested distributional_stream.masked_quantile_pool, applied to RAW values, no BERT)
          -> small MLP -> 2 logits. Target-column ONLY (matched to the raw-ECDF baseline; predictor
          conditioning is Stage 2, explicitly out of scope here). 3 seeds.

SUCCESS (pre-registered): φ materially exceeds the old-encoder probe AND approaches/exceeds the raw-ECDF
LR on LOD, WHILE own-value stays flat. Then the rewrite/Option-C is earned. FAIL: φ does not clear the
old-encoder probe / raw-ECDF baseline -> the bottleneck is likely transfer regime / dataset diversity /
limited practical signal, NOT merely the old backbone -> do NOT rewrite (DECISION-MEMO §9).

Run: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u scripts/probe_stage0_column_primary.py
"""

import json
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from lacuna.core.rng import RNGState
from lacuna.data.catalog import create_default_catalog
from lacuna.survey.batching import collate, make_example, make_lod_example
from lacuna.survey.coarse_bins import assign_bins
from lacuna.survey.conditioned_head import create_target_conditioned_model
from lacuna.survey.distributional_stream import masked_quantile_pool
from lacuna.survey.example_source import LODSurveyExampleSource
from lacuna.survey.leakage import assess_leakage, leakage_pass
from lacuna.survey.loss import rps_loss
from lacuna.survey.train import TrainConfig, train_delta_prior

TRAIN = ["survey_bfi", "survey_cars93", "survey_computers", "survey_cps1988",
         "survey_psid1976", "survey_psid7682", "survey_survey", "survey_yrbss"]
VAL = ["survey_chile", "survey_hmda"]
TEST = ["survey_cps1985", "survey_workinghours"]
GRID = [0.0, 2.5]
MAX_ROWS, MAX_COLS = 384, 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
QLEVELS = tuple(np.round(np.linspace(0.02, 0.98, 12), 3).tolist())  # 12-quantile φ grid (+max in pool)
SEEDS = [2026, 7, 99]


# ---------- shared corpus ----------
def _corpus(pool, rng, n, family):
    out, ds_rng = [], rng.spawn()
    for i in range(n):
        d = GRID[i % len(GRID)]
        raw = pool[int(ds_rng.randint(0, len(pool), (1,)).item())]
        if family == "lod":
            ex = make_lod_example(raw, beta1=1.0, delta=d, target_rate=0.3,
                                  tau_quantile=0.70, rng=rng.spawn(), max_rows=MAX_ROWS)
        else:
            ex = make_example(raw, beta1=1.0, delta=d, target_rate=0.3, rng=rng.spawn(), max_rows=MAX_ROWS)
        out.append(ex)
    return out


def _values(examples):
    """Observed target values per example -> padded (V[N,Rmax,1], MASK[N,Rmax]) standardized within observed."""
    V = torch.zeros(len(examples), MAX_ROWS, 1)
    MK = torch.zeros(len(examples), MAX_ROWS, dtype=torch.bool)
    for i, ex in enumerate(examples):
        t = ex.answer_sheet.target_col_idx
        obs = ex.observed.r[:, t].bool()
        v = ex.observed.x[obs, t].float()
        if v.numel() == 0:
            continue
        sd = v.std(unbiased=False)
        v = (v - v.mean()) / sd if float(sd) > 0 else v - v.mean()
        k = min(v.numel(), MAX_ROWS)
        V[i, :k, 0] = v[:k]; MK[i, :k] = True
    return V, MK


def _raw_and_y(examples):
    raw, y = [], []
    for s in range(0, len(examples), 16):
        db = collate(examples[s:s + 16], max_rows=MAX_ROWS, max_cols=MAX_COLS)
        raw.append(db.consequence); y.append(assign_bins("binary", db.delta))
    return torch.cat(raw).numpy(), torch.cat(y).numpy()


def _auc(Xtr, ytr, Xte, yte):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=5000).fit(sc.transform(Xtr), ytr)
    return float(roc_auc_score(yte, clf.predict_proba(sc.transform(Xte))[:, 1]))


# ---------- old encoder reps probe (row-primary; δ-trained, frozen) ----------
@torch.no_grad()
def _encoder_reps(encoder, examples):
    encoder.eval(); ev, tp = [], []
    for s in range(0, len(examples), 16):
        db = collate(examples[s:s + 16], max_rows=MAX_ROWS, max_cols=MAX_COLS)
        tok = db.tokens
        enc = encoder(tok.tokens.to(DEVICE), tok.row_mask.to(DEVICE), tok.col_mask.to(DEVICE),
                      return_intermediates=True)
        treps = enc["token_representations"]; b, r, c, h = treps.shape
        ti = db.target_idx.to(DEVICE).view(b, 1, 1, 1).expand(b, r, 1, h)
        tgt = treps.gather(2, ti).squeeze(2)
        rm = tok.row_mask.to(DEVICE).unsqueeze(-1).float()
        pooled = (tgt * rm).sum(1) / rm.sum(1).clamp(min=1.0)
        ev.append(enc["evidence"].cpu()); tp.append(pooled.cpu())
    return np.hstack([torch.cat(ev).numpy(), torch.cat(tp).numpy()])


def _trained_encoder(cat, git):
    src = lambda names: LODSurveyExampleSource([cat.load(n) for n in names], tau_quantile=0.70)
    cfg = TrainConfig(delta_grid=GRID, beta1_range=(0.0, 2.0), target_rate=0.3, max_rows=MAX_ROWS,
                      max_cols=MAX_COLS, batch_size=16, train_batches_per_epoch=40, max_epochs=18,
                      patience=6, val_size=120, test_size=120, hidden_dim=128, evidence_dim=64,
                      n_layers=4, n_heads=4, target_conditioned=True, coarse_scheme="binary")
    torch.manual_seed(2026)
    out = train_delta_prior(src(TRAIN), src(VAL), src(TEST), cfg, RNGState(seed=2026), kind="ablation",
                            run_id="stage0-encoder", git_commit=git, timestamp="2026-06-05T12:00:00Z",
                            device=DEVICE)
    enc = out["model"].encoder
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc


# ---------- column-primary φ (learned distribution encoder over observed target values) ----------
class ColumnPhi(nn.Module):
    """DeepSets over observed target values: per-value MLP -> quantile pooling -> small MLP -> 2 logits."""

    def __init__(self, m=16, hidden=48, levels=QLEVELS):
        super().__init__()
        self.levels = levels
        self.h = nn.Sequential(nn.Linear(1, m), nn.GELU(), nn.Linear(m, m), nn.GELU())
        self.rho = nn.Sequential(nn.Linear(m * (len(levels) + 1), hidden), nn.GELU(),
                                 nn.Dropout(0.1), nn.Linear(hidden, 2))

    def forward(self, V, MK):  # V [B,R,1], MK [B,R]
        e = self.h(V)                                   # [B,R,m]
        stats = masked_quantile_pool(e, MK, self.levels)  # [B,m,Q+1]
        return self.rho(stats.reshape(stats.shape[0], -1))


def _train_phi(Vtr, Mtr, ytr, Vva, Mva, yva, seed, epochs=40, bs=32):
    torch.manual_seed(seed)
    model = ColumnPhi().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    ytr_t = torch.tensor(ytr, dtype=torch.long)
    best, best_state, since = 1e9, None, 0
    n = Vtr.shape[0]
    g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=g)
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            logits = model(Vtr[idx].to(DEVICE), Mtr[idx].to(DEVICE))
            loss = rps_loss(logits, ytr_t[idx].to(DEVICE))
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = model(Vva.to(DEVICE), Mva.to(DEVICE))
            vrps = float(rps_loss(vl, torch.tensor(yva, dtype=torch.long).to(DEVICE)).item())
        if vrps < best - 1e-4:
            best, since = vrps, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            since += 1
            if since >= 6:
                break
    if best_state:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def _phi_auc(model, Vte, Mte, yte):
    model.eval()
    p = torch.softmax(model(Vte.to(DEVICE), Mte.to(DEVICE)), -1)[:, 1].cpu().numpy()
    return float(roc_auc_score(yte, p))


def main():
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    Path("runs").mkdir(exist_ok=True)
    cat = create_default_catalog()
    print("=" * 100); print(f"STAGE 0 — COLUMN-PRIMARY REPRESENTATION PROBE  device={DEVICE}  binary LOD  test={TEST}")
    print("=" * 100)
    poolP = {P: [cat.load(n) for n in v] for P, v in (("tr", TRAIN), ("va", VAL), ("te", TEST))}
    rep = {"git": git, "device": DEVICE, "qlevels": list(QLEVELS), "seeds": SEEDS, "families": {}}

    # frozen old encoder (built once, LOD-trained)
    enc = _trained_encoder(cat, git)

    for family in ["lod", "ownvalue"]:
        tr = _corpus(poolP["tr"], RNGState(seed=41), 600, family)
        va = _corpus(poolP["va"], RNGState(seed=43), 200, family)
        te = _corpus(poolP["te"], RNGState(seed=42), 240, family)
        # leakage gate over the corpus (same discipline)
        lk = leakage_pass(assess_leakage([e.answer_sheet for e in tr + va + te], RNGState(seed=9).spawn()))
        raw_tr, ytr = _raw_and_y(tr); raw_va, yva = _raw_and_y(va); raw_te, yte = _raw_and_y(te)
        Vtr, Mtr = _values(tr); Vva, Mva = _values(va); Vte, Mte = _values(te)

        raw_auc = _auc(raw_tr, ytr, raw_te, yte)                                   # 1. raw-ECDF LR
        enc_auc = _auc(_encoder_reps(enc, tr), ytr, _encoder_reps(enc, te), yte)   # 2. old encoder probe
        phi_aucs = []                                                             # 3. learned φ (3 seeds)
        for sd in SEEDS:
            m = _train_phi(Vtr, Mtr, ytr, Vva, Mva, yva, sd)
            phi_aucs.append(_phi_auc(m, Vte, Mte, yte))
        phi_m, phi_s = float(np.mean(phi_aucs)), float(np.std(phi_aucs))
        rep["families"][family] = {"raw_ecdf_LR": raw_auc, "old_encoder_probe": enc_auc,
                                   "column_phi_auc_mean": phi_m, "column_phi_auc_std": phi_s,
                                   "column_phi_seeds": phi_aucs, "leakage_pass": lk}
        print(f"\n[{family}]  leakage_pass={lk}")
        print(f"   1. raw-ECDF LR (fixed column-primary)   : {raw_auc:.3f}")
        print(f"   2. old encoder reps probe (row-primary) : {enc_auc:.3f}")
        print(f"   3. column-primary φ (LEARNED, 3 seeds)  : {phi_m:.3f} ± {phi_s:.3f}  {[round(a,3) for a in phi_aucs]}")

    # verdict
    lod = rep["families"]["lod"]; ov = rep["families"]["ownvalue"]
    beats_encoder = lod["column_phi_auc_mean"] >= lod["old_encoder_probe"] + 0.05
    reaches_raw = lod["column_phi_auc_mean"] >= lod["raw_ecdf_LR"] - 0.03
    ov_flat = ov["column_phi_auc_mean"] <= 0.60
    print("\n" + "=" * 100); print("VERDICT (Stage 0)"); print("=" * 100)
    print(f"  LOD: φ {lod['column_phi_auc_mean']:.3f} vs old-encoder {lod['old_encoder_probe']:.3f} "
          f"vs raw-ECDF {lod['raw_ecdf_LR']:.3f} | own-value φ {ov['column_phi_auc_mean']:.3f}")
    if beats_encoder and reaches_raw and ov_flat:
        verdict = ("EARNED: a learned column-primary distribution φ recovers the LOD signal the old "
                   "backbone loses (φ ≫ old-encoder, ≈/≥ raw-ECDF) while own-value stays flat -> Option-C "
                   "rewrite is justified; proceed to a Stage-1 spec (φ + minimal δ-head) for approval.")
    elif beats_encoder and not reaches_raw:
        verdict = ("PARTIAL: φ beats the old backbone but does not reach the raw-ECDF baseline -> "
                   "column-primary helps but the learned encoder underperforms the fixed features; "
                   "reconsider φ design before committing to the full rewrite.")
    else:
        verdict = ("NOT EARNED: a learned column-primary φ does NOT clear the old-encoder/raw-ECDF bar "
                   "on LOD -> the bottleneck is likely transfer regime / dataset diversity / limited "
                   "practical signal, NOT merely the old backbone. Do NOT rewrite; revisit DECISION-MEMO §9.")
    print(f"  VERDICT: {verdict}")
    rep["verdict"] = verdict
    Path("runs/probe-stage0-column-primary.json").write_text(json.dumps(rep, indent=2))
    print("\nsaved: runs/probe-stage0-column-primary.json")


if __name__ == "__main__":
    main()
