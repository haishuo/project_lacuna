"""
scripts/probe_encoder_representation.py

REPRESENTATION PROBE (ARCHITECTURE-CHANGE-AUDIT §8; PI-approved diagnostic — NOT a new architecture,
NOT a patch). Earns the reuse-vs-rewrite decision the patch's A/B could not: does the UNCHANGED BERT
backbone's representation linearly CONTAIN / PRESERVE the LOD δ signal that raw observed-value ECDF
features carry out-of-family?

Comparison (binary LOD δ0-vs-δ2.5, same held-out leave-datasets-out split as Gate-2):
  1. raw-ECDF LR                          — the known signal reference (Gate-2 measured ~0.754 OOF AUC)
  2. linear probe on FROZEN encoder reps  — evidence-only / target-reps-only / evidence+target
       encoder states: (a) random-init (structural prior, no training) and
                       (b) δ-trained-then-frozen (best case — the backbone got to learn the task)
  3. full trained model                   — reference only (Gate-2: ~0.58–0.59)

Plus a PRESERVATION probe (positive-capable, §4.9-safe): linear (Ridge) reconstruction of the 17 raw
ECDF features FROM the encoder reps; report test R² overall and for the discriminative upper-tail
features. High R² ⇒ the signal is RETAINED in the reps (architecture not the bottleneck → head/objective
issue, salvage/patch). Low R² across BOTH encoder states ⇒ the backbone discards it (→ rewrite / Option C).

Interpretation is reported, not asserted as a kill: a frozen-probe negative is weak-proxy evidence
(§4.9) and is cross-checked against the preservation probe and the random-vs-trained pattern.

Run: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u scripts/probe_encoder_representation.py
"""

import json
import subprocess
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from lacuna.core.rng import RNGState
from lacuna.data.catalog import create_default_catalog
from lacuna.survey.batching import collate, make_lod_example
from lacuna.survey.coarse_bins import assign_bins
from lacuna.survey.conditioned_head import create_target_conditioned_model
from lacuna.survey.consequence_features import FEATURE_NAMES
from lacuna.survey.example_source import LODSurveyExampleSource
from lacuna.survey.train import TrainConfig, train_delta_prior

TRAIN = ["survey_bfi", "survey_cars93", "survey_computers", "survey_cps1988",
         "survey_psid1976", "survey_psid7682", "survey_survey", "survey_yrbss"]
TEST = ["survey_cps1985", "survey_workinghours"]
GRID = [0.0, 2.5]
MAX_ROWS, MAX_COLS = 384, 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# indices of the discriminative upper-tail raw features (truncation lives here)
TAIL_FEATS = [FEATURE_NAMES.index(n) for n in ["zq90", "zq95", "zq99", "z_max", "gap_q99_q90"]]


def _corpus(pool, rng, n):
    """Decoupled (dataset, δ) LOD examples -> list[DeltaExample] (dataset pick is random, not i%len)."""
    out, ds_rng = [], rng.spawn()
    for i in range(n):
        d = GRID[i % len(GRID)]
        raw = pool[int(ds_rng.randint(0, len(pool), (1,)).item())]
        out.append(make_lod_example(raw, beta1=1.0, delta=d, target_rate=0.3,
                                    tau_quantile=0.70, rng=rng.spawn(), max_rows=MAX_ROWS))
    return out


@torch.no_grad()
def _extract(encoder, examples):
    """Frozen-encoder reps for a list of examples -> (evidence[N,E], target_pool[N,H], rawECDF[N,17], y[N])."""
    encoder.eval()
    ev, tp, raw, y = [], [], [], []
    for s in range(0, len(examples), 16):
        db = collate(examples[s:s + 16], max_rows=MAX_ROWS, max_cols=MAX_COLS)
        tok = db.tokens
        enc = encoder(tok.tokens.to(DEVICE), tok.row_mask.to(DEVICE), tok.col_mask.to(DEVICE),
                      return_intermediates=True)
        evidence = enc["evidence"]                       # [B, E]
        treps = enc["token_representations"]             # [B, R, C, H]
        b, r, c, h = treps.shape
        ti = db.target_idx.to(DEVICE).view(b, 1, 1, 1).expand(b, r, 1, h)
        tgt = treps.gather(2, ti).squeeze(2)             # [B, R, H]
        rm = tok.row_mask.to(DEVICE).unsqueeze(-1).float()
        pooled = (tgt * rm).sum(1) / rm.sum(1).clamp(min=1.0)   # masked mean target reps
        ev.append(evidence.cpu()); tp.append(pooled.cpu())
        raw.append(db.consequence); y.append(assign_bins("binary", db.delta))
    return (torch.cat(ev).numpy(), torch.cat(tp).numpy(),
            torch.cat(raw).numpy(), torch.cat(y).numpy())


def _auc(Xtr, ytr, Xte, yte):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=5000).fit(sc.transform(Xtr), ytr)
    return float(roc_auc_score(yte, clf.predict_proba(sc.transform(Xte))[:, 1]))


def _preservation(reps_tr, raw_tr, reps_te, raw_te):
    """Ridge reps -> raw ECDF (17). Return overall test R² and mean R² over the upper-tail features."""
    sc = StandardScaler().fit(reps_tr)
    rg = Ridge(alpha=1.0).fit(sc.transform(reps_tr), raw_tr)
    pred = rg.predict(sc.transform(reps_te))
    overall = float(r2_score(raw_te, pred, multioutput="variance_weighted"))
    tail = float(np.mean([r2_score(raw_te[:, j], pred[:, j]) for j in TAIL_FEATS]))
    return overall, tail


def _trained_encoder(cat, git):
    """Train a binary-LOD target-conditioned mean-pool model end-to-end; return its frozen encoder."""
    src = lambda names: LODSurveyExampleSource([cat.load(n) for n in names], tau_quantile=0.70)
    cfg = TrainConfig(delta_grid=GRID, beta1_range=(0.0, 2.0), target_rate=0.3,
                      max_rows=MAX_ROWS, max_cols=MAX_COLS, batch_size=16,
                      train_batches_per_epoch=40, max_epochs=18, patience=6, val_size=120, test_size=120,
                      hidden_dim=128, evidence_dim=64, n_layers=4, n_heads=4,
                      target_conditioned=True, consequence_features=False, rep_ecdf_pooling=False,
                      coarse_scheme="binary", model_kind="auto")
    torch.manual_seed(2026)
    out = train_delta_prior(src(TRAIN), src(["survey_chile", "survey_hmda"]), src(TEST), cfg,
                            RNGState(seed=2026), kind="ablation", run_id="probe-trained-encoder",
                            git_commit=git, timestamp="2026-06-05T12:00:00Z", device=DEVICE)
    enc = out["model"].encoder
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc, out["results"]["epochs_run"]


def main():
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    Path("runs").mkdir(exist_ok=True)
    cat = create_default_catalog()
    print("=" * 100); print(f"REPRESENTATION PROBE (audit §8)  device={DEVICE}  binary LOD  held-out test={TEST}")
    print("=" * 100)

    tr_ex = _corpus([cat.load(n) for n in TRAIN], RNGState(seed=41), 400)
    te_ex = _corpus([cat.load(n) for n in TEST], RNGState(seed=42), 240)

    # random-init encoder (structural prior; no training)
    rand_model = create_target_conditioned_model(hidden_dim=128, evidence_dim=64, n_layers=4, n_heads=4,
                                                  max_cols=MAX_COLS, num_bins=2, rng=RNGState(seed=7)).to(DEVICE)
    rand_enc = rand_model.encoder
    for p in rand_enc.parameters():
        p.requires_grad_(False)
    # δ-trained encoder (best case)
    trained_enc, ep = _trained_encoder(cat, git)
    print(f"(δ-trained encoder: {ep} epochs)\n")

    rep = {"git": git, "device": DEVICE, "encoders": {}}
    for tag, enc in [("random_init", rand_enc), ("delta_trained", trained_enc)]:
        ev_tr, tp_tr, raw_tr, y_tr = _extract(enc, tr_ex)
        ev_te, tp_te, raw_te, y_te = _extract(enc, te_ex)
        res = {
            "raw_ecdf_LR_auc": _auc(raw_tr, y_tr, raw_te, y_te),     # same each encoder (raw feats); sanity
            "probe_evidence_auc": _auc(ev_tr, y_tr, ev_te, y_te),
            "probe_target_auc": _auc(tp_tr, y_tr, tp_te, y_te),
            "probe_both_auc": _auc(np.hstack([ev_tr, tp_tr]), y_tr, np.hstack([ev_te, tp_te]), y_te),
        }
        ov, tl = _preservation(np.hstack([ev_tr, tp_tr]), raw_tr, np.hstack([ev_te, tp_te]), raw_te)
        res["preservation_R2_overall"] = ov
        res["preservation_R2_tailfeats"] = tl
        rep["encoders"][tag] = res
        print(f"[{tag}]")
        print(f"   raw-ECDF LR AUC      : {res['raw_ecdf_LR_auc']:.3f}  (known signal reference)")
        print(f"   probe evidence-only  : {res['probe_evidence_auc']:.3f}")
        print(f"   probe target-reps    : {res['probe_target_auc']:.3f}")
        print(f"   probe evidence+target: {res['probe_both_auc']:.3f}")
        print(f"   preservation R² (all): {ov:.3f}   tail-feats R²: {tl:.3f}\n")

    # interpretation (reported, not a kill — §4.9)
    rawref = rep["encoders"]["delta_trained"]["raw_ecdf_LR_auc"]
    probe_best = max(rep["encoders"]["delta_trained"][k] for k in
                     ("probe_evidence_auc", "probe_target_auc", "probe_both_auc"))
    pres = rep["encoders"]["delta_trained"]["preservation_R2_tailfeats"]
    print("=" * 100); print("INTERPRETATION (reported; cross-checked, not a standalone kill)"); print("=" * 100)
    print(f"  raw-ECDF LR={rawref:.3f}  best frozen δ-trained probe={probe_best:.3f}  "
          f"tail-feat preservation R²={pres:.3f}")
    if probe_best >= rawref - 0.05:
        msg = ("backbone reps CONTAIN the signal (probe ≈ raw-ECDF) -> bottleneck is head/objective/"
               "training, NOT the backbone -> salvage/patch the readout (Option A/B), reuse encoder.")
    elif pres >= 0.5:
        msg = ("probe << raw-ECDF BUT reps linearly PRESERVE the raw tail features (R²≥0.5) -> info is "
               "retained, the δ-head/objective fails to use it -> head/objective redesign, encoder reusable.")
    else:
        msg = ("probe << raw-ECDF AND reps do NOT preserve the raw tail features -> backbone discards the "
               "within-column order-stat signal -> supports rewrite / Option C (column-primary). "
               "(Weak-proxy negative per §4.9: confirm against the random-init row + Option-C spec.)")
    print(f"  => {msg}")
    rep["interpretation"] = msg
    Path("runs/probe-encoder-representation.json").write_text(json.dumps(rep, indent=2))
    print("\nsaved: runs/probe-encoder-representation.json")


if __name__ == "__main__":
    main()
