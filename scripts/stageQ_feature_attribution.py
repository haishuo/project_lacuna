#!/usr/bin/env python3
"""
Stage Q — feature attribution: WHERE is the loud-subtype signal, and does the encoder help?

Companion to scripts/stageQ_subtype_layer.py. The subtype layer's data channel detects the LOUD MNAR
fingerprints (threshold / detection) vs the non-identifiable reject (quiet self-censoring MNAR + MAR +
MCAR). This script measures the loud-vs-reject CEILING three ways, on matched-rate full-diversity
mixtures, to attribute where the signal lives and justify the layer's architecture choice:

  - deployable distributional features alone (observed-value skew/kurtosis/SMD; no oracle),
  - the FROZEN dataset-task encoder's per-column token reps alone (pooled over rows),
  - both concatenated.

The Stage-Q finding (committed in the layer): the signal is in the deployable features; the frozen
encoder reps are ~chance and DILUTE the features when concatenated — so the deployable-feature
detector (lacuna.models.subtype_likelihood) excludes the encoder. This echoes the Stage-C
"encoder is the bottleneck / under-represents the footprint" result, at subtype granularity.

Deterministic via explicit seeds. Usage:
    python scripts/stageQ_feature_attribution.py --train-batches 120 --eval-batches 50
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
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

from lacuna.core.rng import RNGState
from lacuna.config import load_config
from lacuna.models.encoder import create_encoder
from lacuna.models.column_deployable_features import per_column_deployable_features
from lacuna.data.catalog import create_default_catalog
from lacuna.data.mixed_batch import build_mixed_batch
from lacuna.data.subtype_targets import subtype_targets
from lacuna.priors.subtype_ontology import LIKE_INDETERMINATE

BASELINE = "/mnt/artifacts/project_lacuna/runs/stage0_general_baseline"
MIXTURE_KWARGS = dict(p_observed=0.3, target_miss_rate=0.25,
                      mnar_diverse=True, mar_diverse=True, compensate_rate=True)


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


def collect(encoder, raws, *, seed, n_batches, batch_size, max_rows, max_cols, device):
    rng = RNGState(seed=seed)
    F5, ENC, Y = [], [], []
    with torch.no_grad():
        for _ in range(n_batches):
            mb = build_mixed_batch(raws, rng.spawn(), max_rows=max_rows, max_cols=max_cols,
                                   batch_size=batch_size, **MIXTURE_KWARGS)
            b = mb.batch.to(device)
            feats = per_column_deployable_features(b.tokens, b.row_mask, b.col_mask).cpu().numpy()
            tr = encoder.get_token_representations(b.tokens, b.row_mask, b.col_mask)   # [B,R,C,H]
            rm = b.row_mask.to(tr.dtype)[:, :, None, None]
            pooled = ((tr * rm).sum(1) / rm.sum(1).clamp(min=1)).cpu().numpy()         # [B,C,H]
            like, _onto, sup = subtype_targets(mb)
            B, C = sup.shape
            for i in range(B):
                for j in range(C):
                    if bool(sup[i, j]):
                        F5.append(feats[i, j]); ENC.append(pooled[i, j]); Y.append(int(like[i, j]))
    yloud = (np.array(Y) != LIKE_INDETERMINATE).astype(int)
    return np.array(F5), np.array(ENC), yloud


def evaluate(Xtr, ytr, Xte, yte, seed):
    clf = RandomForestClassifier(n_estimators=400, class_weight="balanced", random_state=seed, n_jobs=-1)
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    prec, rec, _ = precision_recall_curve(yte, p)
    def rap(mp):
        ok = prec >= mp
        return round(float(rec[ok].max()), 4) if ok.any() else 0.0
    return dict(auc=round(float(roc_auc_score(yte, p)), 4),
                ap=round(float(average_precision_score(yte, p)), 4),
                recall_at_p50=rap(0.5), recall_at_p70=rap(0.7))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline-checkpoint", default=f"{BASELINE}/checkpoints/best_model.pt")
    ap.add_argument("--baseline-config", default=f"{BASELINE}/config.yaml")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--train-batches", type=int, default=120)
    ap.add_argument("--eval-batches", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    cfg = load_config(args.baseline_config)
    dims = dict(hidden_dim=cfg.model.hidden_dim, evidence_dim=cfg.model.evidence_dim,
                n_layers=cfg.model.n_layers, n_heads=cfg.model.n_heads,
                max_cols=cfg.data.max_cols, dropout=cfg.model.dropout)
    sd = torch.load(args.baseline_checkpoint, map_location="cpu", weights_only=False)["model_state"]
    enc = create_encoder(**dims)
    enc.load_state_dict({k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")})
    enc = enc.to(args.device).eval()
    cat = create_default_catalog()
    train = load_raws(cat, cfg.data.train_datasets, cfg.data.max_cols)
    val = load_raws(cat, cfg.data.val_datasets, cfg.data.max_cols)
    print(f"Stage Q attribution | loud-vs-reject ceiling (RF) | train {len(train)} / val {len(val)} datasets")

    arms = {"deploy_features": [], "encoder_reps": [], "features+encoder": []}
    base_rates = []
    for s in range(args.seeds):
        seed = 20260601 + 101 * s
        F5tr, ENtr, ytr = collect(enc, train, seed=seed, n_batches=args.train_batches,
                                  batch_size=args.batch_size, max_rows=cfg.data.max_rows,
                                  max_cols=cfg.data.max_cols, device=args.device)
        F5te, ENte, yte = collect(enc, val, seed=seed + 7, n_batches=args.eval_batches,
                                  batch_size=args.batch_size, max_rows=cfg.data.max_rows,
                                  max_cols=cfg.data.max_cols, device=args.device)
        base_rates.append(float(yte.mean()))
        arms["deploy_features"].append(evaluate(F5tr, ytr, F5te, yte, seed))
        arms["encoder_reps"].append(evaluate(ENtr, ytr, ENte, yte, seed))
        arms["features+encoder"].append(
            evaluate(np.concatenate([F5tr, ENtr], 1), ytr, np.concatenate([F5te, ENte], 1), yte, seed))
        print(f"  seed {s+1}/{args.seeds}: feats AUC {arms['deploy_features'][-1]['auc']} | "
              f"enc AUC {arms['encoder_reps'][-1]['auc']} | both AUC {arms['features+encoder'][-1]['auc']}", flush=True)

    def agg(arm, key):
        v = [d[key] for d in arms[arm]]
        return [round(float(np.mean(v)), 4), round(float(np.std(v)), 4)]
    summary = {arm: {k: agg(arm, k) for k in ("auc", "ap", "recall_at_p50", "recall_at_p70")} for arm in arms}
    summary["loud_base_rate"] = [round(float(np.mean(base_rates)), 4), round(float(np.std(base_rates)), 4)]

    print("\n" + "=" * 72)
    print(f"STAGE Q ATTRIBUTION — loud-vs-reject ceiling ({args.seeds} seeds, mean+/-sd)")
    print(f"loud base rate {summary['loud_base_rate']}")
    print("=" * 72)
    for arm in ("deploy_features", "encoder_reps", "features+encoder"):
        a = summary[arm]
        print(f"  {arm:18s} AUC {a['auc']} | AP {a['ap']} | R@P.5 {a['recall_at_p50']} | R@P.7 {a['recall_at_p70']}")
    print("=" * 72)
    out = args.output or Path(f"{BASELINE}/stageQ_feature_attribution.json")
    out.write_text(json.dumps({"config": vars(args) | {"baseline_checkpoint": str(args.baseline_checkpoint),
                                                        "output": str(out)},
                               "summary_mean_sd": summary, "per_seed": arms}, indent=2, default=str))
    print(f"Wrote -> {out}")


if __name__ == "__main__":
    main()
