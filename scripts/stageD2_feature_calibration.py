#!/usr/bin/env python3
"""
Stage D-v2 (ADR-0007 follow-up): FEATURE-CONDITIONAL calibration of the composition posterior.

Stage D's global temperature made the region statements reliable ON AVERAGE but left the per-instance
uncertainty RANKING broken (corr(can't-tell, error) ≈ −0.15): the posterior was not more uncertain on
the datasets it actually got more wrong. This stage asks whether conditioning the temperature on the
observable footprint fixes that — and, FIRST, whether it even can.

Two steps, attribution-first:
  1. FEASIBILITY PROBE. Is the per-dataset composition error predictable from the footprint at all?
     A RandomForest regresses |error| on the 20-D footprint (fit on the calibration split, scored on a
     disjoint test split). Held-out R² > 0 ⇒ a feature-conditional temperature has signal to exploit;
     R² ≈ 0 ⇒ the error is non-identifiable noise and a global temperature is the best attainable (a
     documented null — still a finding).
  2. FEATURE-CONDITIONAL TEMPERATURE. A small net maps footprint → per-dataset temperature, fit on the
     calibration split by minimising the Dirichlet NLL (a proper score). Evaluated on test against the
     UNCALIBRATED posterior and the Stage-D GLOBAL temperature: corr(can't-tell, error), and the query
     ECE (reliability) / Brier (proper score). A temperature cannot change which composition is
     predicted (it preserves the argmax), so any gain is pure calibration, not a re-fit.

Deterministic via explicit seeds. Usage:
    python scripts/stageD2_feature_calibration.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lacuna.core.rng import RNGState
from lacuna.config import load_config
from lacuna.models.composition_head import (
    CompositionHead, ensemble_alpha, composition_mean, cant_tell_mass,
)
from lacuna.data.composition_batch import build_composition_batch, N_FOOTPRINT_FEATURES
from lacuna.training.composition_calibration import (
    apply_temperature, default_query_set, query_ece, collect_query_pairs, brier_score,
)
from lacuna.training.composition_recalibration import fit_feature_temperature
from scripts.stageC_composition_head import init_encoder, forward_alpha, load_raws

BASELINE = "/mnt/artifacts/project_lacuna/runs/stage0_general_baseline"


def _collect(encoder, heads, raws, *, n_batches, batch_size, max_rows, max_cols, device, seed,
             block_rate_share):
    """(footprints [N,F], ensemble alpha [N,3], realised [N,3]) as torch tensors over fresh data."""
    rng = RNGState(seed=seed)
    encoder.eval()
    alpha_models, realised, foot = [[] for _ in heads], [], []
    with torch.no_grad():
        for _ in range(n_batches):
            mb = build_composition_batch(raws, rng.spawn(), max_rows=max_rows, max_cols=max_cols,
                                         batch_size=batch_size, block_rate_share=block_rate_share,
                                         with_footprints=True)
            b = mb.batch.to(device)
            extra = mb.footprints.to(device)
            for m, head in enumerate(heads):
                alpha_models[m].append(forward_alpha(encoder, head, b, extra).cpu())
            realised.append(mb.composition.clone())
            foot.append(mb.footprints.clone())
    per_model = torch.stack([torch.cat(a, 0) for a in alpha_models], 0)
    return torch.cat(foot, 0), ensemble_alpha(per_model), torch.cat(realised, 0)


def _corr(vac, err):
    return float(np.corrcoef(vac, err)[0, 1])


def _report(alpha_np, realised_np, queries):
    err = np.abs(alpha_np / alpha_np.sum(1, keepdims=True) - realised_np).sum(1)
    vac = (alpha_np.shape[1] / alpha_np.sum(1))
    preds, inds = collect_query_pairs(alpha_np, realised_np, queries)
    return {"ece": round(query_ece(alpha_np, realised_np, queries), 4),
            "brier": round(brier_score(preds, inds), 4),
            "cant_tell_mean": round(float(vac.mean()), 4),
            "vacuity_error_corr": round(_corr(vac, err), 4)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline-checkpoint", default=f"{BASELINE}/checkpoints/best_model.pt")
    ap.add_argument("--baseline-config", default=f"{BASELINE}/config.yaml")
    ap.add_argument("--heads-checkpoint",
                    default=f"{BASELINE}/checkpoints/stageC_composition_frozen-probe+footprint.pt")
    ap.add_argument("--global-tau", type=float, default=None,
                    help="Stage-D global temperature (else read from the Stage-D report)")
    ap.add_argument("--calibration-report", default=f"{BASELINE}/stageD_calibration.json")
    ap.add_argument("--cal-batches", type=int, default=60)
    ap.add_argument("--test-batches", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--head-hidden", type=int, default=64)
    ap.add_argument("--block-rate-share", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=20260530)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output", type=Path, default=Path(f"{BASELINE}/stageD2_feature_calibration.json"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    cfg = load_config(args.baseline_config)
    dims = dict(hidden_dim=cfg.model.hidden_dim, evidence_dim=cfg.model.evidence_dim,
                n_layers=cfg.model.n_layers, n_heads=cfg.model.n_heads,
                max_cols=cfg.data.max_cols, dropout=cfg.model.dropout)
    max_rows, max_cols = cfg.data.max_rows, cfg.data.max_cols
    global_tau = args.global_tau
    if global_tau is None:
        global_tau = float(json.loads(Path(args.calibration_report).read_text())["tau"])

    encoder = init_encoder(args.baseline_checkpoint, dims, args.device)
    state = torch.load(args.heads_checkpoint, map_location="cpu", weights_only=False)["head_states"]
    heads = []
    for hs in state:
        h = CompositionHead(cfg.model.evidence_dim, hidden_dim=args.head_hidden,
                            dropout=cfg.model.dropout, n_extra_features=N_FOOTPRINT_FEATURES)
        h.load_state_dict(hs)
        heads.append(h.to(args.device).eval())
    print(f"Loaded {len(heads)} frozen+footprint heads | global tau={global_tau:.3f} | device {args.device}")

    com = dict(batch_size=args.batch_size, max_rows=max_rows, max_cols=max_cols,
               device=args.device, block_rate_share=args.block_rate_share)
    print("collecting calibration split ...", flush=True)
    f_cal, a_cal, r_cal = _collect(encoder, heads, load_raws(cfg.data.val_datasets, max_cols),
                                   n_batches=args.cal_batches, seed=args.seed + 1, **com)
    print("collecting test split ...", flush=True)
    f_te, a_te, r_te = _collect(encoder, heads, load_raws(cfg.data.val_datasets, max_cols),
                                n_batches=args.test_batches, seed=args.seed + 2, **com)

    # --- Step 1: feasibility probe (is |error| predictable from the footprint?) ---
    err_cal = (composition_mean(a_cal) - r_cal).abs().sum(-1).numpy()
    err_te = (composition_mean(a_te) - r_te).abs().sum(-1).numpy()
    rf = RandomForestRegressor(n_estimators=300, random_state=0, n_jobs=-1)
    rf.fit(f_cal.numpy(), err_cal)
    pred_te = rf.predict(f_te.numpy())
    ss_res = float(((err_te - pred_te) ** 2).sum())
    ss_tot = float(((err_te - err_te.mean()) ** 2).sum())
    probe_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    probe_corr = float(np.corrcoef(pred_te, err_te)[0, 1])
    print(f"\nFEASIBILITY PROBE: |error| predictable from footprint? "
          f"held-out R²={probe_r2:.3f}, corr(pred,actual)={probe_corr:.3f}")

    # --- Step 2: fit the feature-conditional temperature, evaluate on test ---
    model = fit_feature_temperature(f_cal, a_cal, r_cal, RNGState(seed=args.seed + 9))
    with torch.no_grad():
        a_te_feat = model.recalibrate(a_te, f_te).numpy()
    queries = default_query_set()
    results = {
        "global_tau": round(float(global_tau), 4), "probe_r2": round(probe_r2, 4),
        "probe_corr": round(probe_corr, 4), "n_cal": int(len(r_cal)), "n_test": int(len(r_te)),
        "uncalibrated": _report(a_te.numpy(), r_te.numpy(), queries),
        "global_temperature": _report(apply_temperature(a_te.numpy(), global_tau), r_te.numpy(), queries),
        "feature_conditional": _report(a_te_feat, r_te.numpy(), queries),
    }

    print("\n" + "=" * 78)
    print("STAGE D-v2 — feature-conditional vs global calibration (test split)")
    print("=" * 78)
    print(f"  {'method':22s} {'ECE':>7s} {'Brier':>7s} {'cant-tell':>10s} {'corr(vac,err)':>14s}")
    for k in ("uncalibrated", "global_temperature", "feature_conditional"):
        m = results[k]
        print(f"  {k:22s} {m['ece']:7.4f} {m['brier']:7.4f} {m['cant_tell_mean']:10.4f} "
              f"{m['vacuity_error_corr']:14.4f}")
    print(f"\n  feasibility: held-out R²(|error| from footprint) = {probe_r2:.3f}")
    print("  goal: corr(vacuity,error) should rise from ~-0.15 (global) toward positive (feature-cond).")
    print("=" * 78)

    args.output.write_text(json.dumps(results, indent=2))
    print(f"\nWrote Stage D-v2 report -> {args.output}")


if __name__ == "__main__":
    main()
