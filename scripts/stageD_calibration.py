#!/usr/bin/env python3
"""
Stage D (ADR-0007): CALIBRATE the composition posterior — the headline metric.

Loads the Stage-C frozen-encoder + footprint-feature composition heads (the best Stage-C config),
fits a single temperature on a held-out CALIBRATION split (minimising query-ECE), and evaluates on a
disjoint TEST split. The deliverable is reliability of the posterior's simplex-region statements:
"when it says p% sure of region Q, it is right p% of the time."

Reported (all over the simplex-region query set P(f_c >= t)):
  - ECE (reliability) and Brier (proper score = reliability AND resolution) for UNCALIBRATED vs
    CALIBRATED vs a PRIOR-ONLY constant-Dirichlet baseline. The pre-registered bar (ADR-0007):
    calibration must beat the uncalibrated posterior, and beat prior-only on the proper score (a
    constant predictor can be trivially reliable but has no resolution).
  - The honest seam: ECE/Brier split by the random-vs-structured (MCAR) queries vs the MAR-vs-MNAR
    queries — the identifiable axis should be both calibrated AND resolved; the non-identifiable
    split should fall back toward prior-only (a correctly wide band).
  - The can't-tell mass after calibration, and the two ADR example statements as worked numbers.

Deterministic via explicit seeds. Usage:
    python scripts/stageD_calibration.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lacuna.config import load_config
from lacuna.models.composition_head import CompositionHead, ensemble_alpha, cant_tell_mass
from lacuna.data.composition_batch import build_composition_batch, N_FOOTPRINT_FEATURES
from lacuna.training.composition_calibration import (
    apply_temperature, default_query_set, collect_query_pairs, expected_calibration_error,
    brier_score, query_ece, fit_temperature, region_prob_ge,
)
from scripts.stageC_composition_head import init_encoder, forward_alpha, load_raws

BASELINE = "/mnt/artifacts/project_lacuna/runs/stage0_general_baseline"


def _collect(encoder, heads, raws, *, n_batches, batch_size, max_rows, max_cols, device, seed,
             block_rate_share):
    """Ensemble Dirichlet alpha [N,3] (numpy) + realised composition [N,3] over fresh data."""
    from lacuna.core.rng import RNGState
    rng = RNGState(seed=seed)
    encoder.eval()
    for h in heads:
        h.eval()
    alpha_models, realised = [[] for _ in heads], []
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
    per_model = torch.stack([torch.cat(a, 0) for a in alpha_models], 0)   # [M,N,3]
    alpha = ensemble_alpha(per_model).numpy()
    return alpha, torch.cat(realised, 0).numpy()


def _metrics(alpha, realised, queries):
    preds, inds = collect_query_pairs(alpha, realised, queries)
    return {"ece": round(expected_calibration_error(preds, inds), 4),
            "brier": round(brier_score(preds, inds), 4)}


def _prior_only_alpha(mean_comp, conc, n):
    return np.tile((conc * mean_comp)[None, :], (n, 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline-checkpoint", default=f"{BASELINE}/checkpoints/best_model.pt")
    ap.add_argument("--baseline-config", default=f"{BASELINE}/config.yaml")
    ap.add_argument("--heads-checkpoint",
                    default=f"{BASELINE}/checkpoints/stageC_composition_frozen-probe+footprint.pt")
    ap.add_argument("--cal-batches", type=int, default=40)
    ap.add_argument("--test-batches", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--head-hidden", type=int, default=64)
    ap.add_argument("--block-rate-share", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=20260530)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output", type=Path, default=Path(f"{BASELINE}/stageD_calibration.json"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    cfg = load_config(args.baseline_config)
    dims = dict(hidden_dim=cfg.model.hidden_dim, evidence_dim=cfg.model.evidence_dim,
                n_layers=cfg.model.n_layers, n_heads=cfg.model.n_heads,
                max_cols=cfg.data.max_cols, dropout=cfg.model.dropout)
    max_rows, max_cols = cfg.data.max_rows, cfg.data.max_cols

    encoder = init_encoder(args.baseline_checkpoint, dims, args.device)
    state = torch.load(args.heads_checkpoint, map_location="cpu", weights_only=False)["head_states"]
    heads = []
    for hs in state:
        h = CompositionHead(cfg.model.evidence_dim, hidden_dim=args.head_hidden,
                            dropout=cfg.model.dropout, n_extra_features=N_FOOTPRINT_FEATURES)
        h.load_state_dict(hs)
        heads.append(h.to(args.device))
    print(f"Loaded {len(heads)} Stage-C heads (frozen+footprint) | device {args.device}")

    val_raws = load_raws(cfg.data.val_datasets, max_cols)
    com = dict(n_batches=args.cal_batches, batch_size=args.batch_size, max_rows=max_rows,
               max_cols=max_cols, device=args.device, block_rate_share=args.block_rate_share)
    print("collecting calibration split ...", flush=True)
    a_cal, r_cal = _collect(encoder, heads, val_raws, seed=args.seed + 1, **{**com, "n_batches": args.cal_batches})
    print("collecting test split ...", flush=True)
    a_test, r_test = _collect(encoder, heads, val_raws, seed=args.seed + 2, **{**com, "n_batches": args.test_batches})

    queries = default_query_set()
    mcar_q = [(c, t) for (c, t) in queries if c == 0]            # random-vs-structured axis
    struct_q = [(c, t) for (c, t) in queries if c in (1, 2)]    # MAR-vs-MNAR axis

    # Fit temperature on calibration (minimise query-ECE).
    tau = fit_temperature(a_cal, r_cal, queries)
    a_test_cal = apply_temperature(a_test, tau)

    # Prior-only: constant Dirichlet at the calibration mean composition; concentration fit on
    # calibration to minimise its query Brier (its best shot).
    mean_comp = r_cal.mean(0)
    concs = np.geomspace(1.0, 200.0, 40)
    best_c = min(concs, key=lambda c: brier_score(
        *collect_query_pairs(_prior_only_alpha(mean_comp, c, len(r_cal)), r_cal, queries)))
    a_prior = _prior_only_alpha(mean_comp, best_c, len(r_test))

    results = {
        "tau": round(float(tau), 4), "prior_conc": round(float(best_c), 3),
        "n_cal": int(len(r_cal)), "n_test": int(len(r_test)),
        "uncalibrated": _metrics(a_test, r_test, queries),
        "calibrated": _metrics(a_test_cal, r_test, queries),
        "prior_only": _metrics(a_prior, r_test, queries),
        "honest_seam": {
            "calibrated_mcar_axis": _metrics(a_test_cal, r_test, mcar_q),
            "calibrated_mar_mnar": _metrics(a_test_cal, r_test, struct_q),
            "prior_only_mcar_axis": _metrics(a_prior, r_test, mcar_q),
            "prior_only_mar_mnar": _metrics(a_prior, r_test, struct_q),
        },
    }
    # can't-tell after calibration + does it track error now?
    vac = cant_tell_mass(torch.tensor(a_test_cal)).numpy()
    mean_cal = a_test_cal / a_test_cal.sum(1, keepdims=True)
    err = np.abs(mean_cal - r_test).sum(1)
    results["cant_tell_mean"] = round(float(vac.mean()), 4)
    results["vacuity_error_corr"] = round(float(np.corrcoef(vac, err)[0, 1]), 4)

    # Two ADR example statements on a few test datasets.
    examples = []
    for i in range(min(4, len(r_test))):
        examples.append({
            "realized": [round(float(x), 3) for x in r_test[i]],
            "P(f_MAR>=0.7)": round(float(region_prob_ge(a_test_cal[i:i+1], 1, 0.7)[0]), 3),
            "P(f_MNAR>=0.5)": round(float(region_prob_ge(a_test_cal[i:i+1], 2, 0.5)[0]), 3),
            "P(f_MCAR<=0.2)": round(float(1 - region_prob_ge(a_test_cal[i:i+1], 0, 0.2)[0]), 3),
        })
    results["examples"] = examples

    print("\n" + "=" * 74)
    print(f"STAGE D — composition-posterior calibration (tau={tau:.3f}, n_test={len(r_test)})")
    print("=" * 74)
    print(f"  query reliability (ECE) / proper score (Brier), lower better:")
    print(f"    uncalibrated : ECE {results['uncalibrated']['ece']}  Brier {results['uncalibrated']['brier']}")
    print(f"    CALIBRATED   : ECE {results['calibrated']['ece']}  Brier {results['calibrated']['brier']}")
    print(f"    prior-only   : ECE {results['prior_only']['ece']}  Brier {results['prior_only']['brier']}")
    print(f"  honest seam (calibrated):")
    print(f"    random-vs-structured (MCAR) : ECE {results['honest_seam']['calibrated_mcar_axis']['ece']}"
          f"  Brier {results['honest_seam']['calibrated_mcar_axis']['brier']}"
          f"   (prior-only Brier {results['honest_seam']['prior_only_mcar_axis']['brier']})")
    print(f"    MAR-vs-MNAR split           : ECE {results['honest_seam']['calibrated_mar_mnar']['ece']}"
          f"  Brier {results['honest_seam']['calibrated_mar_mnar']['brier']}"
          f"   (prior-only Brier {results['honest_seam']['prior_only_mar_mnar']['brier']})")
    print(f"  can't-tell mass mean {results['cant_tell_mean']} | corr(vacuity, error) {results['vacuity_error_corr']}")
    print("=" * 74)

    args.output.write_text(json.dumps(results, indent=2))
    print(f"\nWrote calibration report -> {args.output}")


if __name__ == "__main__":
    main()
