#!/usr/bin/env python3
"""
Stage P5 (ADR-0008): calibrate the COMBINED (prior x likelihood) posterior.

P2 established: combine the prior with the RAW likelihood evidence (to preserve the data's differential
informativeness and the override property), then calibrate the result afterwards. This script does that
"afterwards": it fits a single temperature tau_c on the COMBINED Dirichlet (alpha_cal = 1 + (alpha-1)/tau_c)
by minimising simplex-query ECE on a held-out calibration split, and evaluates reliability (ECE) and the
proper score (Brier) on a disjoint test split.

The honest difficulty: on semi-synthetic data we have the realised composition (ground truth) but the
real metadata prior is MISMATCHED (the Stage-B generator assigns mechanisms randomly, not by column
semantics). So we calibrate against an INJECTED prior of KNOWN reliability rho — it points at the true
dominant mechanism with probability rho, else at a wrong one — simulating a metadata prior whose accuracy
is rho. This isolates the calibration question from the prior-accuracy question (the latter is the P1/P3
face-validity programme). Sweeping rho makes the key point explicit: **a temperature fixes CONFIDENCE, not
CORRECTNESS** — the calibrated combined posterior beats data-only on the proper score only when the prior
is actually reliable; an unreliable prior is well-calibrated but no better (or worse) than data-only.

Baselines: data-only calibrated (the Stage-D temperature on the raw likelihood) and prior-only. Reported by
axis: MCAR queries (identifiable) vs MAR/MNAR queries (the non-identifiable split, where a reliable prior
should add resolution). Deterministic via explicit seeds. Usage:
    python scripts/stageP5_combined_calibration.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lacuna.core.rng import RNGState
from lacuna.config import load_config
from lacuna.data.composition_batch import N_FOOTPRINT_FEATURES
from lacuna.models.composition_head import CompositionHead
from lacuna.priors.metadata_prior import MCAR, MAR, MNAR, N_CLASSES, reliability_to_strength
from lacuna.training.composition_calibration import (
    apply_temperature, fit_temperature, default_query_set, collect_query_pairs,
    expected_calibration_error, brier_score,
)
from scripts.stageC_composition_head import init_encoder, load_raws
from scripts.stageP2_override_audit import collect_likelihood

BASELINE = "/mnt/artifacts/project_lacuna/runs/stage0_general_baseline"


def inject_prior(truth, reliability, rng):
    """Per-dataset prior of known reliability rho: points at argmax(truth) w.p. rho, else a wrong class."""
    n = truth.shape[0]
    dom = truth.argmax(axis=1)
    kappa = reliability_to_strength(reliability)
    alpha = np.ones((n, N_CLASSES))
    for i in range(n):
        if rng.numpy_rng.random() < reliability:
            m = dom[i]
        else:
            others = [c for c in range(N_CLASSES) if c != dom[i]]
            m = others[int(rng.numpy_rng.integers(len(others)))]
        alpha[i, m] += kappa
    return alpha


def combine(prior_alpha, like_alpha):
    """Evidence-pool a batch: alpha_post = 1 + (prior-1) + (like-1)."""
    return 1.0 + (prior_alpha - 1.0) + (like_alpha - 1.0)


def axis_scores(alpha, truth, queries):
    """ECE + Brier overall and split by axis (MCAR queries vs MAR/MNAR queries)."""
    preds, inds = collect_query_pairs(alpha, truth, queries)
    out = {"ece": round(expected_calibration_error(preds, inds), 4),
           "brier": round(brier_score(preds, inds), 4)}
    # per-axis: queries are ordered class-major (c=0 MCAR, 1 MAR, 2 MNAR) x thresholds
    nthr = len(queries) // N_CLASSES
    npts = len(preds) // len(queries)
    mcar_sl = slice(0, nthr * npts)
    split_sl = slice(nthr * npts, 3 * nthr * npts)  # MAR + MNAR queries
    out["brier_mcar_axis"] = round(brier_score(preds[mcar_sl], inds[mcar_sl]), 4)
    out["brier_split_axis"] = round(brier_score(preds[split_sl], inds[split_sl]), 4)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline-checkpoint", default=f"{BASELINE}/checkpoints/best_model.pt")
    ap.add_argument("--baseline-config", default=f"{BASELINE}/config.yaml")
    ap.add_argument("--heads-checkpoint",
                    default=f"{BASELINE}/checkpoints/stageC_composition_frozen-probe+footprint.pt")
    ap.add_argument("--calibration-report", default=f"{BASELINE}/stageD_calibration.json")
    ap.add_argument("--head-hidden", type=int, default=64)
    ap.add_argument("--cal-batches", type=int, default=40)
    ap.add_argument("--test-batches", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--reliabilities", type=float, nargs="+", default=[0.85, 0.65, 0.45])
    ap.add_argument("--seed", type=int, default=20260601)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output", type=Path, default=Path(f"{BASELINE}/stageP5_combined_calibration.json"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    cfg = load_config(args.baseline_config)
    dims = dict(hidden_dim=cfg.model.hidden_dim, evidence_dim=cfg.model.evidence_dim,
                n_layers=cfg.model.n_layers, n_heads=cfg.model.n_heads,
                max_cols=cfg.data.max_cols, dropout=cfg.model.dropout)
    tau_d = float(json.loads(Path(args.calibration_report).read_text())["tau"])
    encoder = init_encoder(args.baseline_checkpoint, dims, args.device)
    state = torch.load(args.heads_checkpoint, map_location="cpu", weights_only=False)["head_states"]
    heads = []
    for hs in state:
        h = CompositionHead(cfg.model.evidence_dim, hidden_dim=args.head_hidden,
                            dropout=cfg.model.dropout, n_extra_features=N_FOOTPRINT_FEATURES)
        h.load_state_dict(hs)
        heads.append(h.to(args.device).eval())

    raws = load_raws(cfg.data.val_datasets, cfg.data.max_cols)
    print(f"Stage P5 — combined-posterior calibration | data-only tau_D={tau_d:.3f} | {len(raws)} datasets")

    like_cal, _, truth_cal = collect_likelihood(encoder, heads, raws, tau_d, n_batches=args.cal_batches,
                                                batch_size=args.batch_size, max_rows=cfg.data.max_rows,
                                                max_cols=cfg.data.max_cols, device=args.device, seed=args.seed + 1)
    like_test, _, truth_test = collect_likelihood(encoder, heads, raws, tau_d, n_batches=args.test_batches,
                                                  batch_size=args.batch_size, max_rows=cfg.data.max_rows,
                                                  max_cols=cfg.data.max_cols, device=args.device, seed=args.seed + 2)
    queries = default_query_set(N_CLASSES)

    # Reference baselines (no prior): data-only calibrated (tau_D) and uncalibrated raw likelihood.
    data_only = axis_scores(apply_temperature(like_test, tau_d), truth_test, queries)
    report = {"tau_D": tau_d, "n_cal": int(truth_cal.shape[0]), "n_test": int(truth_test.shape[0]),
              "data_only_calibrated": data_only, "by_reliability": {}}

    rng = RNGState(seed=args.seed + 99)
    for rho in args.reliabilities:
        pcal = inject_prior(truth_cal, rho, rng.spawn())
        ptest = inject_prior(truth_test, rho, rng.spawn())
        comb_cal = combine(pcal, like_cal)
        comb_test = combine(ptest, like_test)
        tau_c = fit_temperature(comb_cal, truth_cal, queries)
        uncal = axis_scores(comb_test, truth_test, queries)
        cal = axis_scores(apply_temperature(comb_test, tau_c), truth_test, queries)
        report["by_reliability"][f"{rho:.2f}"] = {"tau_c": round(tau_c, 3), "combined_uncal": uncal,
                                                   "combined_cal": cal}

    print("\n" + "=" * 100)
    print("STAGE P5 — COMBINED-POSTERIOR CALIBRATION (lower ECE/Brier better; n_test="
          f"{report['n_test']})")
    print("=" * 100)
    d = data_only
    print(f"  data-only calibrated (tau_D={tau_d:.2f}) : ECE {d['ece']} | Brier {d['brier']}  "
          f"(MCAR-axis {d['brier_mcar_axis']} | split-axis {d['brier_split_axis']})")
    print(f"\n  combined (prior reliability rho), tau_c fit on the combined posterior:")
    for rho_s, r in report["by_reliability"].items():
        c, u = r["combined_cal"], r["combined_uncal"]
        print(f"   rho={rho_s} tau_c={r['tau_c']:.2f}: uncal ECE {u['ece']} -> CAL ECE {c['ece']} | "
              f"Brier {u['brier']}->{c['brier']}  (split-axis Brier {c['brier_split_axis']} "
              f"vs data-only {d['brier_split_axis']})")
    # verdict
    rel = report["by_reliability"]
    hi = rel.get("0.85") or list(rel.values())[0]
    lo = rel.get("0.45") or list(rel.values())[-1]
    helps = hi["combined_cal"]["brier_split_axis"] < d["brier_split_axis"]
    hurts = lo["combined_cal"]["brier_split_axis"] > d["brier_split_axis"]
    verdict = (
        f"The combined posterior CALIBRATES (a temperature drives ECE to ~{hi['combined_cal']['ece']}, "
        f"comparable to data-only {d['ece']}). But calibration fixes CONFIDENCE, not CORRECTNESS: a RELIABLE "
        f"prior (rho=0.85) improves split-axis Brier {d['brier_split_axis']}->{hi['combined_cal']['brier_split_axis']} "
        f"(resolution where the data is silent){' ✓' if helps else ''}; an UNRELIABLE prior (rho=0.45) is "
        f"still well-calibrated but no better/worse ({lo['combined_cal']['brier_split_axis']}){' (worse ✓)' if hurts else ''}. "
        f"So tau_c is real and prior-reliability-agnostic for CALIBRATION; the prior's VALUE is contingent "
        f"on its accuracy (the P1/P3 face-validity programme).")
    report["verdict"] = verdict
    print(f"\nVERDICT: {verdict}")
    print("=" * 100)
    args.output.write_text(json.dumps(report, indent=2))
    print(f"\nWrote -> {args.output}")


if __name__ == "__main__":
    main()
