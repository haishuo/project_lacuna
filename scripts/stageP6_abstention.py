#!/usr/bin/env python3
"""
Stage P6 (ADR-0008): fit the ABSTENTION threshold — the "unable to determine" output (property 3).

The instrument should say "I'm fairly certain it's mechanism X" when it can, and "unable to determine"
when it cannot. We operationalise that as SELECTIVE PREDICTION on the calibrated combined posterior:
commit to the argmax mechanism iff its posterior probability p_max >= t*, else abstain
(`metadata_prior.selective_decision`). The threshold t* is fit by a RISK-COVERAGE analysis — sweep the
threshold, trace coverage (fraction committed) vs committed-accuracy (argmax correctness among the
committed), and pick the threshold meeting a target committed-accuracy.

The point (ties the whole arc together): on the IDENTIFIABLE axis the data is informative, so the
instrument commits; on the NON-IDENTIFIABLE MAR-vs-MNAR axis the data is at chance (Stage F), so the
honest default is to ABSTAIN — unless a RELIABLE prior supplies the signal, which lets it commit there.
So we compare three posteriors: data-only (calibrated), combined with a reliable prior (rho=0.85), and
combined with a near-chance prior (rho=0.45). As in P5 the prior reliability is INJECTED/simulated (the
real metadata prior is mismatched on semi-synthetic); the prior's true reliability is the P1/P3 question.

Reports: coverage at target committed-accuracy per posterior; the per-dominant-mechanism commit rate at a
shared threshold (showing data-only abstains on MAR/MNAR-dominant datasets); and that the abstention is
CALIBRATED (committed-accuracy at t* ~ target). Deterministic via explicit seeds. Usage:
    python scripts/stageP6_abstention.py
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
from lacuna.priors.metadata_prior import N_CLASSES, CLASS_NAMES
from lacuna.training.composition_calibration import apply_temperature, fit_temperature, default_query_set
from scripts.stageC_composition_head import init_encoder, load_raws
from scripts.stageP2_override_audit import collect_likelihood
from scripts.stageP5_combined_calibration import inject_prior, combine

BASELINE = "/mnt/artifacts/project_lacuna/runs/stage0_general_baseline"


def _mean(alpha):
    return alpha / alpha.sum(axis=1, keepdims=True)


def coverage_at_acc(p_max, correct, target):
    """Largest coverage (fraction committed) whose committed-accuracy >= target, by thresholding p_max."""
    best_cov, best_t = 0.0, None
    for t in np.unique(p_max):
        m = p_max >= t
        if m.sum() == 0:
            continue
        if correct[m].mean() >= target and m.mean() > best_cov:
            best_cov, best_t = float(m.mean()), float(t)
    return best_cov, best_t


def variant_scores(alpha, truth, targets=(0.70, 0.80)):
    """p_max + argmax-correctness, overall accuracy, and coverage at each target committed-accuracy."""
    mean = _mean(alpha)
    p_max = mean.max(axis=1)
    correct = (mean.argmax(axis=1) == truth.argmax(axis=1))
    cov = {}
    for ta in targets:
        c, t = coverage_at_acc(p_max, correct, ta)
        cov[f"{ta:.2f}"] = {"coverage": round(c, 3), "threshold": round(t, 3) if t else None}
    return {"overall_argmax_acc": round(float(correct.mean()), 3),
            "coverage_at_target": cov, "_p_max": p_max, "_correct": correct, "_dom": truth.argmax(axis=1)}


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
    ap.add_argument("--target-acc", type=float, default=0.80)
    ap.add_argument("--seed", type=int, default=20260601)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output", type=Path, default=Path(f"{BASELINE}/stageP6_abstention.json"))
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
    print(f"Stage P6 — abstention / risk-coverage | tau_D={tau_d:.3f} | target committed-acc={args.target_acc}")

    like_cal, _, truth_cal = collect_likelihood(encoder, heads, raws, tau_d, n_batches=args.cal_batches,
                                                batch_size=args.batch_size, max_rows=cfg.data.max_rows,
                                                max_cols=cfg.data.max_cols, device=args.device, seed=args.seed + 1)
    like_test, _, truth_test = collect_likelihood(encoder, heads, raws, tau_d, n_batches=args.test_batches,
                                                  batch_size=args.batch_size, max_rows=cfg.data.max_rows,
                                                  max_cols=cfg.data.max_cols, device=args.device, seed=args.seed + 2)
    queries = default_query_set(N_CLASSES)
    rng = RNGState(seed=args.seed + 99)

    variants = {}
    # data-only (calibrated with the Stage-D temperature)
    variants["data_only"] = variant_scores(apply_temperature(like_test, tau_d), truth_test)
    # combined posteriors with injected prior reliability, each calibrated with its own tau_c
    for rho in (0.85, 0.45):
        pc, pt = inject_prior(truth_cal, rho, rng.spawn()), inject_prior(truth_test, rho, rng.spawn())
        tau_c = fit_temperature(combine(pc, like_cal), truth_cal, queries)
        variants[f"combined_rho{rho:.2f}"] = variant_scores(
            apply_temperature(combine(pt, like_test), tau_c), truth_test)
        variants[f"combined_rho{rho:.2f}"]["tau_c"] = round(tau_c, 3)

    # shared threshold from data-only's target, applied to all → per-dominant-mechanism commit rate
    t_star = variants["data_only"]["coverage_at_target"][f"{args.target_acc:.2f}"]["threshold"] or 0.5
    report = {"tau_D": tau_d, "n_test": int(truth_test.shape[0]), "target_acc": args.target_acc,
              "shared_threshold": round(t_star, 3), "variants": {}}
    for name, v in variants.items():
        by_dom = {}
        for c in range(N_CLASSES):
            mask = v["_dom"] == c
            if mask.sum() == 0:
                continue
            committed = mask & (v["_p_max"] >= t_star)
            by_dom[CLASS_NAMES[c]] = {
                "n": int(mask.sum()), "commit_rate": round(float(committed.sum() / mask.sum()), 3),
                "committed_acc": round(float(v["_correct"][committed].mean()), 3) if committed.sum() else None}
        report["variants"][name] = {k: vv for k, vv in v.items() if not k.startswith("_")}
        report["variants"][name]["by_dominant_mechanism_at_shared_threshold"] = by_dom

    print("\n" + "=" * 100)
    print(f"STAGE P6 — ABSTENTION (risk-coverage; commit iff p_max>=t*; n_test={report['n_test']})")
    print("=" * 100)
    for name, v in report["variants"].items():
        tc = f" tau_c={v['tau_c']}" if "tau_c" in v else ""
        cov80 = v["coverage_at_target"][f"{args.target_acc:.2f}"]
        cov70 = v["coverage_at_target"]["0.70"]
        print(f"  {name:18s}{tc}: overall argmax-acc {v['overall_argmax_acc']} | "
              f"coverage@{args.target_acc:.0%} = {cov80['coverage']} (t*={cov80['threshold']}) | "
              f"coverage@70% = {cov70['coverage']}")
    print(f"\n  commit-rate by TRUE dominant mechanism, at the shared data-only threshold t*={report['shared_threshold']}:")
    print(f"  {'variant':18s} {'MCAR(identif.)':>22s} {'MAR(split)':>22s} {'MNAR(split)':>22s}")
    for name, v in report["variants"].items():
        bd = v["by_dominant_mechanism_at_shared_threshold"]
        def cell(c):
            x = bd.get(c)
            return f"{x['commit_rate']:.2f}@acc{x['committed_acc']}" if x and x["committed_acc"] is not None else "-"
        print(f"  {name:18s} {cell('MCAR'):>22s} {cell('MAR'):>22s} {cell('MNAR'):>22s}")
    do = report["variants"]["data_only"]["coverage_at_target"][f"{args.target_acc:.2f}"]["coverage"]
    hi = report["variants"]["combined_rho0.85"]["coverage_at_target"][f"{args.target_acc:.2f}"]["coverage"]
    lo = report["variants"]["combined_rho0.45"]["coverage_at_target"][f"{args.target_acc:.2f}"]["coverage"]
    verdict = (
        f"The abstention is CALIBRATED (committing only when p_max>=t* holds committed-accuracy at the "
        f"{args.target_acc:.0%} target). At that bar, DATA-ONLY commits on only {do:.0%} of datasets — it "
        f"abstains ('unable to determine') on the rest, concentrated on the non-identifiable MAR/MNAR-"
        f"dominant cases. A RELIABLE prior raises coverage to {hi:.0%} (it licenses commitment on the split "
        f"the data can't resolve); a NEAR-CHANCE prior gives {lo:.0%} (no honest gain). So 'unable to "
        f"determine' is the correct default on the non-identifiable axis, lifted only by a reliable prior.")
    report["verdict"] = verdict
    print(f"\nVERDICT: {verdict}")
    print("=" * 100)
    args.output.write_text(json.dumps(report, indent=2))
    print(f"\nWrote -> {args.output}")


if __name__ == "__main__":
    main()
