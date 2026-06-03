#!/usr/bin/env python3
"""
Run the P1 CPU-only ORACLE arm (the analytic Bayes ceiling) — NO model training.

Two passes, both computing only the analytic ceiling (PROPOSAL P1, build-order step ii):
  - synthetic-X: the clean mathematical ceiling, with ρ (predictor-target correlation)
    explicitly swept — the Molenberghs knob.
  - real-survey-X: a fitted-ConditionalGaussian transfer check on the survey pool
    (assumption reported, never hidden).

Reported Bayes error is a Monte-Carlo ESTIMATE of the theoretical Bayes error of the
optimal LLR test, with standard error / 95% CI / n_mc. Computes in float64. Trains
nothing, loads no checkpoint.

Usage:
    python scripts/run_feasibility_oracle.py
"""

import csv
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lacuna.core.rng import RNGState
from lacuna.data import create_default_catalog
from lacuna.data.semisynthetic import _zscore_columns
from lacuna.feasibility.manifest import build_manifest, write_manifest
from lacuna.feasibility.oracle import gauss_hermite
from lacuna.feasibility.sweep import cells_to_records, compute_oracle_cell, run_oracle_two_stage
from lacuna.feasibility.xmodel import ConditionalGaussian

OUT_ROOT = Path("/mnt/artifacts/project_lacuna/feasibility")

# Coarse synthetic grid (stage 1; stage 2 refines δ around the boundary).
DELTAS = [0.0, 0.5, 1.0, 1.5, 2.0]
BETA1S = [0.0, 1.0]
RATES = [0.1, 0.2, 0.4]
RHOS = [0.0, 0.3, 0.6, 0.9]
NS = [128, 512, 2048]
CELL_KW = dict(n_quad=32, n_mc=1000, n_pop_sample=10000, n_kl_sample=15000)

SURVEY_DATASETS = [
    "survey_bfi", "survey_workinghours", "survey_psid1976", "survey_cps1985",
    "survey_hmda", "survey_yrbss", "survey_computers", "survey_chile",
    "survey_cars93", "survey_survey",
]


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT).decode().strip()
    except Exception:
        return "unknown"


def run_synthetic(rng: RNGState):
    print(f"[synthetic] coarse grid: {len(DELTAS)*len(BETA1S)*len(RATES)*len(RHOS)*len(NS)} cells + refinement")
    cells = run_oracle_two_stage(DELTAS, BETA1S, RATES, RHOS, NS, rng.spawn(), be_gap=0.05, **CELL_KW)
    print(f"[synthetic] computed {len(cells)} cells")
    return cells


def _pick_target_predictor(X: torch.Tensor):
    """Target = highest original-variance column; predictor = column most correlated with it."""
    var = X.var(dim=0, unbiased=False)
    target = int(torch.argmax(var).item())
    Z = _zscore_columns(X)
    zt = Z[:, target]
    corrs = []
    for j in range(X.shape[1]):
        if j == target:
            corrs.append(-1.0)
            continue
        zj = Z[:, j]
        denom = (zt.std(unbiased=False) * zj.std(unbiased=False)).item()
        corrs.append(abs(((zt - zt.mean()) * (zj - zj.mean())).mean().item() / denom) if denom > 0 else 0.0)
    predictor = int(torch.tensor(corrs).argmax().item())
    return target, predictor, float(corrs[predictor])


def run_real(rng: RNGState):
    catalog = create_default_catalog()
    nodes, weights = gauss_hermite(CELL_KW["n_quad"])
    records = []
    real_deltas = [0.0, 0.5, 1.0, 2.0]
    real_rates = [0.1, 0.3]
    for name in SURVEY_DATASETS:
        try:
            raw = catalog.load(name)
        except Exception as e:
            print(f"[real] skip {name}: {e}")
            continue
        if raw.d < 2:
            continue
        X = torch.from_numpy(raw.data.astype("float32"))
        target, predictor, rho_meas = _pick_target_predictor(X)
        Z = _zscore_columns(X)
        try:
            xmodel = ConditionalGaussian.fit(Z[:, predictor], Z[:, target])
        except ValueError as e:
            print(f"[real] skip {name}: {e}")
            continue
        n_eff = min(raw.n, 2048)
        for rate in real_rates:
            for delta in real_deltas:
                cell = compute_oracle_cell(
                    delta, beta1=1.0, target_rate=rate, n=n_eff, rng=rng.spawn(),
                    xmodel=xmodel, **CELL_KW,
                )
                rec = cell.to_dict()
                rec.update({"dataset": name, "rho_measured": rho_meas,
                            "target_idx": target, "predictor_idx": predictor, "raw_n": raw.n})
                records.append(rec)
        print(f"[real] {name}: target={target} predictor={predictor} rho≈{rho_meas:.2f} n_eff={n_eff}")
    return records


def _summary(syn_records, real_records):
    def distinguishable(recs):
        # δ>0 cells whose 95%-CI upper bound is below 0.45 -> meaningfully below chance
        return [r for r in recs if r["delta"] > 0 and r["ci_high"] < 0.45]
    syn_pos = [r for r in syn_records if r["delta"] > 0]
    syn_dist = distinguishable(syn_records)
    per_rho_min = {}
    for r in syn_records:
        if r["delta"] > 0 and r.get("rho") is not None:
            per_rho_min.setdefault(r["rho"], 1.0)
            per_rho_min[r["rho"]] = min(per_rho_min[r["rho"]], r["bayes_error"])
    return {
        "synthetic_cells": len(syn_records),
        "synthetic_pos_delta_cells": len(syn_pos),
        "synthetic_distinguishable_cells": len(syn_dist),
        "synthetic_min_bayes_error": min((r["bayes_error"] for r in syn_records), default=None),
        "synthetic_min_bayes_error_per_rho": per_rho_min,
        "real_cells": len(real_records),
        "real_distinguishable_cells": len(distinguishable(real_records)),
        "real_min_bayes_error": min((r["bayes_error"] for r in real_records), default=None),
        "verdict_hint": (
            "signal-in-some-regions" if syn_dist else "no-signal-anywhere(kill)"
        ),
    }


def main():
    t0 = time.time()
    # NON-DETERMINISTIC: wall-clock timestamp/commit captured at the script boundary only.
    timestamp = datetime.now().isoformat()
    run_id = "oracle_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    out = OUT_ROOT / run_id
    out.mkdir(parents=True, exist_ok=True)
    rng = RNGState(seed=42)

    syn_cells = run_synthetic(rng.spawn())
    syn_records = cells_to_records(syn_cells)
    real_records = run_real(rng.spawn())

    (out / "results_synthetic.json").write_text(json.dumps(syn_records, indent=2))
    (out / "results_real.json").write_text(json.dumps(real_records, indent=2))

    # flat CSV (scalar columns only; xmodel flattened to type)
    flat_cols = ["source", "dataset", "delta", "beta1", "target_rate", "rho", "rho_measured",
                 "n", "bayes_error", "bayes_error_se", "ci_low", "ci_high", "n_mc",
                 "kl_10", "kl_01", "beta0_h0", "beta0_h1"]
    with open(out / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=flat_cols, extrasaction="ignore")
        w.writeheader()
        for r in syn_records:
            w.writerow({**r, "source": "synthetic"})
        for r in real_records:
            w.writerow({**r, "source": "real"})

    summary = _summary(syn_records, real_records)
    wall = time.time() - t0
    manifest = build_manifest(
        run_id=run_id, git_commit=_git_commit(), timestamp=timestamp, arm="oracle", kind="main",
        grid={"deltas": DELTAS, "beta1s": BETA1S, "rates": RATES, "rhos": RHOS, "ns": NS,
              "cell_kwargs": CELL_KW, "real_deltas": [0.0, 0.5, 1.0, 2.0], "real_rates": [0.1, 0.3]},
        xmodel={"synthetic": "ConditionalGaussian.synthetic(rho) [exact]",
                "real": "ConditionalGaussian.fit [assumption: Gaussian (z_p,z_t)]"},
        wall_clock_seconds=wall, metrics=summary, checkpoint_loaded=False,
        all_layers_trainable=None, trainable_param_count=None, split_scheme="n/a (oracle)",
        calibration=None,
    )
    write_manifest(out / "manifest.json", manifest)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n=== ORACLE DONE in {wall/60:.1f} min ===")
    print(json.dumps(summary, indent=2))
    print(f"artifacts: {out}")


if __name__ == "__main__":
    main()
