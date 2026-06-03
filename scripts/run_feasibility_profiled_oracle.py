#!/usr/bin/env python3
"""
Run the CPU-only β₁′-PROFILED MAR-null oracle and compare to the point-null — NO model training.

For every synthetic cell, reports the point-null Bayes error (fixed-MAR vs fixed-MNAR) AND the
profiled Bayes error (MNAR vs the best-fitting MAR over β₁′), plus the selected β₁′. The profiled
surface is the deployment-relevant ceiling for the β₁-flexible MAR family. Trains nothing, loads no
checkpoint.

Usage:
    python scripts/run_feasibility_profiled_oracle.py
"""

import csv
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lacuna.core.rng import RNGState
from lacuna.feasibility.manifest import build_manifest, write_manifest
from lacuna.feasibility.profiled_oracle import compute_profiled_cell

OUT_ROOT = Path("/mnt/artifacts/project_lacuna/feasibility")

DELTAS = [0.0, 0.5, 1.0, 2.0]
BETA1S = [0.0, 1.0]
RATES = [0.1, 0.3]
RHOS = [0.0, 0.3, 0.6, 0.9]
NS = [128, 512, 2048]
KW = dict(n_quad=32, n_mc=1000, n_pop_sample=12000, n_fit_sample=30000)


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT).decode().strip()
    except Exception:
        return "unknown"


def main():
    t0 = time.time()
    timestamp = datetime.now().isoformat()  # NON-DETERMINISTIC: script-boundary only
    run_id = "profiled_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    out = OUT_ROOT / run_id
    out.mkdir(parents=True, exist_ok=True)
    rng = RNGState(seed=42)

    total = len(DELTAS) * len(BETA1S) * len(RATES) * len(RHOS) * len(NS)
    print(f"[profiled] {total} cells (point-null + β₁′-profiled each)")
    records = []
    i = 0
    for rho in RHOS:
        for rate in RATES:
            for beta1 in BETA1S:
                for n in NS:
                    for delta in DELTAS:
                        rec = compute_profiled_cell(delta, beta1, rate, n, rng.spawn(), rho=rho, **KW)
                        records.append(rec)
                        i += 1
            print(f"[profiled] rho={rho} rate={rate} done ({i}/{total})")

    (out / "results_profiled.json").write_text(json.dumps(records, indent=2, default=float))
    flat = ["delta", "beta1", "beta1_prime", "target_rate", "rho", "n",
            "point_bayes_error", "point_se", "profiled_bayes_error", "profiled_se",
            "profiled_ci_low", "profiled_ci_high", "beta0_h1", "beta0_h0_point", "beta0_h0_profiled", "n_mc"]
    with open(out / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=flat, extrasaction="ignore")
        w.writeheader()
        for r in records:
            w.writerow(r)

    # Summary: how much profiling absorbs, and whether signal survives.
    pos = [r for r in records if r["delta"] > 0]
    survive = [r for r in pos if r["profiled_ci_high"] < 0.45]
    absorbed = [r for r in pos if (r["profiled_bayes_error"] - r["point_bayes_error"]) > 0.05]
    by_b1_rho = {}
    for r in pos:
        k = f"b1={r['beta1']},rho={r['rho']}"
        d = by_b1_rho.setdefault(k, {"point": [], "profiled": [], "b1p": []})
        d["point"].append(r["point_bayes_error"])
        d["profiled"].append(r["profiled_bayes_error"])
        d["b1p"].append(r["beta1_prime"])
    by_b1_rho = {k: {"mean_point": sum(v["point"]) / len(v["point"]),
                     "mean_profiled": sum(v["profiled"]) / len(v["profiled"]),
                     "mean_beta1_prime": sum(v["b1p"]) / len(v["b1p"])}
                 for k, v in by_b1_rho.items()}
    summary = {
        "n_cells": len(records),
        "pos_delta_cells": len(pos),
        "profiled_distinguishable_cells": len(survive),
        "cells_absorbed_gt_0.05": len(absorbed),
        "min_profiled_bayes_error": min((r["profiled_bayes_error"] for r in records), default=None),
        "selected_beta1_prime_range": [min(r["beta1_prime"] for r in records),
                                       max(r["beta1_prime"] for r in records)],
        "mean_by_beta1_rho": by_b1_rho,
        "verdict_hint": "profiled-signal-survives" if survive else "profiled-collapse(kill)",
    }
    wall = time.time() - t0
    manifest = build_manifest(
        run_id=run_id, git_commit=_git_commit(), timestamp=timestamp, arm="oracle", kind="main",
        grid={"deltas": DELTAS, "beta1s": BETA1S, "rates": RATES, "rhos": RHOS, "ns": NS, "cell_kwargs": KW,
              "null": "beta1-prime-profiled MAR (best-fit over beta1') vs point-null"},
        xmodel={"synthetic": "ConditionalGaussian.synthetic(rho) [exact]"},
        wall_clock_seconds=wall, metrics=summary, checkpoint_loaded=False,
        all_layers_trainable=None, trainable_param_count=None,
        split_scheme="n/a (oracle)", calibration=None,
    )
    write_manifest(out / "manifest.json", manifest)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n=== PROFILED ORACLE DONE in {wall/60:.1f} min ===")
    print(json.dumps(summary, indent=2))
    print(f"artifacts: {out}")


if __name__ == "__main__":
    main()
