#!/usr/bin/env python3
"""
P1R-A: predictor-choice profiled MAR null — representative-regime ρ_a sweep (CPU oracle; no training).

For each P1 representative regime (fixed P1 H1), build a 3-column X [z_p, z_a, z_t] with the added
candidate predictor z_a at correlation ρ_a to the target, and compare the restricted (z_p-only) vs
richer (predictor-choice over {z_p, z_a}) profiled MAR null ON THE SAME 3-column data. Reports
E_restricted, E_richer, Δ = E_richer − E_restricted (the monotonic comparison), selected predictor,
and the 2-column P1 ceiling as secondary context only.

Usage: python scripts/run_feasibility_p1ra.py
"""

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lacuna.core.rng import RNGState
from lacuna.feasibility.delta_generator import SelfCensorParams
from lacuna.feasibility.manifest import build_manifest, write_manifest
from lacuna.feasibility.profiled_oracle_mv import compute_p1ra_cell

OUT_ROOT = Path("/mnt/artifacts/project_lacuna/feasibility")
PROFILED = OUT_ROOT / "profiled_20260603_100549" / "results_profiled.json"

# P1 representative regimes: (rho_orig, delta, n, 2-col P1 ceiling context)
REGIMES = {
    "control":  dict(rho_orig=0.9, delta=0.5, n=128,  p1_ceiling=0.453),
    "boundary": dict(rho_orig=0.9, delta=1.0, n=512,  p1_ceiling=0.251),
    "moderate": dict(rho_orig=0.3, delta=1.0, n=512,  p1_ceiling=0.104),
    "strong":   dict(rho_orig=0.0, delta=1.0, n=2048, p1_ceiling=0.004),
}
RHO_A = [0.0, 0.3, 0.6, 0.9, 0.95, 0.99]
RATE = 0.1


def _git():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT).decode().strip()
    except Exception:
        return "unknown"


def main():
    t0 = time.time()
    ts = datetime.now().isoformat()
    run_id = "p1ra_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    out = OUT_ROOT / run_id
    out.mkdir(parents=True, exist_ok=True)
    rng = RNGState(seed=42)

    recs = json.loads(PROFILED.read_text())

    def h1_for(rho_orig, delta, n):
        c = next(r for r in recs if r["beta1"] == 1.0 and r["rho"] == rho_orig and r["delta"] == delta
                 and r["n"] == n and r["target_rate"] == RATE)
        return SelfCensorParams(beta0=c["beta0_h1"], beta1=1.0, beta2=delta)

    results = []
    print(f"{'regime':>9} {'rho_a':>5} {'sel':>4} {'E_restr':>8} {'E_rich':>8} {'delta':>7} {'P1ceil':>7}")
    for name, rg in REGIMES.items():
        h1 = h1_for(rg["rho_orig"], rg["delta"], rg["n"])
        for rho_a in RHO_A:
            cell = compute_p1ra_cell(rg["rho_orig"], rho_a, h1, rg["n"], RATE, rng.spawn(),
                                     n_quad=32, n_mc=2000, n_fit=30000, kl_sample=30000)
            cell["regime"] = name
            cell["p1_2col_ceiling"] = rg["p1_ceiling"]
            cell["E_restricted_minus_p1ceiling"] = cell["E_restricted"] - rg["p1_ceiling"]
            results.append(cell)
            print(f"{name:>9} {rho_a:>5} {cell['selected_predictor']:>4} {cell['E_restricted']:>8.3f} "
                  f"{cell['E_richer']:>8.3f} {cell['E_richer_minus_restricted']:>+7.3f} {rg['p1_ceiling']:>7.3f}")

    any_viol = any(r["monotonicity_violation"] for r in results)
    max_absorb = max(r["E_richer_minus_restricted"] for r in results)
    n_za = sum(1 for r in results if r["selected_predictor"] == "z_a")
    summary = {
        "n_cells": len(results),
        "any_monotonicity_violation": bool(any_viol),
        "max_absorption_delta": max_absorb,
        "cells_selecting_z_a": n_za,
        "verdict_hint": ("predictor-choice-absorbs" if max_absorb > 0.02
                         else "no-absorption-z_p-dominates"),
    }
    wall = time.time() - t0
    manifest = build_manifest(
        run_id=run_id, git_commit=_git(), timestamp=ts, arm="oracle", kind="main",
        grid={"regimes": {k: {kk: vv for kk, vv in v.items()} for k, v in REGIMES.items()},
              "rho_a": RHO_A, "rate": RATE, "candidates": ["z_p", "z_a"],
              "rho_pa": "rho_orig*rho_a (conditional-independence)"},
        xmodel={"type": "MultivariateGaussianX (3-col [z_p,z_a,z_t])"},
        wall_clock_seconds=wall, metrics={"per_cell": results, "summary": summary},
        checkpoint_loaded=False, all_layers_trainable=None, trainable_param_count=None,
        split_scheme="n/a (oracle)", calibration=None,
        beta0_solver="H1 fixed P1 params; H0 β0′ rate-matched per candidate predictor",
    )
    write_manifest(out / "manifest.json", manifest)
    (out / "results.json").write_text(json.dumps(results, indent=2, default=float))
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"\n=== P1R-A DONE in {wall/60:.1f} min ===")
    print(json.dumps(summary, indent=2, default=float))
    print(f"artifacts: {out}")


if __name__ == "__main__":
    main()
