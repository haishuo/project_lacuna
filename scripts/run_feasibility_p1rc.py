#!/usr/bin/env python3
"""
P1R-C: multi-predictor MAR null — representative-regime ρ_a sweep (CPU oracle; no training).

H0 = σ(β0'+β1'·z_p+β2'·z_a) (least-favorable by max Bayes error; restricted β2'=0 in the grid).
Reports E_restricted, E_multi, Δ, selected coeffs, β0', edge status, Var(z_t|z_p,z_a), SE/CI,
monotonicity check. Final pre-P2 robustness gate.

Usage: python scripts/run_feasibility_p1rc.py
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
from lacuna.feasibility.profiled_oracle_mv_multi import compute_p1rc_cell

OUT_ROOT = Path("/mnt/artifacts/project_lacuna/feasibility")
PROFILED = OUT_ROOT / "profiled_20260603_100549" / "results_profiled.json"

REGIMES = {
    "control":  dict(rho_orig=0.9, delta=0.5, n=128),
    "boundary": dict(rho_orig=0.9, delta=1.0, n=512),
    "moderate": dict(rho_orig=0.3, delta=1.0, n=512),
    "strong":   dict(rho_orig=0.0, delta=1.0, n=2048),
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
    run_id = "p1rc_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    out = OUT_ROOT / run_id
    out.mkdir(parents=True, exist_ok=True)
    rng = RNGState(seed=42)
    recs = json.loads(PROFILED.read_text())

    def h1_for(rho_orig, delta, n):
        c = next(r for r in recs if r["beta1"] == 1.0 and r["rho"] == rho_orig and r["delta"] == delta
                 and r["n"] == n and r["target_rate"] == RATE)
        return SelfCensorParams(beta0=c["beta0_h1"], beta1=1.0, beta2=delta)

    results = []
    print(f"{'regime':>9} {'rho_a':>5} {'Var(zt|o)':>9} {'E_restr':>8} {'E_multi':>8} {'delta':>7} "
          f"{'b1*':>5} {'b2*':>5} {'edge':>6}")
    for name, rg in REGIMES.items():
        h1 = h1_for(rg["rho_orig"], rg["delta"], rg["n"])
        for rho_a in RHO_A:
            cell = compute_p1rc_cell(rg["rho_orig"], rho_a, h1, rg["n"], RATE, rng.spawn(),
                                     n_quad=32, n_grid=6, n_mc_search=400, n_mc_final=2500)
            cell["regime"] = name
            results.append(cell)
            print(f"{name:>9} {rho_a:>5} {cell['var_zt_given_obs']:>9.3f} {cell['E_restricted']:>8.3f} "
                  f"{cell['E_multi']:>8.3f} {cell['E_multi_minus_restricted']:>+7.3f} "
                  f"{cell['beta1p_multi']:>5.2f} {cell['beta2p_multi']:>5.2f} {cell['edge_status']:>6}")

    any_viol = any(r["monotonicity_violation"] for r in results)
    any_edge = any(r["edge_status"] != "ok" for r in results)
    max_absorb = max(r["E_multi_minus_restricted"] for r in results)
    # collapse: E_multi within MC of chance (>= 0.45) for δ>0 cells
    collapse_cells = [(r["regime"], r["rho_a"]) for r in results if r["E_multi"] >= 0.45]
    moderate_rho_collapse = any(r["E_multi"] >= 0.45 and r["rho_a"] <= 0.6 for r in results)
    summary = {
        "n_cells": len(results),
        "any_monotonicity_violation": bool(any_viol),
        "any_grid_edge": bool(any_edge),
        "max_absorption_delta": max_absorb,
        "collapse_cells_E_multi_ge_0.45": collapse_cells,
        "collapse_at_moderate_rho_a(<=0.6)": bool(moderate_rho_collapse),
    }
    wall = time.time() - t0
    manifest = build_manifest(
        run_id=run_id, git_commit=_git(), timestamp=ts, arm="oracle", kind="main",
        grid={"regimes": REGIMES, "rho_a": RHO_A, "rate": RATE,
              "rho_pa": "rho_orig*rho_a", "predictors": ["z_p", "z_a"],
              "profiling": "max-Bayes-error over 2D (b1',b2') grid incl. restricted b2'=0 slice"},
        xmodel={"type": "MultivariateGaussianX (3-col)"},
        wall_clock_seconds=wall, metrics={"per_cell": results, "summary": summary},
        checkpoint_loaded=False, all_layers_trainable=None, trainable_param_count=None,
        split_scheme="n/a (oracle)", calibration=None,
        beta0_solver="H1 fixed P1 params; H0 β0′ rate-matched per (β1′,β2′)",
    )
    write_manifest(out / "manifest.json", manifest)
    (out / "results.json").write_text(json.dumps(results, indent=2, default=float))
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"\n=== P1R-C DONE in {wall/60:.1f} min ===")
    print(json.dumps(summary, indent=2, default=float))
    print(f"artifacts: {out}")


if __name__ == "__main__":
    main()
