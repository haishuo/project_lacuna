#!/usr/bin/env python3
"""
P1 model-arm pilot — train the full LacunaModel from scratch on five regimes and report the
gap to the β₁′-profiled Bayes ceiling. Binary H0-vs-H1 test (MCAR-safe). NO checkpoint loaded.

Regimes use the EXACT recorded params from the profiled oracle run (one-to-one with the ceiling).

Usage:
    python scripts/run_feasibility_model_arm.py [PROFILED_RESULTS_JSON]
"""

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lacuna.config import load_config
from lacuna.core.rng import RNGState
from lacuna.feasibility.delta_generator import SelfCensorParams
from lacuna.feasibility.manifest import build_manifest, write_manifest
from lacuna.feasibility.model_arm import train_regime
from lacuna.feasibility.oracle import gauss_hermite, solve_beta0_population
from lacuna.feasibility.xmodel import ConditionalGaussian
from lacuna.models import create_lacuna_model

OUT_ROOT = Path("/mnt/artifacts/project_lacuna/feasibility")
PROFILED = Path(sys.argv[1]) if len(sys.argv) > 1 else \
    OUT_ROOT / "profiled_20260603_100549" / "results_profiled.json"

REGIME_CELLS = {  # (rho, delta, n)
    "control": (0.9, 0.5, 128),
    "boundary": (0.9, 1.0, 512),
    "moderate": (0.3, 1.0, 512),
    "strong": (0.0, 1.0, 2048),
}


def _find(recs, rho, delta, n, rate=0.1, beta1=1.0):
    for r in recs:
        if r["beta1"] == beta1 and r["rho"] == rho and r["delta"] == delta and r["n"] == n and r["target_rate"] == rate:
            return r
    raise ValueError(f"cell not found: rho={rho} delta={delta} n={n}")


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT).decode().strip()
    except Exception:
        return "unknown"


def main():
    t0 = time.time()
    timestamp = datetime.now().isoformat()
    run_id = "modelarm_pilot_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    out = OUT_ROOT / run_id
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = RNGState(seed=42)

    recs = json.loads(PROFILED.read_text())
    cfg = load_config("configs/training/survey.yaml")

    def factory():
        return create_lacuna_model(
            hidden_dim=cfg.model.hidden_dim, evidence_dim=cfg.model.evidence_dim,
            n_layers=cfg.model.n_layers, n_heads=cfg.model.n_heads, max_cols=2,
            dropout=cfg.model.dropout, mnar_variants=["self_censoring"],
            learn_evidence_attenuation=cfg.model.learn_evidence_attenuation,
            evidence_attenuation_init=cfg.model.evidence_attenuation_init,
        )

    regimes = []
    for name, (rho, delta, n) in REGIME_CELLS.items():
        c = _find(recs, rho, delta, n)
        regimes.append(dict(
            name=name, rho=rho, delta=delta, n=n,
            h1=SelfCensorParams(beta0=c["beta0_h1"], beta1=1.0, beta2=delta),
            h0=SelfCensorParams(beta0=c["beta0_h0_profiled"], beta1=c["beta1_prime"], beta2=0.0),
            ceiling=c["profiled_bayes_error"], ceiling_se=c["profiled_se"],
        ))
    # null control: H0==H1 (identical MAR, δ=0) ⇒ true error 0.5 by construction.
    nodes, weights = gauss_hermite(32)
    mar_null = solve_beta0_population(ConditionalGaussian.synthetic(0.9), 1.0, 0.0, 0.1, rng.spawn(), nodes, weights)
    regimes.append(dict(name="null_control", rho=0.9, delta=0.0, n=512,
                        h1=mar_null, h0=mar_null, ceiling=0.5, ceiling_se=0.0))

    results = []
    for rg in regimes:
        print(f"\n[model-arm] regime={rg['name']} rho={rg['rho']} delta={rg['delta']} n={rg['n']} "
              f"ceiling={rg['ceiling']:.3f}")
        m = train_regime(
            rg["name"], rg["rho"], rg["delta"], rg["n"], rg["h1"], rg["h0"],
            rg["ceiling"], rg["ceiling_se"], factory, rng.spawn(), device,
        )
        state = m.pop("best_state")
        if state is not None:
            torch.save(state, out / "checkpoints" / f"{rg['name']}.pt")  # provenance only; never loaded
        results.append(m)
        print(f"  model_error={m['model_error']:.3f}±{m['model_se']:.3f}  ceiling={m['ceiling']:.3f}  "
              f"gap={m['gap']:+.3f}  epochs={m['epochs_run']}  rates(H0/H1)={m['rate_h0_mean']:.3f}/{m['rate_h1_mean']:.3f}"
              f"  suspicious={m['suspicious_negative_gap']}")

    wall = time.time() - t0
    null = next(r for r in results if r["regime"] == "null_control")
    null_leak = null["model_error"] < 0.5 - 2 * null["model_se"]
    any_suspicious = any(r["suspicious_negative_gap"] for r in results)
    summary = {
        "seeds": 1, "pilot": True, "device": device,
        "null_control_error": null["model_error"], "null_control_leakage": bool(null_leak),
        "any_suspicious_negative_gap": bool(any_suspicious),
        "gaps": {r["regime"]: r["gap"] for r in results},
        "valid": (not null_leak) and (not any_suspicious),
    }
    manifest = build_manifest(
        run_id=run_id, git_commit=_git_commit(), timestamp=timestamp, arm="model", kind="main",
        grid={"regimes": REGIME_CELLS, "rate": 0.1, "beta1": 1.0,
              "counts": {"train": 8000, "val": 2000, "test": 4000}, "seed": 42},
        xmodel={"synthetic": "ConditionalGaussian.synthetic(rho)"},
        wall_clock_seconds=wall, metrics={"per_regime": results, "summary": summary},
        checkpoint_loaded=False, all_layers_trainable=True,
        trainable_param_count=results[0]["trainable_param_count"],
        split_scheme="independent disjoint-seed draws: train 8000 / val 2000 / test 4000 per regime",
        calibration={r["regime"]: r["ece"] for r in results},
        beta0_solver="FIXED recorded oracle params (not solved at train time)",
    )
    write_manifest(out / "manifest.json", manifest)
    (out / "results.json").write_text(json.dumps(results, indent=2, default=float))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n=== MODEL ARM PILOT DONE in {wall/60:.1f} min ===")
    print(json.dumps(summary, indent=2))
    print(f"artifacts: {out}")


if __name__ == "__main__":
    main()
