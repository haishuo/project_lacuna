#!/usr/bin/env python3
"""
P1 model-arm multi-seed follow-up: boundary / moderate / strong across 3 seeds.

Answers whether the pilot's boundary failure (high-ρ, gap +0.222) is seed/optimization
sensitivity (recovers on some seeds) or a robust high-ρ learning gap. Full LacunaModel from
scratch each seed; binary MCAR-safe head; fixed oracle params; NO checkpoint loaded.

Usage: python scripts/run_feasibility_model_arm_multiseed.py
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
from lacuna.models import create_lacuna_model

OUT_ROOT = Path("/mnt/artifacts/project_lacuna/feasibility")
PROFILED = OUT_ROOT / "profiled_20260603_100549" / "results_profiled.json"
SEEDS = [42, 43, 44]
CELLS = {"boundary": (0.9, 1.0, 512), "moderate": (0.3, 1.0, 512), "strong": (0.0, 1.0, 2048)}


def _find(recs, rho, delta, n, rate=0.1, beta1=1.0):
    for r in recs:
        if r["beta1"] == beta1 and r["rho"] == rho and r["delta"] == delta and r["n"] == n and r["target_rate"] == rate:
            return r
    raise ValueError(f"cell not found {rho},{delta},{n}")


def _git():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT).decode().strip()
    except Exception:
        return "unknown"


def main():
    t0 = time.time()
    ts = datetime.now().isoformat()
    run_id = "modelarm_multiseed_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    out = OUT_ROOT / run_id
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
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

    results = []
    for name, (rho, delta, n) in CELLS.items():
        c = _find(recs, rho, delta, n)
        h1 = SelfCensorParams(beta0=c["beta0_h1"], beta1=1.0, beta2=delta)
        h0 = SelfCensorParams(beta0=c["beta0_h0_profiled"], beta1=c["beta1_prime"], beta2=0.0)
        for seed in SEEDS:
            print(f"[multiseed] {name} seed={seed} ceiling={c['profiled_bayes_error']:.3f}")
            m = train_regime(name, rho, delta, n, h1, h0, c["profiled_bayes_error"], c["profiled_se"],
                             factory, RNGState(seed=seed), device)
            m.pop("best_state", None)
            m["seed"] = seed
            results.append(m)
            print(f"  seed={seed} model_error={m['model_error']:.3f} gap={m['gap']:+.3f} epochs={m['epochs_run']}")

    # aggregate per regime
    agg = {}
    for name in CELLS:
        rs = [r for r in results if r["regime"] == name]
        errs = [r["model_error"] for r in rs]
        gaps = [r["gap"] for r in rs]
        mean = sum(errs) / len(errs)
        std = (sum((e - mean) ** 2 for e in errs) / len(errs)) ** 0.5
        agg[name] = {
            "ceiling": rs[0]["ceiling"], "seeds": SEEDS,
            "model_error_mean": mean, "model_error_std": std,
            "model_error_per_seed": errs, "gap_per_seed": gaps,
            "gap_mean": sum(gaps) / len(gaps),
            "boundary_recovers_some_seed": any(g < 0.10 for g in gaps) if name == "boundary" else None,
        }
    wall = time.time() - t0
    manifest = build_manifest(
        run_id=run_id, git_commit=_git(), timestamp=ts, arm="model", kind="main",
        grid={"cells": CELLS, "seeds": SEEDS, "rate": 0.1, "beta1": 1.0,
              "counts": {"train": 8000, "val": 2000, "test": 4000}},
        xmodel={"synthetic": "ConditionalGaussian.synthetic(rho)"},
        wall_clock_seconds=wall, metrics={"per_run": results, "aggregate": agg},
        checkpoint_loaded=False, all_layers_trainable=True,
        trainable_param_count=results[0]["trainable_param_count"],
        split_scheme="independent disjoint-seed draws per (regime,seed); 8000/2000/4000",
        calibration={f"{r['regime']}_s{r['seed']}": r["ece"] for r in results},
        beta0_solver="FIXED recorded oracle params",
    )
    write_manifest(out / "manifest.json", manifest)
    (out / "results.json").write_text(json.dumps(results, indent=2, default=float))
    (out / "aggregate.json").write_text(json.dumps(agg, indent=2, default=float))
    print(f"\n=== MULTISEED DONE in {wall/60:.1f} min ===")
    print(json.dumps(agg, indent=2, default=float))
    print(f"artifacts: {out}")


if __name__ == "__main__":
    main()
