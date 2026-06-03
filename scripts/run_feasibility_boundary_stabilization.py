#!/usr/bin/env python3
"""
P1 boundary-stabilization experiment — training-procedure changes ONLY.

10 restarts (seeds 100-109) of the full LacunaModel from scratch on the FIXED boundary H0/H1 pair
(ρ=0.9, δ=1.0, n=512), recipe: lr 1e-4, 300-step warmup, cosine→1e-5, patience 20, max 80 epochs.
One shared FRESH held-out test set (seed block 9000) across restarts (reporting only; checkpoints
selected by validation). Ceiling held fixed at 0.251. No architecture/generator/oracle changes; no
checkpoint loaded; all layers trainable.

Usage: python scripts/run_feasibility_boundary_stabilization.py
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
from lacuna.feasibility.model_arm import regime_pool, train_regime
from lacuna.models import create_lacuna_model

OUT_ROOT = Path("/mnt/artifacts/project_lacuna/feasibility")
PROFILED = OUT_ROOT / "profiled_20260603_100549" / "results_profiled.json"

RHO, DELTA, N, RATE = 0.9, 1.0, 512, 0.1
CEILING, CEILING_SE = 0.251, 0.010
SEEDS = list(range(100, 110))
TEST_SEED = 9000
RECIPE = dict(lr=1e-4, warmup_steps=300, lr_min=1e-5, patience=20, max_epochs=80,
              n_train=8000, n_val=2000, batch_size=32)
ESCAPE_THRESHOLD = 0.40
SUCCESS_ESCAPE_FRAC = 0.6
SUCCESS_MEDIAN = 0.32


def _git():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT).decode().strip()
    except Exception:
        return "unknown"


def main():
    t0 = time.time()
    ts = datetime.now().isoformat()
    run_id = "boundary_stab_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    out = OUT_ROOT / run_id
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    recs = json.loads(PROFILED.read_text())
    c = next(r for r in recs if r["beta1"] == 1.0 and r["rho"] == RHO and r["delta"] == DELTA
             and r["n"] == N and r["target_rate"] == RATE)
    h1 = SelfCensorParams(beta0=c["beta0_h1"], beta1=1.0, beta2=DELTA)
    h0 = SelfCensorParams(beta0=c["beta0_h0_profiled"], beta1=c["beta1_prime"], beta2=0.0)

    cfg = load_config("configs/training/survey.yaml")

    def factory():
        return create_lacuna_model(
            hidden_dim=cfg.model.hidden_dim, evidence_dim=cfg.model.evidence_dim,
            n_layers=cfg.model.n_layers, n_heads=cfg.model.n_heads, max_cols=2,
            dropout=cfg.model.dropout, mnar_variants=["self_censoring"],
            learn_evidence_attenuation=cfg.model.learn_evidence_attenuation,
            evidence_attenuation_init=cfg.model.evidence_attenuation_init,
        )

    # ONE shared fresh held-out test set (seed 9000) — reporting only.
    test_pool = regime_pool(RHO, h1, h0, N, 4000, RNGState(seed=TEST_SEED))
    test_stats = test_pool[2]

    results = []
    for seed in SEEDS:
        print(f"[stab] seed={seed} recipe=warmup300+cosine lr=1e-4->1e-5 patience=20")
        m = train_regime(
            "boundary", RHO, DELTA, N, h1, h0, CEILING, CEILING_SE,
            factory, RNGState(seed=seed), device, test_pool=test_pool, **RECIPE,
        )
        m.pop("best_state", None)
        m["seed"] = seed
        m["escaped_basin"] = bool(m["model_error"] < ESCAPE_THRESHOLD)
        results.append(m)
        print(f"  seed={seed} test_err={m['model_error']:.3f} best_val={m['best_val_error']:.3f} "
              f"gap={m['gap']:+.3f} epochs={m['epochs_run']} escaped={m['escaped_basin']} "
              f"suspicious={m['suspicious_negative_gap']}")

    errs = sorted(r["model_error"] for r in results)
    mean = sum(errs) / len(errs)
    std = (sum((e - mean) ** 2 for e in errs) / len(errs)) ** 0.5
    median = (errs[len(errs) // 2 - 1] + errs[len(errs) // 2]) / 2 if len(errs) % 2 == 0 else errs[len(errs) // 2]
    val_selected = min(results, key=lambda r: r["best_val_error"])  # the restart you'd pick by val
    n_escape = sum(r["escaped_basin"] for r in results)
    any_leak = any(r["suspicious_negative_gap"] for r in results)
    success = (n_escape / len(results) >= SUCCESS_ESCAPE_FRAC) and (median <= SUCCESS_MEDIAN) and (not any_leak)
    summary = {
        "ceiling": CEILING, "n_restarts": len(SEEDS), "seeds": SEEDS,
        "escape_fraction": n_escape / len(SEEDS), "n_escaped": n_escape,
        "test_error_mean": mean, "test_error_std": std, "test_error_median": median,
        "test_error_best": min(errs),
        "val_selected_seed": val_selected["seed"],
        "val_selected_test_error": val_selected["model_error"],
        "gap_mean": mean - CEILING, "gap_median": median - CEILING,
        "ece_mean": sum(r["ece"] for r in results) / len(results),
        "rate_h0_mean": test_stats["rate_h0_mean"], "rate_h1_mean": test_stats["rate_h1_mean"],
        "any_suspicious_negative_gap": bool(any_leak),
        "success_thresholds": {"escape_frac>=": SUCCESS_ESCAPE_FRAC, "median<=": SUCCESS_MEDIAN},
        "stabilization_success": bool(success),
    }
    wall = time.time() - t0
    manifest = build_manifest(
        run_id=run_id, git_commit=_git(), timestamp=ts, arm="model", kind="main",
        grid={"boundary": {"rho": RHO, "delta": DELTA, "n": N, "rate": RATE},
              "h1": [h1.beta0, h1.beta1, h1.beta2], "h0": [h0.beta0, h0.beta1, h0.beta2],
              "ceiling": CEILING, "seeds": SEEDS, "test_seed": TEST_SEED, "recipe": RECIPE},
        xmodel={"synthetic": "ConditionalGaussian.synthetic(0.9)"},
        wall_clock_seconds=wall, metrics={"per_restart": results, "summary": summary},
        checkpoint_loaded=False, all_layers_trainable=True,
        trainable_param_count=results[0]["trainable_param_count"],
        split_scheme="per-restart fresh train(8000)/val(2000); ONE shared fresh test(4000) seed=9000; val-only selection",
        calibration={f"seed{r['seed']}": r["ece"] for r in results},
        beta0_solver="FIXED recorded oracle params",
    )
    write_manifest(out / "manifest.json", manifest)
    (out / "results.json").write_text(json.dumps(results, indent=2, default=float))
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"\n=== BOUNDARY STABILIZATION DONE in {wall/60:.1f} min ===")
    print(json.dumps(summary, indent=2, default=float))
    print(f"artifacts: {out}")


if __name__ == "__main__":
    main()
