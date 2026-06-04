"""
scripts/run_p2p2_pilot.py

Minimal P2.2 δ-prior pilot: a from-scratch, leave-datasets-out training run on the real survey
pool with own-value self-censoring, reporting calibration-first metrics (before/after temperature)
and the blocking matched-rate leakage gate, then writing a validated run manifest.

Determinism note: the trained weights are not bit-reproducible on CPU (train-time dropout +
nn.Embedding backward; see lacuna/survey/train.py). The data/leakage/manifest provenance IS
seeded. Run:  python -u scripts/run_p2p2_pilot.py
"""

import subprocess
import time
from pathlib import Path

from lacuna.core.rng import RNGState
from lacuna.data.catalog import create_default_catalog
from lacuna.survey.example_source import SurveyExampleSource
from lacuna.survey.run_manifest import write_manifest
from lacuna.survey.train import TrainConfig, train_delta_prior


def main() -> None:
    git = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()

    cat = create_default_catalog()
    load = lambda names: [cat.load(n) for n in names]
    # Disjoint leave-DATASETS-out split (out-of-family PROXY; one mechanism family in P2.2).
    train = load([
        "survey_bfi", "survey_chile", "survey_computers", "survey_cps1988",
        "survey_psid7682", "survey_survey", "survey_yrbss", "survey_cars93",
    ])
    val = load(["survey_hmda", "survey_workinghours"])
    test = load(["survey_cps1985", "survey_psid1976"])

    cfg = TrainConfig(
        delta_grid=[0.0, 0.15, 0.4, 0.75, 1.25, 1.75, 2.5],
        beta1_range=(0.0, 2.0), target_rate=0.3,
        max_rows=160, max_cols=32, batch_size=16,
        train_batches_per_epoch=50, max_epochs=25, patience=6,
        val_size=140, test_size=140,
        hidden_dim=96, evidence_dim=48, n_layers=2, n_heads=4,
    )

    t0 = time.time()
    out = train_delta_prior(
        SurveyExampleSource(train), SurveyExampleSource(val), SurveyExampleSource(test),
        cfg, RNGState(seed=2026),
        kind="main", run_id="p2p2-pilot-001", git_commit=git,
        timestamp="2026-06-04T12:00:00Z",
    )
    wall = time.time() - t0
    m, res, lk = out["manifest"], out["results"], out["leakage"]
    m["wall_clock_seconds"] = round(wall, 1)

    print("=" * 70)
    print("P2.2 PILOT — leave-datasets-out, own-value self-censoring")
    print("=" * 70)
    print(f"wall={wall:.1f}s  params={m['trainable_param_count']}  "
          f"epochs_run={res['epochs_run']}  best_val_rps={res['best_val_rps']:.4f}")
    print(f"temperature(fit on val)={m['temperature']:.3f}   "
          f"uniform_rps reference={res['uniform_rps']:.4f}")

    def show(tag, b):
        print(f"  [{tag}] rps={b['rps']:.4f} log_score={b['log_score']:.4f} "
              f"bin_acc={b['bin_accuracy']:.3f} adj_acc={b['adjacent_accuracy']:.3f} "
              f"E[delta]_mae={b['e_delta']['mae']:.3f} p_delta0={b['p_delta0']:.3f} "
              f"ece={b['ece']:.3f}")
        print(f"        coverage(50/80/90)={b['coverage']}")

    print("\nTEST metrics:")
    show("before-T", res["test_before_temperature"])
    show("after-T ", res["test_after_temperature"])

    print("\nLEAKAGE GATE:")
    print(f"  delta->rate pearson={lk.delta_rate_pearson:+.3f} slope={lk.delta_rate_slope:+.4f}")
    print(f"  rate_only_acc={lk.rate_only_acc:.3f} vs base_rate_acc={lk.base_rate_acc:.3f}")
    print(f"  per-bin mean realized rate (target={lk.target_rate:.3f}):")
    for row in lk.per_bin:
        print(f"    bin {row['bin']}: n={row['n']:3d} mean={row['mean_rate']:.4f} "
              f"dev={row['abs_dev_from_target']:.4f}")
    print(f"  LEAKAGE_PASS = {out['leakage_pass']}")

    Path("runs").mkdir(exist_ok=True)
    p = write_manifest(Path("runs/p2p2-pilot-001.json"), m)
    print(f"\nmanifest written + re-validated: {p}")


if __name__ == "__main__":
    main()
