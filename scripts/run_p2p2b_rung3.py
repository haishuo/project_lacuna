"""
scripts/run_p2p2b_rung3.py

P2.2b ladder — RUNG 3: real survey X, 7 δ-bins, SUPPLIED target with head-side conditioning.

A direct A/B against the failed P2.2 global-evidence run: SAME leave-datasets-out split, SAME
generator, SAME RPS/metrics/leakage/manifest — the ONLY change is target_conditioned=True (the
δ-head also sees the pooled representation of the supplied target column). Isolates real-X
covariance geometry from target localization.

Reports RPS vs uniform/base-rate, bin/adjacent acc, E[δ] MAE, ECE/coverage, leakage gate, manifest
validity, and the comparison to the P2.2 global run. Run: python -u scripts/run_p2p2b_rung3.py
"""

import subprocess
import time
from pathlib import Path

import torch

from lacuna.core.rng import RNGState
from lacuna.data.catalog import create_default_catalog
from lacuna.survey.example_source import SurveyExampleSource
from lacuna.survey.loss import rps_loss
from lacuna.survey.run_manifest import write_manifest
from lacuna.survey.train import TrainConfig, train_delta_prior


def _base_rate_rps(sheets, num_bins=7):
    bins = torch.tensor([s.delta_bin for s in sheets], dtype=torch.long)
    marg = torch.bincount(bins, minlength=num_bins).float()
    marg = marg / marg.sum()
    logits = torch.log(marg.clamp(min=1e-9)).unsqueeze(0).repeat(len(bins), 1)
    return float(rps_loss(logits, bins).item())


def main() -> None:
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    cat = create_default_catalog()
    load = lambda names: [cat.load(n) for n in names]
    # IDENTICAL split to the P2.2 pilot (direct A/B).
    train = SurveyExampleSource(load([
        "survey_bfi", "survey_chile", "survey_computers", "survey_cps1988",
        "survey_psid7682", "survey_survey", "survey_yrbss", "survey_cars93",
    ]))
    val = SurveyExampleSource(load(["survey_hmda", "survey_workinghours"]))
    test = SurveyExampleSource(load(["survey_cps1985", "survey_psid1976"]))

    # IDENTICAL config to the P2.2 pilot, except target_conditioned=True.
    cfg = TrainConfig(
        delta_grid=[0.0, 0.15, 0.4, 0.75, 1.25, 1.75, 2.5], beta1_range=(0.0, 2.0),
        target_rate=0.3, max_rows=160, max_cols=32, batch_size=16,
        train_batches_per_epoch=50, max_epochs=25, patience=6,
        val_size=140, test_size=140,
        hidden_dim=96, evidence_dim=48, n_layers=2, n_heads=4,
        target_conditioned=True,
    )

    t0 = time.time()
    out = train_delta_prior(train, val, test, cfg, RNGState(seed=2026),
                            kind="main", run_id="p2p2b-rung3", git_commit=git,
                            timestamp="2026-06-04T12:00:00Z")
    wall = time.time() - t0
    m, res, lk = out["manifest"], out["results"], out["leakage"]
    m["wall_clock_seconds"] = round(wall, 1)

    # base-rate reference on a fresh seeded test draw
    rng = RNGState(seed=777)
    base_sheets = [test.make_one(cfg, rng.spawn(), delta=float(cfg.delta_grid[i % len(cfg.delta_grid)]),
                                 beta1=1.0).answer_sheet for i in range(cfg.test_size)]
    base_rps = _base_rate_rps(base_sheets)

    print("=" * 70)
    print("P2.2b RUNG 3 — real survey X, 7 δ-bins, TARGET-CONDITIONED (head-side)")
    print("=" * 70)
    print(f"wall={wall:.1f}s  params={m['trainable_param_count']}  epochs_run={res['epochs_run']}  "
          f"best_val_rps={res['best_val_rps']:.4f}")
    print(f"conditioning={m['model_arch']['conditioning']}  temperature={m['temperature']:.3f}")
    print(f"REFERENCES: uniform_rps={res['uniform_rps']:.4f}  base_rate_rps={base_rps:.4f}")

    def show(tag, b):
        print(f"  [{tag}] rps={b['rps']:.4f} log_score={b['log_score']:.4f} "
              f"bin_acc={b['bin_accuracy']:.3f} adj_acc={b['adjacent_accuracy']:.3f} "
              f"E[delta]_mae={b['e_delta']['mae']:.3f} p_delta0={b['p_delta0']:.3f} ece={b['ece']:.3f}")
        print(f"        coverage(50/80/90)={b['coverage']}")

    print("\nTEST metrics:")
    show("before-T", res["test_before_temperature"])
    show("after-T ", res["test_after_temperature"])

    after = res["test_after_temperature"]
    print(f"\nLEARNS ORDERED δ?  beats_uniform={after['rps'] < res['uniform_rps'] - 0.003}  "
          f"beats_base_rate={after['rps'] < base_rps - 0.003}  "
          f"bin_acc>chance(0.143)={after['bin_accuracy'] > 0.143}")
    print("A/B vs P2.2 GLOBAL pilot (same split/config): P2.2 test RPS ~0.1904 (= uniform). "
          f"rung3 test RPS = {after['rps']:.4f}")

    print("\nLEAKAGE GATE:")
    print(f"  delta->rate pearson={lk.delta_rate_pearson:+.3f}  "
          f"rate_only_acc={lk.rate_only_acc:.3f} vs base_rate_acc={lk.base_rate_acc:.3f}")
    print(f"  LEAKAGE_PASS = {out['leakage_pass']}")

    Path("runs").mkdir(exist_ok=True)
    p = write_manifest(Path("runs/p2p2b-rung3.json"), m)
    print(f"\nmanifest written + re-validated: {p}")


if __name__ == "__main__":
    main()
