"""
scripts/run_p2p2b_rung1.py

P2.2b ladder — RUNG 1: oracle-aligned 2-column δ-bin learnability test.

Synthetic 2-col standard bivariate-normal X (ConditionalGaussian(rho)) + own-value self-censoring
on the KNOWN target column (col 1), matched rate, 7 δ-bins. Removes real-X geometry, table width,
and target localization — the cleanest test of whether the δ-bin head + RPS can learn ordered δ
at all. In-distribution (same synthetic process for train/val/test): this is a LEARNABILITY test,
not a generalization test.

Reports RPS vs uniform/base-rate, bin/adjacent accuracy, E[δ] error, ECE/coverage, the leakage
gate, and manifest validity.  Run:  python -u scripts/run_p2p2b_rung1.py
"""

import subprocess
import time
from pathlib import Path

import torch

from lacuna.core.rng import RNGState
from lacuna.survey.example_source import SyntheticTwoColSource
from lacuna.survey.run_manifest import write_manifest
from lacuna.survey.train import TrainConfig, train_delta_prior


def _base_rate_rps(sheets, num_bins, normalize=True):
    """RPS of the 'predict the train δ-bin marginal' baseline, scored on the test bins."""
    from lacuna.survey.loss import rps_loss
    bins = torch.tensor([s.delta_bin for s in sheets], dtype=torch.long)
    marg = torch.bincount(bins, minlength=num_bins).float()
    marg = marg / marg.sum()
    logits = torch.log(marg.clamp(min=1e-9)).unsqueeze(0).repeat(len(bins), 1)
    return float(rps_loss(logits, bins, normalize=normalize).item())


def main() -> None:
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()

    rho_grid = [0.0, 0.3, 0.6]
    src = lambda: SyntheticTwoColSource(rho_grid=rho_grid)

    cfg = TrainConfig(
        delta_grid=[0.0, 0.15, 0.4, 0.75, 1.25, 1.75, 2.5], beta1_range=(0.0, 2.0),
        target_rate=0.3, max_rows=1024, max_cols=2, batch_size=16,
        train_batches_per_epoch=30, max_epochs=20, patience=5,
        val_size=140, test_size=140,
        hidden_dim=64, evidence_dim=32, n_layers=2, n_heads=4,
    )

    t0 = time.time()
    out = train_delta_prior(
        src(), src(), src(), cfg, RNGState(seed=2026),
        kind="main", run_id="p2p2b-rung1", git_commit=git,
        timestamp="2026-06-04T12:00:00Z",
    )
    wall = time.time() - t0
    m, res, lk = out["manifest"], out["results"], out["leakage"]
    m["wall_clock_seconds"] = round(wall, 1)

    # base-rate reference: 'predict the marginal' RPS on a fresh seeded test draw
    base_rps = _base_rate_rps(_make_test_sheets(src(), cfg), num_bins=7)

    print("=" * 70)
    print("P2.2b RUNG 1 — synthetic 2-col, 7 δ-bins, KNOWN target (learnability)")
    print("=" * 70)
    print(f"wall={wall:.1f}s  params={m['trainable_param_count']}  epochs_run={res['epochs_run']}  "
          f"best_val_rps={res['best_val_rps']:.4f}")
    print(f"temperature={m['temperature']:.3f}  rho_grid={rho_grid}  rows/example={cfg.max_rows}")
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
    beats_uniform = after["rps"] < res["uniform_rps"] - 0.003
    beats_baserate = after["rps"] < base_rps - 0.003
    print(f"\nLEARNS ORDERED δ?  beats_uniform={beats_uniform}  beats_base_rate={beats_baserate}  "
          f"bin_acc>chance(0.143)={after['bin_accuracy'] > 0.143}")

    print("\nLEAKAGE GATE:")
    print(f"  delta->rate pearson={lk.delta_rate_pearson:+.3f}  "
          f"rate_only_acc={lk.rate_only_acc:.3f} vs base_rate_acc={lk.base_rate_acc:.3f}")
    print(f"  LEAKAGE_PASS = {out['leakage_pass']}")

    Path("runs").mkdir(exist_ok=True)
    p = write_manifest(Path("runs/p2p2b-rung1.json"), m)
    print(f"\nmanifest written + re-validated: {p}")


def _make_test_sheets(source, cfg):
    """A fresh stratified test draw of answer sheets for the base-rate RPS reference."""
    rng = RNGState(seed=777)
    sheets = []
    for i in range(cfg.test_size):
        delta = float(cfg.delta_grid[i % len(cfg.delta_grid)])
        sheets.append(source.make_one(cfg, rng.spawn(), delta=delta, beta1=1.0).answer_sheet)
    return sheets


if __name__ == "__main__":
    main()
