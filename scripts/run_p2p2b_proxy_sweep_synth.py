"""
scripts/run_p2p2b_proxy_sweep_synth.py

P2.2b proxy-strength sweep — PART A: synthetic ρ-sweep (mechanism test, no discreteness confound).

For each ρ, train a fresh in-distribution δ-prior on synthetic 2-col ConditionalGaussian(ρ) X +
own-value self-censoring (the rung-1 setup, which is known to learn at low ρ). Report the full
metric block per ρ in ONE table and check monotonicity. Prediction if proxy absorption is real:
as ρ→1, RPS worsens AND adjacent accuracy falls AND entropy rises AND P(δ=0) rises — discrimination
and uncertainty moving together. Run: python -u scripts/run_p2p2b_proxy_sweep_synth.py
"""

import json
import subprocess
import time
from pathlib import Path

import torch

from lacuna.core.rng import RNGState
from lacuna.survey.example_source import SyntheticTwoColSource
from lacuna.survey.train import TrainConfig, train_delta_prior

RHO_GRID = [0.0, 0.3, 0.6, 0.8, 0.9, 0.95, 0.99]


def _spearman(a, b):
    """Spearman rank correlation (Pearson on ranks), no scipy."""
    ra = torch.argsort(torch.argsort(torch.tensor(a, dtype=torch.float64))).double()
    rb = torch.argsort(torch.argsort(torch.tensor(b, dtype=torch.float64))).double()
    ra -= ra.mean(); rb -= rb.mean()
    denom = (ra.norm() * rb.norm()).item()
    return 0.0 if denom == 0 else float((ra * rb).sum().item() / denom)


def main() -> None:
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    rows = []
    for i, rho in enumerate(RHO_GRID):
        src = lambda: SyntheticTwoColSource(rho_grid=[rho])
        cfg = TrainConfig(
            delta_grid=[0.0, 0.15, 0.4, 0.75, 1.25, 1.75, 2.5], beta1_range=(0.0, 2.0),
            target_rate=0.3, max_rows=1024, max_cols=2, batch_size=16,
            train_batches_per_epoch=25, max_epochs=12, patience=4,
            val_size=140, test_size=140,
            hidden_dim=64, evidence_dim=32, n_layers=2, n_heads=4,
            target_conditioned=False,  # global head; on d=2 the target is unambiguous (matches rung 1)
        )
        t0 = time.time()
        out = train_delta_prior(src(), src(), src(), cfg, RNGState(seed=2026 + i),
                                kind="ablation", run_id=f"proxyA-rho{rho}", git_commit=git,
                                timestamp="2026-06-04T12:00:00Z")
        b = out["results"]["test_after_temperature"]
        uni = out["results"]["uniform_rps"]
        rows.append({
            "rho": rho, "rps": b["rps"], "rps_adv": uni - b["rps"],
            "bin_acc": b["bin_accuracy"], "adj_acc": b["adjacent_accuracy"],
            "entropy_bits": b["entropy_bits"], "p_delta0": b["p_delta0"],
            "ece": b["ece"], "cov50": b["coverage"]["50"],
            "leakage_pass": out["leakage_pass"], "wall": round(time.time() - t0, 1),
        })
        print(f"  done rho={rho} ({rows[-1]['wall']}s)")

    print("\n" + "=" * 92)
    print("P2.2b PART A — synthetic ρ-sweep (uniform_rps reference = 0.1905, max entropy = log2(7)=2.807)")
    print("=" * 92)
    hdr = f"{'rho':>5} | {'rps':>7} | {'rps_adv':>8} | {'bin_acc':>7} | {'adj_acc':>7} | {'entropy':>7} | {'P(d=0)':>6} | {'ece':>6} | {'cov50':>6}"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['rho']:>5.2f} | {r['rps']:>7.4f} | {r['rps_adv']:>+8.4f} | {r['bin_acc']:>7.3f} | "
              f"{r['adj_acc']:>7.3f} | {r['entropy_bits']:>7.3f} | {r['p_delta0']:>6.3f} | "
              f"{r['ece']:>6.3f} | {r['cov50']:>6.3f}")

    rhos = [r["rho"] for r in rows]
    sp_ent = _spearman(rhos, [r["entropy_bits"] for r in rows])
    sp_adv = _spearman(rhos, [r["rps_adv"] for r in rows])
    sp_adj = _spearman(rhos, [r["adj_acc"] for r in rows])
    sp_p0 = _spearman(rhos, [r["p_delta0"] for r in rows])
    print("\nMONOTONICITY (Spearman vs ρ):")
    print(f"  entropy↑  : {sp_ent:+.3f} (expect > 0)")
    print(f"  rps_adv↓  : {sp_adv:+.3f} (expect < 0)")
    print(f"  adj_acc↓  : {sp_adj:+.3f} (expect < 0)")
    print(f"  P(δ=0)↑   : {sp_p0:+.3f} (expect > 0)")
    part_a_pass = sp_ent > 0 and sp_adv < 0
    print(f"\nPART A VERDICT: {'SUPPORTS proxy story (discrimination down with entropy up)' if part_a_pass else 'DOES NOT support — proxy mechanism weakened'}")

    Path("runs").mkdir(exist_ok=True)
    Path("runs/p2p2b-proxyA-synth.json").write_text(json.dumps(
        {"git": git, "rho_grid": RHO_GRID, "rows": rows,
         "spearman": {"entropy": sp_ent, "rps_adv": sp_adv, "adj_acc": sp_adj, "p_delta0": sp_p0}},
        indent=2))
    print("\nsaved: runs/p2p2b-proxyA-synth.json")


if __name__ == "__main__":
    main()
