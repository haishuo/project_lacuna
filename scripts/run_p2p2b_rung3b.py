"""
scripts/run_p2p2b_rung3b.py

P2.2b — RUNG 3b: high-row real-X diagnostic (separates evidence SCALE from real-X GEOMETRY).

In-distribution (like the rung-1 pass) on LARGE NARROW real survey datasets, max_rows=1024, 7
δ-bins, target-conditioned head. Reuses the existing path — NO model/tokenization changes. If real-X
learns here it was a scale problem; if it floors it is real-X geometry. See
docs/findings/feasibility-p2p2b-rung3b-note.md.  Run:  python -u scripts/run_p2p2b_rung3b.py
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
from lacuna.survey.train import TrainConfig, train_delta_prior, _forward_examples, _make_examples


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
    pool = [cat.load(n) for n in (
        "survey_cps1988", "survey_yrbss", "survey_computers",
        "survey_psid7682", "survey_chile", "survey_hmda",
    )]
    src = lambda: SurveyExampleSource(pool)  # in-distribution: shared pool for train/val/test

    cfg = TrainConfig(
        delta_grid=[0.0, 0.15, 0.4, 0.75, 1.25, 1.75, 2.5], beta1_range=(0.0, 2.0),
        target_rate=0.3, max_rows=1024, max_cols=8, batch_size=16,
        train_batches_per_epoch=40, max_epochs=20, patience=5,
        val_size=105, test_size=105,
        hidden_dim=96, evidence_dim=48, n_layers=2, n_heads=4,
        target_conditioned=True,
    )

    t0 = time.time()
    out = train_delta_prior(src(), src(), src(), cfg, RNGState(seed=2026),
                            kind="main", run_id="p2p2b-rung3b", git_commit=git,
                            timestamp="2026-06-04T12:00:00Z")
    wall = time.time() - t0
    m, res, lk = out["manifest"], out["results"], out["leakage"]
    m["wall_clock_seconds"] = round(wall, 1)
    model = out["model"]

    # Recompute test logits to get the per-example RPS SE for the 2-SE success criterion.
    test_ex = _make_examples(src(), cfg, cfg.test_size, RNGState(seed=4242), stratify=True)
    t_logits, t_labels, _, t_sheets = _forward_examples(model, test_ex, cfg)
    T = m["temperature"]
    per_ex = rps_loss(t_logits / T, t_labels, reduction="none")
    rps_mean = float(per_ex.mean().item())
    rps_se = float((per_ex.std(unbiased=True) / (len(per_ex) ** 0.5)).item())
    uniform = res["uniform_rps"]
    base = _base_rate_rps(t_sheets)
    after = res["test_after_temperature"]

    print("=" * 70)
    print("P2.2b RUNG 3b — high-row real-X, 7 δ-bins, target-conditioned (in-distribution)")
    print("=" * 70)
    print(f"wall={wall:.1f}s  params={m['trainable_param_count']}  epochs_run={res['epochs_run']}  "
          f"best_val_rps={res['best_val_rps']:.4f}  T={T:.3f}")
    print(f"pool={[r.name for r in pool]}  max_rows={cfg.max_rows}")
    print(f"REFERENCES: uniform_rps={uniform:.4f}  base_rate_rps={base:.4f}")
    print(f"TEST rps(after-T)={rps_mean:.4f} ± {rps_se:.4f} (SE)  "
          f"bin_acc={after['bin_accuracy']:.3f}  adj_acc={after['adjacent_accuracy']:.3f}  "
          f"E[delta]_mae={after['e_delta']['mae']:.3f}  ece={after['ece']:.3f}")
    print(f"coverage(50/80/90)={after['coverage']}")

    beats_uniform_2se = (uniform - rps_mean) > 2 * rps_se
    beats_base_2se = (base - rps_mean) > 2 * rps_se
    print(f"\nSUCCESS CRITERION (RPS beats refs by >=2 SE):")
    print(f"  (uniform - rps)/SE = {(uniform - rps_mean) / rps_se:+.2f}  -> beats_uniform_2SE={beats_uniform_2se}")
    print(f"  (base    - rps)/SE = {(base - rps_mean) / rps_se:+.2f}  -> beats_base_2SE={beats_base_2se}")
    print(f"  adj_acc nontrivial (>0.45)? {after['adjacent_accuracy'] > 0.45}")
    verdict = "LEARNS -> bottleneck was SCALE" if (beats_uniform_2se and beats_base_2se) \
        else "FLOOR -> bottleneck is real-X GEOMETRY"
    print(f"  VERDICT: {verdict}")

    print(f"\nLEAKAGE: pearson={lk.delta_rate_pearson:+.3f} rate_only={lk.rate_only_acc:.3f} "
          f"vs base={lk.base_rate_acc:.3f}  LEAKAGE_PASS={out['leakage_pass']}")

    Path("runs").mkdir(exist_ok=True)
    p = write_manifest(Path("runs/p2p2b-rung3b.json"), m)
    print(f"\nmanifest written + re-validated: {p}")


if __name__ == "__main__":
    main()
