"""
scripts/run_p2p2b_proxy_sweep_realx.py

P2.2b proxy-strength sweep — PART B: real-X R² stratification (the gate).

SAFEGUARD FIRST: characterize + save the empirical R²(target|observed) distribution over all
candidate (dataset, target) columns (min/quartiles/median/max) and the chosen stratum boundaries,
BEFORE training — so "low R²" and "high R²" are verified to be genuinely different.

Then train ONE target-conditioned δ-prior with R²-STRATIFIED sampling (so every stratum is
learnable — a low-R² stratum at the floor must mean unidentifiable, not undertrained), and report
the full metric block PER R² stratum. Gate: the same model must be SHARP at low R² and COLLAPSE
(high entropy) at high R². Run: python -u scripts/run_p2p2b_proxy_sweep_realx.py
"""

import json
import subprocess
import time
from pathlib import Path

import torch

from lacuna.core.rng import RNGState
from lacuna.data.catalog import create_default_catalog
from lacuna.survey.example_source import StratifiedRealXSource
from lacuna.survey.proxy_score import r2_distribution, target_r2_table
from lacuna.survey.train import (
    TrainConfig, _forward_examples, _make_examples, train_delta_prior,
)
from lacuna.survey import metrics as M

# Narrow (d<=7) large datasets so high-row training is tractable at max_cols=8 (matches rung 3b).
# The R2 safeguard below reports whether this pool spans a genuine low->high R2 range.
POOL = ["survey_cps1988", "survey_yrbss", "survey_computers",
        "survey_psid7682", "survey_chile", "survey_hmda"]
N_STRATA = 5


def _make_strata(table, n_strata):
    """Quantile strata over candidate-target R²; returns (strata_pairs, boundaries, per_stratum_range)."""
    vals = sorted(row["r2"] for row in table)
    qs = [vals[min(len(vals) - 1, int(round(q * (len(vals) - 1))))]
          for q in [i / n_strata for i in range(n_strata + 1)]]
    boundaries = qs
    strata, ranges = [], []
    for k in range(n_strata):
        lo, hi = boundaries[k], boundaries[k + 1]
        if k == n_strata - 1:
            members = [r for r in table if lo <= r["r2"] <= hi]
        else:
            members = [r for r in table if lo <= r["r2"] < hi]
        strata.append([(r["dataset"], r["target_idx"]) for r in members])
        rr = [r["r2"] for r in members]
        ranges.append({"lo": lo, "hi": hi, "n_pairs": len(members),
                       "r2_min": min(rr) if rr else None, "r2_max": max(rr) if rr else None,
                       "r2_mean": (sum(rr) / len(rr)) if rr else None})
    return strata, boundaries, ranges


def main() -> None:
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    cat = create_default_catalog()
    datasets = {n: cat.load(n) for n in POOL}
    dlist = list(datasets.values())

    # ---- SAFEGUARD: empirical R² distribution + strata BEFORE training ----
    table = target_r2_table(dlist)
    dist = r2_distribution(table)
    strata, boundaries, ranges = _make_strata(table, N_STRATA)

    print("=" * 78)
    print("P2.2b PART B — real-X candidate-target R²(target|observed) distribution")
    print("=" * 78)
    print(f"  n candidate targets = {dist['n_targets']} over {len(POOL)} datasets")
    print(f"  min={dist['min']:.3f}  q25={dist['q25']:.3f}  median={dist['median']:.3f}  "
          f"q75={dist['q75']:.3f}  max={dist['max']:.3f}")
    print(f"  stratum boundaries (quintiles): {[round(b, 3) for b in boundaries]}")
    for k, rg in enumerate(ranges):
        print(f"    stratum {k}: R2 in [{rg['lo']:.3f}, {rg['hi']:.3f}]  n_pairs={rg['n_pairs']}  "
              f"mean={rg['r2_mean'] if rg['r2_mean'] is None else round(rg['r2_mean'], 3)}")
    spread = dist["max"] - dist["min"]
    print(f"  R2 spread (max-min) = {spread:.3f}  "
          f"{'(strata genuinely differ)' if spread > 0.2 else '(WARNING: narrow R2 range — strata may not differ)'}")

    Path("runs").mkdir(exist_ok=True)
    Path("runs/p2p2b-proxyB-r2dist.json").write_text(json.dumps(
        {"git": git, "distribution": dist, "boundaries": boundaries, "strata_ranges": ranges,
         "table": table}, indent=2))
    print("  saved: runs/p2p2b-proxyB-r2dist.json")

    # ---- train ONE model with R2-stratified sampling ----
    src = lambda: StratifiedRealXSource(datasets, strata)
    cfg = TrainConfig(
        delta_grid=[0.0, 0.15, 0.4, 0.75, 1.25, 1.75, 2.5], beta1_range=(0.0, 2.0),
        target_rate=0.3, max_rows=1024, max_cols=8, batch_size=16,
        train_batches_per_epoch=40, max_epochs=18, patience=5,
        val_size=105, test_size=105,
        hidden_dim=96, evidence_dim=48, n_layers=2, n_heads=4,
        target_conditioned=True,
    )
    t0 = time.time()
    out = train_delta_prior(src(), src(), src(), cfg, RNGState(seed=2026),
                            kind="main", run_id="p2p2b-proxyB", git_commit=git,
                            timestamp="2026-06-04T12:00:00Z")
    out["manifest"]["wall_clock_seconds"] = round(time.time() - t0, 1)
    model = out["model"]
    print(f"\ntrained: params={out['manifest']['trainable_param_count']} "
          f"epochs={out['results']['epochs_run']} wall={out['manifest']['wall_clock_seconds']}s "
          f"leakage_pass={out['leakage_pass']}")

    # ---- per-stratum evaluation ----
    print("\n" + "=" * 100)
    print("PER-R2-STRATUM metrics (uniform_rps=0.1905, max entropy=2.807 bits)")
    print("=" * 100)
    hdr = (f"{'stratum':>7} | {'R2_mean':>7} | {'rps':>7} | {'(uni-rps)/SE':>12} | {'bin':>5} | "
           f"{'adj':>5} | {'entropy':>7} | {'P(d=0)':>6} | {'ece':>6}")
    print(hdr); print("-" * len(hdr))
    per_stratum = []
    for k, stratum in enumerate(strata):
        if len(stratum) == 0:
            continue
        ssrc = StratifiedRealXSource(datasets, [stratum])
        ex = _make_examples(ssrc, cfg, 120, RNGState(seed=900 + k), stratify=True)
        logits, labels, deltas, _ = _forward_examples(model, ex, cfg)
        T = out["manifest"]["temperature"]
        from lacuna.survey.loss import rps_loss
        per = rps_loss(logits / T, labels, reduction="none")
        rps_m = float(per.mean()); se = float(per.std(unbiased=True) / (len(per) ** 0.5))
        probs = torch.softmax(logits / T, dim=-1)
        row = {
            "stratum": k, "r2_mean": ranges[k]["r2_mean"], "rps": rps_m, "rps_se": se,
            "uni_minus_rps_over_se": (0.190476 - rps_m) / se if se > 0 else 0.0,
            "bin_acc": M.bin_accuracy(probs, labels), "adj_acc": M.adjacent_accuracy(probs, labels),
            "entropy_bits": M.mean_predictive_entropy(probs), "p_delta0": M.p_delta_zero(probs),
            "ece": M.ece(probs, labels)["ece"],
        }
        per_stratum.append(row)
        print(f"{k:>7} | {row['r2_mean']:>7.3f} | {rps_m:>7.4f} | {row['uni_minus_rps_over_se']:>+12.2f} | "
              f"{row['bin_acc']:>5.3f} | {row['adj_acc']:>5.3f} | {row['entropy_bits']:>7.3f} | "
              f"{row['p_delta0']:>6.3f} | {row['ece']:>6.3f}")

    # ---- gate ----
    sharp_low = any(r["uni_minus_rps_over_se"] > 2 and r["adj_acc"] > 0.45
                    for r in per_stratum if r["r2_mean"] is not None and r["r2_mean"] < 0.4)
    rmeans = [r["r2_mean"] for r in per_stratum if r["r2_mean"] is not None]
    ents = [r["entropy_bits"] for r in per_stratum if r["r2_mean"] is not None]
    ent_rises = (len(rmeans) >= 3) and (ents[-1] > ents[0])
    print("\nGATE:")
    print(f"  sharp at low R2 (some stratum R2<0.4 beats uniform >2SE & adj>0.45)? {sharp_low}")
    print(f"  entropy rises with R2 (top stratum entropy > bottom)? {ent_rises}")
    verdict = ("PASS -> proxy strength is a validated governance variable (-> P2.3)"
               if (sharp_low and ent_rises)
               else "FAIL -> flat/overconfident; reject proxy strength as primary; revisit geometry/discreteness")
    print(f"  VERDICT: {verdict}")

    from lacuna.survey.run_manifest import write_manifest
    out["manifest"]["proxy_sweep"] = {"r2_distribution": dist, "boundaries": boundaries,
                                      "per_stratum": per_stratum, "sharp_low": sharp_low,
                                      "entropy_rises": ent_rises}
    p = write_manifest(Path("runs/p2p2b-proxyB.json"), out["manifest"])
    print(f"\nmanifest written + re-validated: {p}")


if __name__ == "__main__":
    main()
