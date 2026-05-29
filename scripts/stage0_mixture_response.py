#!/usr/bin/env python3
"""
Stage 0 (ADR-0006): how does the frozen DATASET-LEVEL model respond to column mixtures?

The current model emits one posterior P(MCAR/MAR/MNAR) per dataset, trained only on
single-mechanism datasets. This script characterises its behaviour when a dataset is a
*mixture* of column-level mechanisms — the question being whether the dataset-level
posterior (a) tracks the mechanism composition, (b) collapses to the most-severe mechanism
present, (c) follows the majority-by-count, or (d) just diffuses (entropy up). This is a
finding regardless of outcome and needs no new training (the control arm is frozen).

Method
------
For each held-out dataset we sweep the column composition between two mechanisms (and a
3-way point), holding the per-column missing rate ~constant to control the miss-rate
confound, generate the mixture with `lacuna.data.mixed_missingness` (clean-MAR regime),
run the frozen model, and record the calibrated posterior + entropy + decision against the
known composition. Deterministic: every trial draws from an explicit RNGState seed.

Usage
-----
    python scripts/stage0_mixture_response.py \
        --checkpoint /mnt/artifacts/project_lacuna/runs/stage0_general_baseline/checkpoints/calibrated.pt \
        --config     /mnt/artifacts/project_lacuna/runs/stage0_general_baseline/config.yaml
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lacuna.core.rng import RNGState
from lacuna.core.types import MCAR, MAR, MNAR, CLASS_NAMES
from lacuna.config import load_config
from lacuna.models import create_lacuna_model
from lacuna.training import load_model_weights
from lacuna.data.catalog import create_default_catalog
from lacuna.data.tokenization import tokenize_and_batch
from lacuna.data.semisynthetic import subsample_raw
from lacuna.data.mixed_missingness import compose_mixed_missingness, OBSERVED

ACTION_NAMES = ("GREEN", "YELLOW", "RED")

# Pairwise composition sweeps + a 3-way point. Each pairwise sweep assigns a fraction
# `frac_b` of the missing-eligible columns to class_b and the rest to class_a.
SWEEPS = (
    ("MAR_to_MNAR", MAR, MNAR),
    ("MCAR_to_MAR", MCAR, MAR),
    ("MCAR_to_MNAR", MCAR, MNAR),
)
FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)


# =========================================================================
# Model loading (mirrors scripts/evaluate.py; asserts calibration loaded)
# =========================================================================

def load_frozen_model(checkpoint: Path, config_path: Path, device: str):
    """Build the model at the run's dims and load the calibrated checkpoint.

    Fails loud if the calibration temperature did not load (guards against silently
    characterising an uncalibrated model).
    """
    config = load_config(str(config_path))
    model = create_lacuna_model(
        hidden_dim=config.model.hidden_dim,
        evidence_dim=config.model.evidence_dim,
        n_layers=config.model.n_layers,
        n_heads=config.model.n_heads,
        max_cols=config.data.max_cols,
        dropout=config.model.dropout,
        mnar_variants=["self_censoring"],
    )
    load_model_weights(model, str(checkpoint), device=device)
    model.eval()
    model.to(device)

    loaded_t = float(torch.exp(model.moe.gating.log_temperature).item())
    cal_json = checkpoint.parent / "calibrated.json"
    if cal_json.exists():
        expected_t = json.loads(cal_json.read_text())["optimal_temperature"]
        if abs(loaded_t - expected_t) > 0.05:
            raise RuntimeError(
                f"Loaded temperature {loaded_t:.3f} != calibrated.json {expected_t:.3f}. "
                f"Wrong/uncalibrated checkpoint?"
            )
    elif abs(loaded_t - 1.0) < 1e-6:
        raise RuntimeError(
            "Model temperature is 1.0 and no calibrated.json found — this looks "
            "uncalibrated. Point --checkpoint at a calibrated.pt."
        )
    return model, config, loaded_t


def load_raw_datasets(catalog, names, max_cols):
    """Load complete datasets by name, skipping any that exceed max_cols or fail."""
    out = []
    for name in names:
        try:
            raw = catalog.load(name)
        except Exception as e:  # noqa: BLE001 — report and skip, per loader convention
            print(f"  warning: could not load '{name}': {e}")
            continue
        if raw.d > max_cols:
            print(f"  skip '{name}': d={raw.d} > max_cols={max_cols}")
            continue
        out.append(raw)
    return out


# =========================================================================
# Composition construction
# =========================================================================

def build_pairwise_classes(d, class_a, class_b, frac_b, rng):
    """Assign columns: col 0 = OBSERVED clean predictor; the rest split a/b by frac_b.

    Eligible columns are shuffled so the a/b split is not positionally confounded.
    """
    classes = [OBSERVED] * d
    eligible = list(range(1, d))  # reserve col 0 as a clean MAR predictor
    m = len(eligible)
    n_b = int(round(frac_b * m))
    perm = rng.choice(m, size=m, replace=False)  # permutation of eligible positions
    shuffled = [eligible[i] for i in perm]
    for k, col in enumerate(shuffled):
        classes[col] = class_b if k < n_b else class_a
    return tuple(classes)


def build_threeway_classes(d, rng):
    """Split the missing-eligible columns ~equally across MCAR/MAR/MNAR."""
    classes = [OBSERVED] * d
    eligible = list(range(1, d))
    perm = rng.choice(len(eligible), size=len(eligible), replace=False)
    cycle = (MCAR, MAR, MNAR)
    for k, i in enumerate(perm):
        classes[eligible[i]] = cycle[k % 3]
    return tuple(classes)


def composition_counts(column_classes):
    """Counts and fractions of MCAR/MAR/MNAR among non-OBSERVED columns."""
    counts = {MCAR: 0, MAR: 0, MNAR: 0}
    for c in column_classes:
        if c in counts:
            counts[c] += 1
    total = sum(counts.values())
    fracs = {c: (counts[c] / total if total else 0.0) for c in counts}
    return counts, fracs, total


# =========================================================================
# Single trial
# =========================================================================

def run_trial(model, raw, column_classes, rng, max_rows, max_cols, device,
              target_miss_rate, mar_strength, mnar_strength):
    """Generate one mixed dataset and return the frozen model's response + truth."""
    raw_sub = subsample_raw(raw, max_rows=max_rows, rng=rng.spawn())
    res = compose_mixed_missingness(
        raw_sub, column_classes, rng.spawn(),
        target_miss_rate=target_miss_rate,
        mar_strength=mar_strength, mnar_strength=mnar_strength,
    )
    batch = tokenize_and_batch([res.observed], max_rows=max_rows, max_cols=max_cols)
    with torch.no_grad():
        out = model.forward(batch.to(device), compute_reconstruction=True, compute_decision=True)

    p = out.posterior.p_class[0].cpu().tolist()
    counts, fracs, total = composition_counts(column_classes)
    miss = [m for m, c in zip(res.per_column_miss_rate, column_classes) if c != OBSERVED]
    return {
        "source": res.source_name,
        "true_counts": {CLASS_NAMES[c]: counts[c] for c in counts},
        "true_fracs": {CLASS_NAMES[c]: round(fracs[c], 4) for c in counts},
        "n_missing_cols": total,
        "mean_missing_col_rate": round(sum(miss) / len(miss), 4) if miss else 0.0,
        "p_class": {CLASS_NAMES[i]: round(p[i], 4) for i in range(3)},
        "argmax": CLASS_NAMES[int(max(range(3), key=lambda i: p[i]))],
        "entropy": round(out.posterior.entropy_class[0].item(), 4),
        "action": ACTION_NAMES[out.decision.action_ids[0].item()],
    }


# =========================================================================
# Aggregation + reporting
# =========================================================================

def summarise(records):
    """Mean posterior + entropy per (sweep, frac_b), averaged over datasets and seeds."""
    groups = defaultdict(list)
    for r in records:
        if r["kind"] == "pairwise":
            groups[(r["sweep"], r["frac_b"])].append(r)
    summary = []
    for (sweep, frac_b), rs in sorted(groups.items()):
        n = len(rs)
        mean_p = {c: round(sum(x["p_class"][c] for x in rs) / n, 4) for c in CLASS_NAMES}
        summary.append({
            "sweep": sweep, "frac_b": frac_b, "n_trials": n,
            "mean_p_class": mean_p,
            "mean_entropy": round(sum(x["entropy"] for x in rs) / n, 4),
        })
    return summary


def print_summary(summary):
    print("\n" + "=" * 78)
    print("STAGE 0 — dataset-level posterior vs. column composition (means)")
    print("=" * 78)
    cur = None
    for row in summary:
        if row["sweep"] != cur:
            cur = row["sweep"]
            a, b = cur.split("_to_")
            print(f"\n{cur}  (frac_b = fraction of missing columns that are {b})")
            print(f"  {'frac_b':>7} | {'P(MCAR)':>8} {'P(MAR)':>8} {'P(MNAR)':>8} | "
                  f"{'entropy':>7} | n")
        p = row["mean_p_class"]
        print(f"  {row['frac_b']:>7.2f} | {p['MCAR']:>8.3f} {p['MAR']:>8.3f} "
              f"{p['MNAR']:>8.3f} | {row['mean_entropy']:>7.3f} | {row['n_trials']}")
    print("=" * 78)


# =========================================================================
# Main
# =========================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--output", type=Path, default=None,
                    help="Results JSON (default: <run_dir>/stage0_mixture_response.json)")
    ap.add_argument("--datasets", nargs="+", default=None,
                    help="Override datasets (default: config.data.val_datasets)")
    ap.add_argument("--seeds", type=int, default=5, help="Trials per (dataset, sweep, frac)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--target-miss-rate", type=float, default=0.25)
    ap.add_argument("--mar-strength", type=float, default=1.5)
    ap.add_argument("--mnar-strength", type=float, default=1.5)
    ap.add_argument("--base-seed", type=int, default=20260529)
    args = ap.parse_args()

    model, config, temp = load_frozen_model(args.checkpoint, args.config, args.device)
    print(f"Loaded control arm: {args.checkpoint}  (T={temp:.3f}, device={args.device})")

    names = args.datasets or config.data.val_datasets
    max_rows, max_cols = config.data.max_rows, config.data.max_cols
    raws = load_raw_datasets(create_default_catalog(), names, max_cols)
    print(f"Datasets: {[r.name for r in raws]}")

    records = []
    for di, raw in enumerate(raws):
        # Pairwise sweeps
        for si, (sweep, ca, cb) in enumerate(SWEEPS):
            for fi, frac_b in enumerate(FRACTIONS):
                for seed in range(args.seeds):
                    rng = RNGState(seed=args.base_seed + di*100000 + si*10000 + fi*1000 + seed)
                    classes = build_pairwise_classes(raw.d, ca, cb, frac_b, rng.spawn())
                    rec = run_trial(model, raw, classes, rng, max_rows, max_cols,
                                    args.device, args.target_miss_rate,
                                    args.mar_strength, args.mnar_strength)
                    rec.update(kind="pairwise", sweep=sweep, frac_b=frac_b, seed=seed)
                    records.append(rec)
        # 3-way equal point
        for seed in range(args.seeds):
            rng = RNGState(seed=args.base_seed + di*100000 + 99999 + seed)
            classes = build_threeway_classes(raw.d, rng.spawn())
            rec = run_trial(model, raw, classes, rng, max_rows, max_cols,
                            args.device, args.target_miss_rate,
                            args.mar_strength, args.mnar_strength)
            rec.update(kind="threeway", sweep="3way_equal", frac_b=None, seed=seed)
            records.append(rec)

    summary = summarise(records)
    print_summary(summary)

    out_path = args.output or (args.checkpoint.parent.parent / "stage0_mixture_response.json")
    payload = {
        "control_checkpoint": str(args.checkpoint),
        "temperature": temp,
        "datasets": [r.name for r in raws],
        "n_trials": len(records),
        "config": {
            "target_miss_rate": args.target_miss_rate,
            "mar_strength": args.mar_strength,
            "mnar_strength": args.mnar_strength,
            "seeds_per_cell": args.seeds,
            "fractions": list(FRACTIONS),
        },
        "summary": summary,
        "records": records,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {len(records)} trial records to {out_path}")


if __name__ == "__main__":
    main()
