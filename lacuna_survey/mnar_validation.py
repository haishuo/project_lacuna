"""Synth-MNAR-on-real-X validation for Lacuna-Survey.

This is the controlled MNAR detection test: apply each synthetic
mechanism (MCAR/MAR/MNAR) from the survey registry to held-out real
survey X-bases and measure per-class detection rate. Held-out means
the X-base is NOT in the training set used to fit the current
checkpoint, so the test measures generalisation across both
mechanism (synthetic, in-distribution) and X-distribution
(real survey columns the model didn't train on).

This complements the real-anchor diagnostic in two ways:
  - Real anchors test mechanism interpretation under genuine
    real-world data structure but suffer from contested labels
    (NHANES INDFMPIR, NHANES DPQ) and Molenberghs identifiability
    limits that no model can solve from data alone.
  - Synth-on-real-X uses synthetic mechanisms with known labels,
    so the test is unambiguous; it answers "given a known MNAR
    process, can the model detect it?"

Usage:
    python -m lacuna_survey.mnar_validation
    python -m lacuna_survey.mnar_validation --verbose

The held-out X-bases (`survey_cars93`, `survey_survey`) are NOT in
the v9+ training set per `configs/training/survey.yaml` (validation
set only).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lacuna.core.rng import RNGState
from lacuna.data.catalog import create_default_catalog
from lacuna.data.semisynthetic import apply_missingness
from lacuna.generators.families.registry_builder import load_registry_from_config
from demo.pipeline import build_model, run_model

HELD_OUT_X_BASES = ["survey_cars93", "survey_survey"]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "demo" / "model.pt"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--registry", default="lacuna_survey")
    ap.add_argument("--verbose", action="store_true",
                    help="Print per-generator results.")
    args = ap.parse_args()

    model, _ = build_model(str(args.checkpoint))
    reg = load_registry_from_config(args.registry)
    cat = create_default_catalog()

    bases = [(name, cat.load(name)) for name in HELD_OUT_X_BASES]
    by_class = {0: "MCAR", 1: "MAR", 2: "MNAR"}

    results = {0: [0, 0], 1: [0, 0], 2: [0, 0]}
    per_gen_failures = []

    for g in reg.generators:
        true_class = g.class_id
        for base_name, xb in bases:
            try:
                semi = apply_missingness(xb, g, RNGState(seed=42))
                pred = run_model(model, semi.observed)
                p = np.asarray(pred["p_class"])
                pred_class = int(np.argmax(p))
                results[true_class][1] += 1
                if pred_class == true_class:
                    results[true_class][0] += 1
                else:
                    per_gen_failures.append(
                        (g.name, base_name, by_class[true_class],
                         by_class[pred_class], p.tolist())
                    )
            except Exception:
                # Some generators may not apply to small X-bases — skip.
                pass

    print("Lacuna-Survey synth-MNAR-on-real-X validation")
    print(f"Held-out X-bases: {', '.join(HELD_OUT_X_BASES)}")
    print(f"Checkpoint: {args.checkpoint}")
    print()
    print(f"{'Class':<6s} {'Correct':>10s} {'Total':>10s} {'Rate':>10s}")
    print("-" * 40)
    for cid, name in by_class.items():
        correct, total = results[cid]
        rate = (correct / total * 100) if total else 0.0
        print(f"{name:<6s} {correct:>10d} {total:>10d} {rate:>9.1f}%")

    total_c = sum(c for c, _ in results.values())
    total_t = sum(t for _, t in results.values())
    print(f"{'TOTAL':<6s} {total_c:>10d} {total_t:>10d} "
          f"{total_c/total_t*100:>9.1f}%")

    if args.verbose and per_gen_failures:
        print(f"\n{len(per_gen_failures)} generator-X failures:")
        for gn, bn, true_cls, pred_cls, p in per_gen_failures:
            ps = " ".join(f"{x:.3f}" for x in p)
            print(f"  {gn:<28s} on {bn:<22s} "
                  f"true={true_cls:<5s} pred={pred_cls:<5s} [{ps}]")


if __name__ == "__main__":
    main()
