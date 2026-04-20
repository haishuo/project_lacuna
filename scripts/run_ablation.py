#!/usr/bin/env python3
"""
Run the missingness-feature ablation sweep on Lacuna.

For each (spec, seed) combination, train a model, evaluate it, and
append one row of metrics to a tidy CSV. Results feed directly into
`lacuna.analysis.ablation_stats` for paired comparisons.

Usage:
    # Minimal smoke test on iris (CPU, fast)
    python scripts/run_ablation.py --minimal --seeds 1 2 --csv ablation_smoke.csv

    # Full sweep — committee-grade, on Forge
    python scripts/run_ablation.py --config configs/training/ablation.yaml \\
        --seeds 1 2 3 4 5 --csv ablation.csv

    # Only a subset of specs
    python scripts/run_ablation.py --minimal --seeds 1 \\
        --specs baseline disable_littles --csv partial.csv
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lacuna.config.schema import LacunaConfig
from lacuna.config.load import load_config
from lacuna.data.littles_cache import load_cache
from lacuna.analysis.ablation_harness import (
    DEFAULT_SPECS,
    run_ablation_sweep,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--config", type=str, help="Path to training YAML config")
    src.add_argument(
        "--minimal",
        action="store_true",
        help="Use LacunaConfig.minimal() — CPU, tiny, for smoke tests.",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", required=True,
        help="Seeds to run. Each spec is evaluated at every seed (paired design).",
    )
    parser.add_argument(
        "--specs", type=str, nargs="+", default=None,
        help=(
            "Subset of spec names to run. Default: all DEFAULT_SPECS. "
            f"Available: {[s.name for s in DEFAULT_SPECS]}"
        ),
    )
    parser.add_argument("--csv", type=Path, required=True, help="Output CSV path.")
    parser.add_argument("--device", type=str, default=None, help="Override device.")
    parser.add_argument(
        "--littles-cache", type=Path, default=None,
        help="Path to the Little's MCAR JSON cache "
             "(see scripts/build_littles_cache.py). Required whenever any "
             "spec has include_littles_approx=True, which is the baseline.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.minimal:
        base = LacunaConfig.minimal()
    else:
        base = load_config(args.config)

    if args.device is not None:
        base.device = args.device

    # Filter specs if requested.
    if args.specs is not None:
        known = {s.name: s for s in DEFAULT_SPECS}
        unknown = [n for n in args.specs if n not in known]
        if unknown:
            raise SystemExit(
                f"Unknown spec name(s): {unknown}. "
                f"Available: {list(known.keys())}"
            )
        specs = [known[n] for n in args.specs]
    else:
        specs = DEFAULT_SPECS

    total = len(specs) * len(args.seeds)
    print(f"Running ablation sweep: {len(specs)} specs × {len(args.seeds)} seeds "
          f"= {total} runs")
    print(f"  specs:   {[s.name for s in specs]}")
    print(f"  seeds:   {list(args.seeds)}")
    print(f"  device:  {base.device}")
    print(f"  epochs:  {base.training.epochs}, "
          f"batches/epoch: {base.training.batches_per_epoch}")
    print(f"  csv:     {args.csv}")
    print()

    n_done = 0

    def _log(result):
        nonlocal n_done
        n_done += 1
        print(
            f"[{n_done:>3}/{total}] {result.spec_name:<22}  seed={result.seed}  "
            f"acc={result.accuracy:.4f}  ece={result.ece:.4f}  "
            f"t={result.train_time_s:.1f}s"
        )

    cache = load_cache(args.littles_cache) if args.littles_cache else None
    if cache is not None:
        print(f"  Little's cache: {args.littles_cache} ({len(cache.entries)} entries)")
        print()

    run_ablation_sweep(
        base_config=base,
        seeds=args.seeds,
        specs=specs,
        csv_path=args.csv,
        on_result=_log,
        littles_cache=cache,
    )

    print()
    print(f"Done. Wrote {args.csv}")


if __name__ == "__main__":
    main()
