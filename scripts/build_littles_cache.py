#!/usr/bin/env python3
"""
Build the Little's MCAR cache for a given (dataset pool, generator registry).

Runs `pystatistics.mvnmle.little_mcar_test` once per (dataset, generator)
pair at a large sample size, stores the result in a JSON cache file.
Training pulls from this cache rather than recomputing per-batch — the
real Little's test is far too slow for the hot path.

Usage:
    # Build cache for the full semi-synthetic pipeline
    python scripts/build_littles_cache.py \\
        --config configs/training/ablation.yaml \\
        --output /mnt/artifacts/project_lacuna/cache/littles_mcar_v1.json

    # Smaller cache for iris-only smoke tests
    python scripts/build_littles_cache.py \\
        --datasets iris \\
        --registry lacuna_minimal_6 \\
        --output /tmp/littles_iris.json

Outputs a JSON file with a `version`, `generator_registry`,
`sample_rows_per_evaluation`, `seed_base`, and an array of `entries`,
each holding `{dataset, generator_id, generator_name, statistic, p_value,
df, n_used, rejected}`. Human-readable, diffable.
"""

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lacuna.config.load import load_config
from lacuna.data.catalog import create_default_catalog
from lacuna.data.littles_cache import build_cache, save_cache
from lacuna.generators.families.registry_builder import load_registry_from_config


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--config", type=str,
        help="Training YAML. Uses its train_datasets + val_datasets + generator.config_name.",
    )
    src.add_argument(
        "--datasets", type=str, nargs="+",
        help="Explicit list of catalog dataset names (used with --registry).",
    )
    parser.add_argument(
        "--registry", type=str, default=None,
        help="Generator registry name (required iff --datasets is used).",
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Output JSON path.",
    )
    parser.add_argument(
        "--sample-rows", type=int, default=1000,
        help="Rows to subsample per evaluation (clamped to dataset size).",
    )
    parser.add_argument(
        "--max-cols", type=int, default=None,
        help="Filter out datasets exceeding this column count. "
             "Defaults to config.data.max_cols or 48 when --datasets is used.",
    )
    parser.add_argument(
        "--seed-base", type=int, default=20260418,
        help="Base seed for deterministic computation.",
    )
    parser.add_argument(
        "--backend", choices=("auto", "cpu", "gpu"), default="auto",
        help="pystatistics compute backend for Little's test. Default 'auto' "
             "picks GPU when available (CUDA/MPS), otherwise CPU.",
    )
    args = parser.parse_args()

    if args.datasets and not args.registry:
        parser.error("--datasets requires --registry")
    return args


def main():
    args = parse_args()

    # Resolve datasets + registry.
    if args.config:
        cfg = load_config(args.config)
        dataset_names = list(cfg.data.train_datasets or []) + list(cfg.data.val_datasets or [])
        # De-dup but preserve order
        seen = set()
        dataset_names = [n for n in dataset_names if not (n in seen or seen.add(n))]
        registry_name = cfg.generator.config_path or cfg.generator.config_name
        max_cols = args.max_cols if args.max_cols is not None else cfg.data.max_cols
    else:
        dataset_names = list(args.datasets)
        registry_name = args.registry
        max_cols = args.max_cols if args.max_cols is not None else 48

    print(f"Building Little's cache:")
    print(f"  Output:      {args.output}")
    print(f"  Registry:    {registry_name}")
    print(f"  Datasets:    {len(dataset_names)} ({dataset_names[:5]}{'...' if len(dataset_names) > 5 else ''})")
    print(f"  max_cols:    {max_cols}")
    print(f"  Sample rows: {args.sample_rows}")
    print(f"  Seed base:   {args.seed_base}")
    print(f"  Backend:     {args.backend}")
    print()

    registry = load_registry_from_config(registry_name)
    generators = list(registry.generators)

    catalog = create_default_catalog()
    raw_datasets = []
    skipped = []
    for name in dataset_names:
        raw = catalog.load(name)
        if raw.d <= max_cols:
            raw_datasets.append(raw)
        else:
            skipped.append((name, raw.d))
    if skipped:
        print(f"Skipping datasets exceeding max_cols={max_cols}: {skipped}")
        print()

    total_pairs = len(raw_datasets) * len(generators)
    print(f"Total (dataset × generator) pairs to compute: {total_pairs}")
    print()

    start = time.time()
    n_done = 0

    def _log(entry):
        nonlocal n_done
        n_done += 1
        if n_done % 25 == 0 or n_done == total_pairs:
            elapsed = time.time() - start
            rate = n_done / elapsed
            remaining = (total_pairs - n_done) / rate if rate > 0 else float("inf")
            print(
                f"  [{n_done:>5}/{total_pairs}] {entry.dataset:<20} × "
                f"{entry.generator_name[:28]:<28}  "
                f"MLE: stat={entry.mle_statistic:>8.2f} p={entry.mle_p_value:.3f}  "
                f"MoM: stat={entry.mom_statistic:>8.2f} p={entry.mom_p_value:.3f}  "
                f"rate={rate:.1f}/s  eta={remaining/60:.1f} min"
            )

    cache = build_cache(
        raw_datasets=raw_datasets,
        generators=generators,
        generator_registry_name=registry_name,
        sample_rows=args.sample_rows,
        seed_base=args.seed_base,
        backend=args.backend,
        on_entry=_log,
    )

    elapsed = time.time() - start
    print()
    print(f"Done in {elapsed:.1f}s ({elapsed/60:.1f} min).")

    save_cache(cache, args.output)
    print(f"Wrote {args.output}")

    # Summary statistics for sanity check.
    mle_rejected = sum(1 for e in cache.entries.values() if e.mle_rejected)
    mom_rejected = sum(1 for e in cache.entries.values() if e.mom_rejected)
    n = len(cache.entries)
    print(f"  Total entries: {n}")
    print(f"  MLE rejected (p < 0.05): {mle_rejected} ({100 * mle_rejected / n:.1f}%)")
    print(f"  MoM rejected (p < 0.05): {mom_rejected} ({100 * mom_rejected / n:.1f}%)")


if __name__ == "__main__":
    main()
