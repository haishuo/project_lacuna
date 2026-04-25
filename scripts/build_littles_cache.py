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

Outputs a schema v3 JSON cache with MLE, MoM, propensity, HSIC, and
MissMech results per (dataset, generator) pair. See
`lacuna/data/littles_cache.py` for the full entry schema.
"""

# Set BLAS/OpenMP thread counts BEFORE importing numpy/torch/sklearn.
# Every worker process spawned under --jobs > 1 inherits this environment,
# so libgomp / MKL / OpenBLAS / torch all see single-threaded at their
# library-init time. Setting these from `_init_worker` in the worker
# would be too late: ProcessPoolExecutor's initializer reference pulls
# in `lacuna.data.littles_cache` during resolution, which transitively
# imports the scientific stack — BLAS pools are created at that point
# using whatever the env vars said *then*. Setting them here in the
# parent before any import closes that timing gap.
#
# N parallel workers each using all-N-cores internally oversubscribes
# catastrophically: we measured 6 workers × libgomp pools on a 12-core
# box as a hang (load avg 50+, zero pair completions in 5 min). With
# per-worker threads pinned to 1, each worker is genuinely
# single-threaded and the ``jobs`` parameter controls total CPU use.
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"

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
        help="pystatistics compute backend for Little's MLE / MoM. "
             "Default 'auto' picks GPU when available (CUDA/MPS). "
             "Parallel workers get their own CUDA contexts via the "
             "'spawn' start method; the GPU scheduler serialises the "
             "MLE calls naturally.",
    )
    parser.add_argument(
        "--jobs", type=int, default=1,
        help="Number of parallel worker processes for the "
             "(dataset × generator) sweep. Default 1 (sequential). "
             "Recommended: ~`nproc // 2` on multi-core boxes, leaving "
             "headroom for sklearn HGB's internal threading. With "
             "--backend auto, each worker gets its own CUDA context "
             "(one-time ~3-5 s init) and shares the GPU scheduler "
             "naturally.",
    )
    # Schema v3 nonparametric-test tuning. Propensity uses its
    # analytical (Mann-Whitney-U) null for cache builds — zero
    # permutations — so there's no propensity-permutations flag.
    parser.add_argument(
        "--hsic-permutations", type=int, default=199,
        help="Permutation count for hsic_mcar_test. Default 199.",
    )
    parser.add_argument(
        "--missmech-permutations", type=int, default=199,
        help="Permutation count for missmech_mcar_test. Default 199.",
    )
    parser.add_argument(
        "--propensity-model", choices=("hgb", "rf", "gbm"), default="hgb",
        help="Classifier for propensity_mcar_test. Default 'hgb' "
             "(HistGradientBoostingClassifier — 3–5x faster than 'rf').",
    )
    parser.add_argument(
        "--propensity-cv-folds", type=int, default=3,
        help="StratifiedKFold count for propensity_mcar_test. Default 3.",
    )
    parser.add_argument(
        "--propensity-n-estimators", type=int, default=50,
        help="n_estimators for the RF/GBM in propensity_mcar_test. "
             "Default 50.",
    )
    parser.add_argument(
        "--propensity-n-jobs", type=int, default=-1,
        help="n_jobs for RandomForestClassifier in propensity_mcar_test. "
             "Default -1 (all cores). Ignored for model='gbm'.",
    )
    parser.add_argument(
        "--missmech-n-neighbors", type=int, default=5,
        help="k for missmech_mcar_test's KNNImputer. Default 5.",
    )
    parser.add_argument(
        "--missmech-min-pattern-size", type=int, default=6,
        help="Minimum rows per missingness pattern for MissMech to use it. "
             "Default 6.",
    )
    parser.add_argument(
        "--nonparametric-seed", type=int, default=0,
        help="Seed for nonparametric_mcar permutation draws. Default 0.",
    )
    parser.add_argument(
        "--log-every", type=int, default=10,
        help="Print a per-pair progress line every N completions. "
             "Default 10 (trade-off: smaller N = more terminal output, "
             "more confidence the build is alive; larger N = cleaner "
             "logs). A 30-second heartbeat runs independently so N=1 "
             "isn't needed for liveness.",
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
    print(f"  Jobs:        {args.jobs}")
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
        if n_done % args.log_every == 0 or n_done == total_pairs:
            elapsed = time.time() - start
            rate = n_done / elapsed
            remaining = (total_pairs - n_done) / rate if rate > 0 else float("inf")
            print(
                f"  [{n_done:>5}/{total_pairs}] {entry.dataset:<20} × "
                f"{entry.generator_name[:28]:<28}  "
                f"MLE p={entry.mle_p_value:.2f}  MoM p={entry.mom_p_value:.2f}  "
                f"prop p={entry.propensity_p_value:.2f}  "
                f"hsic p={entry.hsic_p_value:.2f}  "
                f"mm p={entry.missmech_p_value:.2f}  "
                f"rate={rate:.2f}/s eta={remaining/60:.1f} min",
                flush=True,
            )

    cache = build_cache(
        raw_datasets=raw_datasets,
        generators=generators,
        generator_registry_name=registry_name,
        sample_rows=args.sample_rows,
        seed_base=args.seed_base,
        backend=args.backend,
        jobs=args.jobs,
        on_entry=_log,
        hsic_permutations=args.hsic_permutations,
        missmech_permutations=args.missmech_permutations,
        propensity_model=args.propensity_model,
        propensity_cv_folds=args.propensity_cv_folds,
        propensity_n_estimators=args.propensity_n_estimators,
        propensity_n_jobs=args.propensity_n_jobs,
        missmech_n_neighbors=args.missmech_n_neighbors,
        missmech_min_pattern_size=args.missmech_min_pattern_size,
        nonparametric_seed=args.nonparametric_seed,
    )

    elapsed = time.time() - start
    print()
    print(f"Done in {elapsed:.1f}s ({elapsed/60:.1f} min).")

    save_cache(cache, args.output)
    print(f"Wrote {args.output}")

    # Summary statistics for sanity check.
    n = len(cache.entries)
    mle_rej = sum(1 for e in cache.entries.values() if e.mle_rejected)
    mom_rej = sum(1 for e in cache.entries.values() if e.mom_rejected)
    prop_rej = sum(1 for e in cache.entries.values() if e.propensity_p_value < 0.05)
    hsic_rej = sum(1 for e in cache.entries.values() if e.hsic_p_value < 0.05)
    mm_rej = sum(1 for e in cache.entries.values() if e.missmech_p_value < 0.05)
    print(f"  Total entries: {n}")
    print(f"  MLE        rejected (p < 0.05): {mle_rej}  ({100*mle_rej/n:.1f}%)")
    print(f"  MoM        rejected (p < 0.05): {mom_rej}  ({100*mom_rej/n:.1f}%)")
    print(f"  Propensity rejected (p < 0.05): {prop_rej} ({100*prop_rej/n:.1f}%)")
    print(f"  HSIC       rejected (p < 0.05): {hsic_rej} ({100*hsic_rej/n:.1f}%)")
    print(f"  MissMech   rejected (p < 0.05): {mm_rej}   ({100*mm_rej/n:.1f}%)")


if __name__ == "__main__":
    main()
