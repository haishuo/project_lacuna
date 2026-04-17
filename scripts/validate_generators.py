#!/usr/bin/env python3
"""
Validate that Lacuna's synthetic generators produce patterns consistent
with their declared MCAR / MAR / MNAR class.

Runs `pystatistics.little_mcar_test` on N independent samples from each
generator in a registry and reports the reject-rate alongside the expected
rate. MCAR generators should reject at ~alpha; MAR/MNAR generators should
reject at ~1.

Usage:
    python scripts/validate_generators.py --registry lacuna_minimal_6
    python scripts/validate_generators.py --registry lacuna_minimal_6 \
        --n-datasets 100 --n-rows 300 --n-cols 6 --seed 42
    python scripts/validate_generators.py --registry lacuna_minimal_6 \
        --output generator_validation.json
"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lacuna.generators.families.registry_builder import load_registry_from_config
from lacuna.analysis.generator_validation import (
    validate_registry,
    format_results_table,
    summarize,
    VERDICT_FAIL,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        required=True,
        help="Generator registry config name (e.g. lacuna_minimal_6) or path to YAML.",
    )
    parser.add_argument("--seed", type=int, default=20260417)
    parser.add_argument("--n-datasets", type=int, default=50)
    parser.add_argument("--n-rows", type=int, default=200)
    parser.add_argument("--n-cols", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--mcar-tolerance",
        type=float,
        default=0.10,
        help="MCAR passes if reject-rate <= alpha + tolerance.",
    )
    parser.add_argument(
        "--non-mcar-lower",
        type=float,
        default=0.80,
        help="MAR/MNAR passes if reject-rate >= this.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path.",
    )
    args = parser.parse_args()

    print(f"Loading registry: {args.registry}")
    registry = load_registry_from_config(args.registry)
    print(f"  {registry!r}")
    print(
        f"Sampling {args.n_datasets} datasets of shape "
        f"({args.n_rows}, {args.n_cols}) per generator at alpha={args.alpha}..."
    )
    print()

    results = validate_registry(
        registry,
        seed=args.seed,
        n_datasets=args.n_datasets,
        n_rows=args.n_rows,
        n_cols=args.n_cols,
        alpha=args.alpha,
        mcar_upper_tolerance=args.mcar_tolerance,
        non_mcar_lower_bound=args.non_mcar_lower,
    )

    print(format_results_table(results))
    print()

    n_pass, n_fail, n_indet = summarize(results)
    print(f"Pass: {n_pass}  Fail: {n_fail}  Indeterminate: {n_indet}  "
          f"(Total: {len(results)})")
    if n_indet > 0:
        print("  Indeterminate = MNAR generators. Little's test is "
              "underpowered against self-censoring MNAR; validate these "
              "with a mechanism-specific check.")

    failures = [r for r in results if r.verdict == VERDICT_FAIL]
    if failures:
        print()
        print("FAILED generators:")
        for r in failures:
            print(
                f"  - [{r.generator_id}] {r.generator_name} "
                f"(class={r.class_id}): reject_rate={r.reject_rate:.3f}, "
                f"expected={r.expected_reject_rate:.3f}, skipped={r.skipped_count}"
            )

    if args.output is not None:
        payload = {
            "config": {
                "registry": args.registry,
                "seed": args.seed,
                "n_datasets": args.n_datasets,
                "n_rows": args.n_rows,
                "n_cols": args.n_cols,
                "alpha": args.alpha,
                "mcar_upper_tolerance": args.mcar_tolerance,
                "non_mcar_lower_bound": args.non_mcar_lower,
            },
            "summary": {
                "n_pass": n_pass,
                "n_fail": n_fail,
                "n_indeterminate": n_indet,
                "n_total": len(results),
            },
            "results": [asdict(r) for r in results],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2))
        print(f"\nWrote {args.output}")

    # Exit non-zero if any generator failed. Indeterminate is not failure.
    sys.exit(1 if n_fail > 0 else 0)


if __name__ == "__main__":
    main()
