"""Validate and stage a new calibration anchor.

Drops a user-provided CSV into `lacuna_survey/evaluation_data/` after
sanity-checking that it (a) is parseable, (b) contains numeric
columns, (c) has actual missingness in those columns. After import,
the user must add an `Anchor(...)` row to `lacuna_survey/anchors.py`
to make it visible to `calibrate.py`.

Usage:
    python -m lacuna_survey.import_anchor \
        path/to/source.csv \
        --slug nhanes_2018_demographics \
        --max-rows 500     # cap the file size for storage

The script prints a one-line `Anchor(...)` row you can paste into
anchors.py — it does NOT auto-edit anchors.py because the consensus
mechanism label requires a human judgment call backed by citation.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
EVAL_DIR = HERE / "evaluation_data"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", type=Path, help="Source CSV (must be readable by pandas).")
    ap.add_argument("--slug", required=True,
                    help="Identifier for the anchor — used as filename "
                         "stem (will be saved as <slug>_real.csv).")
    ap.add_argument("--max-rows", type=int, default=None,
                    help="Optional row cap (random subsample, seed=42).")
    ap.add_argument("--keep-non-numeric", action="store_true",
                    help="By default we strip non-numeric columns since the "
                         "demo loader does. Use this flag to keep them in "
                         "the CSV for inspection.")
    args = ap.parse_args()

    if not args.source.is_file():
        raise SystemExit(f"Not a file: {args.source}")

    df = pd.read_csv(args.source)
    if not args.keep_non_numeric:
        n_dropped = df.shape[1] - df.select_dtypes(include="number").shape[1]
        df = df.select_dtypes(include="number")
        if n_dropped:
            print(f"Stripped {n_dropped} non-numeric columns")
    if df.shape[1] < 2:
        raise SystemExit("Need at least 2 numeric columns; got "
                         f"{df.shape[1]}")

    # Subsample if needed
    if args.max_rows and df.shape[0] > args.max_rows:
        df = df.sample(n=args.max_rows, random_state=42).reset_index(drop=True)
        print(f"Subsampled to {args.max_rows} rows")

    n_total = df.size
    n_nan = int(df.isna().sum().sum())
    miss_pct = n_nan / n_total * 100 if n_total else 0
    cols_w_nan = int((df.isna().sum() > 0).sum())

    print(f"\nValidated anchor candidate:")
    print(f"  shape:        {df.shape[0]} rows × {df.shape[1]} cols")
    print(f"  NaN cells:    {n_nan:,} / {n_total:,} ({miss_pct:.2f} %)")
    print(f"  cols w/ NaN:  {cols_w_nan} / {df.shape[1]}")

    if n_nan == 0:
        print("\n  ⚠  No missing values — this dataset has no signal for "
              "Lacuna to classify. Aborting.")
        raise SystemExit(1)

    # Save
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EVAL_DIR / f"{args.slug}_real.csv"
    if out_path.exists():
        print(f"\n  ⚠  {out_path.name} already exists — overwriting.")
    df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")

    # Print template Anchor row
    print("\n--- paste this into lacuna_survey/anchors.py "
          "(after filling in the label, source, citation, notes): ---\n")
    print(f"""    Anchor(
        slug="{args.slug}",
        label=?,                       # 0=MCAR, 1=MAR, 2=MNAR
        label_name="?",                # MCAR / MAR / MNAR
        source="?",                    # URL or institution
        citation="?",                  # textbook reference / paper
        notes="{df.shape[0]}×{df.shape[1]}, {miss_pct:.1f}% missing across {cols_w_nan} cols",
    ),""")


if __name__ == "__main__":
    main()
