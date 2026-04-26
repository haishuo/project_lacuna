#!/usr/bin/env python3
"""Fetch real public survey datasets for the Lacuna-Survey training catalog
and the demo evaluation suite.

INVARIANT — DO NOT BREAK:
    Training-side files (/mnt/data/lacuna/raw/survey_*.csv) MUST contain
    zero NaN cells. Semi-synthetic training applies a KNOWN mechanism to
    clean X; if X already has natural missingness of unknown mechanism,
    the synthetic mechanism mask gets superimposed on an unlabeled mask
    and the training labels are no longer trustworthy. Always go through
    `dropna()` (i.e. complete-case) before writing the training file.

    Evaluation-side files (demo/sample_data/survey_*_real.csv) are the
    opposite: they keep native missingness because that pattern IS the
    test signal — we want to see what the trained model says about real
    survey item nonresponse.

For each dataset we save TWO versions:

    /mnt/data/lacuna/raw/survey_<name>.csv
        Complete-case numeric subset. This is the X-base distribution
        the semi-synthetic pipeline applies survey-flavoured mechanisms
        to during training. The catalog auto-discovers files in this
        directory at startup.

    /mnt/projects/project_lacuna/demo/sample_data/survey_<name>_real.csv
        Original numeric data with its native missingness preserved.
        These are evaluation cases — apples-to-apples with how a real
        researcher would receive a survey dataset (real item nonresponse
        already present).

Run from the repo root:
    python scripts/fetch_survey_data.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path("/mnt/data/lacuna/raw")
EVAL_DIR = Path(__file__).parent.parent / "demo" / "sample_data"

RDATASETS = "https://vincentarelbundock.github.io/Rdatasets/csv"


# (slug, package, name, description, has_native_nan, eval_consensus)
DATASETS = [
    ("bfi",       "psych",     "bfi",
     "Big Five personality inventory (28 items, 2800 respondents).",
     True, "MAR"),
    ("yrbss",     "openintro", "yrbss",
     "Youth Risk Behavior Surveillance System.",
     True, "MAR"),
    ("survey",    "MASS",      "survey",
     "Adelaide student survey (already in diagnostic suite as mass_survey).",
     True, "MAR"),
    ("chile",     "carData",   "Chile",
     "1988 Chilean plebiscite survey (vote intention).",
     True, "MAR"),
    ("cars93",    "MASS",      "Cars93",
     "Consumer car survey (93 models, 19 numeric attributes).",
     True, "MAR"),
    # No-NaN survey-shaped X-bases (training only — provides survey-style
    # distributions for the semi-synthetic mechanism injection):
    ("psid7682",  "AER",       "PSID7682",
     "Panel Study of Income Dynamics 1976-82 (training X-base).",
     False, None),
    ("psid1976",  "AER",       "PSID1976",
     "Mroz labour-force survey 1976 (training X-base).",
     False, None),
    ("cps1985",   "AER",       "CPS1985",
     "Current Population Survey 1985 (training X-base).",
     False, None),
    ("cps1988",   "AER",       "CPS1988",
     "Current Population Survey 1988 (training X-base).",
     False, None),
    ("hmda",      "AER",       "HMDA",
     "Home Mortgage Disclosure Act applicant survey.",
     False, None),
    ("computers", "Ecdat",     "Computers",
     "1993 consumer computer-purchase survey.",
     False, None),
    ("workinghours", "Ecdat",  "Workinghours",
     "Quinn time-use survey (training X-base).",
     False, None),
]


def numeric_only(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the rownames index column and keep numeric columns only."""
    df_num = df.select_dtypes(include="number").copy()
    if "rownames" in df_num.columns:
        df_num = df_num.drop(columns=["rownames"])
    return df_num


def fetch(package: str, name: str) -> pd.DataFrame:
    return pd.read_csv(f"{RDATASETS}/{package}/{name}.csv")


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Training-ready CSVs → {RAW_DIR}")
    print(f"Evaluation CSVs    → {EVAL_DIR}\n")

    for slug, package, name, desc, has_nan, consensus in DATASETS:
        try:
            df = fetch(package, name)
        except Exception as e:
            print(f"  SKIP  survey_{slug:14s}  fetch error: {type(e).__name__}")
            continue

        df_num = numeric_only(df)
        n_total = df_num.size
        n_nan = int(df_num.isna().sum().sum())

        # Training file: complete-case rows only.
        cc = df_num.dropna()
        if len(cc) < 30:
            print(f"  SKIP  survey_{slug:14s}  only {len(cc)} complete-case rows")
            continue
        # Hard postcondition — see module docstring INVARIANT.
        assert int(cc.isna().sum().sum()) == 0, (
            f"BUG: training-side survey_{slug}.csv has NaN cells. "
            f"Semi-synthetic training requires complete-case X."
        )
        train_path = RAW_DIR / f"survey_{slug}.csv"
        cc.to_csv(train_path, index=False)

        # Evaluation file: native missingness preserved (only if there is some).
        eval_str = ""
        if has_nan and n_nan > 0:
            eval_path = EVAL_DIR / f"survey_{slug}_real.csv"
            df_num.to_csv(eval_path, index=False)
            eval_str = f" + eval ({n_nan}/{n_total} NaN, consensus={consensus})"

        print(f"  OK    survey_{slug:14s}  cc={len(cc):>5d}  cols={cc.shape[1]:>2d}{eval_str}  — {desc}")


if __name__ == "__main__":
    main()
