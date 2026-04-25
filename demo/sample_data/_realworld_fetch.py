"""Fetch two real-world datasets with documented expert opinion on the
missingness mechanism. Both are smoke tests, NOT ground-truth labels:
no real dataset has a verified mechanism.

  pima_real.csv         — Pima Indians Diabetes (UCI). Cells coded as 0
                          in glucose / blood_pressure / skin_thickness /
                          insulin / bmi are recoded to NaN. Consensus
                          reading: MNAR — patients with the worst
                          readings disproportionately skipped tests.

  airquality_real.csv   — New York 1973 air-quality measurements (R's
                          `datasets::airquality`, via the Rdatasets
                          mirror). Ozone has 37/153 missing, Solar.R
                          has 7/153 missing. Textbook reading: MAR
                          (Ozone missingness is argued to depend on
                          observed Wind/Temp). The textbook reading is
                          itself contested — instrument failure on
                          extreme-ozone days could equally argue MNAR.
                          Lacuna currently leans MNAR on this one;
                          treat as a discussion point, not a smoke-test
                          pass/fail.

Run from the repo root:
    python demo/sample_data/_realworld_fetch.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).parent

PIMA_URL = (
    "https://raw.githubusercontent.com/jbrownlee/Datasets/"
    "master/pima-indians-diabetes.data.csv"
)
PIMA_COLS = [
    "pregnancies", "glucose", "blood_pressure", "skin_thickness",
    "insulin", "bmi", "diabetes_pedigree", "age", "outcome",
]
PIMA_ZERO_AS_NAN = [
    "glucose", "blood_pressure", "skin_thickness", "insulin", "bmi",
]

AIRQUALITY_URL = (
    "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/"
    "airquality.csv"
)
AIRQUALITY_KEEP_COLS = ["Ozone", "Solar.R", "Wind", "Temp", "Month", "Day"]


def fetch_pima() -> pd.DataFrame:
    df = pd.read_csv(PIMA_URL, header=None, names=PIMA_COLS)
    for col in PIMA_ZERO_AS_NAN:
        df.loc[df[col] == 0, col] = np.nan
    # Drop the outcome label so the demo can't see it.
    df = df.drop(columns=["outcome"])
    return df


def fetch_airquality() -> pd.DataFrame:
    df = pd.read_csv(AIRQUALITY_URL)
    return df[AIRQUALITY_KEEP_COLS].copy()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for name, fn, consensus in [
        ("pima_real.csv",       fetch_pima,       "MNAR"),
        ("airquality_real.csv", fetch_airquality, "MAR"),
    ]:
        df = fn()
        path = OUT_DIR / name
        df.to_csv(path, index=False)
        n_missing = int(df.isna().sum().sum())
        n_total = df.size
        print(f"{name:18s}  consensus={consensus:5s}  "
              f"{df.shape[0]}×{df.shape[1]}  "
              f"{n_missing}/{n_total} missing ({n_missing/n_total*100:.1f}%)")


if __name__ == "__main__":
    main()
