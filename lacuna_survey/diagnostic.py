#!/usr/bin/env python3
"""Run Lacuna against textbook real-world datasets to scope the MAR
detection problem.

Each dataset is paired with its textbook-consensus mechanism reading.
We run inference with the demo's installed checkpoint and report
agreement / disagreement plus the model's confidence and per-class
probabilities. The goal is diagnostic, not validation: if MAR-consensus
datasets systematically land in MNAR, that signals a distribution-
shift problem in the training generators; if the failures are mixed,
that points at an architectural issue.

Usage:
    python scripts/diagnose_mar.py
    python scripts/diagnose_mar.py --checkpoint demo/model.pt --csv diagnose.csv

References / consensus citations are in CONSENSUS_NOTES below — they
are textbook readings, not ground truth (no real dataset has a verified
mechanism). Use the output as evidence, not proof.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from demo.pipeline import build_model, csv_to_dataset, run_model, MODEL_DEFAULTS

# Optional Lacuna-Survey calibration (vector scaling). If a calibration
# file is present in deployment/calibration.json, predictions are
# post-processed before reporting. Enable with --calibrated.
DEFAULT_CALIBRATION_PATH = Path(__file__).parent / "deployment" / "calibration.json"


def _load_calibration(path: Path):
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    return payload["temperature"], payload["bias"]


def _apply_calibration(p_class, T: float, bias):
    """Apply vector scaling (logits' = (logits - bias) / T) to a
    posterior probability vector. Returns calibrated probabilities."""
    p = np.asarray(p_class, dtype=np.float64)
    logits = np.log(np.clip(p, 1e-8, 1.0))
    cal = (logits - np.asarray(bias)) / T
    e = np.exp(cal - cal.max())
    return e / e.sum()


RDATASETS = "https://vincentarelbundock.github.io/Rdatasets/csv"


def _fetch(pkg: str, name: str) -> pd.DataFrame:
    return pd.read_csv(f"{RDATASETS}/{pkg}/{name}.csv")


# --- Dataset preparers ------------------------------------------------------
# Each preparer returns a numeric DataFrame with NaN in missing cells.

def prep_airquality() -> pd.DataFrame:
    df = _fetch("datasets", "airquality")
    return df[["Ozone", "Solar.R", "Wind", "Temp", "Month", "Day"]].copy()


def prep_pima_uci() -> pd.DataFrame:
    url = ("https://raw.githubusercontent.com/jbrownlee/Datasets/"
           "master/pima-indians-diabetes.data.csv")
    cols = ["pregnancies", "glucose", "blood_pressure", "skin_thickness",
            "insulin", "bmi", "diabetes_pedigree", "age", "outcome"]
    df = pd.read_csv(url, header=None, names=cols)
    for c in ["glucose", "blood_pressure", "skin_thickness", "insulin", "bmi"]:
        df.loc[df[c] == 0, c] = np.nan
    return df.drop(columns=["outcome"])


def prep_pima_tr2() -> pd.DataFrame:
    df = _fetch("MASS", "Pima.tr2")
    # type column is the diabetes label; drop. Other 7 are numeric.
    return df.select_dtypes(include="number").copy()


def prep_hitters() -> pd.DataFrame:
    df = _fetch("ISLR", "Hitters")
    return df.select_dtypes(include="number").copy()


def prep_mass_survey() -> pd.DataFrame:
    df = _fetch("MASS", "survey")
    return df.select_dtypes(include="number").copy()


def prep_pbc() -> pd.DataFrame:
    df = _fetch("survival", "pbc")
    # Drop id, time, status (outcome columns); keep clinical/lab columns.
    drop = ["rownames", "id", "time", "status"]
    df = df.drop(columns=[c for c in drop if c in df.columns])
    return df.select_dtypes(include="number").copy()


# --- Real survey datasets (saved by lacuna_survey.fetch_data) ----------

def _survey_real(name: str) -> pd.DataFrame:
    path = Path(__file__).parent / "evaluation_data" / f"survey_{name}_real.csv"
    return pd.read_csv(path)


def prep_survey_bfi() -> pd.DataFrame:
    return _survey_real("bfi")


def prep_survey_yrbss() -> pd.DataFrame:
    return _survey_real("yrbss")


def prep_survey_chile() -> pd.DataFrame:
    return _survey_real("chile")


def prep_survey_cars93() -> pd.DataFrame:
    return _survey_real("cars93")


def prep_survey_survey() -> pd.DataFrame:
    return _survey_real("survey")


# --- Diagnostic registry ----------------------------------------------------

CONSENSUS_NOTES = {
    "airquality": (
        "MAR",
        "Schafer / van Buuren texts — Ozone missingness depends on "
        "observed Wind/Temp. Contested: MNAR is also defensible if "
        "missingness reflects instrument failure on extreme days.",
    ),
    "pima_uci": (
        "MNAR",
        "Patients with worst readings disproportionately skipped tests. "
        "Standard reading.",
    ),
    "pima_tr2": (
        "MNAR",
        "Variant of Pima Indians (MASS::Pima.tr2). Same mechanism "
        "argument as the UCI version.",
    ),
    "hitters": (
        "MNAR",
        "ISLR baseball Salary; players without salary likely had "
        "non-random career situations (free agency, retirement).",
    ),
    "mass_survey": (
        "MAR",
        "Student survey (Venables & Ripley). Item nonresponse depends "
        "on demographic predictors that are observed.",
    ),
    "pbc": (
        "MAR",
        "Mayo Clinic primary biliary cirrhosis lab values. Most "
        "textbook treatments use MICE under MAR; some lab values "
        "are arguably MNAR.",
    ),
    "survey_bfi": (
        "MAR",
        "psych::bfi — Big Five 28-item personality inventory, "
        "n=2800. Item nonresponse spreads across 25 items; "
        "treated as MAR for MICE in the psych package.",
    ),
    "survey_yrbss": (
        "MAR",
        "openintro::yrbss — Youth Risk Behavior Surveillance "
        "System, n=13583. Item nonresponse on physical/behavior "
        "items, plausibly MAR conditional on demographics.",
    ),
    "survey_chile": (
        "MAR",
        "carData::Chile — 1988 plebiscite vote-intention survey, "
        "n=2700. Item nonresponse on income / vote-intention; "
        "MAR conditional on demographics.",
    ),
    "survey_cars93": (
        "MAR",
        "MASS::Cars93 — consumer car-attribute survey. Sparse "
        "nonresponse on a few attributes.",
    ),
    "survey_survey": (
        "MAR",
        "MASS::survey — Adelaide student survey (also in the "
        "diagnostic suite as `mass_survey`).",
    ),
}

REGISTRY: dict[str, Callable[[], pd.DataFrame]] = {
    # Cross-domain textbook cases (originally probed against generic Lacuna)
    "airquality":   prep_airquality,
    "pima_uci":     prep_pima_uci,
    "pima_tr2":     prep_pima_tr2,
    "hitters":      prep_hitters,
    "mass_survey":  prep_mass_survey,
    "pbc":          prep_pbc,
    # Real survey datasets — apples-to-apples for Lacuna-Survey
    "survey_bfi":    prep_survey_bfi,
    "survey_yrbss":  prep_survey_yrbss,
    "survey_chile":  prep_survey_chile,
    "survey_cars93": prep_survey_cars93,
    "survey_survey": prep_survey_survey,
}


# --- Runner -----------------------------------------------------------------

CLASS_NAMES_TUPLE = ("MCAR", "MAR", "MNAR")


def run_one(model, name: str, calibration=None) -> dict:
    df = REGISTRY[name]()
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    loaded = csv_to_dataset(csv_bytes, name, max_rows=MODEL_DEFAULTS["max_rows"])
    overview_n_missing = int((~loaded.dataset.r).sum().item())
    if overview_n_missing == 0:
        return {"name": name, "error": "no missing values after preprocessing"}

    out = run_model(model, loaded.dataset)
    p_class = out["p_class"]
    if calibration is not None:
        T, bias = calibration
        p_class = _apply_calibration(p_class, T, bias).tolist()
    pred_idx = int(np.argmax(p_class))
    pred = CLASS_NAMES_TUPLE[pred_idx]
    consensus, _ = CONSENSUS_NOTES[name]
    return {
        "name": name,
        "n": loaded.dataset.n,
        "d": loaded.dataset.d,
        "n_missing": overview_n_missing,
        "pct_missing": overview_n_missing / (loaded.dataset.n * loaded.dataset.d) * 100,
        "consensus": consensus,
        "pred": pred,
        "agree": pred == consensus,
        "p_mcar": float(p_class[0]),
        "p_mar":  float(p_class[1]),
        "p_mnar": float(p_class[2]),
        "confidence": float(max(p_class)),
        "action": out["action_label"],  # action is from raw posterior; calibration affects p only
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default=str(PROJECT_ROOT / "demo" / "model.pt"))
    ap.add_argument("--csv", default=None,
                    help="Optional path to write the result table as CSV.")
    ap.add_argument("--calibrated", action="store_true",
                    help=f"Apply Lacuna-Survey calibration from {DEFAULT_CALIBRATION_PATH}")
    args = ap.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    model, n_params = build_model(args.checkpoint)
    print(f"  {n_params:,} parameters")
    calibration = None
    if args.calibrated:
        calibration = _load_calibration(DEFAULT_CALIBRATION_PATH)
        if calibration is None:
            print(f"  WARNING: --calibrated requested but no calibration at "
                  f"{DEFAULT_CALIBRATION_PATH} — running raw")
        else:
            T, bias = calibration
            print(f"  Calibration: T={T:.3f}  bias={[round(b, 3) for b in bias]}")
    print()

    rows = []
    for name in REGISTRY:
        print(f"Running {name}…", flush=True)
        try:
            row = run_one(model, name, calibration=calibration)
        except Exception as e:
            row = {"name": name, "error": f"{type(e).__name__}: {e}"}
        rows.append(row)

    print()
    print("=" * 100)
    print(f"{'dataset':14s}  {'n×d':10s}  {'miss%':6s}  "
          f"{'consensus':10s}  {'pred':5s}  {'agree':5s}  "
          f"{'p_MCAR':>7s} {'p_MAR':>7s} {'p_MNAR':>7s}  {'conf':>6s}  action")
    print("-" * 100)
    for row in rows:
        if "error" in row:
            print(f"{row['name']:14s}  ERROR: {row['error']}")
            continue
        agree_mark = "✓" if row["agree"] else "✗"
        print(
            f"{row['name']:14s}  "
            f"{row['n']:>4d}×{row['d']:<3d}    "
            f"{row['pct_missing']:5.1f}%  "
            f"{row['consensus']:10s}  "
            f"{row['pred']:5s}  "
            f"{agree_mark:5s}  "
            f"{row['p_mcar']:7.3f} {row['p_mar']:7.3f} {row['p_mnar']:7.3f}  "
            f"{row['confidence']*100:5.1f}%  "
            f"{row['action']}"
        )
    print("=" * 100)

    # Summary by consensus class
    print()
    valid = [r for r in rows if "error" not in r]
    by_consensus = {}
    for r in valid:
        by_consensus.setdefault(r["consensus"], []).append(r)
    for cons, items in sorted(by_consensus.items()):
        n_total = len(items)
        n_agree = sum(1 for r in items if r["agree"])
        misroutes = {}
        for r in items:
            if not r["agree"]:
                misroutes[r["pred"]] = misroutes.get(r["pred"], 0) + 1
        misroute_str = ", ".join(f"{k}: {v}" for k, v in misroutes.items()) or "—"
        print(f"  consensus={cons:5s}  agreement {n_agree}/{n_total}  "
              f"misroutes: {misroute_str}")

    if args.csv:
        out_df = pd.DataFrame(valid)
        out_df.to_csv(args.csv, index=False)
        print(f"\nWrote table → {args.csv}")


if __name__ == "__main__":
    main()
