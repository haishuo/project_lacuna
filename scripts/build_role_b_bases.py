"""
scripts/build_role_b_bases.py

Produce CLEAN role-B complete-case bases from the v1.0 CURATED role-A extracts
(`lacuna_survey/evaluation_data/*_real.csv`) — already codebook-curated to substantive survey items
(cross-domain plan §2; MASTER §7). Role-A sources are preserved in place. Each base + provenance JSON
is written to /mnt/data/lacuna/role_b/ with the binding `projected_from_naturally_missing` flag.

Why the curated extracts and not the raw NHANES/ESS files: the raw files are dominated by survey-
weight/design columns and per-VARIABLE refuse codes that automated heuristics cannot separate from
valid values (age 77 vs refuse 77) without codebooks — they yield ~zero clean continuous targets. The
curated extracts already did that codebook work. NOTE (recorded finding): clean CONTINUOUS δ-targets
are SCARCE in survey data — most items are ordinal/Likert/binary; top-coding/self-censoring idioms
apply to the continuous minority (poverty, income, weight, age, news-minutes).

Curation here is light (pre-cleaned): recode only high (≥3-digit) refuse codes, complete-case project,
keep continuous targets (card>=15), reject any target with a HIGH-code spike. Prints clean targets.

Run: python -u scripts/build_role_b_bases.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from lacuna.survey.role_b_projection import (
    PROJECTION_FLAG, complete_case_project, continuous_targets, recode_sentinels,
)

EVAL = Path("/mnt/projects/project_lacuna/lacuna_survey/evaluation_data")
OUT = Path("/mnt/data/lacuna/role_b")
SENTINELS = (777, 888, 999, 6666, 7777, 8888, 9999, 99999, 999999)         # high codes only
SPIKE_CODES = (999, 6666, 7777, 8888, 9999, 99999, 999999)                 # NOT 66/77/88/99 (valid ages)
SPIKE_TOL = 0.01

# Curated extracts with >=1 clean CONTINUOUS target, tagged by NEW domain (not in the genuine-9).
SOURCES = [
    {"file": "survey_nhanes_demographics_real.csv", "name": "rb_nhanes_demographics",
     "domain": "demographics", "block": "NHANES-2017-18"},
    {"file": "survey_gssvocab_real.csv", "name": "rb_gssvocab", "domain": "social", "block": "GSS"},
    {"file": "nhanes_inq_income_real.csv", "name": "rb_nhanes_income",
     "domain": "income", "block": "NHANES-2017-18"},
    {"file": "nhanes_whq_weight_real.csv", "name": "rb_nhanes_weight",
     "domain": "health", "block": "NHANES-2017-18"},
]


def _spike_frac(col):
    v = col[~np.isnan(col)]
    return float(np.isin(v, np.asarray(SPIKE_CODES, float)).mean()) if v.size else 1.0


def build(spec):
    df = pd.read_csv(EVAL / spec["file"]).select_dtypes("number")
    x = recode_sentinels(df.to_numpy(dtype=float), SENTINELS)
    try:
        sub, names, _ = complete_case_project(x, list(df.columns), tau_col=0.6)
    except ValueError as e:
        print(f"\n### {spec['name']}: projection failed ({e}) — SKIP"); return None
    clean = [j for j in continuous_targets(sub, min_card=15) if _spike_frac(sub[:, j]) <= SPIKE_TOL]
    if not clean or sub.shape[1] < 2:
        print(f"\n### {spec['name']}: no clean continuous target / <2 cols — SKIP"); return None
    prov = {"source": spec["block"], "name": spec["name"], "domain": spec["domain"],
            "source_block": spec["block"], "flag": PROJECTION_FLAG,
            "projection": "complete_case_from_curated_role_a", "sentinels_recoded": list(SENTINELS),
            "tau_col": 0.6, "rows_retained": int(sub.shape[0]), "cols_kept": int(sub.shape[1]),
            "n_continuous_targets": len(clean), "targetable_cols": [names[j] for j in clean],
            "survey_weights": "n/a (curated extract)"}
    print(f"\n### {spec['name']}  ({spec['domain']}, {spec['block']})  role-B {sub.shape[0]}x{sub.shape[1]} "
          f"| {len(clean)} continuous targets")
    for j in clean:
        col = sub[:, j].astype(float)
        print(f"      {names[j]:18} min={np.nanmin(col):9.2f} max={np.nanmax(col):9.2f} "
              f"card={len(np.unique(col)):5d} spike={_spike_frac(col):.3f}")
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(sub, columns=names).to_csv(OUT / f"{spec['name']}.csv", index=False)
    (OUT / f"{spec['name']}.provenance.json").write_text(json.dumps(prov, indent=2))
    return prov


def main():
    print("=" * 92); print("ROLE-B from CURATED role-A extracts (light recode; complete-case; spike-reject)")
    print("=" * 92)
    provs = [p for p in (build(s) for s in SOURCES) if p]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "INDEX.json").write_text(json.dumps({"flag": PROJECTION_FLAG, "bases": provs}, indent=2))
    print(f"\nsaved {len(provs)} clean role-B bases + index: {OUT}/INDEX.json  "
          f"(domains: {sorted({p['domain'] for p in provs})})")


if __name__ == "__main__":
    main()
