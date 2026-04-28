"""Build a curated demo sample set from the canonical real-survey anchors.

Copies a story-ordered subset of `lacuna_survey/evaluation_data/` into this
directory with descriptive filenames, so the demo's sample picker can walk
a viewer through:

  1.  A clean MAR win — Chilean political survey (P(MAR) ~ 0.91 ✓)
  2.  A clean MCAR win — PISA 2018 rotated booklet (P(MCAR) ~ 0.83 ✓)
  3.  A clean MNAR win — NHANES drug-use module (P(MNAR) ~ 0.52, argmax MNAR ✓)
  4.  A probabilistic-story case — NHANES depression screener (P(MNAR) ~ 0.33;
      argmax MAR but with elevated MNAR posterior, demonstrating Lacuna's
      calibrated answer to a Molenberghs-underdetermined case)
  5.  Same — NHANES income module (P(MNAR) ~ 0.43)
  6.  Same — NHANES weight history (P(MNAR) ~ 0.30)
  7.  An out-of-scope demo — Pima diabetes (cross-domain, OOD-flagged)

The eval anchors are public-data subsets (NHANES is US-government public domain,
PISA is OECD-licensed redistributable, R-package datasets are open-licensed).

Run from the repo root:
    python demo/sample_data/_canonical_anchors.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

OUT_DIR = Path(__file__).parent
EVAL_DIR = Path(__file__).resolve().parent.parent.parent / "lacuna_survey" / "evaluation_data"

# (output filename, source filename in evaluation_data, description)
CANONICAL_SET = [
    ("01_MAR_chile_voting.csv",
     "survey_chile_real.csv",
     "MAR win — 1988 Chilean political survey, item nonresponse driven by demographics"),
    ("02_MCAR_pisa_2018.csv",
     "pisa2018_gbr_rotation_real.csv",
     "MCAR win — PISA 2018 rotated booklet design, missingness by random booklet assignment"),
    ("03_MNAR_nhanes_drug.csv",
     "nhanes_duq_drug_real.csv",
     "MNAR win — NHANES 2017-18 drug-use module, refusal-driven missingness"),
    ("04_borderline_nhanes_depression.csv",
     "nhanes_dpq_phq9_real.csv",
     "Probabilistic case — NHANES PHQ-9 depression screener, contested MNAR/MAR-cond"),
    ("05_borderline_nhanes_income.csv",
     "nhanes_inq_income_real.csv",
     "Probabilistic case — NHANES income module, high-earner refusal"),
    ("06_borderline_nhanes_weight.csv",
     "nhanes_whq_weight_real.csv",
     "Probabilistic case — NHANES weight history, heavy-respondent refusal"),
]


def main() -> None:
    if not EVAL_DIR.exists():
        raise SystemExit(f"Evaluation data not found: {EVAL_DIR}. "
                         "Run anchor-fetch scripts first.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for out_name, src_name, desc in CANONICAL_SET:
        src = EVAL_DIR / src_name
        if not src.exists():
            print(f"  SKIP {out_name}: source missing ({src_name})")
            continue
        dst = OUT_DIR / out_name
        shutil.copy2(src, dst)
        print(f"  wrote {out_name}  ({desc})")


if __name__ == "__main__":
    main()
