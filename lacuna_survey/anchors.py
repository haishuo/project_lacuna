"""Anchor registry for Lacuna-Survey real-data calibration.

A small declarative table of (slug, consensus_label, source, citation,
notes). Each anchor must be a numeric CSV with native missingness in
`evaluation_data/`. The calibration script reads this list rather than
hard-coded paths so adding an anchor is one row + one CSV.

To add a new anchor:
  1. Drop the CSV into `lacuna_survey/evaluation_data/<slug>.csv`.
  2. Add an `Anchor(...)` row below with the consensus mechanism and
     a citation justifying why that mechanism is the textbook reading
     for this dataset.
  3. Re-run `python -m lacuna_survey.calibrate`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

# Class IDs match lacuna.core.types: MCAR=0, MAR=1, MNAR=2.


@dataclass(frozen=True)
class Anchor:
    slug: str           # filename stem (without _real.csv)
    label: int          # 0=MCAR, 1=MAR, 2=MNAR
    label_name: str     # readable
    source: str         # where it came from (URL or institution)
    citation: str       # textbook / paper justifying the mechanism reading
    notes: str = ""


# ---------------------------------------------------------------------------
# Currently fetched (programmatic Rdatasets mirrors). All consensus-MAR.
# This is the v1 anchor set; calibration here is structurally biased toward
# MAR because we have no real-survey MNAR or MCAR examples yet.
# ---------------------------------------------------------------------------

ANCHORS: List[Anchor] = [
    Anchor(
        slug="survey_bfi",
        label=1, label_name="MAR",
        source="psych::bfi (CRAN R package)",
        citation="Revelle 2018 'psych' package; Schafer & Graham 2002 §6.1 use bfi as the canonical MAR-via-MICE example.",
        notes="2236 respondents × 28 personality items. Item nonresponse spread across 25 items.",
    ),
    Anchor(
        slug="survey_chile",
        label=1, label_name="MAR",
        source="carData::Chile (CRAN R package)",
        citation="Fox 2008 'Applied Regression' uses Chile as an MAR example with demographic predictors driving income nonresponse.",
        notes="2700 respondents × 4 numeric items, 1988 Chilean plebiscite vote-intention survey.",
    ),
    Anchor(
        slug="survey_cars93",
        label=1, label_name="MAR",
        source="MASS::Cars93 (CRAN R package)",
        citation="Venables & Ripley 2002 'Modern Applied Statistics with S' use Cars93 as a tractable MAR example.",
        notes="93 cars × 18 numeric attributes. Sparse nonresponse on a few attributes.",
    ),
    Anchor(
        slug="survey_survey",
        label=1, label_name="MAR",
        source="MASS::survey (CRAN R package)",
        citation="Adelaide student survey, Venables & Ripley 2002.",
        notes="237 students × 5 numeric items.",
    ),
    Anchor(
        slug="survey_yrbss",
        label=1, label_name="MAR",
        source="openintro::yrbss (CRAN R package)",
        citation="Diez, Çetinkaya-Rundel, Barr 2019 OpenIntro Statistics treat YRBSS as MAR. Note that "
                 "phone-survey nonresponse can also be defensibly MCAR; the consensus is the weakest of the bunch.",
        notes="Youth Risk Behavior Surveillance System, 13583 respondents × 5 numeric items.",
    ),

    # 2026-04-26 expansion (post-OOD work):
    Anchor(
        slug="survey_nhanes_demographics",
        label=2, label_name="MNAR",
        source="HuggingFace nguyenvy/cleaned_nhanes_1988_2018 (NHANES 1988-2018 demographics, "
               "filtered to rows where demographics fully administered, leaving NaN concentrated "
               "in INDFMPIR income-to-poverty ratio).",
        citation="NHANES Continuous methodology reports document income refusal patterns "
                 "(NCHS 2013); INDFMPIR refusal is the canonical MNAR example for income items "
                 "in the missing-data literature (Allison 2001 §6.4; Schafer & Graham 2002).",
        notes="500 rows × 5 cols; ~8% NaN concentrated in INDFMPIR. The first real-survey MNAR "
              "anchor in the calibration corpus.",
    ),
    Anchor(
        slug="survey_gssvocab",
        label=1, label_name="MAR",
        source="carData::GSSvocab (CRAN R package; General Social Survey vocabulary module).",
        citation="Fox 2008 'Applied Regression' treats GSS item nonresponse as MAR conditional on "
                 "demographic predictors (age, education, gender). NORC GSS methodology reports "
                 "document item nonresponse patterns by demographics.",
        notes="500-row sample × 4 numeric items; ~1% NaN. Light missingness — primarily on "
              "year-of-birth and other demographic items.",
    ),
    Anchor(
        slug="survey_ucla_textbooks",
        label=1, label_name="MAR",
        source="openintro::ucla_textbooks_f18 (UCLA student textbook-purchase survey, Fall 2018).",
        citation="Diez et al. OpenIntro Statistics document student-survey item nonresponse as "
                 "MAR conditional on observed enrollment and price-quote characteristics.",
        notes="201 students × 7 numeric items; ~43% NaN concentrated on price-quote items "
              "(students didn't quote prices for textbooks they didn't purchase).",
    ),
]


# ---------------------------------------------------------------------------
# Candidate anchors NOT YET FETCHED — drop in CSVs and uncomment the rows.
# ---------------------------------------------------------------------------
#
# MCAR-by-design candidates:
#   - PISA 2018/2022 student questionnaire: rotated booklet design.
#     Free OECD download; pick one item module.
#     https://www.oecd.org/pisa/data/
#
#   - NAEP three-form planned-missing design.
#     https://nces.ed.gov/nationsreportcard/about/datafiles.aspx
#
#   - ESS rotation modules (each respondent gets random module subset).
#     https://www.europeansocialsurvey.org/
#
# MNAR candidates:
#   - CPS-ASEC income items (high-earner refusal documented in Census
#     Bureau methodology reports). Free Census download.
#     https://www.census.gov/data/datasets/time-series/demo/cps/cps-asec.html
#
#   - NHANES sensitive-item modules (drug use, sexual behavior).
#     128 MB CSV on HuggingFace nguyenvy/cleaned_nhanes_1988_2018,
#     or per-cycle via CDC.
#
#   - NLSY97 wealth/income supplements with documented refusal.
#     https://www.bls.gov/nls/nlsy97.htm
#
#   - HRS Wave 14+ wealth-question refusal items.
#     https://hrs.isr.umich.edu/data-products
#
# ---------------------------------------------------------------------------


def by_class(label: int) -> List[Anchor]:
    """Return anchors with a given class label."""
    return [a for a in ANCHORS if a.label == label]


def class_balance() -> dict:
    """Return a count of anchors per class. Useful for checking that the
    calibration corpus has enough class diversity to fit non-degenerate
    parameters."""
    return {
        "MCAR": len(by_class(0)),
        "MAR": len(by_class(1)),
        "MNAR": len(by_class(2)),
        "total": len(ANCHORS),
    }
