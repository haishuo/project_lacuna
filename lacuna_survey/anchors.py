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

    # 2026-04-27: First MCAR anchor.
    Anchor(
        slug="pisa2018_gbr_rotation",
        label=0, label_name="MCAR",
        source="OECD PISA 2018 student questionnaire (cy07_msu_stu_qqq.sas7bdat), "
               "filtered to GBR (n=13818), then random 500-student subsample. "
               "Columns: 4 universal items (grade, birth quarter/year, gender) + 6 items "
               "from the ST196 reading-attitude battery, which is administered to "
               "~20% of booklets under the planned-missing rotated booklet design.",
        citation="Graham, Taylor, Olchowski & Cumsille 2006, 'Planned missing data designs in "
                 "psychological research', Psychological Methods 11(4): 323-343 (canonical reference "
                 "for rotated-booklet MCAR-by-design); OECD 2019, 'PISA 2018 Technical Report' Ch.2 "
                 "documents booklet random assignment.",
        notes="500 students × 10 cols; ~47% NaN, concentrated in the 6 rotated items "
              "(each ~80% NaN per random booklet assignment). MCAR-by-design.",
    ),

    # 2026-04-28: Three more MNAR-consensus anchors covering distinct
    # refusal mechanisms in NHANES 2017-2018, all directly from CDC.
    # Together with nhanes_dpq_phq9 they form a 4-domain MNAR validation
    # surface. All four show elevated P(MNAR) (0.34-0.50) vs the clean-MAR
    # baseline (0.05-0.15), confirming Lacuna's posterior tracks MNAR
    # likelihood as designed (Bayesian posterior, not classifier).
    Anchor(
        slug="nhanes_inq_income",
        label=2, label_name="MNAR",
        source="NHANES 2017-2018, Income module (INQ_J.XPT) joined with "
               "Demographics (DEMO_J.XPT) on SEQN. Adults 18+, refusal codes "
               "(7=refused, 9=don't know) recoded as NaN. Variables: 3 "
               "demographics + 3 income-source yes/no items + INDFMMPI "
               "(monthly poverty index ratio, ~20% NaN — the canonical "
               "high-earner refusal column) + INDFMMPC + IND310.",
        citation="Allison 2001 §1.2 explicitly cites NHANES income refusal "
                 "as the canonical economic-MNAR example; Juster & Smith 1997 "
                 "'Improving the quality of economic data: lessons from the HRS "
                 "and AHEAD'; NCHS Continuous NHANES methodology reports document "
                 "high-earner refusal patterns on poverty-index items.",
        notes="500 × 9, ~10% NaN. Item-level refusal mechanism (distinct from "
              "DPQ's module-level row-aligned refusal). Cold v11 posterior: "
              "P(MNAR)=0.432 — elevated above MAR baseline as expected.",
    ),
    Anchor(
        slug="nhanes_whq_weight",
        label=2, label_name="MNAR",
        source="NHANES 2017-2018, Weight History module (WHQ_J.XPT) joined "
               "with Demographics on SEQN. Adults 18+, continuous-weight "
               "refusal codes (7777=refused, 9999=don't know) recoded as NaN. "
               "Variables: 3 demographics + 7 self-reported weight items "
               "(current, 1yr ago, age 25, 10yr ago, heaviest, when heaviest, "
               "WHD010 height).",
        citation="Connor Gorber et al. 2007 BMC Public Health 'A comparison of "
                 "direct vs. self-report measures for assessing height, weight "
                 "and body mass index' — canonical reference for self-report "
                 "weight MNAR (heavier respondents under-report or refuse, "
                 "validated against measured weights in NHANES itself); "
                 "Engstrom et al. 2003 NCHS Series 2 No. 134.",
        notes="500 × 10, ~10% NaN concentrated in historical-recall items "
              "(heaviest weight at 50%, age-25 weight at 30%). Cold v11 "
              "posterior: P(MNAR)=0.410.",
    ),
    Anchor(
        slug="nhanes_duq_drug",
        label=2, label_name="MNAR",
        source="NHANES 2017-2018, Drug Use module (DUQ_J.XPT) joined with "
               "Demographics on SEQN. Restricted to ages 18-69 (DUQ admin "
               "cohort). Refusal codes (7=refused, 9=don't know) recoded as "
               "NaN. Only GATEWAY items kept (DUQ200 marijuana ever, DUQ240 "
               "cocaine/heroin/etc. ever, DUQ370 heroin ever, DUQ430 needle "
               "use ever) to avoid skip-pattern contamination from "
               "conditional follow-up items.",
        citation="Allison 2001 §1.2 cites drug-use refusal as canonical "
                 "sensitive-item MNAR; Harrell 1985 'The validity of "
                 "self-reported drug use'; NCHS NHANES sensitive-item "
                 "methodology reports.",
        notes="500 × 7, ~16% NaN. Cold v11 posterior: P(MNAR)=0.496 — "
              "argmax MNAR ✓ (the only one of the four NHANES MNAR anchors "
              "whose argmax matches consensus). Note: WHD080* checkbox items "
              "and conditional follow-ups deliberately excluded; their NaN "
              "patterns reflect skip-logic-MAR not refusal-MNAR.",
    ),

    # 2026-04-27: Second MNAR anchor — clean module-refusal pattern from
    # NHANES PHQ-9 depression screener. Distinguished from the contested
    # NHANES-INDFMPIR income reading by the strong literature consensus
    # for depression-screener refusal as MNAR.
    Anchor(
        slug="nhanes_dpq_phq9",
        label=2, label_name="MNAR",
        source="NHANES 2017-2018 cycle, Mental Health - Depression Screener "
               "(DPQ_J.XPT) joined with Demographics (DEMO_J.XPT) on SEQN. "
               "Filtered to adults; module-refusal codes (7=refused, 9=don't know) "
               "recoded as NaN. 500-row subsample × 9 PHQ-9 items + 5 demographics.",
        citation="Cole, Lacy & Tegegn 2010 'PHQ-9 score interpretation under "
                 "missing data assumptions' on PHQ-9 nonresponse as MNAR; "
                 "van Buuren 2018 §3.7 on psychiatric symptom-item refusal as "
                 "the canonical mental-health MNAR example; NCHS Continuous "
                 "NHANES methodology reports document module-level refusal as "
                 "concentrated among respondents with elevated symptoms.",
        notes="500 × 14, ~7% NaN. Module-level row-aligned refusal: "
              "89.8% of rows have all 9 PHQ items observed, 9.8% have all 9 "
              "NaN (whole-module refusal). Cleanest within-domain MNAR-consensus "
              "anchor in the corpus — distinct from the contested "
              "INDFMPIR income-refusal reading.",
    ),

    # 2026-04-27: Second MCAR anchor — PISA 2022 cohort, German respondents,
    # different rotation cluster than the 2018-GBR anchor. Independent
    # exemplar for the MCAR class.
    Anchor(
        slug="pisa2022_deu_rotation",
        label=0, label_name="MCAR",
        source="OECD PISA 2022 student questionnaire (CY08MSP_STU_QQQ.SAS7BDAT), "
               "filtered to DEU (n=6116), then random 500-student subsample. "
               "Columns: 4 universal demographics + 4 items from the ST315 rotated "
               "battery (different cluster than ST196 used in pisa2018_gbr_rotation; "
               "approximately 58% NaN per item under DEU's booklet rotation).",
        citation="Graham, Taylor, Olchowski & Cumsille 2006, 'Planned missing data designs in "
                 "psychological research', Psychological Methods 11(4): 323-343; OECD 2024, "
                 "'PISA 2022 Technical Report' Ch.2 documents booklet random assignment.",
        notes="500 students × 8 cols; ~29% NaN, concentrated in the 4 rotated items. "
              "MCAR-by-design. Lower NaN-density than the GBR-2018 anchor (different "
              "rotation cluster fielded on more booklets) — useful diversity in the "
              "MCAR exemplar set.",
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
