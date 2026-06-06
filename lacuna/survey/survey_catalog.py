"""
lacuna.survey.survey_catalog

The curated GENUINE survey catalog for Lacuna-Survey training (MASTER §8; DATA-INVENTORY §0;
Stage-1 spec §2).

ONE job: name the survey datasets eligible for the role-B supervised δ corpus, EXCLUDING the three
non-survey contaminants (cars93, computers, survey — DATA-INVENTORY §1), and fail loud if the δ path
is ever asked to use a contaminant. Also tags each dataset with its domain + source block, for the
block-aware leave-DOMAIN-out split (cross-domain plan §3): same-block datasets must move together
across train/test to avoid same-respondent leakage.

Pure constants + guards; no I/O, no RNG.
"""

# The 3 non-survey tables that pollute the survey-manifold prior (PI decision 2026-06-06).
CONTAMINANTS = frozenset({"survey_cars93", "survey_computers", "survey_survey"})

# The 9 genuine survey datasets currently projected to role-B (complete-case) bases.
GENUINE_SURVEYS = (
    "survey_bfi",          # psychology (Big-Five Likert)
    "survey_chile",        # political attitudes
    "survey_cps1985",      # labor/income
    "survey_cps1988",      # labor/income
    "survey_hmda",         # finance
    "survey_psid1976",     # labor/income
    "survey_psid7682",     # labor/income
    "survey_workinghours", # labor/income
    "survey_yrbss",        # health-risk behavior
)

# Domain tag per dataset (for the cross-domain learning curve; MASTER §8).
DOMAIN = {
    "survey_cps1985": "labor", "survey_cps1988": "labor", "survey_psid1976": "labor",
    "survey_psid7682": "labor", "survey_workinghours": "labor",
    "survey_bfi": "psychology", "survey_chile": "political", "survey_hmda": "finance",
    "survey_yrbss": "health",
}

# Source block per dataset (independent surveys = their own block; projected NHANES/ESS bases will
# share a block — added by the role-B projection). Block-aware splits keep a block together.
SOURCE_BLOCK = {name: name for name in GENUINE_SURVEYS}


def assert_genuine(name: str) -> str:
    """Return `name` if it is a genuine survey; fail loud if it is a known contaminant or unknown."""
    if name in CONTAMINANTS:
        raise ValueError(
            f"{name!r} is a non-survey contaminant excluded from the Lacuna-Survey δ corpus "
            f"(DATA-INVENTORY §1); it must not enter the supervised stream"
        )
    if name not in GENUINE_SURVEYS:
        raise ValueError(f"{name!r} is not a genuine survey dataset; known: {GENUINE_SURVEYS}")
    return name


def load_genuine(catalog, name: str):
    """Load a genuine survey RawDataset through the catalog, guarding against contaminants."""
    return catalog.load(assert_genuine(name))


def domain_of(name: str) -> str:
    """Domain tag for a genuine survey (fail loud if unknown)."""
    assert_genuine(name)
    return DOMAIN[name]
