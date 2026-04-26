# Data acquisition guide for Lacuna-Survey calibration anchors

**Goal**: expand the calibration corpus from current `1 MNAR + 7 MAR + 0 MCAR`
to roughly `5 MNAR + 5 MAR + 5 MCAR`. Each new anchor reduces structural
bias in the deployed calibration.

For each candidate below, the entry gives:
- **What to download** (specific dataset / module)
- **Where** (URL, registration requirements)
- **Format** (CSV / SAS / SPSS / Stata — and how to convert if needed)
- **Estimated effort** (minutes for direct download, hours for restricted access)
- **What to extract** (specific variables / items)
- **Consensus mechanism** (the textbook reading you'd label it as)

After downloading, run:
```bash
python -m lacuna_survey.import_anchor path/to/downloaded.csv --slug <name> --max-rows 500
# Add a row to lacuna_survey/anchors.py with the consensus label + citation
python -m lacuna_survey.calibrate
python -m lacuna_survey.diagnostic --calibrated --ood
```

---

## MNAR candidates (top priority — we have only 1)

### 1. NHANES sensitive-item modules

**What to download**: NHANES questionnaire items that record sensitive
information — drug use (DUQ), sexual behavior (SXQ), alcohol (ALQ), or
mental health (DPQ). Documented refusal patterns are MNAR by NHANES
methodology reports (NCHS Series 2 reports).

**Where**:
- Direct from CDC: <https://wwwn.cdc.gov/nchs/nhanes/Default.aspx>
- Pick a cycle (e.g. 2017–2018), download the `.XPT` file for the
  module of interest. Free, no registration.
- Alternatively: HuggingFace `nguyenvy/cleaned_nhanes_1988_2018` —
  the file `comments_clean.csv` (180 MB) merges sensitive items
  across cycles. Already on Forge if you ran the demographics fetch
  (`/mnt/data/lacuna/nhanes/`).

**Format**: SAS XPT (CDC) → use `pyreadstat`
(`pip install pyreadstat`); or CSV (HuggingFace) — direct.

**Effort**: ~30 min if you already have the HuggingFace file. ~1 hr
if downloading per-cycle XPT files.

**What to extract**: 200–500 rows × ~10 numeric items from one of
DUQ010 (ever used marijuana), DUQ240 (ever used cocaine), SXQ021
(ever had sex), ALQ110 (12+ drinks lifetime), DPQ020 (felt down).
Keep rows where demographics are observed; the NaN you want is on
the sensitive items themselves.

**Label**: MNAR

**Citation**: Allison 2001 §6.4; NCHS Continuous NHANES methodology
reports.

---

### 2. NLSY97 income / wealth supplements

**What to download**: NLSY97 (National Longitudinal Survey of Youth
1997) income and wealth items. The wealth supplement has documented
high-earner refusal patterns.

**Where**: <https://www.nlsinfo.org/investigator/>

- Free with registration (NLS Investigator login)
- Use the variable picker to select INCOME / NET WORTH variables
- Export as CSV (or Stata, then convert)

**Format**: CSV download direct from Investigator UI.

**Effort**: ~2 hours (registration + variable selection + download).
The NLS Investigator UI is clunky but free.

**What to extract**: Round 16+ income (`INC_REGULAR_JOB`), wealth
(`NET_WORTH`), specific refusal-prone items. Export 1000–2000 rows
of those columns.

**Label**: MNAR

**Citation**: Allison 2001 §1.2 cites NLSY income refusal as the
canonical economic-MNAR example.

---

### 3. HRS (Health and Retirement Study) wealth questions

**What to download**: HRS Core wealth and income items from any
recent wave (2018, 2020). Wealth-question refusal is well-documented
MNAR.

**Where**: <https://hrs.isr.umich.edu/data-products>

- Free with registration (University of Michigan ISR)
- "RAND HRS Longitudinal File" is the most accessible — already
  cleaned and merged across waves
- Stata / SAS / SPSS / CSV all available

**Format**: CSV preferred. The RAND file is large (multi-GB) — stream
or use the per-wave files.

**Effort**: ~2–3 hours (registration + RAND file download + Stata-to-
CSV conversion if needed).

**What to extract**: 500 rows × wealth columns (`H{wave}ATOTB`,
`H{wave}ATOTW`, etc.) plus demographics for context. NaN on wealth
items is the MNAR signal.

**Label**: MNAR

**Citation**: Juster & Smith 1997, "Improving the quality of economic
data: lessons from the HRS and AHEAD".

---

## MCAR candidates (we have zero — 1 high-priority)

### 4. PISA student questionnaire (rotated booklet)

**What to download**: PISA 2018 or 2022 student questionnaire data.
Rotated booklet design = each student gets a random subset of items.
The missingness on items NOT IN A STUDENT'S BOOKLET is MCAR by
design.

**Where**: <https://www.oecd.org/pisa/data/>

- Free, no registration for downloads
- Download "Student questionnaire data file" SPSS (.sav) or SAS
- The 2018 release is fully public; 2022 has a delayed-public schedule

**Format**: SPSS `.sav` or SAS XPT → use `pyreadstat`:
```python
import pyreadstat
df, meta = pyreadstat.read_sav("CY07_MSU_STU_QQQ.sav")
```

**Effort**: ~1–2 hours. The file is ~600 MB; you only need a subset
of 500 students × ~20 items.

**What to extract**: Pick a few item-clusters that vary across booklets
(e.g. `ST036*` math attitude, `ST157*` reading enjoyment). The
booklet rotation produces structural NaN that's MCAR-by-design.

**Label**: MCAR (planned-missing rotated booklet design)

**Citation**: OECD 2019, "PISA 2018 Technical Report", §8 (rotated
booklet design).

---

### 5. NAEP three-form design

**What to download**: NAEP (National Assessment of Educational
Progress) student-questionnaire files. Three-form planned-missing
design assigns each student to one of three subsets of items.

**Where**: <https://nces.ed.gov/nationsreportcard/researchcenter/datatools.aspx>

- Free, no registration for restricted-use license? Actually, NAEP
  *student-level* files require a restricted-use data license
  application (~6-week turnaround, free for academic researchers).
- Public-use NAEP (NDE Online) gives aggregate stats, not item-level
  data — not useful for our purpose.
- **Alternative**: ECLS-K 1998 / 2010 (Early Childhood Longitudinal
  Study Kindergarten cohort) has similar rotated-form designs and
  the public-use file is direct download.
  <https://nces.ed.gov/ecls/dataproducts.asp>

**Format**: SAS or Stata.

**Effort**: ECLS-K public-use ~1 hr. NAEP restricted-use ~6 weeks
of waiting + research-plan paperwork.

**Label**: MCAR (three-form / rotated planned-missing)

**Citation**: Graham et al. 2006, "Planned missing data designs in
psychological research", *Psychological Methods*.

---

### 6. ESS module randomization (European Social Survey)

**What to download**: ESS Round 9 or 10 cumulative file. Each
respondent gets a randomly-assigned rotating module (out of 2-3
options); the rotated module's items are MCAR-by-design for
respondents who didn't get assigned to it.

**Where**: <https://ess.sikt.no/en/>

- Free with registration (one-time, instant approval)
- Direct CSV / SPSS / Stata download

**Format**: CSV available directly.

**Effort**: ~1 hour.

**What to extract**: The rotating-module items for one round. Filter
to one country (e.g. Germany, ~3000 respondents) for tractable size.

**Label**: MCAR (random module assignment)

**Citation**: ESS Round Documentation, "Module Randomization" section.

---

## Additional MAR candidates (lower priority — we have 7)

### 7. Pew Research Center datasets

**Where**: <https://www.pewresearch.org/our-methods/u-s-surveys/data-downloads/>

- Free, no registration for archived datasets (~1 yr after release)
- Direct CSV / SPSS / Stata download
- Many polls have item-nonresponse documented as MAR

**Effort**: ~1 hr per dataset.

**Label**: MAR (item nonresponse driven by demographics).

---

### 8. ANES (American National Election Studies)

**Where**: <https://electionstudies.org/data-center/>

- Free with registration
- Time Series Cumulative Data File or single-year files
- CSV / Stata / SPSS

**Effort**: ~2 hrs.

**Label**: MAR.

---

## What you actually need (priorities)

For *balanced* calibration, the highest-leverage additions are:
- **2 more MNAR anchors** (NHANES sensitive items + NLSY income or HRS wealth)
- **2 MCAR anchors** (PISA rotated booklet + ESS module randomization)

Total expected effort: ~6–8 hours of manual data work. The framework
absorbs each anchor in <1 minute of code (`import_anchor` + edit
`anchors.py` + refit `calibrate`).

If grant funding becomes available, a research assistant could
reasonably handle this in a week including:
- Verifying consensus mechanism per dataset against literature
- Producing citation files
- Cross-checking the labeled corpus against a missing-data textbook
  authority (Allison 2001 / van Buuren 2018 / Schafer 1997)

## Storage policy

- Training X-bases: `/mnt/data/lacuna/raw/survey_*.csv` on Forge
  (gitignored). Never travels to client machines.
- Evaluation anchors: `lacuna_survey/evaluation_data/survey_*_real.csv`
  (gitignored, but small enough to copy to Mac if you want to verify
  predictions). Currently 7 files, ~50 KB total.
- Large source files (e.g. NHANES 122 MB demographics) live in
  `/mnt/data/lacuna/<source>/` on Forge only. Subsets get extracted
  to evaluation_data/ for the calibration corpus.

## Workflow once new data arrives

```bash
# 1. Validate and stage the CSV
python -m lacuna_survey.import_anchor /path/to/downloaded.csv \
    --slug nhanes_drug_use --max-rows 500

# 2. Edit lacuna_survey/anchors.py to add the row with the
#    consensus mechanism label + citation

# 3. Refit calibration
python -m lacuna_survey.calibrate

# 4. Re-run diagnostic to see the new state
python -m lacuna_survey.diagnostic --calibrated --ood
```
