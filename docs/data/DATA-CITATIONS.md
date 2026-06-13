# Data Citations — sources actually used by Lacuna

*Every external dataset used in experiments must be cited here, with the on-disk path and the
role it plays. Add an entry BEFORE first experimental use.*

## European Social Survey Round 11

European Social Survey European Research Infrastructure (ESS ERIC) (2026) *ESS11 - integrated
file, edition 4.1* [Data set]. Sikt - Norwegian Agency for Shared Services in Education and
Research. https://doi.org/10.21338/ess11e04_1

- On disk: `/mnt/data/lacuna/incoming/ESS11e04_1.dta` / `.sav` (labels);
  `/mnt/data/lacuna/rejected/ESS11e04_1.csv` (coded values; same edition 4.1)
- Role: real documented item-nonresponse labels (refusal 77-family / don't-know 88-family /
  no-answer 99-family / not-applicable 66-family) + variable label text. Used in the
  real-missingness Stage 1/2 showdown (`docs/findings/REAL-MISSINGNESS-stage1-findings.md`) and the
  semantic text→behavior corpus.

## NHANES 2017–2018 (cycle J)

Centers for Disease Control and Prevention (CDC), National Center for Health Statistics (NCHS).
*National Health and Nutrition Examination Survey Data, 2017–2018* (DEMO_J, DPQ_J, DUQ_J,
INQ_J, WHQ_J). Hyattsville, MD: U.S. Department of Health and Human Services, CDC.
https://wwwn.cdc.gov/nchs/nhanes/

- On disk: `/mnt/data/lacuna/incoming/{DEMO,DPQ,DUQ,INQ,WHQ}_J.xpt`;
  codebooks `/mnt/data/lacuna/incoming/codebooks/*_J.htm`
- Role: role-B projection bases (weight/poverty); refusal/DK codes (7777/9999-family); question
  text for the semantic corpus (`/mnt/data/lacuna/role_b/nhanes_text_corpus.csv`).

## Survey of Consumer Finances 2022

Board of Governors of the Federal Reserve System. *2022 Survey of Consumer Finances, Summary
Extract Public Data*. https://www.federalreserve.gov/econres/scfindex.htm

- On disk: `/mnt/data/lacuna/incoming/scf2022_summary/rscfp2022.dta` (sha256 3bb4d890…, see
  `docs/proposals/PROPOSAL-SCF-wealth-acquisition-plan.md`)
- Role: role-B wealth base `rb_scf2022_wealth_cont` (implicate 1 only; role-B-only — public file
  imputes item nonresponse).

## General Social Survey 1972–2024

Davern, Michael; Bautista, Rene; Freese, Jeremy; Herd, Pamela; and Morgan, Stephen L. *General
Social Survey 1972–2024 Cumulative File* (Release 3, March 2026) [Data set]. Chicago: NORC at the
University of Chicago. https://gss.norc.org/get-the-data

- On disk: `/mnt/data/lacuna/incoming/gss7224_r3.dta` (+ `GSS 2024 Codebook R3.pdf`,
  from `gss_stata.zip`, downloaded 2026-06-12, no registration required)
- Role: third instrument for the semantic channel — item label text + EXPLICITLY TYPED
  missingness (Stata extended missing .r/.d/.n/.i/.s = refused / don't know / no answer /
  IAP-routing / skipped; no sentinel inference needed). Corpus:
  `/mnt/data/lacuna/role_b/gss_text_corpus.csv`.

## PISA 2018 / 2022 (downloaded, not yet used in experiments)

OECD. *Programme for International Student Assessment (PISA) 2018 and 2022 student questionnaire
data files*. https://www.oecd.org/pisa/data/

- On disk: `/mnt/data/lacuna/incoming/cy07_msu_stu_qqq.sas7bdat` (2018),
  `CY08MSP_STU_QQQ.SAS7BDAT` (2022) + SAS format catalogs
- Role: deferred (requires SAS extraction); intended role-B education domain.

## Classic survey extracts (v1.0-era training corpus)

The `survey_*` bases under `/mnt/data/lacuna/raw/` derive from public R package datasets
(AER/Ecdat: PSID 1976/7682, CPS 1985/1988, HMDA, working hours; psych: bfi; carData: Chile;
plus yrbss, GSS vocab extracts). Cite the originating packages/studies in any publication that
uses them; provenance recorded in `lacuna_survey/` and `docs/data/DATA-INVENTORY-ground-truth.md`.
