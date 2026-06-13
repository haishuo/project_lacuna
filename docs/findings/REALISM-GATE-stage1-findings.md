# Realism Gate — Stage 1 findings (L1 footprint gate, first sweep)

**Date:** 2026-06-13
**Branch:** `p2/delta-prior-rearchitecture`
**Code:** `lacuna/analysis/realism_gate/` (harness, committed before results) +
`scripts/run_realism_gate.py`. Raw results: `runs/realism_gate_{ess,nhanes}.json`.
**Spec executed:** `docs/proposals/GENERATOR-DESIGN-charter.md` §6 (L1 gate) /
`docs/findings/GENERATOR-AUDIT.md` §5 step 1.

---

## 0. Headline

**The existing 113-generator family (`lacuna_tabular_110`) fails the L1 realism
gate universally: 0/113 pass against ESS, 0/113 pass against NHANES.** The dominant
blocker is **cross-column co-missingness** — real survey nonresponse is strongly
correlated across items (whole modules/dispositions go missing together), and *no*
current generator reproduces it. The secondary blocker is **rate (saturation /
under-production)**: most generators are parameterised for standard-normal X and
over- or under-produce on real survey scales. This is exactly the gap the audit
predicted, now measured against real masks for the first time.

This is a *finding about the generators*, not a defect in the gate. It is the
expected precondition the reformulation must clear before any coverage claim
transports, and it concretely prioritises the next build step (anti-saturation +
co-missingness fitting, charter §9 step 3 / audit §5 steps 2 & 5).

## 1. What the gate measures (and the honest direction of the claim)

For each generator we inject missingness into a corpus's **complete-case real-X
substrate** (`generator.apply_to`), then compare the resulting masks to the real
masks on the **same column set**:

- **C2ST** (Lopez-Paz & Oquab 2016): a block-aware (leave-country-out for ESS)
  classifier two-sample test. Pooled OOF AUC ≈ 0.5 ⇒ indistinguishable. We report
  the exact finite-sample binomial p-value, but with tens of thousands of rows it
  is ~0 for any real difference, so the **effect size (AUC), not the p-value, is
  the practical gate.**
- **Fidelity panel** (Dankar 2022): attribute (per-column rate), bivariate
  (co-missingness correlation), population (run-length / row-count distributions),
  plus an Alaa (2021) authenticity guard (vacuous for the current generators — they
  never see real masks; load-bearing only for the future *learned* generator).
- **Verdict (conjunctive, charter §6.1 "the gate is the panel"):** PASS iff
  `AUC < 0.60` **and** `attribute gap ≤ 0.05` **and** `bivariate gap ≤ 0.30`. The
  conjunction is necessary: a generator that injects almost no missingness slides
  under an AUC threshold while badly missing the real rates and co-missingness —
  the fidelity tolerances catch that degeneracy (it did occur on NHANES, see §3).

**Direction caveat (load-bearing).** A near-chance C2ST **certifies realism only in
the falsification direction** — it fails to *refute* realism at the achieved power;
it does not *certify* it (Barber et al. 2022: coverage gap ≤ divergence, but finite
power only bounds *detectable* divergence). See
`docs/findings/LITREVIEW-conformal-under-shift.md`. Here it does not bite: every
generator is *falsified*, so the weaker direction is moot — but it governs how a
future PASS must be worded ("not falsified at this power", never "certified real").

## 2. Setup

| corpus | source | block | d | real rate | substrate (complete-case rows) |
|---|---|---|---|---|---|
| ESS R11 | `ESS11e04_1.csv`, width-correct sentinels (`ess_codes.py`) | country (30, leave-country-out) | 12 | 0.086 | 27,634 |
| NHANES J | `INQ/DPQ/DUQ/WHQ_J.xpt`, merged on SEQN | one cycle (no block ⇒ random K-fold, **flagged**) | 12 | 0.065 | 7,990 |

Columns are selected deterministically: missing rate in [0.01, 0.40], closest to a
target footprint rate (0.08), most-complete-case-friendly. This evaluates the
**moderate-rate regime**; the very-high-refusal columns (income) whose substrate
needs imputation are a documented follow-up, not silently included. (Note NHANES
landed on the INQ income module, which is why its real co-missingness is extreme.)

## 3. Results

| metric | ESS | NHANES |
|---|---|---|
| **PASS** | **0 / 113** | **0 / 113** |
| C2ST AUC (min / median / max) | 0.592 / 0.886 / 1.000 | 0.565 / 0.935 / 1.000 |
| within attribute tol (≤0.05) | 2 / 113 | 2 / 113 |
| within bivariate tol (≤0.30) | **0 / 113** | **0 / 113** |
| rate over-produces (>1.5× real) | 57 | 60 |
| rate under-produces (<0.5× real) | 18 | 24 |
| family AUC medians (MCAR/MAR/MNAR) | 0.81 / 0.98 / 0.83 | 0.88 / 0.98 / 0.85 |

**Failure mode 1 — co-missingness (the universal blocker).** Real ESS masks have
co-missingness correlation up to **0.964** (4 of 66 pairs > 0.5); NHANES income
items are near-perfectly co-missing (**median 0.92, all 66 pairs > 0.5** — the whole
module is skipped together). Every current generator produces **near-independent**
masks (max co-missingness ≈ 0.00–0.02). So the bivariate gap is ≥ 0.57 (ESS) / ≥
0.76 (NHANES) for *every* generator — **0/113 within tolerance on either corpus.**
This is the structural signature of unit/section/disposition nonresponse that the
generators were never built to model (they punch holes per-cell, near-independently).

**Failure mode 2 — rate / saturation.** Only 2/113 land within the attribute-rate
tolerance. The MAR-logistic and MNAR-latent families saturate worst (latent →
overall rate 0.984; MAR logistic/probit/poly → AUC ~0.98 even near the right
marginal because their *conditional-on-observed* structure is highly detectable).
The hard-percentile MNAR families (Q70–Q90, detection, social) **under**-produce
(rate 0.01–0.05) — fixed percentile thresholds clip only a tiny tail of real bounded
ordinal scales. This is the documented saturation/scale problem, now quantified.

**MAR is the most separable family** (median AUC 0.98 both corpora): even at a
matched marginal rate, its dependence on other observed columns is a strong,
easily-detected fingerprint that real ESS/NHANES missingness does not share at that
strength — consistent with the Stage-2 finding that real refusal is ~0.90
covariate-predictable but not via these hand-set MAR weights.

**The NHANES degeneracy the conjunction caught.** Under an AUC-only verdict, 8
NHANES generators "passed" — all by injecting ~0 missingness (rate 0.000–0.001),
sliding under AUC 0.60 on the weaker (no-block) NHANES fold while missing attribute
(0.11) and bivariate (0.95) fidelity entirely. The conjunctive verdict correctly
fails them. (Lesson recorded: AUC alone is not the gate; the panel is.)

## 4. What this means for the build

1. **Co-missingness must be fitted from real data** (charter §9.3 / audit §5.5,
   Radosavljević method): learn the block / disposition / section structure
   (correlated drop) from ESS/NHANES and inject correlated masks. Until then, no
   generator can clear L1 — this is now the **highest-priority** generator fix,
   above the rate fix.
2. **Universal anti-saturation** (audit §5.2): lift `_zscore_columns` to a base
   guarantee and add a uniform `target_rate` calibration wrapper so every generator
   hits a specified rate on given X (bisection on the intercept/threshold). This
   clears failure mode 2 and makes matched-rate a generator property, not a harness
   hack.
3. **Re-run this gate as the acceptance test** after each fix; the harness reports
   per-column localisation (C2ST coefficients + rate gaps) to target the work.

## 5. Scope, caveats, honest limits

- **Falsification direction** (§1): these are refutations; a future PASS is "not
  falsified at this power", never "certified real".
- **NHANES is not block-aware** (single cycle J ⇒ random K-fold) — a weaker test
  than ESS leave-country-out; flagged in the JSON. Leave-instrument-out across
  NHANES cycles is a follow-up.
- **GSS not yet wired** (597 MB `.dta`, typed `.r/.d/.n/.i` codes via pyreadstat
  `user_missing`) — a separate loader, the obvious next corpus. The harness is
  generic; adding GSS is one `load_gss_corpus` function.
- **Moderate-rate regime only** (§2): the income-type high-refusal columns need an
  imputed substrate to enter the gate (their complete-case projection is empty).
- **One verdict tolerance set** (AUC 0.60 / attr 0.05 / bivar 0.30): chosen before
  seeing per-generator detail to make the gate faithful to §6.1; the result (0/113)
  is robust to any reasonable loosening because the bivariate gap floor is ≥ 0.57.

## 6. Bottom line

The realism-gate harness works (positive control: a rate- and structure-matched
Bernoulli scores AUC 0.495, PASS; saturated/degenerate generators are FAILED). Its
first verdict on the real corpora is unambiguous and was expected: **the current
generator family is not realistic at L1 — universally — because it lacks real
survey co-missingness and saturates on real scales.** The reformulation's coverage
guarantee cannot transport on this family as-is. Next: fit co-missingness + uniform
rate calibration, then re-gate. No generator should be used for certification until
it clears this gate.
