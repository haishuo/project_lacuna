# PROPOSAL — The Real-Missingness Showdown + The Learned Generator (close-the-book spec)

*Companion to `docs/findings/T-review-findings.md` and `docs/proposals/PROPOSAL-semantic-channel-pivot.md`. Status:
SPEC ONLY — nothing built, downloaded, or run. Purpose: settle, on REAL documented missingness
(not our generators), the one question the synthetic T-review could not — does a trained network
beat enumerable statistics where the complexity is real and we did not author it — and specify the
learned-generator ("pseudo-semi-synthetic") training amplifier that may itself be the load-bearing
neural contribution. If adopted, the locked sections become a pre-registration in the 7714f0c
discipline.*

---

## 0. Why this exists (the cat argument, steelmanned and split)

The T-review verdict (fail/fail/fail) was measured on CLEAN mathematical generators. The standing
objection: a logistic regression that matches a network *on our generators* has only been shown
equal *within our synthetic universe* — real missingness is a tangled mixture (value-dependence +
interface + fatigue + language + glitch) whose footprint may be high-dimensional in ways 37
statistics miss and a network captures. The objection is correct in general and must be tested on
real labels. It splits cleanly along the detectability dichotomy:

- **Flat idioms (own-value self-censoring):** the objection FAILS. On these, the truth-access
  ORACLE scores ~1.0 while BOTH the observed-data network AND LR floor together at ~0.68 — the
  signature of information absent from the observed data, not of LR being coincidentally
  sufficient. Real-world messiness ADDS MAR/MCAR noise on top of an already-flat likelihood ⇒
  strictly harder, never easier. This is Molenberghs, not generator-simplicity. Not retested here.
- **Detectable idioms (top-coding, skip-logic, MCAR, refusal-vs-DK):** the objection HAS TEETH.
  Signal genuinely exists in the observed data; we only ever tested clean threshold/gate
  generators; real versions are messy (soft brackets, digit heaping, routing errors, partial
  compliance). And — decisively — **real labels EXIST for exactly this subset** (documented
  topcodes, printed skip logic, item-level refusal codes). This is the testable corner. This
  document specifies the test.

**The escapability principle:** the cat fallacy is escapable only where real labels exist. They
exist for the detectable subset and not for the flat subset. So this is simultaneously the most
the network can be asked to do on real data and the most it can be honestly credited for.

---

## 1. PART A — The real-missingness detector showdown (locked design)

### 1.1 Task

Real survey columns, **variable names and text stripped** (pure matrix + mask, the canonical
no-metadata view), pooled across many instruments. Per column, classify the missingness footprint
into a documented class:

| class | real-data definition | documented ground truth |
|---|---|---|
| ordinary / MAR-ish | item-nonresponse with no censoring/gate/refusal signature | default (no sentinel, no gate, no topcode) |
| top-coded / censored | observed values capped/bracketed at a published threshold | codebook topcode (e.g. NHANES INDFMPIR capped at 5; CPS/ACS topcodes) |
| skip-gated (MAR-structural) | missingness deterministically routed by an observed gate item | printed skip logic (NHANES DPQ/DUQ gates; PISA routing) |
| refusal (MNAR-suspected) | item-level **refused** code, distinct from don't-know / not-applicable | ESS 77/88/99 + 7/8/9 + 7777/8888/9999; NHANES 7777/9999 |

The refusal task is the sharp one: **strip the sentinel codes** (recode all to plain missing),
then ask whether the observed-data footprint alone recovers which masked cells were *refusals* vs
*don't-knows* vs *skips*. Ground truth = the held-out codes. This is the pure matrix-channel
question on real labels: if features win, the matrix channel is exhausted even on real data; if the
network wins, it is load-bearing on real complexity.

### 1.2 Arms (frozen before any network result, anti-handicap as in T1)

- **Feature null:** the full frozen ceiling (17 consequence + 16 H/S/R no-truth + 37 topology,
  per column where applicable) + LR and GBM, better-of. Assembled maximally.
- **Network:** the design-of-record column-φ + mask-topology stream (the build the prereg already
  authorized), one fixed config, 5 seeds. No hand-fed statistics.

### 1.3 Splits, metric, bar

- **Block-aware leave-INSTRUMENT-out** (ESS = one block; each NHANES cycle = one block; PISA cohort
  = one block — same-respondent / same-survey items never cross train/test).
- **Paraphrase/near-duplicate dedup** across instruments before splitting (recurring standard items
  leak otherwise).
- Confound covariates (item position, response-option count, module size, mode where coded)
  included in ALL arms so they cannot masquerade as learned signal.
- Primary: pooled OOF macro-AUC over the documented classes, per held-out instrument.
- **Locked bar:** network (mean − SE over 5 seeds) beats the better shallow arm by **≥ +0.05**
  pooled OOF macro-AUC on held-out instruments. (Margin to be frozen at this value unless the PI
  sets otherwise before any run.) Recorded prior the network clears it: **~30–40%** — lower than
  even odds, because every prior measurement says detectable footprints are low-dimensional; but
  UNTESTED on real data, which is the whole point.

### 1.4 What each outcome means (close-the-book)

- **Network wins (≥ +0.05 on real held-out instruments):** the network is load-bearing on real
  complexity that statistics cannot enumerate — a genuine representation-learning result on real
  labels, not table-lookup. Lacuna lives, on real data, with the strongest possible claim.
- **Parity:** the matrix channel is exhausted *on real labels*, not merely on our generators — the
  cat objection is answered and closed. The verdict is then airtight, not scope-limited, and the
  honest path is the semantic channel (`PROPOSAL-semantic-channel-pivot.md`) or sunset.

Either way the book closes, because the labels are real and not authored by us.

---

## 2. PART B — The learned generator ("pseudo-semi-synthetic"): the training amplifier that may BE the thesis

### 2.1 The idea (PI's), stated precisely

Instead of hand-coding mechanisms (`topology_generators.py`: clean threshold, clean gate, clean
own-value), **learn** a generative model of real missingness from real data that carries
ground-truth labels, then sample it to produce a near-limitless, realistically-messy, **labeled**
training corpus. Real data is reserved for validation/test (Part A); the generator only amplifies
training. This directly attacks the cat critique: the training mechanisms become real-footprint, not
clean-math.

### 2.2 THE TRAP (must be designed around, not discovered later)

A generator trained to reproduce the **observed missingness distribution** learns P(mask | X). But
mechanism labels are **not a function of the observed distribution** — that is exactly Molenberghs
(same observed law, different mechanisms). So an *unconditional* "realistic-missingness GAN"
produces patterns you cannot truthfully label ⇒ unlimited training data with unidentified labels ⇒
worse than useless. The generator cannot manufacture identification absent from its source.

**The fix — CONDITIONAL generation on documented labels.** Train on real triples
`(X, mask, mechanism_label)` where the label is DOCUMENTED (refusal code / skip structure / topcode
threshold — i.e. exactly the Part-A detectable subset). Learn `G(mask | X, label)`. Sample
`label ~ balanced`, `X ~ real columns`, generate `mask` ⇒ the label is KNOWN because you conditioned
on it. This yields balanced, realistically-messy, correctly-labeled training data — a learned,
data-grounded, auditable replacement for the hand generators. **It works exactly where Part A works
(the detectable, documented subset) and fails exactly where the flat idioms are, for the same
identifiability reason.** No overclaim is possible.

### 2.3 The one crack toward the flat idioms: bracket / follow-up δ ground truth

Own-value censoring has no per-cell documented label on real data (that IS unidentifiability) —
EXCEPT where **bracket follow-ups** exist: surveys that ask refusers "is it above or below X?"
(SCF, CPS-ASEC) yield a measured refuser-vs-respondent value gap = real δ. This does NOT make
own-value detectable from observed data (still floored), but it lets the generator be **calibrated
to real δ magnitudes** instead of our arbitrary δ-grid, and it supplies the cited ledger anchors the
semantic-channel δ-prior needs. High-value, harder-to-get; flagged, not required for v1.

### 2.4 Why this is the load-bearing neural contribution (the dissertation point)

If the learned conditional generator produces training data that **measurably improves real
held-out detection (Part A) versus the hand-coded generators**, then the GENERATOR is a trained,
non-decorative neural network whose value is demonstrated on real labels:

> *A generative model of real survey missingness, grounded in documented ground truth, that
> produces training data improving mechanism detection on real held-out surveys, relative to
> hand-specified mechanisms.*

That is a method contribution (a novel conditional generative model for tabular missingness), it is
falsifiable (beat hand generators on real test, or not), it satisfies the trained-network
requirement non-trivially, and Part A is its built-in evaluation harness. This is plausibly a
cleaner ML thesis than the detector itself — the detector tests the question; the generator IS the
answer, and it is a network you trained.

### 2.5 Architecture honesty (GAN vs alternatives)

"GAN" names the idea; for tabular value+mask data a vanilla GAN is the *worst* concrete choice
(instability, mode collapse, mixed-type discretes). For a dissertation, prefer auditable conditional
generators and report the comparison:
- **Conditional VAE / conditional diffusion over masks** — stable, likelihood-ish, samples diverse
  footprints conditioned on (X, label).
- **Learned mechanism-parameterization** — fit the *distributions of mechanism parameters*
  (threshold location/softness, gate strength, refusal-propensity-by-covariate, heaping) from real
  documented columns, then sample a hand-form generator with learned parameters. Least black-box,
  most auditable, strongest "we learned the real footprint" story; a natural baseline the deeper
  generators must beat.
- A conditional GAN may be included as one arm, not the headline, and only if it beats the
  cVAE/parameterization on realism metrics.

### 2.6 Realism + leakage discipline (locked)

- Generator trained ONLY on train-split instruments (leave-instrument-out preserved end-to-end);
  generated data must never see test-instrument footprints.
- Realism gates before any downstream use: (a) a real-vs-generated discriminator should be near
  chance on held-out real columns; (b) generated footprint statistics (spike fractions, gate
  AUCs, heaping, refusal-by-covariate curves) must match held-out real distributions within
  tolerance; (c) report failures, never silently pass.
- The downstream claim is strictly: train-on-generated → test-on-REAL-held-out ≥ train-on-hand-
  generators → test-on-REAL-held-out, by a locked margin.

### 2.7 Honest joint failure branch

If real detectable footprints are genuinely low-dimensional (the standing prior), then Part A shows
features win AND the hand generators already capture the footprint, so the learned generator adds
nothing downstream — **both fail together**, and that convergence is itself the airtight close-the-
book result (the matrix channel is exhausted on real labels, by two independent routes). Recorded
prior the generator beats hand-generators on real test: **~35%**.

---

## 3. Data inventory — local / algorithmic / manual

### 3.1 ALREADY LOCAL (showdown needs zero downloads)

| asset | path | gives |
|---|---|---|
| ESS R11 | `/mnt/data/lacuna/rejected/ESS11e04_1.csv` (691 cols, ~37% NaN) | item-level refusal/DK/NA codes (77/88/99, 7/8/9, 7777/8888/9999) across ~75–94 cols — richest real refusal labels |
| NHANES 2017–18 raw | `/mnt/data/lacuna/incoming/{DEMO,DPQ,DUQ,INQ,WHQ}_J.xpt` (+ `DUQ_J.htm` codebook, `nhanes_dpq_phq9.csv`) | refusal codes (7777/9999), skip-gated modules (DPQ/DUQ), income top-coding (INQ), self-report (WHQ) |
| NHANES role-B + role-A | `/mnt/data/lacuna/role_b/`, `/role_a/` | INDFMPIR capped@5 = documented topcode; natural-missingness archives |
| PISA 2018/2022 | `/mnt/data/lacuna/incoming/CY07*.sas7bdat (3.5GB), CY08*.SAS7BDAT (4GB)` + FORMAT.SAS catalogs | education routing/skip logic, large N — needs SAS extraction (heavy, deferrable) |
| SCF 2022 | `/mnt/data/lacuna/incoming/scf2022_summary/rscfp2022.dta` (+ zip) | wealth; bracket-follow-up route for real δ (public file imputed — verify bracket fields) |

ESS + NHANES alone = ≥2 independent instrument blocks with all four documented classes ⇒ a valid
leave-instrument-out showdown TODAY. **No downloads, no logins, no acquisition gate for Part A v1.**

### 3.2 ALGORITHMIC (free, no login — nice-to-have for more top-coding domains)

- **NHANES other cycles/modules:** CDC publishes `.xpt` + codebooks at fixed URLs
  (`wwwn.cdc.gov/Nchs/Nhanes/...`) — fully scriptable, no key.
- **Census CPS-ASEC / ACS** (published income topcodes — the canonical real top-coding): Census
  Data API is scriptable and free; needs a free instant API key (no human approval). Adds genuine
  top-coding domains beyond NHANES.
- **ESS codebook** (exact per-variable sentinel map): public, scriptable from the ESS data portal.
- **PISA/OECD, SCF/Fed:** public, scriptable (already local).

### 3.3 MANUAL (registration / extract-builder — DEFERRED, not needed to close the book)

- **IPUMS-CPS/USA:** free account + interactive extract builder (semi-manual) — only if we want
  harmonized multi-year income topcodes beyond the Census API.
- **HRS / PSID restricted follow-ups** (richest bracket/linkage δ truth): registration-gated —
  only pursue if §2.3 real-δ calibration becomes the headline.

---

## 4. Sequencing (gated on PI/advisor sign-off; nothing authorized yet)

1. **Codebook curation** (read-only, ~days): extract `(X, mask, documented-label)` from ESS +
   NHANES with per-variable sentinel maps (the lesson stands: codebook-driven, never heuristic —
   age-77 ≠ refuse-77). Deliverable: labeled real-missingness corpus + block map + dedup report.
2. **Part A showdown** (frozen arms, 5 seeds, leave-instrument-out) → the real-data verdict.
3. **Part B generator** (cVAE / learned-parameterization first; realism gates; train-on-generated
   → test-on-real) → does a learned generator beat hand generators on real test; is the generator
   the load-bearing network.
4. Verdict computed mechanically from §1.3 / §2.6 bars. Book closed either way.

*Spec only. This is the experiment the cat argument demands and the project has not run: real
documented missingness, real labels, frozen-feature null, leave-instrument-out — plus the learned
generator that, if it works, is the trained neural contribution the dissertation requires, and if it
doesn't, closes the matrix channel for good on real data rather than on our own generators.*
