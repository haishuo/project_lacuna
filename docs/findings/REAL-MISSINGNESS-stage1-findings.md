# Real-Missingness Stage 1/2 — Findings (the cat argument, tested on real labels)

> **RIG-REPAIR (2026-06-12, post-first-run; documented per discipline).** The first-run curation
> contained a second variant of the age-77 trap: ESS sentinel codes are FIELD-WIDTH based, so on
> 0–10 scale items the valid answers 7/8/9 were being counted as refusal/DK/no-answer (e.g.
> `happy`=7 → "refusal"). Fixed by `lacuna/survey/ess_codes.py` (width-correct per-column
> resolution: wide codes present ⇒ wide-only sentinels; single-digit sentinels only for true
> 1-digit fields; ambiguous columns REJECTED; 12 unit tests). Corrected corpus: 349 items, 45,057
> true refusals (was 333k inflated); top refusal items now face-valid (party-voted-for 16–26%,
> religion — the canonical sensitive items). **All numbers below are from the corrected re-run;
> first-run (contaminated) numbers are shown struck for the record. The verdict direction is
> unchanged by the repair.**

*Stage 1 of `docs/proposals/PROPOSAL-real-missingness-showdown-and-learned-generator.md`. Runs 2026-06-12.
Read alongside `docs/findings/T-review-findings.md` — this is the real-data test the synthetic T-review
could not run. Scripts: `scripts/run_real_missingness_feasibility.py`,
`scripts/run_real_missingness_transfer.py`. Artifacts: `runs/real_missingness_*.json`.*

## What was tested

On **real ESS R11 documented missingness** (50,116 respondents, 293 clean bounded items after a
conservative out-of-range sentinel filter — the age-77 trap handled, see below), variable **names
stripped**: from the observed data alone, can the documented missingness MECHANISM of a cell be
recovered — refusal vs don't-know vs skip(not-applicable)? Feature null only (LR + GBM on
respondent-, column-, and gate-level statistics). This is the **upper bound**: if even the
strongest features can't, no network can.

## Curation (the trap, handled)

The naive ESS standardized-sentinel scan over-counts massively (e.g. `emplno` 90% "refusal" —
count/code variables whose legitimate values collide with the 7/77/7777 family). Fix: admit a
column only if it is a **bounded integer item** (valid block 0..vmax≤30) AND the sentinel family
is **strictly out of range** (gap above vmax) — codebook-free, conservative, excludes
counts/codes/continuous. Yields 293 clean items, 333k documented refusals / 340k DK / 164k
no-answer / 2.48M not-applicable, including the sensitive-item gold (`hinctnta` household income
11.7k refusals, `alcfreq`, `pray`, `rlgatnd`).

## Results

**In-distribution (held-out respondents):**

| feature set | model | macro | refusal-vs-DK | DK-vs-skip | refusal-vs-skip |
|---|---|---|---|---|---|
| full (14) | GBM | 0.881 | **0.746** | 0.954 | 0.952 |
| no column-base-rate (11) | GBM | 0.879 | **0.740** | 0.953 | 0.950 |

**Leave-country-out transfer** (5 folds over 30 ESS countries; held-out populations/languages/
modes; column stats train-countries only; genuine-signal features, no base-rate shortcut),
refusal-vs-DK, GBM: **pooled OOF AUC 0.897** (corrected labels; first-run contaminated value was
~~0.767~~ — label noise was *diluting* the signal).

## Reading (precise, and bounded)

1. **The synthetic verdict does NOT transfer to real data — the cat argument is vindicated.** The
   T-review's flat-idiom result (own-value ≈ chance) does *not* mean real missingness is flat.
   Real refusal-vs-DK carries **genuine, transferable signal (~0.76 OOF leave-country-out)** from
   observed data alone, names stripped. The matrix channel is **not exhausted on real labels** for
   this distinction.
2. **It is not a base-rate artifact.** The signal survives removing column base-rate features
   (0.746→0.740 GBM) ⇒ it lives in respondent-level disposition (serial-refuser vs genuine-DK
   structure) and value-distribution shape, not in "recognize the column."
3. **It transfers** (0.767 OOF ≈ in-distribution) — the *opposite* of the P2.2c collapse
   (in-dist 0.94 / OOF 0.43). Across populations the footprint is stable. This is real human
   behavior with a portable signature.
4. **Skip is the feature-trivial detectable idiom on real data** (0.95), exactly as the synthetic
   work predicted — that part of the T-review *does* replicate.

## What this does and does NOT establish

- **DOES:** real missingness has a transferable, non-trivial, name-free mechanism signal that our
  synthetic generator vocabulary did not contain. This reframes the scrap verdict: it was a
  verdict about *our generators*, not about real missingness.
- **Does NOT (the open load-bearing question):** the feature null now *gets* 0.767. Whether a
  **trained network beats it** (the +0.05 bar ⇒ ~0.82 OOF) is **untested** — this is exactly the
  project's recurring situation: real signal exists, shallow captures a lot, the network must earn
  the rest. Stage 2.
- **Caveats:** (a) leave-COUNTRY-out shares the ESS questionnaire across folds; true
  leave-INSTRUMENT-out (ESS→NHANES, different items) is the harder transfer test and is not yet
  run. (b) refusal-vs-DK predicts the *type* of nonresponse (a proxy for "is this the worrying
  MNAR-suspected kind"), not δ magnitude directly. (c) single seed; Stage 2 uses 5.

## Stage 2 — THE SHOWDOWN (network vs feature null on real signal)

`scripts/run_real_missingness_showdown.py`, `runs/real_missingness_showdown.json`. Real ESS
refusal-vs-DK, names stripped, block-aware leave-country-out, 5 seeds. Feature null: GBM on the 10
genuine-signal features. Network: permutation-invariant DeepSets row encoder — each column a token
[value_z, status onehot, is_target, column population stats], target cell's status hidden, context
= masked mean over non-target tokens; the network sees the **full raw respondent row** (strictly
more information than the 10 features). Bar: network (mean−SE) − GBM ≥ +0.05.

| arm | pooled leave-country-out OOF AUC (corrected labels) |
|---|---|
| GBM feature null | **0.901** |
| network seed 2026 / 7 / 99 / 13 / 41 | 0.877 / 0.881 / 0.875 / 0.882 / 0.880 |
| network mean − SE | **0.878** |

**Margin = −0.023 (bar +0.05) ⇒ network NOT load-bearing.** It trails the shallow null despite
more information; seeds tight (no degenerate seeds, not an optimization artifact). First-run
(contaminated) values for the record: GBM ~~0.764~~, network ~~0.751~~, margin ~~−0.014~~ — the
repair strengthened both arms and left the verdict unchanged.

## Verdict of the real-data test

1. **The cat argument was half-right, and now we know which half.** Real missingness DOES carry
   transferable, name-free mechanism signal the synthetic generators lacked (refusal-vs-DK 0.76 OOF)
   — so the synthetic scrap verdict was about *our generators*, not real data. BUT the extra real
   complexity is **still feature-capturable**: the network does not beat 37-statistics-style
   shallow learning on real labels either. The matrix channel is feature-sufficient **synthetic AND
   real**.
2. **The matrix-channel book is closed, on real labels.** Across the whole review — T1/T2/T3
   (synthetic) and Stage 1/2 (real) — no trained network is load-bearing for missingness-mechanism
   inference from the numeric matrix. This is no longer a statement about our generators; it now
   holds where real, transferable signal genuinely exists.
3. **The remaining neural hope is OFF the matrix** — the semantic channel (read the item text;
   load-bearing by type signature, statistics cannot read language) and the learned generator
   (Part B; load-bearing by construction). Both untested; both are about information the matrix
   does not contain. This is where a dissertation-grade network must now be sought, if anywhere.

### Caveats (honest)
- One network config (DeepSets), modest training; but the burden-of-proof design gave the network
  *more* information than the null and it lost — the direction is not a tuning artifact.
- Leave-COUNTRY-out shares the ESS questionnaire; true leave-INSTRUMENT-out (ESS→NHANES) remains a
  harder transfer test, **confounded by items-per-respondent** (ESS ~293 items vs NHANES ~12, so
  the strongest feature — respondent disposition — is much thinner in NHANES). Worth running to
  characterize cross-instrument robustness of the *signal*, but it does not change the
  network-vs-feature verdict.
- refusal-vs-DK is mechanism-type (a proxy for "the MNAR-suspected kind"), not δ magnitude.
