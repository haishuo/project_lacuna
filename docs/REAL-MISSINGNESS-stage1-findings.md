# Real-Missingness Stage 1 — Findings (the cat argument, tested on real labels)

*Stage 1 of `docs/PROPOSAL-real-missingness-showdown-and-learned-generator.md`. Runs 2026-06-12.
Read alongside `docs/T-review-findings.md` — this is the real-data test the synthetic T-review
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
refusal-vs-DK, GBM: per-fold 0.752 / 0.784 / 0.781 / 0.756 / 0.763 ⇒ **pooled OOF AUC 0.767**.

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

## Next (Stage 2 — the real showdown)

1. **Network vs frozen-feature null**, refusal-vs-DK (and the 3-way), on bit-identical examples,
   5 seeds — does the network beat 0.767 by ≥ +0.05.
2. **True leave-instrument-out** (ESS train → NHANES test, using NHANES 7/9 refusal/DK codes) —
   the hard transfer test.
3. If the network wins on real transferable signal: load-bearing on real complexity, the strongest
   possible dissertation position. If parity: the matrix channel is feature-sufficient even where
   real signal genuinely exists — and the learned generator (Part B) / semantic channel become the
   neural contribution. Either way the verdict is now grounded on real labels, not our generators.
