# P2.2b — Learnability Ladder — RUNG 3 Findings

*Branch `p2/delta-prior-rearchitecture`. Spec: `docs/proposals/PROPOSAL-P2.2b-learnability-ladder-audit.md`.
Rung 3 only — stop-for-review per the approved plan. Read with the rung-1 pass
(`docs/findings/feasibility-p2p2b-rung1-findings.md`).*

## Result: RUNG 3 FAILS — head-side target conditioning does NOT rescue real-X

Real survey X, leave-datasets-out (8 train / 2 val / 2 test), 7 δ-bins, **supplied target with
head-side conditioning** (gather the target column's encoder token-reps, masked-mean pool over
rows, concat with global evidence → δ-head). **Same split and config as the failed P2.2 global
run — the ONLY change is `target_conditioned=True`** (clean A/B). 270,249 params, 18 epochs,
~1244 s CPU. `runs/p2p2b-rung3.json` (validated, `kind=main`, `leakage_pass=True`).

| metric | rung 3 (after-T) | reference | reading |
|---|---|---|---|
| **test RPS** | **0.1910** | uniform/base-rate **0.1905** | **at the floor — does NOT beat either** |
| bin accuracy | 0.129 | chance 0.143 | ≈ chance |
| adjacent accuracy | 0.414 | — | ≈ chance-level for 7 ordered bins |
| E[δ] MAE | 0.738 | — | ≈ marginal |
| coverage 50/80/90 | 0.71 / 1.00 / 1.00 | nominal | marginal-driven |
| ECE (before/after T) | 0.058 / 0.025 | — | low (trivially — predicts ≈ marginal) |
| leakage_pass | **True** | — | corr(δ,rate)=+0.04; rate-only 0.18 vs base 0.11 |

### Direct A/B vs the P2.2 global-evidence run (the comparison requested)

| run | conditioning | split/config | test RPS | bin-acc |
|---|---|---|---|---|
| P2.2 pilot (global evidence) | none | leave-datasets-out, max_rows 160 | ~0.1904 | 0.16 |
| **rung 3 (target-conditioned)** | head-side target pool | **identical** | **0.1910** | 0.13 |

Both sit at the uniform floor. Adding the supplied target column to the head changed nothing.

## Interpretation (per the approved decision rules)

> *If rung 3 fails, the bottleneck is real-X geometry or generator/data scale, not merely target
> localization.*

**Target localization / interface is NOT the bottleneck.** Telling the model exactly which column
to report δ for — the human-parity affordance — does not move real-X performance off the floor.
Combined with rung 1 (which learned strongly on synthetic 2-col with adj-acc 0.90), the bottleneck
is now isolated to what differs between rung 1 (pass) and rung 3 (fail): **real survey-X covariance
geometry and/or per-example data scale**, not the head, the loss, or target conditioning.

This also fits the earlier reframing: because only the target column carries missingness, the
censored column was already observable — so conditioning was expected to help only modestly, and
it did not help at all here. The signal the model must extract (own-value truncation of the
observed target conditional on the predictors) is evidently not recoverable from real survey X at
this scale, whereas it was trivially recoverable from clean synthetic Gaussian X at n=1024.

## Honest caveat — geometry vs scale is NOT yet separated

The clean A/B holds **rows-per-example fixed at 160** (to match P2.2). But rung 1 succeeded at
**n=1024**. So two candidates remain confounded in the rung-1-pass / rung-3-fail contrast:
1. **Real-X geometry** — real survey columns are discrete/skewed/heteroscedastic, not standard
   bivariate normal; the self-censoring footprint may be far weaker or harder to read.
2. **Per-example scale** — 160 rows (with ~48 missing target cells) may simply carry too little
   evidence to estimate δ on real X, independent of geometry.

Rung 3 cleanly rejects *localization* as the cause but does **not** separate geometry from scale.

## Machinery / discipline

Head-side conditioning implemented exactly as approved: **tokenization unchanged (4 channels, no
target-marker channel)**; target index flows out-of-band in the batch; the target column's encoder
`token_representations` are gathered + masked-mean pooled and concatenated with global evidence;
no `LacunaModel`/reconstruction/MoE/3-class/binary head in the graph (asserted). Same RPS loss,
δ-bins, calibration metrics, leakage gate (clean), manifest validation (`kind=main`). 12 new
`conditioned_head` tests; full suite green (1277 passed, 1 skipped).

## Recommended next step (await review) — separate geometry from scale

Per the decision rule, the bottleneck is real-X geometry or scale. The single most informative
next experiment is to **disambiguate them**, before any large compute spend:
- **(a) Real-X at high rows** — rerun rung 3 (and/or the global model) at `max_rows` ≈ 1024 on a
  couple of large survey datasets (e.g. `survey_cps1988` n=28k, `survey_yrbss` n=11k). If it now
  learns → the P2.2 floor was **per-example scale**, and rung 5 (scale) is the path. If it still
  floors → the bottleneck is **real-X geometry**.
- **(b) Narrow real-X probe** — restrict to real datasets with d≈2–4 (`survey_chile` d=4,
  `survey_cps1985` d=4, `survey_hmda` d=6) at high rows, target known. Removes width while keeping
  real geometry.
- The deferred **coarse 3-bin real-X** diagnostic (audit rung 2) remains optional and is only
  worth running if a real-X config first shows *partial* signal.

Do not scale compute blindly (charter §4.6 / the standing instruction): the (a)/(b) probes are
small, targeted disambiguations, not a brute-force scale-up.

## Status

Rung 1 PASS (head/loss sound) · **Rung 3 FAIL (conditioning does not rescue real-X; floor matches
the P2.2 global run).** Bottleneck localized to **real-X geometry or per-example scale**, not the
objective and not target localization. Stop here for review; recommended next is the small
geometry-vs-scale disambiguation above.
