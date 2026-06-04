# P2.2 — δ-Prior Model + Loss — First Pilot Findings

*Branch `p2/delta-prior-rearchitecture`. Implementation commit `8d4210e`; this doc reports the
first pilot runs. Governed by `docs/PROPOSAL-P2.2-model-loss-audit.md`,
`docs/NORTH-STAR.md`. **Honest first result — not a success claim.***

## TL;DR

The P2.2 machinery is **built, tested, and behaves correctly**: a fresh-from-scratch
`DeltaPriorModel` (LacunaEncoder + new δ-bin head, no v1.0 heads in the graph) trains on the
RPS objective, calibrates, passes the matched-rate leakage gate, and writes a validating
`kind="main"` manifest. **But across three pilot configurations the model does NOT learn a
discriminating δ-prior — its test RPS sits at the uniform-prediction floor.** P2.2 therefore
establishes that *the objective trains and calibrates* (audit §0 interpretation constraint), and
explicitly does **not** yet demonstrate out-of-family δ generalization or governance performance.

## What ran

`scripts/run_p2p2_pilot.py` — leave-datasets-out split (8 train / 2 val / 2 test disjoint survey
datasets), own-value self-censoring, matched rate 0.30, δ-grid `{0, .15, .4, .75, 1.25, 1.75,
2.5}`, β₁∈[0,2], RPS loss, post-hoc temperature on val. Plus an in-distribution high-`max_rows`
learnability diagnostic.

| Run | split | params | rows/ex | epochs | test RPS | uniform RPS | bin-acc | adj-acc | leakage_pass |
|---|---|---|---|---|---|---|---|---|---|
| pilot v1 | leave-datasets-out | 112,585 | 128 | 6 | 0.1904 | 0.1905 | 0.190 | 0.448 | ✅ |
| pilot v2 (main, on disk) | leave-datasets-out | 251,049 | 160 | 10 | 0.1904 | 0.1905 | 0.164 | 0.493 | ✅ |
| diagnostic | **in-distribution** | 112,585 | **512** | 8 | 0.1926 | 0.1905 | 0.141 | 0.438 | n/a (smoke) |

`chance bin-acc = 1/7 ≈ 0.143`. The recorded main manifest is `runs/p2p2-pilot-001.json`
(pilot v2).

## Reading the result (attributable, not hedged)

1. **Machinery is correct.** Forward returns `[B,7]` logits; RPS computes and backprops; no
   reconstruction/MoE/3-class/binary head is present; from-scratch + all-trainable asserted;
   labels flow out-of-band; the leakage gate and manifest validation work. (All covered by 59
   passing unit tests.)
2. **Calibration is good — but trivially so.** ECE ≈ 0.006–0.04 and interval coverage ≥ nominal,
   because the model defaults to ≈ the marginal δ-bin distribution. A near-marginal predictor is
   automatically calibrated; this is NOT evidence of a useful prior.
3. **No discrimination.** test RPS ≈ uniform-RPS and bin-acc ≈ chance in every config. The model
   is not reading the own-value self-censoring footprint into a 7-way δ estimate.
4. **The "too few rows per example" hypothesis is REJECTED.** Raising rows/example to 512 and
   training in-distribution (train = test datasets, the easiest possible case) did not beat
   uniform either. So the floor is not simple per-example evidence starvation at the row counts
   tried, nor a train/test distribution-shift artifact.
5. **Leakage is genuinely clean** (|corr(δ, realized_rate)| ≤ 0.05; per-bin rates all within
   ~0.01 of 0.30; rate-only baseline ≈ base rate). The null result is NOT a matched-rate failure
   — there is no rate cue to exploit, and the footprint cue is evidently not being extracted.

## Why this likely differs from P1 (hypotheses for review — NOT fixes applied)

- **The P1 ceiling was a BINARY H0-vs-H1 probe on 2-column synthetic data** with controlled ρ and
  large per-example n, where the censored column was one of only two. P2.2 asks for a **7-way
  ordinal δ on real, wide survey X (d up to 28)** where the model must *both* localize the
  censored column *and* infer its self-censoring strength from a single column's masked cells.
  That is a much harder estimation problem; the per-example signal may be near the detection floor.
- **RPS near the marginal is locally flat** — the optimizer may settle at the base-rate
  distribution. Worth probing the loss landscape / an easier curriculum (e.g. δ=0 vs large-δ
  binary warm-up, or a single-column-table setup that removes the localization burden).
- The encoder may not be **isolating the censored column's footprint** when it is diluted among
  many fully-observed columns.

## Recommended next steps (for PI decision, before any P2.2 "success" claim)

1. **Learnability probe with the localization burden removed**: restrict to narrow datasets
   (d≈2–4) and/or feed only (target, top predictor) columns, large n/example — does the δ-prior
   beat uniform *there*? This isolates "can the head learn δ at all" from "can the encoder find
   the censored column in a wide table".
2. **Easier target first**: a 3-bin or binary δ=0-vs-δ>0 warm-up to confirm gradient signal, then
   widen to 7 bins. (Stays within ordinal/proper-scoring; not a revert to the 3-class objective.)
3. **More rows/example × longer schedule** on GPU (CPU made 250k-param × 512-row runs ~7–11 min);
   the CPU budget capped what was tried here.
4. Only after a config beats the uniform/base-rate references by a margin outside CI does a P2.2
   out-of-(dataset)-family calibration number become meaningful (audit §13 success gate).

## Status

P2.2 code: **done, green (1254 passed, 1 skipped), manifest-gated, leakage-gated.** First pilot:
**at the uniform floor — no learned δ discrimination yet.** Per the interpretation constraint this
is reported as-is; it does not claim generalization. Awaiting PI review on the modeling direction
(learnability probe / curriculum / GPU scale) before declaring the P2.2 model successful or
proceeding to P2.3.
