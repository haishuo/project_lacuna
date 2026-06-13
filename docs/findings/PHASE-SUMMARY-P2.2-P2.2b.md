# Phase Summary — P2.2 / P2.2b: a flat-likelihood boundary condition

*Project-history record. Status: **accepted (PI, 2026-06-05)** as the closing conclusion of the
P2.2/P2.2b investigation. Governed by `NORTH-STAR.md` (esp. §2 identification line, §3 reframed
estimand, §8 validation-ladder addendum). This document is the canonical summary; the per-rung
findings docs carry the tables.*

## The one-paragraph conclusion (the line for the project history)

> We selected what was likely one of the hardest survey missingness mechanisms, and found that its δ
> footprint is effectively unrecoverable from **semi-synthetic missingness imposed on real survey
> data at matched rates**. This identifies a **boundary condition** for Lacuna. It does **not** yet
> determine whether other survey missingness idioms occupy more detectable regions of the mechanism
> manifold.

## What was established (narrow, precise)

> **Own-value self-censoring, at matched rate, on a single survey column, is a flat-likelihood idiom
> under our current semi-synthetic-on-real-X validation framework** (NORTH-STAR §8.2).

Concretely, across the P2.2b ladder — all runs being *semi-synthetic missingness on real survey X*
(real values, holes we imposed, δ known; the legitimate supervised test, NORTH-STAR §8.1 rung 2):

| ruled out as the cause of the floor | evidence |
|---|---|
| δ-prior head / RPS loss / 7-bin / objective | rung 1 PASS on synthetic Gaussian X (adj-acc 0.90) |
| target localization / pooling interface | rung 3 (head-side target conditioning did not help) |
| per-example evidence scale | rung 3b (1024 rows/example still floored) |
| observed-column proxy absorption | proxy sweep (floors at R²≈0.02, where there is no proxy) |
| target discreteness / cardinality | cardinality probe Part A (no cardinality gradient) |
| ordinal resolution (7 bins too fine) | cardinality probe Part B (δ=0-vs-strong AUC ≈ 0.5) |

The head/loss can learn this idiom where the signal exists (synthetic). On real survey X the
footprint→δ channel is empirically near-flat for this idiom, even for a strong effect.

**Why this is a *demonstrated boundary condition*, not an implementation failure.** The result earns
the word "boundary" precisely because of the **breadth of the rule-out list** above. Each rung removed
a candidate *implementation* explanation (objective, localization, scale, proxy, cardinality,
resolution); what survived is the **mechanism itself**. A single floored run could be a bug; a floor
that persists after objective, interface, scale, proxy, discreteness, and resolution have each been
independently excluded — and after the *same* stack provably learns the idiom on synthetic X — is
evidence **about the mechanism's location in identification space**, not about our code. That is the
difference between "we failed to extract a signal" and "we measured that the signal is not there."

## What was NOT established (guard against overgeneralization)

- **NOT** that missingness is generally undetectable. Detectability is **mechanism-specific**, not
  universal (NORTH-STAR §2).
- **NOT** that Lacuna's footprint-learning channel is dead. One idiom — the hardest one — is flat;
  the spectrum is untested.
- **NOT** a real-data performance claim. The runs are semi-synthetic; natural missingness remains a
  face-validity check only (NORTH-STAR §8.1 rung 3).
- **NOT** a license to re-found the project on unconstrained metadata / LLM priors (NORTH-STAR §8.3;
  the earlier "pivot to prior-driven governance" recommendation was withdrawn).

## Statistical context (why this was disappointing but not shocking)

We arguably started with one of the **hardest possible** survey idioms: missingness depends on the
**unobserved value itself**, we **explicitly removed the rate cue** (matched-rate β₀ solve), and we
restricted to a **single censored column**. By construction this sits near the non-identifiable
boundary (NORTH-STAR §2; Molenberghs). The result is therefore:

- disappointing from an **engineering** standpoint;
- not shocking from a **statistical** standpoint;
- consistent with the very concerns that motivated the North Star.

A useful inversion, recorded so we remember it: **had this specific idiom produced a strong, robust
held-out signal, the correct first reaction would have been suspicion** — a hunt for leakage,
shortcuts, or a hidden cue — not celebration. A flat result here is what an honest pipeline near the
identification boundary is supposed to produce.

## PI position (decisions of record)

1. **Keep the validation ladder exactly as corrected** — semi-synthetic missingness on **held-out**
   real survey data is the primary validation target (NORTH-STAR §8.1).
2. **Record own-value self-censoring as a flat-likelihood idiom** under current evidence (§8.2).
3. **Do not conclude Lacuna is dead.**
4. **Do not conclude metadata priors are now the product.** A metadata prior is admissible only if
   calibrated against held-out semi-synthetic truth (§8.3) — not automated expert opinion.
5. **Test at least one structurally detectable survey idiom** before any strategic judgment about the
   footprint-learning channel as a whole.

## The next scientific question

> **Does the detectability spectrum actually exist on real survey X?** — i.e. are there survey
> missingness idioms that leave **stronger observable footprints** than smooth own-value
> self-censoring?

NORTH-STAR §2 predicts yes (sharp truncation / limit-of-detection is "blatantly detectable"). The
agreed next experiment is to test a **structurally detectable idiom** (LOD / top-coding, etc.) on the
**same semi-synthetic, held-out real-survey ladder**, with the same δ-bins / RPS / calibration /
leakage gate / manifest. Outcomes:

- **Detectable idiom recovers δ out-of-family** → the footprint channel is **alive**; own-value
  self-censoring is simply one flat idiom in the vocabulary, and the manifold/detectability-spectrum
  thesis (§3½) gains direct support.
- **Even a detectable idiom is flat on real survey X** → the footprint channel is broadly weak on
  this manifold, and the metadata channel (with its §8.3 semi-synthetic evaluation protocol) becomes
  the next necessary build — known to be necessary, not assumed.

No strategic judgment on the footprint channel is made until at least that one detectable-idiom test
is run.

## Pointers

NORTH-STAR §8 (validation ladder, flat-likelihood idioms, metadata guardrail); rung findings
`feasibility-p2p2b-rung1-findings.md`, `-rung3b-findings.md`, `-proxy-sweep-findings.md`,
`-cardinality-probe-findings.md` (the last carries the binding framing correction). Code/artifacts on
branch `p2/delta-prior-rearchitecture`; full suite green (1296 passed, 1 skipped) — no architecture
or objective changes were made to reach this conclusion.
