# P2.2b — Rung 3b — High-Row Real-X Diagnostic — Findings

*Branch `p2/delta-prior-rearchitecture`. Pre-run note: `docs/findings/feasibility-p2p2b-rung3b-note.md`.
Read with rung 1 (pass) and rung 3 (fail). Stop-for-review per the approved plan.*

## Result: FLOOR — real-X does NOT learn even at rung-1-like row evidence ⇒ the bottleneck is real-X GEOMETRY, not scale

In-distribution, 6 large NARROW real survey datasets, **max_rows=1024**, 7 δ-bins,
target-conditioned head. 269,673 params, 14 epochs, ~1369 s CPU. `runs/p2p2b-rung3b.json`
(validated, `kind=main`, `leakage_pass=True`).

| metric | rung 3b (after-T) | reference | reading |
|---|---|---|---|
| **test RPS** | **0.1905 ± 0.0081 (SE)** | uniform/base-rate **0.1905** | **exactly at the floor** |
| (uniform − RPS)/SE | **−0.00** | need ≥ +2 | **fails the 2-SE criterion by every margin** |
| (base − RPS)/SE | **−0.00** | need ≥ +2 | fails |
| bin accuracy | 0.181 | chance 0.143 | a whisper above chance |
| adjacent accuracy | 0.505 | — | marginally above the 0.45 bar — a faint ordinal whisper, but RPS gains nothing |
| temperature | **10.0 (pinned ceiling)** | — | calibration drove predictions to ≈ uniform |
| E[δ] MAE | 0.738 | — | ≈ marginal |
| leakage_pass | True | — | corr(δ,rate)=−0.05; rate-only 0.16 vs base 0.12 |

**Verdict (per the approved interpretation rules):** real-X at high rows **still sits at the
uniform floor ⇒ the bottleneck is real-X geometry / generator-signal expression, not row count or
localization.** The scale hypothesis is rejected: giving real-X the same 1024-row evidence that let
synthetic 2-col (rung 1) reach RPS 0.06 / adj-acc 0.90 produced no measurable discrimination here.

## The ladder, complete

| rung | setting | result | rules out |
|---|---|---|---|
| 1 | synthetic 2-col, n=1024, target known | **PASS** (RPS 0.06, adj 0.90) | head / loss / calibration / objective |
| 3 | real-X, n=160, target-conditioned | FAIL (floor) | — |
| 3b | **real-X, n=1024, target-conditioned, in-dist** | **FAIL (floor)** | **per-example SCALE; target localization** |

Head/loss sound (rung 1); localization not the cause (rung 3); evidence scale not the cause
(rung 3b). **What remains is real survey-X geometry / how the own-value self-censoring signal is
expressed on real columns.**

## Leading explanation (grounded in prior project findings)

The own-value self-censoring footprint is crisp on continuous standard-Gaussian X (rung 1) but
appears **weak-to-unidentifiable on real survey columns**, for two compounding reasons:

1. **Proxy absorption (the P1R-C phenomenon).** The generator's predictor is the *most-correlated*
   observed column; on real survey data those correlations are substantial (e.g. bfi C4~C5 r≈0.48,
   psid feducation~meducation r≈0.57). When the observed predictors strongly reconstruct the
   target, own-value MNAR ≈ MAR-on-proxy and collapses toward non-identifiability at matched rate —
   exactly the P1R-C result ("a strong observed proxy makes the MNAR claim unreliable"). Rung 1's
   ρ∈{0,0.3,0.6} kept the proxy moderate, so the own-value signal survived.
2. **Discrete / skewed real columns.** Many survey columns are ordinal/discrete with few levels;
   z-scored own-value self-censoring on a 3–5-level column leaves a coarse, low-information
   footprint vs the smooth Gaussian truncation in rung 1.

**Charter-consistent reframing (important).** If own-value δ is genuinely near-unidentifiable from
real survey footprints at matched rate, then a δ-prior that returns ≈ the marginal (high-entropy,
wide) is **partly the CORRECT calibrated behavior** — the model is declining to invent a δ it
cannot see (NORTH-STAR §4.3: abstain, don't confabulate; the project never claims to beat
non-identifiability). The "uniform floor" is then not purely a modeling failure but, in part, the
honest answer — which is precisely why the destination is a calibrated prior **with abstention**
(P2.3), not a point estimator. The open question is whether ANY real-X regime carries recoverable
own-value δ signal, or whether it is essentially always absorbed.

## Recommended next diagnostics (await review — small, targeted)

To confirm proxy-absorption-geometry and find any identifiable regime, before model redesign:
1. **Proxy-strength sweep on real-X** — per generated example, record R²(target | observed
   predictors) (the §6 proxy-score). Re-run rung-3b style but stratify/select datasets+targets by
   LOW proxy R² (weak proxy). If δ signal returns when R² is low → confirms proxy absorption is the
   geometry bottleneck (and directly motivates the P2.3 proxy-aware abstention trigger).
2. **Continuous-column real-X** — restrict targets to genuinely continuous survey columns (e.g.
   wages, hours) rather than discrete items; tests the discreteness factor.
3. **Strong-δ-only real-X** — collapse to a δ=0-vs-large-δ contrast (labeled diagnostic) to see if
   even a STRONG own-value effect is recoverable on real X; if not, geometry/absorption is decisive.

These also feed P2.3 directly: if the bottleneck is proxy absorption, the proxy-score abstention
trigger (PROPOSAL §6) is not just a safety feature but the core governance behavior — the δ-prior
should widen/abstain exactly where real-X absorbs the signal.

## Machinery / discipline

No model/loss/tokenization changes — rung 3b is a run config of the existing target-conditioned
path. Same RPS, δ-bins, calibration, leakage gate (clean), manifest (`kind=main`, validated). Full
suite remains green (1277 passed, 1 skipped). 2-SE success criterion computed from per-example RPS.

## Status

Ladder complete: **bottleneck localized to real survey-X geometry / signal expression** (head,
loss, localization, and scale all ruled out). Leading cause = proxy absorption (P1R-C) + discrete
columns ⇒ own-value δ is weak/unidentifiable on real survey footprints, and the near-marginal
output is partly correct calibrated behavior. **Stop for review** — recommended next is the
proxy-strength sweep (which also seeds the P2.3 proxy-aware abstention design), not a blind
model/scale change.
