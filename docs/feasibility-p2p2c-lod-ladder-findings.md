# P2.2c — LOD Model Ladder (held-out A/B) — Findings

*Branch `p2/delta-prior-rearchitecture`. Spec: `PROPOSAL-P2.2c-LOD-detectability-spectrum-audit.md`;
oracle gate: `feasibility-p2p2c-lod-oracle-findings.md`. Stop-for-review.*

## Result: BOTH arms floor in the learned channel — but for the FIRST time with proven signal

Held-out (leave-datasets-out) A/B, target-conditioned, same δ-bins / RPS / calibration / leakage /
manifest / split. Only the mechanism differs.

| arm | family | held-out RPS | (uni−RPS)/SE | bin | adj | entropy (bits) | leakage_pass |
|---|---|---|---|---|---|---|---|
| own-value | own_value_self_censoring | 0.1905 | +0.00 | 0.114 | 0.350 | 2.807 | True |
| **LOD** | lod_top_coding | **0.1903** | **+0.03** | 0.157 | 0.364 | 2.807 | True |

(uniform RPS = 0.1905; max entropy = 2.807.) **LOD does NOT beat uniform out-of-family** — it floors
like own-value, with entropy pinned at maximum.

## Why this is NOT a repeat of P2.2b — the oracle changed the epistemic situation

At the end of P2.2b the lingering question was *"is the footprint absent, or is the model failing?"*
For LOD that question is **answered**: the §3 oracle proved (Bayes-optimal, profiled MAR null) that
the matched-rate LOD footprint is **trivially separable** (BE→0 at δ≥1 on all 6/6 real-fitted
X-models). By Neyman–Pearson the information is **definitively present**. So the LOD floor is **not**
non-identifiability — it is a **representation / transfer gap**: the learned channel fails to extract
a signal that provably exists. (Exactly the situation the oracle-first gate was built to create:
*"we are no longer allowed to say perhaps there was never any signal."*)

### Confirmation that the signal is accessible in the observed values (not just to the oracle)

A simple observed-marginal consequence-feature — the 95th percentile of the **observed** target
column — moves monotonically with δ on held-out cps1985:

| δ | 0.0 | 0.4 | 1.0 | 2.5 |
|---|---|---|---|---|
| observed-target 95th pct (z) | +2.00 | +1.94 | +1.90 | **+1.71** |

The upper tail of the observed target is truncated more as δ rises (top-coding) — a coarse but real
feature in the raw observed values. The encoder extracts **none** of it (RPS = uniform). The oracle's
full power is greater still (per-row likelihood, not just this marginal), but even the *coarse*
marginal feature is being missed.

## Diagnosis: the encoder does not compute the consequence-feature

The current `DeltaPriorModel` = `LacunaEncoder` + head reads tokenized (value, observed, …) cells
with row-wise attention and row→dataset pooling. The LOD signal lives in an **observed-marginal /
order-statistic property of one column** (the truncated upper tail). The encoder's pooling is not
surfacing that order-statistic to the head — so even the easy (LOD) end of the spectrum is invisible
to the learned channel. This is precisely the failure NORTH-STAR §6 names: *"a discriminative
objective that learned generator fingerprints instead of the consequence (observed-marginal
distortion) … fix with … consequence-features."* P2.2 cut at the `evidence` seam and uses the encoder
alone — neither it nor the (dropped) v1.0 missingness-feature extractor computes observed-**value**
distribution features.

## Honest scoping of the spectrum claim

- **At the ORACLE level the detectability spectrum EXISTS**: LOD (BE→0) ≫ own-value (model-flat) under
  the same matched-rate protocol. NORTH-STAR §2/§3½ gains support.
- **In the CURRENT learned channel the spectrum is NOT yet demonstrated**: both idioms floor, because
  the encoder representation cannot read even the easy end. We have demonstrated that **representation
  is the binding constraint for the learned channel**, with the signal proven present.

This is a productive negative: the first result that cleanly separates *"signal absent"* (own-value,
P2.2b — for that idiom) from *"signal present but unrepresented"* (LOD, here), and points at a
specific, North-Star-prescribed fix rather than a vague "model failed."

## Recommended next (await review) — a REPRESENTATION change, now motivated

The A/B is done; it motivates exactly the consequence-feature change the spec deferred:

1. **Add explicit consequence-features** of the supplied target column to the head input —
   observed-value order statistics / tail-shape (e.g. observed quantiles, max, fraction near the
   censoring edge, observed-vs-predicted residual quantiles), computed from the observed data and
   concatenated with `[evidence ; pooled_target]`. This is the NORTH-STAR §6 "consequence-features"
   prescription, isolated to `lacuna/survey/` (no tokenization change).
2. **Re-run the held-out A/B.** Decisive outcomes:
   - **LOD learns, own-value stays flat** → the detectability spectrum is demonstrated **in the
     learned channel**, and consequence-features are validated as the missing representation. Major
     result (validates the manifold/spectrum thesis empirically).
   - **LOD still floors with consequence-features** → a deeper representation problem; escalate.
3. Keep δ-bins / RPS / calibration / leakage / manifest / held-out ladder fixed; the only change is
   the head's input features. Oracle has already established a floor there is a representation result.

## Discipline / status

Same harness; both arms leakage-clean and manifest-validated (`kind=main`, family recorded honestly
as own_value_self_censoring / lod_top_coding). No architecture change was made to reach this result
(the consequence-feature change is the *proposed next step*). Full suite green (1318 passed, 1
skipped). `runs/p2p2c-ladder-{ownvalue,lod}.json` + summary saved.

**Bottom line:** LOD's signal is proven present (oracle) but unread by the current encoder (model
floors) — a representation gap, not non-identifiability. The recommended next step is the NORTH-STAR
§6 consequence-feature representation, then re-run the A/B. Stop for review.
