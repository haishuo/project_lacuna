# Stage 3b — Richer deployable features + multi-seed reality check

- **Date:** 2026-05-29
- **ADR:** docs/decisions/0006, Stage 3 (b1).
- **Setup:** deployable features extended 3 → 5 (added **signed skewness** = truncation direction,
  **SMD-to-others** = MAR axis). Frozen and fine-tune, multi-seed.
- **Status:** complete — the single-seed headline did NOT reproduce; **and** a confound was found.

## Multi-seed result (fine-tune + deploy5, n=5 seeds; mean ± sd)

| metric | mean ± sd | range |
|---|---|---|
| overall accuracy | 0.577 ± **0.023** | [0.54, 0.60] |
| MCAR recall | 0.896 ± **0.026** | [0.85, 0.92] |
| composition L1 | 0.391 ± **0.023** | [0.35, 0.42] |
| MAR recall | 0.559 ± **0.229** | [0.20, 0.82] |
| MNAR recall | 0.273 ± **0.235** | [0.00, 0.64] |

The single-seed "MNAR 0.331, beats the oracle" was **variance**, not signal. Stable across seeds:
overall accuracy, MCAR recall, composition L1 (all sd ≈ 0.02). Unstable: the **MAR↔MNAR split**
(each sd ≈ 0.23, anti-correlated) — the total non-MCAR mass is stable but its *partition* lands
near-randomly each seed. Frozen + deploy5 (n=3) shows the same pattern. This is non-identifiability
as optimizer instability: the boundary mass has no stable gradient to anchor it.

## Important confound (→ Stage 4)

This instability was measured on mixtures built from a **single MNAR subtype** (logistic
self-censoring) and a single MAR subtype — and self-censoring is plausibly the *hardest* MNAR
(own-value dependence = the core non-identifiability), and the only MNAR family that supports
per-column targeting for splicing. So this result characterizes **self-censoring**, not MNAR in
general. Stage 4 (full registry, per-generator breakdown) shows per-column detectability
**stratifies sharply by subtype** — threshold / detection-limit / quantile MNAR are detected at
near-100% per column. The Stage 3b instability is therefore specific to the hard subtype, not a
property of all per-column MNAR. See `docs/experiments/2026-05-29-stage4-*.md`.

## Takeaway

Report distributions, not single seeds (this is why the verification mattered). The stable,
deployable signal remains MCAR-triage + composition; the MAR/MNAR *split* is subtype-dependent —
hard for self-censoring, easy for threshold-family MNAR.
