# Stage P1 — the frozen metadata→prior table

- **Date:** 2026-06-01
- **ADR:** docs/decisions/0008-metadata-prior-likelihood.md, Stage P1.
- **Branch:** `experiment/metadata-prior`.
- **Modules:** `lacuna/priors/metadata_prior.py` (+ `tests/unit/priors/test_metadata_prior.py`, 22 tests).
  **Script:** `scripts/stageP1_build_prior_table.py`. **Artifact:** `scripts/metadata_prior/prior_table.json`.

## What P1 builds

The prior channel as a **Dirichlet pseudo-count vector** over `(MCAR, MAR, MNAR)`, one per semantic class,
that pools with the data-likelihood's Dirichlet evidence by **summation** — `alpha_post = 1 + (alpha_prior −
1) + (alpha_like − 1)` — the same conjugate/evidential operation the composition deep-ensemble already uses.
Two ADR-0008 properties fall out for free: a flat prior (opaque metadata) is a **no-op** (graceful
degradation), and a strong contrary likelihood **outweighs** a modest prior (the override property). The
local model (Stage P0 bakeoff) supplies only the semantic *label*; this frozen table supplies the prior.

## The frozen table (the auditable artifact)

| semantic class | favored | α (MCAR/MAR/MNAR) | P(favored) |
|---|---|---|--:|
| lab_lod | MNAR | (1, 1, 4.67) | 0.70 |
| skip_gated | MAR | (1, 4.67, 1) | 0.70 |
| planned_random | MCAR | (4.67, 1, 1) | 0.70 |
| sensitive_disclosure | MNAR | (1, 1, 3.71) | 0.65 |
| administrative | MCAR | (3.26, 1, 1) | 0.62 |
| demographic_core | MAR | (1, 2.17, 1) | 0.52 |
| routine_measure | MAR | (1, 2.17, 1) | 0.52 |
| indeterminate | — | (1, 1, 1) | 0.33 |

**Key design decision (documented):** every favoured-class probability is **capped at ≤ 0.70** so the data
likelihood can always override the prior (ADR-0008 commitment 1). The prior is a *nudge*, not a lock —
strongly-grounded mechanisms (a real detection limit, explicit skip-logic) don't need a near-certain prior
because an *agreeing* data signal reinforces them, whereas a mislabelled column must stay overridable. The
strengths are set from the benchmark's grounding tiers (strong ≈ 0.70, consensus ≈ 0.65, weak ≈ 0.52,
indeterminate flat); they are a conservative starting point, re-fit against real likelihoods + the override
test in Stage P2.

## Validation (prior-only, on the grounded benchmark)

| path | clear | strong | consensus | weak | abstain-recall |
|---|--:|--:|--:|--:|--:|
| CURATED (gold semantic labels) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| **LLM-DRIVEN (Qwen2.5-14B, name+desc)** | **0.837** | **0.917** | **1.00** | 0.444 | 0.333 |

- **Curated = 1.0** is a *consistency* check (the table faithfully encodes the grounded semantic→mechanism
  map), not independent accuracy.
- **LLM-driven is the realistic deployed-prior accuracy:** the local model's semantic labels → this table →
  **0.92 on strongly-grounded columns, 1.0 on consensus (sensitive items)**. It is correctly *weak* on the
  weak-grounding tier (0.44 — demographic/routine, where metadata genuinely under-determines the mechanism),
  and under-abstains on the indeterminate columns (0.33) — the abstention threshold is unfit (next).

## Caveats / what P1 does not yet do

- **Strengths are documented, not data-fit.** A data-driven per-class reliability fit needs real labelled
  mechanisms, which don't exist at scale (the recurring arc limitation); the grounding-tier targets are the
  honest stand-in, validated for *consistency* here and to be pressure-tested by the override audit in P2.
- **Abstention is unfit** (Qwen under-abstains on indeterminate, abstain-recall 0.33) — Stage P2 fits the
  abstain rule (semantic=indeterminate ∪ a calibrated confidence/agreement gate).
- **Prior-only, no likelihood yet.** The combination, calibration, and the override audit are Stage P2.

## Reproduce

```
python scripts/stageP1_build_prior_table.py     # reuses the P0 bakeoff predictions; no model calls
```

## Files

- Module: `lacuna/priors/metadata_prior.py`; tests: `tests/unit/priors/test_metadata_prior.py`.
- Builder: `scripts/stageP1_build_prior_table.py`; frozen artifact: `scripts/metadata_prior/prior_table.json`.
