# Stage P5 — calibrating the combined (prior × likelihood) posterior

- **Date:** 2026-06-01
- **ADR:** docs/decisions/0008-metadata-prior-likelihood.md (P2-deferred lead 2).
- **Branch:** `experiment/metadata-prior`.
- **Script:** `scripts/stageP5_combined_calibration.py`. **Report:** `runs/stage0_general_baseline/stageP5_combined_calibration.json`.

## What P5 does

Completes the recipe P2 set up: combine the prior with the **raw** likelihood evidence (to preserve the
data's differential informativeness and the override property), then **calibrate the combined posterior
afterwards** with its own temperature τ_c (`α_cal = 1 + (α−1)/τ_c`), fit by minimising simplex-query ECE on
a held-out calibration split and evaluated on a disjoint test split (the Stage-D machinery, now applied to
the *combined* Dirichlet).

**The honest difficulty:** on semi-synthetic data we have the realised composition (ground truth) but the
real metadata prior is *mismatched* (the generator assigns mechanisms randomly, not by column semantics).
So we calibrate against an **injected prior of known reliability ρ** — it points at the true dominant
mechanism with probability ρ, else at a wrong one — simulating a metadata prior whose accuracy is ρ.
Sweeping ρ separates the **calibration** question from the **prior-accuracy** question (the latter is the
P1/P3 face-validity programme).

## Result (n_test = 640; lower ECE / Brier is better)

| | τ_c | ECE uncal→cal | Brier cal | MCAR-axis Brier | split-axis Brier |
|---|--:|--:|--:|--:|--:|
| data-only calibrated (τ_D=4.99) | — | — / 0.041 | 0.143 | 0.129 | 0.150 |
| combined, ρ=0.85 (reliable) | 6.83 | 0.106 → **0.022** | **0.112** | — | **0.117** |
| combined, ρ=0.65 (gut) | 4.99 | 0.071 → 0.040 | 0.134 | — | 0.141 |
| combined, ρ=0.45 (near-chance) | 8.00 | 0.062 → 0.041 | 0.148 | — | 0.153 |

1. **The combined posterior calibrates.** A single temperature τ_c drives ECE to data-only levels or better
   for *every* ρ (0.022–0.041). Calibration is **reliability-agnostic** — a temperature can always make the
   combined posterior's region statements reliable. τ_c > τ_D (the combined posterior is sharper than the
   data-only one — prior evidence on top of the raw likelihood — so it needs more flattening). (At ρ=0.45,
   τ_c hits the grid max 8.0; ECE is still fine, so the grid is adequate.)
2. **The prior's VALUE (resolution) is reliability-contingent.** A reliable prior (ρ=0.85) improves the
   split-axis Brier **0.150 → 0.117** — it adds resolution on the non-identifiable MAR-vs-MNAR axis, exactly
   where the data is at chance (Stage F). A near-chance prior (ρ=0.45) stays well-calibrated but is *worse*
   on resolution (0.153). The gut-tier reliability (ρ=0.65) gives a modest gain (0.141).
3. **Calibration fixes CONFIDENCE, not CORRECTNESS.** This is the user's "priors amplify error" in
   calibration terms: τ_c makes the combined posterior honest (reliable) regardless of the prior's accuracy,
   but only a *reliable* prior makes it more *informative*. An unreliable prior is honestly-calibrated noise.

## Interpretation

The full ADR-0008 combination recipe is now closed: **combine prior + raw likelihood (P2) → fit τ_c on the
combined posterior (P5).** The result is always calibrated; whether it beats data-only on the proper score
depends on the prior being reliable — which the instrument cannot self-certify, and which is therefore the
job of the P1/P3 grounded/face-validity checks (and the P4 confidence gate, which de-rates the priors most
likely to be wrong). The instrument is honest by construction; its usefulness is earned by the prior's
accuracy, reported separately.

## Caveats

- **τ_c is fit against a *simulated* prior reliability ρ**, not the real metadata prior (mismatched on
  semi-synthetic). The real prior's reliability is the separately-measured face-validity question; for
  deployment one would pick τ_c at the ρ matching the operating prior's measured accuracy (fact-tier ~0.85,
  gut-tier ~0.65), or fit τ_c online once labelled real data exists.
- **Global temperature only** — per-instance recalibration was shown infeasible at the dataset level
  (Stage D-v2: composition error is homoscedastic in the footprint); the same limit applies here.
- Semi-synthetic, survey-scoped, as throughout.

## Reproduce

```
python scripts/stageP5_combined_calibration.py     # sweeps prior reliability rho in {0.85, 0.65, 0.45}
```

## Files

- Script: `scripts/stageP5_combined_calibration.py`; report `runs/stage0_general_baseline/stageP5_combined_calibration.json`.
- Reused: `lacuna/training/composition_calibration.py`, `lacuna/priors/metadata_prior.py`.
