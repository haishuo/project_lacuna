# Stage P6 — the abstention threshold ("unable to determine")

- **Date:** 2026-06-01
- **ADR:** docs/decisions/0008-metadata-prior-likelihood.md (lead 3, property 3).
- **Branch:** `experiment/metadata-prior`.
- **Module:** `lacuna/priors/metadata_prior.py` (`selective_decision`; +4 tests). **Script:**
  `scripts/stageP6_abstention.py`. **Report:** `runs/stage0_general_baseline/stageP6_abstention.json`.

## What P6 does

Operationalises property 3 — "fairly certain it's mechanism X" vs "unable to determine" — as **selective
prediction** on the calibrated combined posterior: commit to the argmax mechanism iff its posterior
probability `p_max >= t*`, else abstain (`selective_decision`). The threshold `t*` is fit by a
**risk-coverage** analysis (sweep the threshold; trace coverage = fraction committed vs committed-accuracy =
argmax correctness among the committed; pick the threshold meeting a target committed-accuracy of 80%).
Three posteriors are compared — data-only (calibrated), combined with a reliable injected prior (ρ=0.85),
combined with a near-chance prior (ρ=0.45) — the prior reliability simulated as in P5 (the real prior's
reliability is the P1/P3 question).

## Result (n_test = 640; target committed-accuracy 80%)

| posterior | overall argmax-acc | coverage @80% acc | coverage @70% acc |
|---|--:|--:|--:|
| data-only (calibrated) | 0.681 | **0.66** (t*=0.37) | 0.93 |
| combined, ρ=0.85 (reliable) | 0.838 | **1.00** | 1.00 |
| combined, ρ=0.45 (near-chance) | 0.686 | 0.56 | 0.93 |

Commit-rate and committed-accuracy **by the true dominant mechanism**, at the shared data-only threshold:

| posterior | MCAR (identifiable) | MAR (split) | MNAR (split) |
|---|--:|--:|--:|
| data-only | **1.00 @acc 1.00** | 0.58 @acc 0.62 | 0.41 @acc 0.60 |
| combined, ρ=0.85 | 1.00 @acc 0.84 | **1.00 @acc 0.86** | **1.00 @acc 0.81** |
| combined, ρ=0.45 | 0.95 @acc 1.00 | 0.46 @acc 0.52 | 0.29 @acc 0.58 |

## Findings

1. **The abstention is calibrated.** Committing only when `p_max >= t*` holds committed-accuracy at the 80%
   target — when the instrument says "fairly certain," it is right ~80% of the time.
2. **"Unable to determine" is the correct default on the non-identifiable axis.** Data-only commits
   **100% — perfectly — on the identifiable MCAR axis**, but abstains on ~half the MAR/MNAR-dominant
   datasets, and is barely above chance (acc ~0.60) when it does commit there. This is Stage F's honest seam
   turned into an explicit, calibrated *abstention*: the instrument declines to name MAR-vs-MNAR from data
   alone, rather than guess.
3. **A reliable prior lifts the abstention.** It raises coverage@80% from **0.66 → 1.00** and the MAR/MNAR
   commit-accuracy from **~0.60 → 0.81–0.86**: the prior supplies the signal the data cannot, turning
   "unable to determine" into "fairly certain" exactly on the split — the prior×likelihood payoff.
4. **A near-chance prior does not help, and the data stays protected.** Coverage@80% falls to 0.56 (an
   unreliable prior adds no honest commitment), yet the data **still overrides it on the identifiable axis**
   (MCAR 0.95 @acc 1.00) — the P2 safety property. Calibration + abstention prevent a bad prior from
   manufacturing false confidence on the part the data can measure.

## Scope note

This abstention is at the **mechanism (MCAR/MAR/MNAR)** granularity (the composition estimand). Property 3's
literal example — "fairly certain 20% is *threshold* MNAR" — is at the **subtype** granularity, which is the
per-column subtype-detector layer (ADR-0008 commitment 6; the restored-Q3 direction). The selective-decision
machinery here transfers directly to it; the subtype layer is the natural next build.

## Caveats

- **Simulated prior reliability** (ρ injected; the real prior's reliability is the P1/P3 face-validity
  question), and **semi-synthetic** truth, as in P5.
- `t*` is fit to a target committed-accuracy on semi-synthetic; deployment would pick the target and confirm
  on whatever labelled data exists.

## Reproduce

```
python scripts/stageP6_abstention.py     # risk-coverage for data-only vs combined (rho 0.85 / 0.45)
```

## Files

- Decision: `lacuna/priors/metadata_prior.py` (`selective_decision`) + tests.
- Script: `scripts/stageP6_abstention.py`; report `runs/stage0_general_baseline/stageP6_abstention.json`.
