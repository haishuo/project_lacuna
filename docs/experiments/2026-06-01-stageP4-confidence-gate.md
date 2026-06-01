# Stage P4 — gating the fact-tier prior on classifier confidence

- **Date:** 2026-06-01
- **ADR:** docs/decisions/0008-metadata-prior-likelihood.md (P3 lead 1).
- **Branch:** `experiment/metadata-prior`.
- **Module:** `lacuna/priors/metadata_prior.py` (`gated_reliability`, `gated_semantic_prior_alpha`; +5 tests).
  **Script:** `scripts/stageP4_confidence_gate.py`. **Report:** `runs/stage0_general_baseline/stageP4_confidence_gate.json`.

## The risk this closes

P3's tiered strength fixed the PISA MCAR-by-design blind spot but exposed a risk: a **strong fact-tier
prior amplifies a misclassification *into* the fact tier** (`survey_bfi`, personality items, was mislabelled
`skip_gated` → a strong MAR prior the data struggled to override). A fact-tier prior is only as trustworthy
as the fact-tier *label*.

## The fix — gate the fact-tier strength on self-consistency confidence

Confidence = **self-consistency**: classify each anchor's metadata k=7 times under sampling (T=0.7, each
seeded), take the modal-label frequency. The fact-tier strength then interpolates from an overridable floor
(P(favored)=0.60, confidence 0) to full strength (0.85, confidence 1). Non-fact tiers ignore confidence
(already capped/weak). Self-consistency is **metadata-only**, so the gate does **not** couple the prior to
the data channel (ADR-0008 commitments 1–3).

## Result — the gate preserves genuine facts and de-rates the misclassification

Self-consistency cleanly separates the two:

| anchor | modal label | confidence | ungated combined (M/A/N) | gated combined (M/A/N) |
|---|---|--:|---|---|
| pisa2018_gbr_rotation | planned_random | **1.00** (7/7) | .37/.36/.26 (MCAR) | .37/.36/.26 (MCAR) |
| pisa2022_deu_rotation | planned_random | **1.00** (7/7) | .79/.09/.13 (MCAR) | .79/.09/.13 (MCAR) |
| survey_ucla_textbooks | skip_gated | **1.00** (7/7) | .04/.70/.26 (MAR) | .04/.70/.26 (MAR) |
| **survey_bfi** | skip_gated | **0.57** (4/7; 3/7 sensitive) | .29/**.59**/.12 (MAR) | .39/**.46**/.16 (MAR) |

- **PISA fix preserved.** The genuine design facts classify *stably* (planned_random / skip_gated 7/7,
  confidence 1.0) → full strength → PISA stays MCAR, UCLA stays MAR. The fix from P3 is intact.
- **The bfi misclassification is de-rated.** It classifies *unstably* (skip_gated 4/7, sensitive_disclosure
  3/7 → confidence 0.57) → the strong MAR prior drops toward the floor → the combined read pulls back from a
  confident MAR (f_MAR 0.59) to an ambiguous near-tie (0.46 MAR / 0.39 MCAR). The over-drive is removed; the
  read now honestly reflects that the *label itself* was uncertain.
- **Surgical.** The gate touches only the fact tier. Gut/moderate tiers are unchanged even when their own
  confidence is low (e.g. `nhanes_demographics` self-consistency 0.43) — correctly, because those priors are
  already capped and overridable; only the dangerous fact tier needs gating.

## Honest limit

Self-consistency catches an *unstable* misclassification (bfi flipped between labels), **not a
confidently-wrong one**: if the model labelled bfi `skip_gated` 7/7, the gate would pass it through at full
strength. Catching stable-but-wrong fact labels would require a corroborating signal — but a *data-side*
corroboration would re-couple the two channels (defeating the separation), so it would have to be another
*metadata-side* check (e.g. an independent ontology, or an abstain-if-rare-label rule). Left as a lead.

## Verdict

The confidence gate resolves P3's amplification risk **without cost to the genuine-fact case**: stably
classified design facts keep full strength (PISA fix intact), unstably classified ones are de-rated toward
an overridable hunch (bfi over-drive removed). The fact-tier strength is now earned by label stability, not
asserted.

## Caveats

- Small N (14 anchors); self-consistency adds k× classification cost (cheap here).
- The confidently-wrong limit (above).
- Combined-posterior calibration and the abstention threshold remain open (P-arc leads 2, 3).

## Reproduce

```
python scripts/stageP4_confidence_gate.py     # k=7 self-consistency classify, then ungated-vs-gated combine
```

## Files

- Gate: `lacuna/priors/metadata_prior.py` (`gated_reliability`, `gated_semantic_prior_alpha`) + tests.
- Script: `scripts/stageP4_confidence_gate.py`; report `runs/stage0_general_baseline/stageP4_confidence_gate.json`.
