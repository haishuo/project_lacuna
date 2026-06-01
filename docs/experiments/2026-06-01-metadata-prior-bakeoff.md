# Metadata-prior bakeoff — can a LOCAL model author the missingness-mechanism prior?

- **Date:** 2026-06-01
- **Branch:** `experiment/metadata-prior` (off `experiment/composition-stageb`).
- **Script:** `scripts/metadata_prior_bakeoff.py`. **Benchmark:** `scripts/metadata_prior/benchmark.json`
  (49 columns). **Results:** `scripts/metadata_prior/bakeoff_results.json`.
- **Status:** feasibility probe for a new direction (a metadata-authored *prior* combined with the
  data-fingerprint *likelihood*); informs a future ADR, not yet a committed estimand change.

## The question

MAR-vs-MNAR is non-identifiable from the data alone (Molenberghs) — but an analyst never works from the
data alone; they know what the columns *mean*. "Income" raises the prior on self-censoring MNAR; a lab
assay with a detection limit → detection MNAR; a skip-gated follow-up → MAR-by-design; a randomized
booklet rotation → MCAR-by-design. That metadata-authored **prior** is the extra-data channel Molenberghs
says you must supply. **Does it require a frontier model, or can a small, local, HIPAA-safe, reproducible
model author it?** If local suffices, the deployed prior needs no frontier API in the loop.

## Design

The model sees only the metadata an analyst already has — `column name + codebook description + domain`,
**no data values, no gold** — and outputs `{mechanism ∈ MCAR/MAR/MNAR/INDETERMINATE, semantic class,
confidence}`. Scored against a 49-column benchmark whose gold is **grounded in real structure** where
possible: published limit-of-detection flags → detection MNAR; explicit skip-logic → MAR-by-design;
randomized administration (PISA rotation, NHANES random subsamples) → MCAR-by-design; the survey-anchor
consensus for sensitive items → self-censoring MNAR. Weaker/abstain cases included to probe
over-confidence. Two conditions isolate the codebook's value: **name+desc** (the realistic analyst case)
vs **name-only** (cryptic-code stress test). Deterministic (greedy, fixed seed). 16 GB RTX 5070 Ti;
≥7 B in 4-bit (bitsandbytes nf4 — confirmed working on Blackwell sm_120).

Benchmark: 49 columns (19 MNAR / 14 MAR / 10 MCAR / 6 indeterminate), 24 strongly-grounded, across
NHANES + clinical EHR + survey + education. Majority-class baseline = **0.442** on the 43 clear columns.

## Result — the capability turns on at 3B; descriptions are load-bearing; no frontier needed

Mechanism accuracy (mean over clear columns; **strong** = the 24 structurally-grounded columns):

| model | name+desc clear | **name+desc strong** | semantic | name-only clear | abstain behaviour |
|---|--:|--:|--:|--:|--|
| Qwen2.5-0.5B | 0.186 | 0.25 | 0.21 | 0.233 | collapses → MCAR |
| Qwen2.5-1.5B | 0.233 | 0.292 | 0.093 | 0.233 | **degenerate: predicts MCAR for ALL** |
| Qwen2.5-3B | 0.605 | **0.875** | 0.442 | 0.07 | over-abstains (recall 1.0) |
| Qwen2.5-7B (4bit) | 0.628 | 0.792 | 0.721 | 0.279 | never abstains; MCAR-lean |
| **Qwen2.5-14B (4bit)** | **0.814** | **0.917** | **0.814** | 0.535 | under-abstains (recall 0.33) |
| Phi-3.5-mini | — | — | — | — | env failure (see caveat) |

1. **No frontier model is needed.** Qwen2.5-14B (4-bit, on a 16 GB consumer GPU) authors the prior at
   **0.92 on strongly-grounded columns and 0.81 semantic** — and gets every signature case right (below).
2. **The capability turns on between 1.5B and 3B.** ≤1.5B models are *degenerate* — Qwen-1.5B predicts
   MCAR for **all 43** clear columns (worse than the 0.442 majority baseline). 3B is the floor where real
   semantic reasoning appears (0.875 on strong columns); 3B→7B→14B then trade abstention for precision.
3. **Codebook descriptions are the load-bearing input.** name-only collapses (3B 0.605→0.07; 14B
   0.814→0.535). Bigger models recover *some* from cryptic codes via world knowledge (14B name-only 0.535
   still beats baseline), but the realistic "analyst has the codebook" condition is where small models win.
   *The more metadata you supply, the smaller the model you need* — confirmed.
4. **4-bit on a 16 GB consumer card is sufficient up to 14B** — genuinely deployable on-prem / HIPAA-safe,
   reproducible (pinned local checkpoint, temperature 0), no API.

### Signature columns (name+desc) — all models ≥3B agree on the grounded cases

| column | gold | 3B | 7B | 14B |
|---|---|---|---|---|
| PISA rotated booklet item | MCAR (by-design) | MCAR ✓ | MCAR ✓ | MCAR ✓ |
| NHANES random-subsample chemical | MCAR (by-design) | MCAR ✓ | MCAR ✓ | MCAR ✓ |
| PHQ-9 suicidal-ideation item | MNAR (self-censor) | MNAR ✓ | MNAR ✓ | MNAR ✓ |
| HIV viral load (detection limit) | MNAR (detection) | MNAR ✓ | MNAR ✓ | MNAR ✓ |
| cigarettes/day (skip-gated) | MAR (by-design) | MAR ✓ | MAR ✓ | MAR ✓ |
| `V37` (no codebook) | INDETERMINATE | abstain ✓ | abstain ✓ | abstain ✓ |
| **household income** | MNAR (self-censor) | abstain | **MCAR ✗** | **MNAR ✓** |

- **The metadata prior fixes the documented Stage-E PISA blind spot.** The footprint reads MCAR-by-design
  rotated booklets as "structured" (value-independence is invisible to the data channel); every model
  ≥3B correctly calls it MCAR *from the metadata* — concrete proof the prior supplies what the data cannot.
- **Income is the discriminating case.** The textbook MNAR example is where the smaller models stumble (7B
  bizarrely calls it MCAR; 3B abstains); only 14B nails it. The canonical self-censoring item — the one
  the data footprint also can't see (Stage F) — needs the larger model *and* is exactly where prior and
  likelihood are both weakest, so it warrants the most caution.

### Confusion structure

14B's residual error is concentrated on **MAR→MCAR** (4/14: skip-logic and demographic columns it treats
as random) while **MNAR is near-perfect (18/19)** and MCAR strong (9/10). 7B carries an **MCAR-lean**
(over-calls MCAR for several MAR/MNAR) yet nails all 10 MCAR. The errors are interpretable, not random.

## Caveats / limitations

- **Abstention is real but uncalibrated out-of-the-box.** 3B *over*-abstains (abstain-recall 1.0 but
  abstains on 23% of clear columns); 7B/14B *under*-abstain (recall 0.33–0.5 — they commit a mechanism on
  genuinely-ambiguous columns). No size is well-calibrated raw → confirms the design rule that the LLM is a
  **feature extractor, not an oracle**: the abstention threshold (and the prior strength) must be *fit*,
  not taken from the model's stated confidence.
- **Qwen-family only.** Phi-3.5-mini failed on an environment incompatibility (`DynamicCache.get_max_length`
  removed in transformers 4.53; its `trust_remote_code` modeling calls it), so the cross-family check is
  missing. The size-ladder result is clean within one family; a cross-family confirmation (Llama once a HF
  token is available, or Granite — both standard-arch) is the obvious next step.
- **Gold is an analyst's grounded prior, not a measured per-dataset fact.** It is defensible (LOD flags,
  skip-logic, by-design randomization, anchor consensus) but it encodes the assumption the prior is meant
  to encode. The bakeoff measures "can a local model reproduce a competent analyst's prior," not "is the
  prior correct for a specific dataset" (which is the prior×likelihood combination's job).

## Verdict

A **local 3–14B model authors the missingness-mechanism prior well** (14B: 0.92 on grounded columns, every
signature case right, including the PISA blind spot the data channel cannot fix) — **no frontier model
required, and none wanted** in a HIPAA-shaped, reproducibility-bound loop. Recommended operating point on a
16 GB card: **Qwen2.5-14B in 4-bit** for best accuracy + semantic precision; 3–7B as a lighter floor. The
endpoint is plausibly a small fine-tuned classifier distilled from this, with a *fitted* abstention
threshold. This clears the feasibility bar for an ADR on the prior×likelihood instrument.

## Reproduce

```
python scripts/metadata_prior_bakeoff.py                       # full lineup, both conditions
python scripts/metadata_prior_bakeoff.py --models Qwen2.5-14B --conditions name+desc
```

## Files

- Benchmark: `scripts/metadata_prior/benchmark.json` (grounded gold + the semantic→mechanism map).
- Harness: `scripts/metadata_prior_bakeoff.py`. Results: `scripts/metadata_prior/bakeoff_results.json`.
</content>
