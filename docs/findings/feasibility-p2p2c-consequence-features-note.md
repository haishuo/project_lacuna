# P2.2c — Consequence-Feature Module — Implementation Note

*Short pre-implementation note (feature list FIXED before the run). Tests the NORTH-STAR §6
hypothesis that the learned channel needs explicit **consequence features** — statistics that
directly expose the observed-marginal distortion — to surface a signal the encoder misses but the
oracle proved present. Tightly scoped: no tokenization/encoder/loss/generator/metadata changes;
the ONLY change is concatenating a fixed consequence-feature vector to the existing
target-conditioned head input.*

## Scope (hard limits, per approval)

- New module `lacuna/survey/consequence_features.py` ONLY; concatenate its output to
  `[evidence ; pooled_target]` in `TargetConditionedDeltaModel`. No other architecture change.
- No tokenization change, no encoder redesign, no metadata, no loss change, no generator change, no
  new mechanisms. Same δ-bins / RPS / calibration / leakage gate / manifest / held-out split.
- Goal is NOT to hand-code LOD detection forever — it is to test whether *any* fixed observed-marginal
  consequence vector lets the learned channel read the LOD truncation it currently misses.

## Feature vector (FIXED — 17 features, computed per example from the SUPPLIED target column)

Computed from the **observed** target values only (what the model legitimately sees), made
**scale-invariant** by z-scoring within the observed values (`z = (obs−mean)/(std+ε)`) so features
transfer across datasets/columns. `n_obs` = observed count, `n_total` = rows.

| # | feature | rationale |
|---|---|---|
| 0 | `missing_rate = 1 − n_obs/n_total` | matched across δ (≈ target_rate) ⇒ carries no δ cue (anti-leakage anchor) |
| 1–8 | z-quantiles at 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99 | observed-marginal shape; upper-tail emphasis |
| 9 | `z_min` | lower edge |
| 10 | `z_max` | upper edge — top-coding truncates it |
| 11 | `skew = mean(z³)` | upper truncation skews observed left |
| 12–14 | `frac(z>1.0)`, `frac(z>1.5)`, `frac(z>2.0)` | upper-tail mass — depleted under top-coding |
| 15 | `q99 − q90` | extreme-upper-tail compression under truncation |
| 16 | `q95 − q50` | upper-half spread compression |

`N_FEATURES = 17`. (Optional observed-vs-predicted residual quantiles are **excluded** this pass to
avoid a feature-engineering project, per approval.) Order/contents frozen before the run; recorded in
the manifest.

**Anti-leakage reasoning (interpretation rule 2):** `missing_rate` ≈ `target_rate` for all δ (matched
solve) ⇒ it cannot encode δ. The quantile/tail features encode the *observable consequence* of δ (the
truncation shape) — that is the legitimate signal, not leakage. The existing δ→realized-rate leakage
gate still runs; if the consequence A/B shows BOTH arms learning, we audit whether a feature
accidentally encodes δ/rate directly.

## Computation & edge handling

- `compute_consequence_features(x, r, target_idx) -> Tensor[17]` (float32). Pure, deterministic
  (quantiles/moments; no RNG).
- Observed rows = `r[:, target_idx]`. If `n_obs < 8`, return a zero vector with `missing_rate` set
  (degenerate column; should not occur — generator keeps ~70% observed). `std==0` ⇒ z=0 vector.
- Computed at batch-build time from the original-scale `ObservedDataset.x` (NOT the tokenized values),
  carried out-of-band on `DeltaBatch.consequence [B, 17]` (like the δ-bin label and target index).

## Model wiring (the only architecture change)

- `TargetConditionedDeltaModel` gains `n_consequence_features` (0 = off). When > 0, the head input is
  `concat([evidence, pooled_target, LayerNorm(consequence)])` → `DeltaBinHead`; head input dim =
  `evidence_dim + hidden_dim + N_FEATURES`. A LayerNorm on the consequence block for scale stability.
- `forward(batch, target_idx, consequence=None)`; require `consequence` when `n_consequence_features>0`.
  `DeltaPriorModel.forward` keeps an ignored `consequence=None` for interface uniformity.
- `TrainConfig.consequence_features: bool` (default False); when True the target-conditioned model is
  built with `N_FEATURES` and the loop/eval pass `db.consequence`.

## Manifest

`model_arch` records `consequence_features_enabled=True`, `n_consequence_features=17`, and the frozen
`consequence_feature_schema` (the FEATURE_NAMES list). Validates as before; family recorded honestly.

## Tests (before the run)

`tests/unit/survey/test_consequence_features.py`: shape (=17) & dtype; **missing handling** (censored
cells excluded; few-observed degenerate → safe); **determinism** (same input → same vector);
**scale-invariance** (scaling/shifting the column leaves features unchanged); **discriminative
sanity** (top-coded column ⇒ lower `z_max`/`q95`/`frac(z>1.5)` than uncensored) — a unit check, not a
training target; failure cases (bad target_idx). Plus model wiring tests (head input dim, forward
with/without consequence, no v1.0 heads).

## A/B run (unchanged except the feature flag)

Exact same held-out leave-datasets-out A/B as `run_p2p2c_lod_ladder.py`, with
`consequence_features=True` for BOTH arms (own-value and LOD), same δ-bins / RPS / calibration /
leakage / manifest / split / config. The 95th-percentile diagnostic stays in the findings as an
**external sanity check** (evidence the signal is present), NOT a training target — the model learns
only from the fixed 17-feature vector + the standard RPS objective.

## Interpretation rules (pre-registered, per approval)

- **LOD learns, own-value flat** → detectability spectrum demonstrated **in the learned channel**;
  consequence features validated as the missing representation. (Headline success.)
- **Both learn** → audit leakage / whether a feature encodes δ or rate directly.
- **LOD still floors** → representation problem deeper than simple observed-marginal consequence
  features.
- **own-value learns but LOD does not** → treat as a bug/artifact; investigate.

## Order

Note (this) → implement module + wiring + tests (suite green) → run the same A/B → report → stop.
