# P2.2b — Cardinality / Strong-δ Probe — Implementation Note

*Short note (design specifics only). Separates the two surviving explanations for the real-X floor:
(1) own-value δ signal **present but not represented** vs (2) **intrinsically near-absent** on real
survey columns. The proxy hypothesis was rejected; this is the last cheap diagnostic before any
representation/architecture change. No architecture changes, no residual features yet.*

## Safeguard confirmed (pre-run reconnaissance)

The narrow real pool spans a genuine cardinality range — continuous/high-cardinality targets exist
(wage n_unique 5970/1017/238, price 808, statusquo 2012, lvrat 1537, pirat 519) alongside
low-cardinality coded items (binary `nonwhite`/`owned`, `screen`=3, `ram`/`speed`=6, age-coded).
So a low-vs-high cardinality contrast is realizable on this data.

## Part A — cardinality-stratified 7-bin task

- **Cardinality metric:** `n_unique` of the target column on full data (with `unique_frac =
  n_unique/n` recorded). New `lacuna/survey/column_stats.py::target_cardinality_table` +
  `cardinality_distribution` (mirrors `proxy_score.target_r2_table`).
- **Strata (fixed, interpretable thresholds):** **low** `n_unique ≤ 8` (coded/binary/few-level),
  **medium** `9–60`, **high/continuous** `> 60`. The runner prints the distribution + per-stratum
  member counts/ranges BEFORE training (same safeguard discipline as the R² sweep).
- **Model/task:** target-conditioned δ-prior, 7 δ-bins, full δ-grid, matched rate — identical to
  the proxy sweep except strata are by cardinality not R². Reuse `StratifiedRealXSource` (it
  stratifies arbitrary `(dataset,target)` pair lists). Cardinality-stratified training so every
  stratum is learnable (high-card not undertrained).
- **Question:** do **high-cardinality/continuous** targets recover δ signal (beat uniform) while
  **low-cardinality** targets floor?

## Part B — strong-δ-only contrast (labeled DIAGNOSTIC)

- **Setup:** the same cardinality-stratified real-X, but δ-grid = **{0.0, 2.5}** only (no-censoring
  vs extreme self-censoring). Kept inside the existing 7-bin RPS pipeline (δ=0→bin 0, δ=2.5→bin 6) —
  **no new binary objective**; the binary read-out is a reported diagnostic, not the P2 objective.
- **Diagnostic metric:** **AUC** of `P(δ>0) = 1 − p[bin 0]` vs the binary label, per cardinality
  stratum (rank-based AUC, no sklearn), plus RPS-vs-uniform and bin accuracy. Clearly labeled
  `kind="ablation"`/diagnostic in the manifest.
- **Question:** is even a STRONG own-value effect unreadable on real columns? If strong-δ separates
  (AUC≫0.5) but the 7-bin task floors → resolution/representation issue. If strong-δ does **not**
  separate even on high-cardinality targets → the footprint is intrinsically near-absent under real
  survey geometry at matched rate.

## Metrics (per cardinality stratum)

Part A: RPS, **(uniform−RPS)/SE**, bin accuracy, adjacent accuracy, **mean entropy (bits)**,
**P(δ=0)**, ECE; leakage gate per run; manifest validated. Part B: **AUC** (diagnostic) + RPS-vs-
uniform + bin accuracy + entropy, per stratum.

## Decision rules (the user's, restated)

- **High-cardinality/continuous learns while low-cardinality floors** → P2 needs cardinality-aware
  abstention or feature engineering (and the signal IS present, just representation-gated).
- **Strong-δ separates but 7-bin floors** → move to a coarse/curriculum δ objective or add
  residual/truncation features.
- **Neither high-cardinality NOR strong-δ learns** → real-X own-value self-censoring at matched
  rate is effectively invisible under the current observable footprint; revise P2 expectations
  toward **wide priors + sensitivity reporting**, and stop chasing δ extraction on this idiom
  without new information/features.

## Expected runtime

Two training runs (Part A 7-bin, Part B strong-δ), high-row narrow pool, max_cols=8 ≈ **25–35 min
CPU** total. No scale-up.

## Code required

`column_stats.py` (cardinality table + distribution, tested); one combined runner
`scripts/run_p2p2b_cardinality_probe.py` (safeguard → Part A → Part B, rank-AUC helper). Reuses
`StratifiedRealXSource`, `train`, `metrics`, `leakage`, `run_manifest`. No model/loss/tokenization
changes; binary only as a labeled diagnostic. ≤500 LOC each, RNG-injected, fail-loud,
leakage-gated, manifest-validated.

## Stop

Implement the above, run the probe, report Part A (cardinality-stratified) + Part B (strong-δ AUC)
against the decision rules, then stop for review.
