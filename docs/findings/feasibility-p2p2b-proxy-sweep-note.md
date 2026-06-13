# P2.2b — Proxy-Strength Sweep — Implementation Note

*Short pre-implementation note (not a full audit). Tests whether δ-learnability tracks proxy
strength — converting the rung-3/3b "floor" into a falsifiable scientific claim and, if confirmed,
into the P2.3 abstention trigger. Gate-style: the result decides whether proxy absorption is the
leading explanation. Stop after this note for approval.*

## Hypothesis under test (and how it can fail)

H: **low R²(target | observed) ⇒ δ signal survives ⇒ model sharp; high R² ⇒ proxy absorption ⇒
model collapses toward a high-entropy prior.** Calibration-first form: as proxy strength rises,
discrimination must fall **AND uncertainty must rise** (entropy ↑, P(δ=0) ↑). Accuracy falling
*without* entropy rising = overconfident collapse = a failure, not correct abstention.

**Falsification conditions (pre-registered):**
- Part A: if clean synthetic data (where rung 1 proved the model *can* learn) does **not** show
  monotone degradation as ρ→1, the proxy mechanism is weakened before we touch real-X.
- Part B (the gate): if **every** real-X R² stratum sits at the floor — i.e. the model is **not
  sharp on any low-R² stratum** — proxy absorption is **rejected**; the bottleneck is
  geometry/discreteness/representation, not absorption. Likewise, if discrimination falls while
  entropy stays flat (overconfident), "correct abstention" is rejected.

## 1. Part A — synthetic ρ-sweep (mechanism, no discreteness confound)

- Reuse `SyntheticTwoColSource` (the rung-1 source that learns). Sweep
  **ρ ∈ {0.0, 0.3, 0.6, 0.8, 0.9, 0.95, 0.99}**.
- **Per-ρ fresh in-distribution model** (each is rung 1 at one fixed ρ): train/val/test all drawn
  from `SyntheticTwoColSource(rho_grid=[ρ])`, so each ρ gets a clean "is δ recoverable here?"
  measurement with full opportunity to learn (removes a mixed-capacity confound). 7 δ-bins,
  matched rate, n=1024 rows/example, target-conditioned head (parity with rung 3/3b).
- Output: the curve **{discrimination, entropy, P(δ=0)} vs ρ**. Prediction: at ρ=0 sharp + low
  entropy; as ρ→0.99, RPS→uniform and entropy→near-max. Mechanistic rationale: at high ρ the
  observed predictor ≈ the latent target, so missingness driven by z_t is re-expressible via z_p
  (MAR-on-proxy) → δ unidentifiable from the footprint.
- Near-free pre-check: also break the **existing rung-1** test set down by its ρ∈{0,0.3,0.6} draw
  for early corroboration before the full sweep.

## 2. Part B — real-X R² stratification (product, the gate)

- **Proxy-score / R² estimator** (new module `lacuna/survey/proxy_score.py` — this IS the planned
  P2.3 §6 module, built now and reused there): for one example, regress the **observed** rows of
  the target column on the other **observed** columns (OLS / ridge) → R². Operational proxy-score
  `≈ 1 − R²`. **Caveat (recorded):** under MNAR the observed target is truncated, so R² estimated
  on observed rows is biased — a *conservative* operational signal, not the true
  Var(z_t | z_t-unobserved). Documented, not silently used.
- **Characterize the available R² range first:** precompute R² over a large bank of (dataset,
  target) draws from the real survey pool so we know the empirical R² distribution before training.
  If the pool offers essentially no low-R² targets, that is itself a finding (real single-column
  own-value MNAR is almost always near-unidentifiable → product is mostly abstention).
- **Falsifiability requires low-R² to be learnable:** train ONE model on the full real pool (high
  rows, target-conditioned, in-distribution) with **R²-stratified sampling** so every R² stratum is
  well represented in training — otherwise a low-R² stratum at the floor is *undertrained*, not
  *unidentifiable*, and proves nothing.
- **Eval per R² stratum** (e.g. bins [0,0.1], (0.1,0.3], (0.3,0.5], (0.5,0.7], (0.7,1.0]): report
  the full metric block per stratum. The gate is whether the SAME model is **sharp at low R²** and
  **collapses (high entropy) at high R²**.

## 3. Exact metrics (per ρ / per R² stratum)

Discrimination: **RPS**, **(uniform − RPS)/SE** and **(base_rate − RPS)/SE** (per-example RPS for
the SE), **bin accuracy**, **adjacent accuracy**. Uncertainty: **mean predictive entropy**,
**mean P(δ=0)**. Calibration: **ECE**, **interval coverage (50/80/90)**. All reuse `survey.metrics`
(+ one small addition, §4). Leakage gate (`survey.leakage`) evaluated and recorded per run; manifest
validated (`kind=main`/`diagnostic`).

## 4. Entropy and P(δ=0) reporting (calibration-first)

- Add `mean_predictive_entropy(probs)` to `survey.metrics` — H = −Σ p_k log₂ p_k averaged over
  examples, reported in **bits** (max = log₂7 ≈ 2.807). Small, tested.
- `P(δ=0)` reuses existing `p_delta_zero` (mean mass on bin 0).
- **The decisive plot is a dual curve:** discrimination AND entropy vs proxy strength on the same
  axis. The North-Star-correct signature is the two moving *together* — discrimination ↓ while
  entropy ↑. A discrimination ↓ with entropy flat is flagged explicitly as overconfident collapse.

## 5. Success / falsification criteria (the gate)

**PASS (proxy absorption confirmed → proceed to P2.3, make proxy strength a first-class governance
variable):**
- Part A: monotone trend — Spearman(ρ, entropy) > 0 and Spearman(ρ, RPS-advantage) < 0; at ρ≤0.3
  the model beats uniform by ≥2 SE with low entropy; at ρ≥0.95 RPS≈uniform with entropy near max.
- Part B: **at least one low-R² stratum is sharp** (beats uniform ≥2 SE, adjacent-acc clearly above
  chance) AND a high-R² stratum collapses; **entropy rises monotonically with R²** across strata.

**FAIL / FALSIFIED (stop treating proxy absorption as leading; revisit geometry/discreteness/
representation):**
- Part B shows **no** sharp low-R² stratum (flat-at-floor everywhere), OR
- discrimination falls with proxy strength while **entropy does not rise** (overconfident collapse —
  not calibrated abstention), OR
- Part A fails to degrade on clean synthetic data.

This is a hard gate: a PASS routes to P2.3 abstention built on R²; a FAIL redirects the whole
diagnosis. The criteria are fixed before the run so the floor cannot be rationalized after the fact.

## 6. Expected runtime (CPU)

- Part A: 7 per-ρ models × ~2–3 min (rung-1-class, max_rows 1024, ~12 epochs) ≈ **18–25 min**.
- Part B: R² precompute (seconds–1 min) + one real-X training run (~15–20 min) + stratified eval ≈
  **~25 min**.
- **Total ≈ 45–50 min.** All CPU; no scale-up (charter §4.6 / standing instruction).

## Code required (scope of the implementation cycle, for approval)

Small and mostly P2.3-reused: `lacuna/survey/proxy_score.py` (observed-R² estimator + tests);
`mean_predictive_entropy` in `survey.metrics` (+ test); R²-stratified sampling + per-stratum eval
(a thin helper, likely in a `proxy_sweep` runner using the existing `train`/`example_source`/
`metrics` path); two runner scripts (`run_p2p2b_proxy_sweep_synth.py`, `..._realx.py`). No model,
loss, tokenization, or objective changes; no abstention/OOD/reporting layer yet (P2.3). All files
≤500 LOC, RNG-injected determinism, fail-loud, leakage-gated, manifest-validated.

## Stop

This is the note. On approval I will implement the small proxy-score + entropy additions and the
two runners, run Part A then Part B, and report the dual discrimination/entropy-vs-proxy curves
against the pre-registered gate — then stop for review.
