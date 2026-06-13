# P2.2b — Proxy-Strength Sweep — Findings

*Branch `p2/delta-prior-rearchitecture`. Pre-registered note: `feasibility-p2p2b-proxy-sweep-note.md`.
Gate result. Stop-for-review.*

## Headline: the gate FAILED — proxy strength is REJECTED as the primary explanation

Per the pre-registered decision rule ("if Part B is flat at the floor across all strata, reject
proxy strength as the primary explanation and return to geometry/discreteness/representation"),
**proxy absorption is not what causes the real-X floor.** The falsifiability safeguard fired exactly
as designed: real-X floors **even where there is essentially no proxy** (R²≈0.02), so the comforting
"correct abstention because the proxy absorbed the signal" story is **disconfirmed, not confirmed.**

## Part B — real-X R² stratification (the gate): FAIL

Safeguard (printed before training): the candidate-target R² range is **genuine** — min 0.006,
median 0.177, max 0.776; stratum 0 mean R²=0.023 (essentially no proxy), stratum 4 mean R²=0.648.
"Low R²" really means ~0.02, not 0.62. One model, R²-stratified training (low-R² *was* learnable,
not undertrained), per-stratum eval:

| stratum | R²_mean | test RPS | (uni−RPS)/SE | bin | adj | entropy (bits) | P(δ=0) |
|---|---|---|---|---|---|---|---|
| 0 | **0.023** | 0.1920 | **−0.20** | 0.142 | 0.317 | **2.804** | 0.144 |
| 1 | 0.083 | 0.1917 | −0.16 | 0.158 | 0.433 | 2.803 | 0.139 |
| 2 | 0.183 | 0.1909 | −0.06 | 0.183 | 0.467 | 2.805 | 0.147 |
| 3 | 0.307 | 0.1914 | −0.11 | 0.125 | 0.400 | 2.806 | 0.143 |
| 4 | 0.648 | 0.1907 | −0.03 | 0.142 | 0.442 | 2.803 | 0.146 |

(uniform RPS = 0.1905; max entropy = log₂7 = 2.807.)

- **Every stratum is at the uniform floor** — no stratum beats uniform (all (uni−RPS)/SE ≤ 0);
  bin-acc ≈ chance everywhere.
- **The lowest-R² stratum (0.023) — where there is no proxy to absorb anything — floors just as
  hard** as the highest. This is the decisive falsification.
- **Entropy is pinned at the maximum (≈2.80 ≈ 2.807) in every stratum and does NOT vary with R².**

Gate (pre-registered): *sharp at some low-R² stratum?* **No.** *Entropy rises with R²?* **No** (flat
at max). ⇒ **FAIL.**

## Part A — synthetic ρ-sweep (mechanism): WEAK / EQUIVOCAL (does not cleanly support either)

| ρ | RPS | rps_adv | bin | adj | entropy | P(δ=0) |
|---|---|---|---|---|---|---|
| 0.00 | 0.064 | +0.126 | 0.486 | 0.914 | 1.99 | 0.225 |
| 0.30 | 0.065 | +0.126 | 0.493 | 0.829 | 1.67 | 0.234 |
| 0.60 | **0.190** | +0.000 | 0.143 | 0.429 | 2.81 | 0.144 |
| 0.80 | 0.089 | +0.102 | 0.386 | 0.871 | 1.95 | 0.150 |
| 0.90 | 0.150 | +0.040 | 0.243 | 0.686 | 2.66 | 0.096 |
| 0.95 | 0.113 | +0.077 | 0.300 | 0.743 | 2.39 | 0.137 |
| 0.99 | 0.121 | +0.070 | 0.293 | 0.643 | 2.23 | 0.146 |

Honest reading (overriding the script's naive "SUPPORTS" auto-verdict, which only checked two weak
Spearman signs): the trend is **noisy and non-monotonic.** Spearman(ρ, rps_adv)=−0.54 and
Spearman(ρ, entropy)=+0.36 point weakly in the predicted direction, but **P(δ=0) moves the wrong
way** (−0.64), ρ=0.60 is an **anomalous full collapse** (a single unstable training run, since
ρ=0.80 recovers), and — crucially — **the model still learns at ρ=0.99** (RPS 0.12 ≪ uniform 0.19,
adj 0.64). On clean Gaussian data even extreme proxy strength does **not** floor the model, because
z_t retains a √(1−ρ²) independent component that keeps δ identifiable. So Part A shows at most a
mild, noisy degradation and notably **no clean collapse** — it does not strongly support proxy
absorption either.

## Interpretation (decisive, and deliberately not rationalized)

1. **Proxy strength is not the governance variable we hoped.** The hypothesis predicted low-R² →
   sharp, high-R² → collapse. Part B finds **floors at all R²**, including R²≈0.02. The predicted
   relationship does not exist. Rejected.
2. **This is the "floors everywhere" branch, NOT "correct abstention."** The user's own criterion:
   a floor counts as correct abstention only if the model is **sharp somewhere** and uncertainty
   tracks identifiability. On real-X the model is sharp on **no** stratum, and entropy is uniformly
   pinned at maximum — it is **not** higher in unidentifiable regions because it is maxed
   *everywhere*. Meanwhile the *same head* is sharp on synthetic (rung 1, adj 0.90). So on real
   survey X this is a **failure to extract the footprint**, not calibrated abstention. We do **not**
   get to call it virtue.
3. **The bottleneck is intrinsic to the real-X own-value footprint**, present even with no proxy.
   The leading candidates are now real-column **discreteness / low cardinality / skew** and/or a
   **representation** that does not expose the observed-vs-expected truncation on real data — not
   proxy absorption.

## What the ladder has now ruled out (cumulative)

| ruled out as the cause | by |
|---|---|
| δ-prior concept / RPS / 7-bin / calibration / objective | rung 1 (PASS) |
| target localization / global-pooling interface | rung 3 (conditioning didn't help) |
| per-example evidence scale | rung 3b (1024 rows still floored) |
| **observed-column proxy absorption** | **proxy sweep Part B (floors at R²≈0.02)** |

Remaining: **real survey-X footprint expression** — discreteness/cardinality/skew and/or
representation of the truncation signal.

## Honest implication for the project

The own-value self-censoring idiom, applied to a **single real survey column at matched rate**,
appears to leave a footprint the encoder cannot read — and this is **not** because a proxy absorbed
it. Two possibilities the next step must distinguish: (a) the signal is *present but not represented*
(fixable by features/representation), or (b) the signal is *intrinsically near-absent* on real
discrete survey columns (then real-data behavior is honestly a wide prior + sensitivity report, and
expectations about discrimination must be reset — charter-consistent, but it changes the product
story). Either way, **proxy strength is not the lever**, and we should not build P2.3 abstention
around R².

## Recommended next (await review — small, targeted; not a model rewrite)

Shift the question from proxy to **footprint expression**:
1. **Cardinality / continuity probe** — does real-X learn when the TARGET is a genuinely continuous
   real column (e.g. wage, hours) vs a low-cardinality coded item? Stratify the same sweep by target
   cardinality instead of R². This directly tests the discreteness hypothesis.
2. **Strong-δ-only contrast** (labeled diagnostic) — δ=0 vs large-δ on real-X: is even a STRONG
   own-value effect unreadable? If yes → the footprint is intrinsically near-absent on real columns
   (possibility b); if no → it is a resolution/representation issue (possibility a).
3. Only if (a): add observed-vs-expected residual features to the representation.

## Discipline note

The pre-registered gate was honored to the letter: low-R² was made genuinely present (R²=0.02) and
genuinely learnable (stratified training), and the result **falsified** the hypothesis we found most
attractive. We are recording the rejection, not the rationalization. Same RPS/δ-bins/leakage(clean,
all runs)/manifest; entropy reported alongside discrimination throughout. Full suite green (1292
passed, 1 skipped).

## Status

**Proxy-absorption hypothesis REJECTED.** The real-X floor is a real-column footprint/representation
problem, present even without a proxy, and the uniform max-entropy output on real-X is failure-to-
extract (not calibrated abstention, by the sharp-somewhere criterion). Stop for review; recommended
next is the cardinality/strong-δ probe to separate "present-but-unrepresented" from
"intrinsically-absent."
