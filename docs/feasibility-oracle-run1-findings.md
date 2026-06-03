# Feasibility Oracle — Run 1 Findings (synthetic + real-X)

*Run `oracle_20260603_084703` (CPU, 5.7 min, no training, no checkpoint). 498 synthetic cells
(coarse 360 + 138 boundary-refinement) + 80 real-survey-X cells. Bayes errors are Monte-Carlo
estimates of the theoretical Bayes error of the optimal LLR test, with SE/95%-CI/n_mc. Computed in
float64. Artifacts: `/mnt/artifacts/project_lacuna/feasibility/oracle_20260603_084703/`.*

**Status: oracle surface presented for review. NO model training run. A design gap was found that
must be resolved before any MAR-vs-MNAR pass/kill claim (see §3).**

## 1. Mechanics validated, surface mapped

The probe + oracle machinery behaves exactly as the unit tests require on the live grid: δ=0 ⟹
Bayes error 0.5; error decreases monotonically with δ; SE/CI reported; ran in 5.7 min. The
distinguishability **boundary** (near-chance region) is confined to the hardest corner — small
δ (0.25), low rate (0.1), small n (128), low ρ — where Bayes error ≈ 0.42–0.44 (CI upper < 0.46).
Everywhere else the point-null oracle is clearly below chance; at n≥512 and δ≥1 it approaches 0.

## 2. The finding: predictor–target correlation ρ *improves* distinguishability (shadow-variable identification)

The naive expectation (and the v1-era memory note) was the opposite: "MNAR on correlated columns
leaves a MAR-like footprint → collapses at matched rate." The oracle says the reverse — **higher ρ
makes MNAR MORE distinguishable**, monotonically:

| δ=0.5, rate=0.1, n=512 | ρ=0.0 | ρ=0.3 | ρ=0.6 | ρ=0.9 |
|---|---|---|---|---|
| β1=0 (MCAR null) Bayes err | 0.308 | 0.233 | 0.143 | 0.061 |
| β1=1 (MAR null) Bayes err  | 0.267 | 0.215 | 0.173 | 0.116 |

**Diagnosis — this is genuine shadow-variable identification, not an artifact.** The oracle sees the
predictor `z_p` (always observed). When `z_p` correlates with the censored target `z_t`, observing
`z_p` carries information about the *missing* `z_t`, which sharpens the test of whether the censoring
was own-value-driven. In the **β1=0** case this is textbook: the missingness depends only on `z_t`
(R ⟂ z_p | z_t), so `z_p` is a *valid shadow variable* (Miao & Tchetgen Tchetgen; Wang–Shao–Kim;
d'Haultfœuille), and MNAR becomes point-identified — better as the shadow strengthens (higher ρ).
Even at ρ=0 there is signal (0.308 < 0.5) from the **truncation of the observed `z_t` marginal**;
ρ adds shadow information on top.

If this holds up, it is a **positive** result and directly supports the "use the rest of the table"
thesis: auxiliary survey columns correlated with a sensitive item help identify its MNAR-ness.

## 3. The design gap that BLOCKS a MAR-vs-MNAR verdict (must fix before the model arm)

The current oracle compares **two point hypotheses** at a *fixed* β1:
  H0 = MAR(β1, β2=0)   vs   H1 = MNAR(β1, β2=δ).
The genuinely non-identifiable (Molenberghs) question — and the one the *deployment* model faces,
having been trained on a *distribution* of MARs — is **MNAR vs the best-fitting MAR FAMILY**: H0 =
{ MAR(β1′, β2=0) : maximize over β1′ (and ideally the MAR predictor/link) }. A composite/profiled
null can *re-fit* its predictor coefficient to absorb apparent slope.

Why this matters precisely here: at **β1>0**, `z_p` is **not** a valid shadow variable (R depends on
`z_p` via β1), so the high-ρ "signal" may be partly a refittable predictor-slope difference that a
profiled MAR null would absorb — which would pull the high-ρ Bayes error back up toward chance (the
real collapse). The point-null cannot see this. So:

- The point-null oracle is a *correct* ceiling for the *easier* question ("is this specific MAR
  distinguishable from this specific MNAR?"). It is **not** the ceiling for the deployment-relevant
  question ("is MNAR distinguishable from *any* MAR?").
- **We must not** read the §2 ρ-helps result as a MAR-vs-MNAR identifiability conclusion until the
  composite/profiled-null oracle is computed. The β1=0 shadow-variable result is clean (z_p is a
  valid shadow there); the β1>0 result is the one at risk.

## 4. Real-survey-X transfer (secondary)

Fitted-ConditionalGaussian oracle on 10 survey datasets (assumption: Gaussian (z_p, z_t), reported).
Measured ρ spans 0.05 (hmda) to 0.98 (cps1985). 60/80 cells distinguishable in the point-null sense,
consistent with the synthetic surface. Same point-vs-composite caveat applies; the fitted-Gaussian
assumption is a second caveat for real-X specifically.

## 5. Recommendation (next step — still oracle arm, CPU, no training)

Before the model arm: **upgrade the oracle to a composite/profiled MAR null** — at minimum profile
over β1′ (and the rate match); ideally over the MAR predictor choice and link. Recompute the
ρ-surface and check whether shadow-variable identification *survives* a re-fitting MAR. That profiled
surface is the deployment-relevant ceiling. Then, and only then, run the model arm in (a) a
near-chance control region, (b) the boundary, (c) moderate-signal, (d) strong-signal — and measure
the model-vs-ceiling gap.

This is a feasibility-design correction surfaced by the data, exactly the kind the charter's
no-weak-instrument / attributable-result discipline exists to catch. Nothing is concluded about
MAR-vs-MNAR until the profiled null is in.
