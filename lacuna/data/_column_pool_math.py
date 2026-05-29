"""
lacuna.data._column_pool_math

Marginal-rate math shared by the per-column MNAR and MAR generator pools
(`mnar_column_pool`, `mar_column_pool`). One job: turn a desired marginal missing rate into the
generator intercepts / radii needed to hit it on a standard-normal predictor view, so mechanism
TYPE can be varied across pooled subtypes while missing QUANTITY is held ~fixed (the confound
control both pools rely on; ADR-0006).

Pure functions, no torch, fully deterministic (Coding Bible Rule 6).
"""

import math


def logit(p: float) -> float:
    """Inverse sigmoid. Raises on p outside (0, 1) (Rule 1)."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"logit requires 0 < p < 1, got {p}")
    return math.log(p / (1.0 - p))


_PROBIT_K = 0.416  # variance-inflation constant for E[sigmoid(a+bZ)] ≈ sigmoid(a/sqrt(1+k b^2))


def comp_beta0(r: float, slope: float) -> float:
    """Logistic intercept so E[sigmoid(beta0 + slope*Z)] ≈ r for Z ~ N(0, 1).

    Probit-style approximation E[sigmoid(a + bZ)] ≈ sigmoid(a / sqrt(1 + k b^2)), inverted:
    beta0 = logit(r) * sqrt(1 + k b^2). The constant k = 0.416 was empirically fit to minimise the
    marginal-rate error over r in [0.1, 0.4], slope in [0.5, 2.5] (worst |error| ≈ 0.013, vs ≈ 0.04
    for the textbook 0.61) — tight enough that every pooled subtype's realised rate sits within the
    confound-control band. Centres a sigmoid-on-predictor subtype's marginal rate at ~r regardless
    of slope.
    """
    return logit(r) * math.sqrt(1.0 + _PROBIT_K * slope * slope)


def norm_ppf(p: float) -> float:
    """Standard-normal quantile via Acklam's rational approximation (no SciPy dependency).

    Accuracy ~1e-9, far beyond what marginal-rate tuning needs. Raises on p outside (0, 1).
    Used for the probit MAR intercept and the centre-censoring radius of gaming/volunteer MNAR.
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"norm_ppf requires 0 < p < 1, got {p}")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]
    plow, phigh = 0.02425, 1.0 - 0.02425
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
    q = p - 0.5
    rr = q * q
    return (((((a[0]*rr+a[1])*rr+a[2])*rr+a[3])*rr+a[4])*rr+a[5])*q / (((((b[0]*rr+b[1])*rr+b[2])*rr+b[3])*rr+b[4])*rr+1.0)
