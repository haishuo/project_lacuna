"""Method-of-moments MCAR test (Lacuna-local variant).

**This is not Little's test.** Little (1988) defines the MCAR
chi-square with the observed-data MLE plug-in, and the asymptotic
chi-square distribution theory leans on MLE's efficiency. This
function computes a statistic of the same shape but with
*pairwise-deletion sample moments* substituted for the MLE:

  - :math:`\\hat\\mu_j` = sample mean of column j over all observations
    where j is observed.
  - :math:`\\hat\\Sigma_{ij}` = sample covariance of columns (i, j) over
    all observations where both are observed.

Under MCAR these moment estimators are consistent, so the statistic is
approximately chi-square with the same degrees of freedom as Little's
test at moderate-to-large n. It is not asymptotically efficient, and
the finite-sample distribution deviates from chi-square more than
Little's does. In exchange it is dramatically faster: no EM / BFGS
iteration, just one pairwise-moment pass + per-pattern chi-square
contribution.

When to prefer this over ``pystatistics.mvnmle.little_mcar_test``:

- You are sweeping MCAR over many (thousands+) datasets for
  diagnostic screening.
- You need a p-value that is order-of-magnitude correct, not exact
  to 3 decimals.

When not to use this:

- Regulated submissions, published papers that cite Little 1988, or
  anywhere the precise asymptotic distribution matters. Use
  ``pystatistics.mvnmle.little_mcar_test`` — it matches Little's
  specification and the R ``BaylorEdPsych::LittleMCAR`` reference
  exactly.

Provenance
----------

Previously lived as ``pystatistics.mvnmle.mcar_test.mom_mcar_test``.
Moved to Lacuna on 2026-04-20 as part of the scope-boundary cleanup:
MoM was added to pystatistics in service of Lacuna's cache-scale
screening use case, not as a general-purpose statistical method.
See CLAUDE.md rule 8.

The original pystatistics version had a GPU-batched chi-square path
for large-pattern datasets. Lacuna's scale (n=1000, small numbers of
patterns per pair) doesn't benefit from that, so this implementation
is a simple per-pattern loop — ~100 LOC instead of ~500. Bit-for-bit
equal to the old CPU path on a single thread.

References
----------

Little, R. J. A. (1988). A test of missing completely at random for
multivariate data with missing values. JASA, 83(404), 1198-1202.
(Original MLE-plug-in formulation.)

Park, T. & Davis, C. S. (1993). A test of the missing data mechanism
for repeated categorical data. Biometrics, 49(2), 631-638. (Moment-
based variants in a related family.)
"""

import warnings
from typing import List, Tuple

import numpy as np
from scipy import stats

from pystatistics.mvnmle import MCARTestResult, PatternInfo, analyze_patterns


def _pairwise_deletion_moments(
    data: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pairwise-deletion mean (per column) and covariance (per column pair).

    Implementation: single matmul + elementwise division, O(n v^2) with
    no Python loop over columns.
    """
    mask = ~np.isnan(data)
    n_per_col = mask.sum(axis=0).astype(np.float64)
    if np.any(n_per_col < 1):
        raise ValueError("At least one column is fully missing")
    means = np.where(mask, data, 0.0).sum(axis=0) / n_per_col
    X_centered = np.where(mask, data - means, 0.0)
    pair_counts = mask.astype(np.float64).T @ mask.astype(np.float64)
    # Guard against no-co-observation pairs; well-formed data shouldn't
    # hit this, but clamp rather than divide by zero.
    pair_counts_safe = np.maximum(pair_counts, 1.0)
    cov = (X_centered.T @ X_centered) / pair_counts_safe
    # Symmetrise to wash out FP asymmetry from division.
    cov = 0.5 * (cov + cov.T)
    return means, cov


def _chi_square_contribution(
    mu: np.ndarray,
    sigma: np.ndarray,
    pattern: PatternInfo,
    condition_threshold: float,
    regularize: bool,
) -> Tuple[float, int]:
    """Per-pattern contribution to the MoM chi-square.

    Returns ``(contribution, was_regularized)`` where ``was_regularized``
    is 1 if the observed-covariance sub-block was too ill-conditioned
    for a direct solve and we fell back to ``np.linalg.pinv``, 0 otherwise.
    """
    if pattern.n_observed == 0:
        return 0.0, 0

    obs = pattern.observed_indices
    n_k = pattern.n_cases
    y_bar = pattern.data.mean(axis=0)  # (n_observed,)

    sigma_oo = sigma[np.ix_(obs, obs)]
    cond = float(np.linalg.cond(sigma_oo))

    if cond > condition_threshold:
        if not regularize:
            from pystatistics.core.exceptions import NumericalError
            raise NumericalError(
                f"Covariance sub-block for a missingness pattern is "
                f"ill-conditioned (cond={cond:.2e} > threshold="
                f"{condition_threshold:.0e}). Pass regularize=True to "
                f"fall back to Moore-Penrose pseudo-inverse."
            )
        warnings.warn(
            f"Covariance matrix for a missingness pattern is ill-"
            f"conditioned (cond={cond:.2e} > threshold="
            f"{condition_threshold:.0e}). Using Moore-Penrose pseudo-"
            f"inverse; chi-square contribution for this pattern may "
            f"have reduced precision. Pass regularize=False to raise "
            f"instead.",
            UserWarning, stacklevel=3,
        )
        sigma_oo_inv = np.linalg.pinv(sigma_oo)
        was_regularized = 1
    else:
        sigma_oo_inv = np.linalg.solve(
            sigma_oo, np.eye(len(obs), dtype=sigma_oo.dtype)
        )
        was_regularized = 0

    diff = y_bar - mu[obs]
    contribution = n_k * float(diff @ sigma_oo_inv @ diff)
    return contribution, was_regularized


def mom_mcar_test(
    data,
    alpha: float = 0.05,
    regularize: bool = True,
    condition_threshold: float = 1e12,
    drop_all_missing_rows: bool = True,
    verbose: bool = False,
) -> MCARTestResult:
    """Method-of-moments MCAR test.

    Parameters and return value mirror
    ``pystatistics.mvnmle.little_mcar_test``. The returned
    ``MCARTestResult.method`` is set to
    ``"Method-of-moments (pairwise-deletion plug-in)"`` so downstream
    code can disambiguate.

    Parameters
    ----------
    data : array-like, shape (n_observations, n_variables)
        Data matrix with ``np.nan`` marking missing entries.
    alpha : float, default 0.05
        Significance level for the ``rejected`` flag.
    regularize : bool, default True
        If True, ill-conditioned per-pattern covariance sub-blocks fall
        back to Moore-Penrose pseudo-inverse with a ``UserWarning``.
        If False, raises ``pystatistics.core.exceptions.NumericalError``.
    condition_threshold : float, default 1e12
        Matrices with ``cond > condition_threshold`` are treated as
        ill-conditioned.
    drop_all_missing_rows : bool, default True
        If True (default), rows that are entirely NaN are silently
        dropped before the test — they carry no information. Emits a
        ``UserWarning`` reporting how many were dropped.
    verbose : bool, default False
        Print per-step progress; intended for interactive use.
    """
    if hasattr(data, "values"):
        data_array = np.asarray(data.values, dtype=float)
    else:
        data_array = np.asarray(data, dtype=float)

    if data_array.ndim != 2:
        raise ValueError("Data must be 2-dimensional")

    if drop_all_missing_rows:
        all_nan_mask = np.all(np.isnan(data_array), axis=1)
        n_dropped = int(np.sum(all_nan_mask))
        if n_dropped > 0:
            warnings.warn(
                f"Dropping {n_dropped} row(s) with all values missing. "
                f"Such rows carry no information for the MCAR test. "
                f"Pass drop_all_missing_rows=False to opt out.",
                UserWarning, stacklevel=2,
            )
            data_array = data_array[~all_nan_mask]

    n_obs, n_vars = data_array.shape

    if verbose:
        print("Step 1: Computing pairwise-deletion moments...")
    mu_mom, sigma_mom = _pairwise_deletion_moments(data_array)

    if verbose:
        print("Step 2: Identifying missingness patterns...")
    patterns: List[PatternInfo] = analyze_patterns(data_array)

    if verbose:
        print("Step 3: Assembling per-pattern chi-square contributions...")

    # Short-circuit: no missing data → degenerate test.
    if len(patterns) == 1 and patterns[0].n_observed == n_vars:
        return MCARTestResult(
            statistic=0.0,
            df=0,
            p_value=1.0,
            rejected=False,
            alpha=alpha,
            patterns=patterns,
            n_patterns=1,
            n_patterns_used=0,
            ml_mean=mu_mom,
            ml_cov=sigma_mom,
            convergence_warnings=["No missing data - MCAR test not applicable"],
            method="Method-of-moments (pairwise-deletion plug-in)",
        )

    test_statistic = 0.0
    n_patterns_used = 0
    n_regularized = 0
    for pattern in patterns:
        contrib, was_reg = _chi_square_contribution(
            mu_mom, sigma_mom, pattern, condition_threshold, regularize,
        )
        if pattern.n_observed > 0:
            n_patterns_used += 1
        test_statistic += contrib
        n_regularized += was_reg

    df = sum(p.n_observed for p in patterns) - n_vars
    if df <= 0:
        raise ValueError(f"Invalid degrees of freedom: {df}")

    p_value = float(1.0 - stats.chi2.cdf(test_statistic, df))
    rejected = p_value < alpha

    convergence_warnings: List[str] = []
    if n_regularized > 0:
        convergence_warnings.append(
            f"{n_regularized} pattern(s) used Moore-Penrose pseudo-inverse "
            f"(ill-conditioned covariance sub-block)."
        )

    return MCARTestResult(
        statistic=test_statistic,
        df=df,
        p_value=p_value,
        rejected=rejected,
        alpha=alpha,
        patterns=patterns,
        n_patterns=len(patterns),
        n_patterns_used=n_patterns_used,
        ml_mean=mu_mom,
        ml_cov=sigma_mom,
        convergence_warnings=convergence_warnings,
        method="Method-of-moments (pairwise-deletion plug-in)",
    )
