"""HSIC-based MCAR test.

Idea
----

Under MCAR, the observed values ``X`` are independent of the
missingness indicator matrix ``R``. The Hilbert-Schmidt Independence
Criterion (Gretton et al. 2005) is a kernel-based measure of
dependence that equals zero iff two random variables are independent
(for characteristic kernels like the Gaussian RBF) and is strictly
positive otherwise. We use HSIC(X_mean_imputed, R) with a Gaussian
RBF kernel and median-heuristic bandwidth, calibrated against a
permutation null (rows of R shuffled).

Test statistic
--------------

Biased HSIC estimator (Gretton et al. 2008):

    HSIC_b(X, R) = (1 / n^2) * tr(K_c L_c)

where ``K_c = H K H``, ``L_c = H L H``, ``H = I - (1/n) 1 1^T`` is the
centring matrix, ``K`` is the RBF kernel on ``X``, and ``L`` is the
RBF kernel on ``R``. Larger values indicate stronger dependence.

Null distribution
-----------------

Permutation: shuffle rows of ``R`` (which destroys any dependence on
``X`` while preserving the marginal distribution of missingness
patterns), recompute HSIC, and use the add-one-smoothed tail
probability as the p-value. Permutation is always valid; we avoid the
Gretton gamma approximation here to keep the implementation auditable
and numerically robust on small / degenerate inputs.

References
----------

Gretton, A., Bousquet, O., Smola, A., & Schölkopf, B. (2005).
Measuring statistical dependence with Hilbert-Schmidt norms. ALT.

Gretton, A., Fukumizu, K., Teo, C.H., Song, L., Schölkopf, B., &
Smola, A.J. (2008). A kernel statistical test of independence. NIPS.

Provenance
----------

Moved to Lacuna on 2026-04-20 from the (now-removed) pystatistics
``nonparametric_mcar`` subpackage. See CLAUDE.md rule 8 for the
scope-boundary rationale.
"""

import numpy as np
from scipy import stats

from lacuna.analysis.mcar.result import NonparametricMCARResult


_VALID_NULLS = ("gamma", "permutation")


def _validate_inputs(data: np.ndarray, alpha: float, n_permutations: int,
                     null: str) -> np.ndarray:
    if not isinstance(data, np.ndarray):
        data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValueError(
            f"data must be 2D (n_observations, n_variables); got shape {data.shape}"
        )
    if data.shape[0] < 10:
        raise ValueError(
            f"hsic_mcar_test needs at least 10 rows; got {data.shape[0]}"
        )
    if data.shape[1] < 2:
        raise ValueError(
            f"hsic_mcar_test needs at least 2 columns; got {data.shape[1]}"
        )
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1); got {alpha}")
    if null not in _VALID_NULLS:
        raise ValueError(f"null must be one of {_VALID_NULLS}; got {null!r}")
    if null == "permutation" and n_permutations < 1:
        raise ValueError(f"n_permutations must be >= 1; got {n_permutations}")
    return data


def _stochastic_impute(data: np.ndarray, seed: int) -> np.ndarray:
    """Stochastic single imputation: replace NaNs with draws from
    ``N(col_mean, col_std)`` using a seeded RNG.

    Plain column-mean imputation introduces a systematic artefact that
    breaks the HSIC null: rows with many missings get pulled toward the
    column means, which clusters them in ``X``-space AND correlates them
    with their rows in the missingness matrix ``R``. That spurious
    coupling rejects MCAR even when MCAR holds. Stochastic imputation
    with column std preserves the marginal column distribution and
    removes the centroid-clustering artefact; the seed keeps the test
    deterministic (Rule 6).
    """
    X = data.copy()
    col_means = np.nanmean(X, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    col_stds = np.nanstd(X, axis=0, ddof=0)
    col_stds = np.where(np.isnan(col_stds) | (col_stds == 0.0), 1.0, col_stds)
    nan_mask = np.isnan(X)
    rng = np.random.default_rng(seed)
    n, d = X.shape
    noise = rng.standard_normal((n, d)) * col_stds + col_means
    X[nan_mask] = noise[nan_mask]
    return X


def _pairwise_sq_distances(X: np.ndarray) -> np.ndarray:
    """Squared Euclidean pairwise distance matrix [n, n]."""
    sq_norms = np.sum(X * X, axis=1)
    # (a-b)^2 = a^2 - 2ab + b^2
    D = sq_norms[:, None] + sq_norms[None, :] - 2.0 * (X @ X.T)
    np.maximum(D, 0.0, out=D)  # clip tiny negative numerical noise
    return D


def _median_bandwidth(D_sq: np.ndarray, eps: float = 1e-12) -> float:
    """Median-heuristic bandwidth: sigma = sqrt(median of non-zero pairwise
    squared distances / 2). The /2 convention makes the RBF kernel
    exp(-d^2 / (2 sigma^2)) match the standard form in Gretton's
    papers."""
    tri = D_sq[np.triu_indices_from(D_sq, k=1)]
    tri = tri[tri > 0]
    if tri.size == 0:
        # Degenerate: all points coincide. Fall back to a tiny positive
        # bandwidth so the kernel is defined (will produce HSIC ~= 0).
        return eps
    med = float(np.median(tri))
    return float(np.sqrt(max(med, eps) / 2.0))


def _rbf_kernel(X: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian RBF kernel matrix. sigma is the bandwidth (not sigma^2)."""
    D_sq = _pairwise_sq_distances(X)
    return np.exp(-D_sq / (2.0 * sigma * sigma))


def _centre(M: np.ndarray) -> np.ndarray:
    """Return H M H where H = I - (1/n) 1 1^T, without materialising H."""
    return (
        M
        - M.mean(axis=0, keepdims=True)
        - M.mean(axis=1, keepdims=True)
        + M.mean()
    )


def _hsic_from_centred(Kc: np.ndarray, Lc: np.ndarray) -> float:
    """Biased HSIC estimator given already-centred K_c, L_c.

    HSIC_b = (1/n²) Σ_ij K_c[i,j] · L_c[i,j]
    """
    n = Kc.shape[0]
    return float(np.sum(Kc * Lc) / (n * n))


def _gamma_null_pvalue(K: np.ndarray, L: np.ndarray, Kc: np.ndarray,
                       Lc: np.ndarray, observed_hsic: float) -> float:
    """Gretton et al. 2008 gamma-approximation p-value for HSIC_b.

    Under H₀ (X ⊥ Y), HSIC_b has approximate distribution
    ``Gamma(α, β)`` with

        α = E[HSIC_b]² / Var[HSIC_b]
        β = Var[HSIC_b] / E[HSIC_b]

    where (from Theorem 1 of the paper + the accompanying MATLAB code
    distributed with it):

        μ_K = (Σ_{i≠j} K_ij) / (n(n-1))            # off-diagonal mean
        μ_L = (Σ_{i≠j} L_ij) / (n(n-1))
        E[HSIC_b] = (1 + μ_K μ_L − μ_K − μ_L) / n

        B_ij = (K_c_ij · L_c_ij / 6)²
        Var[HSIC_b] = ((Σ_ij B_ij − Σ_i B_ii) / (n (n−1))) ·
                      (72 (n−4)(n−5) / (n (n−1)(n−2)(n−3)))

    Returns the upper-tail p-value of the observed HSIC_b under that
    gamma fit.
    """
    n = K.shape[0]

    # Off-diagonal means of raw K, L.
    tr_K = np.trace(K)
    tr_L = np.trace(L)
    mu_K = (K.sum() - tr_K) / (n * (n - 1))
    mu_L = (L.sum() - tr_L) / (n * (n - 1))
    mean_hsic = (1.0 + mu_K * mu_L - mu_K - mu_L) / n

    # Variance via the B matrix.
    B = (Kc * Lc / 6.0) ** 2
    b_sum_off = B.sum() - np.trace(B)
    var_hsic = (
        b_sum_off / (n * (n - 1))
    ) * (72.0 * (n - 4) * (n - 5) / (n * (n - 1) * (n - 2) * (n - 3)))

    # Guard degenerate cases.
    if mean_hsic <= 0.0 or var_hsic <= 0.0:
        return 1.0

    shape = (mean_hsic ** 2) / var_hsic
    scale = var_hsic / mean_hsic
    return float(stats.gamma.sf(observed_hsic, a=shape, scale=scale))


def hsic_mcar_test(
    data,
    *,
    alpha: float = 0.05,
    null: str = "gamma",
    n_permutations: int = 199,
    seed: int = 0,
) -> NonparametricMCARResult:
    """Kernel-based (HSIC) MCAR test.

    Parameters
    ----------
    data : array-like, shape (n_observations, n_variables)
        Data matrix with ``np.nan`` marking missing entries.
    alpha : float, default 0.05
        Significance level.
    null : {'gamma', 'permutation'}, default 'gamma'
        'gamma' — Gretton et al. 2008 two-parameter gamma approximation
        to the H₀ distribution of HSIC_b. Closed-form, zero permutations,
        well-calibrated for n ≥ ~100. The standard null for HSIC
        independence testing.
        'permutation' — exact (up to Monte-Carlo noise) null via row
        permutations of the missingness matrix. ~50× slower for the same
        precision at n_permutations=199.
    n_permutations : int, default 199
        Permutation count — ignored when ``null='gamma'``.
    seed : int, default 0
        Seed for the stochastic imputation and (if used) permutation
        draws. The stochastic imputation makes the whole test
        deterministic given this seed.

    Returns
    -------
    NonparametricMCARResult
        ``statistic`` = biased HSIC estimator between stochastically-
        imputed observed values and the missingness-indicator matrix.
        ``extra`` contains the observed HSIC, the two median-heuristic
        bandwidths, the chosen null mode, and mode-specific diagnostics
        (``gamma_shape`` / ``gamma_scale`` when ``null='gamma'``;
        ``permutation_null_mean_hsic`` when ``null='permutation'``).

    Raises
    ------
    ValueError
        On malformed inputs or a fully-observed matrix (test undefined).
    """
    data = _validate_inputs(data, alpha, n_permutations, null)

    miss_mask = np.isnan(data)
    n_missing = int(miss_mask.sum())
    if n_missing == 0:
        raise ValueError(
            "hsic_mcar_test requires at least one missing cell in data; "
            "got a fully-observed matrix."
        )

    n, d = data.shape

    X = _stochastic_impute(data, seed)
    R = miss_mask.astype(float)

    D_sq_X = _pairwise_sq_distances(X)
    D_sq_R = _pairwise_sq_distances(R)
    sigma_X = _median_bandwidth(D_sq_X)
    sigma_R = _median_bandwidth(D_sq_R)

    K = np.exp(-D_sq_X / (2.0 * sigma_X * sigma_X))
    L = np.exp(-D_sq_R / (2.0 * sigma_R * sigma_R))

    # Centre ONCE. Row/column permutation of a matrix commutes with its
    # centring (P M P^T then centre = centre then P M_c P^T), so the
    # permutation-null loop can reuse the pre-centred matrices instead
    # of re-centring per permutation.
    Kc = _centre(K)
    Lc = _centre(L)

    observed_hsic = _hsic_from_centred(Kc, Lc)

    if null == "gamma":
        p_value = _gamma_null_pvalue(K, L, Kc, Lc, observed_hsic)
        # Echo the fit parameters for diagnostics.
        tr_K = np.trace(K); tr_L = np.trace(L)
        mu_K = (K.sum() - tr_K) / (n * (n - 1))
        mu_L = (L.sum() - tr_L) / (n * (n - 1))
        mean_hsic = (1.0 + mu_K * mu_L - mu_K - mu_L) / n
        B = (Kc * Lc / 6.0) ** 2
        b_sum_off = B.sum() - np.trace(B)
        var_hsic = (b_sum_off / (n * (n - 1))) * (
            72.0 * (n - 4) * (n - 5) / (n * (n - 1) * (n - 2) * (n - 3))
        )
        extra_null = {
            "null": "gamma",
            "gamma_mean_null": float(mean_hsic),
            "gamma_var_null": float(max(var_hsic, 0.0)),
            "gamma_shape": float(mean_hsic ** 2 / var_hsic) if var_hsic > 0 else float("nan"),
            "gamma_scale": float(var_hsic / mean_hsic) if mean_hsic > 0 else float("nan"),
        }
    else:
        # Permutation null: gather pre-centred L by the permutation.
        rng = np.random.default_rng(seed)
        n_ge = 0
        null_sum = 0.0
        for _ in range(n_permutations):
            perm = rng.permutation(n)
            # P L_c P^T; numpy's fancy indexing trick.
            Lc_perm = Lc[perm][:, perm]
            h_null = _hsic_from_centred(Kc, Lc_perm)
            null_sum += h_null
            if h_null >= observed_hsic:
                n_ge += 1
        p_value = (1 + n_ge) / (1 + n_permutations)
        extra_null = {
            "null": "permutation",
            "permutation_null_mean_hsic": float(null_sum / n_permutations),
            "n_permutations": n_permutations,
        }

    tag = "gamma" if null == "gamma" else f"perm={n_permutations}"
    return NonparametricMCARResult(
        statistic=float(observed_hsic),
        p_value=float(p_value),
        rejected=p_value < alpha,
        alpha=alpha,
        method=f"HSIC (Gaussian RBF, median bandwidth, {tag})",
        n_observations=n,
        n_variables=d,
        n_missing_cells=n_missing,
        extra={
            "observed_hsic": float(observed_hsic),
            "bandwidth_X": float(sigma_X),
            "bandwidth_R": float(sigma_R),
            "seed": seed,
            **extra_null,
        },
    )
