"""Propensity-score MCAR test.

Idea
----

Under MCAR the missingness indicator ``R_j`` for column ``j`` is
independent of the observed values. Therefore, a classifier trying to
predict ``R_j`` from the OTHER columns should perform no better than
chance (out-of-fold AUC = 0.5). If a flexible, mixed-type-aware
classifier (HistGradientBoosting / random forest) achieves AUC > 0.5
by more than chance, MCAR is rejected.

Null distribution
-----------------

Two calibrations are supported, selected by the ``null`` argument:

- ``null='analytical'`` (default, fast): under H₀ the pooled-across-
  columns AUC is asymptotically Normal with mean 0.5 and variance
  ``(n_pos + n_neg + 1) / (12 · n_pos · n_neg)`` (the Mann-Whitney-U
  asymptotic form). Zero refits beyond the observed OOF predictions.
- ``null='permutation'`` (slow, exact up to Monte Carlo noise): shuffle
  each column's missingness indicator before scoring, re-fit the
  classifier, accumulate the null distribution across permutations. Use
  when the asymptotic approximation's calibration is in doubt (e.g.
  very small n, extreme class imbalance).

The analytical null is the right default for bulk cache construction
(thousands of pairs) where the p-value is used as a feature scalar
rather than for formal hypothesis testing. The permutation null is
available for the rare case where exact calibration matters more than
throughput.

Why HistGradientBoosting default
--------------------------------

sklearn's ``HistGradientBoostingClassifier`` is typically 3–5× faster
than ``RandomForestClassifier`` on tabular data with comparable AUC,
natively handles NaN inputs (though we impute explicitly anyway), and
is well-suited to mixed integer-encoded / continuous columns. ``rf``
and ``gbm`` remain selectable for reference or A/B comparison.

Dependencies
------------

Requires scikit-learn. Lacuna depends on sklearn directly for other
paths (model training, ablation harness), so no extras gating is
needed.

Provenance
----------

Moved to Lacuna on 2026-04-20 from the (now-removed) pystatistics
``nonparametric_mcar`` subpackage. These tests were motivated
specifically by Lacuna's mcar-alternatives-bakeoff and are not
general-purpose statistical tools — see CLAUDE.md rule 8 (cross-
project scope boundary) for the rationale.
"""

from typing import List, Optional, Tuple
import warnings

import numpy as np
from scipy import stats

from lacuna.analysis.mcar.result import NonparametricMCARResult


_VALID_NULLS = ("analytical", "permutation")
_VALID_MODELS = ("hgb", "rf", "gbm")


def _require_sklearn():
    try:
        import sklearn  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "propensity_mcar_test requires scikit-learn. Install with "
            "`pip install scikit-learn`."
        ) from e


def _validate_inputs(data: np.ndarray, alpha: float, n_permutations: int,
                     cv_folds: int, model: str, n_estimators: int,
                     n_jobs: int, null: str) -> np.ndarray:
    """Validate and normalise the data matrix; raise on any issue."""
    if not isinstance(data, np.ndarray):
        data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValueError(
            f"data must be 2D (n_observations, n_variables); got shape {data.shape}"
        )
    if data.shape[0] < 10:
        raise ValueError(
            f"propensity_mcar_test needs at least 10 rows; got {data.shape[0]}"
        )
    if data.shape[1] < 2:
        raise ValueError(
            f"propensity_mcar_test needs at least 2 columns (one to predict, "
            f"one or more to predict from); got {data.shape[1]}"
        )
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1); got {alpha}")
    if null not in _VALID_NULLS:
        raise ValueError(f"null must be one of {_VALID_NULLS}; got {null!r}")
    if null == "permutation" and n_permutations < 1:
        raise ValueError(f"n_permutations must be >= 1; got {n_permutations}")
    if cv_folds < 2:
        raise ValueError(f"cv_folds must be >= 2; got {cv_folds}")
    if model not in _VALID_MODELS:
        raise ValueError(f"model must be one of {_VALID_MODELS}; got {model!r}")
    if n_estimators < 1:
        raise ValueError(f"n_estimators must be >= 1; got {n_estimators}")
    if n_jobs == 0:
        raise ValueError(
            f"n_jobs must be -1 (all cores) or a positive integer; got {n_jobs}"
        )
    return data


def _build_classifier(model: str, seed: int, n_estimators: int, n_jobs: int):
    """Construct a seeded sklearn classifier.

    ``hgb`` / ``gbm`` are inherently sequential; ``n_jobs`` is passed
    only to ``rf``. RandomForest is deterministic under parallel
    execution because each tree carries its own seeded RNG derived from
    ``random_state`` and the final score averages trees (commutative).
    """
    from sklearn.ensemble import (
        GradientBoostingClassifier,
        HistGradientBoostingClassifier,
        RandomForestClassifier,
    )

    if model == "hgb":
        return HistGradientBoostingClassifier(
            max_iter=n_estimators, random_state=seed,
        )
    if model == "rf":
        return RandomForestClassifier(
            n_estimators=n_estimators, random_state=seed, n_jobs=n_jobs,
        )
    return GradientBoostingClassifier(
        n_estimators=n_estimators, random_state=seed,
    )


def _full_feature_bank(data: np.ndarray):
    """Build the (n, d) imputed matrix + (n, d) missing-indicator matrix
    for ALL columns, once per call. Downstream per-target-column feature
    matrices are produced by slicing out two columns (the target's
    imputed value + its indicator) via ``_features_excluding``.

    This replaces a per-column ``_imputed_features`` that redundantly
    rebuilt (n, 2*(d-1)) features d times per call — an O(d) waste on
    wide datasets (credit_card_default: d=25, so ~24 redundant rebuilds
    per pair). Amortising once saves the factor-of-d.
    """
    miss = np.isnan(data)
    col_means = np.nanmean(data, axis=0)
    # Guard: fully-NaN columns impute to 0 (carry no signal).
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    imputed = np.where(miss, col_means, data)
    indicators = miss.astype(float)
    return imputed, indicators


def _features_excluding(imputed: np.ndarray, indicators: np.ndarray,
                        col: int) -> np.ndarray:
    """Return [imputed_without_col, indicators_without_col] as (n, 2(d-1))."""
    d = imputed.shape[1]
    keep = np.ones(d, dtype=bool)
    keep[col] = False
    return np.hstack([imputed[:, keep], indicators[:, keep]])


def _oof_proba(
    X: np.ndarray, y: np.ndarray, model: str, seed: int, cv_folds: int,
    n_estimators: int, n_jobs: int,
) -> Optional[np.ndarray]:
    """Out-of-fold P(y=1) predictions. Returns None if y is single-class
    or the minority class lacks enough rows for CV."""
    from sklearn.model_selection import StratifiedKFold

    if len(np.unique(y)) < 2:
        return None
    min_class_count = int(min(np.sum(y == 0), np.sum(y == 1)))
    k = min(cv_folds, min_class_count)
    if k < 2:
        return None

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    oof = np.zeros(len(y), dtype=float)
    for train_idx, test_idx in skf.split(X, y):
        clf = _build_classifier(model, seed, n_estimators, n_jobs)
        clf.fit(X[train_idx], y[train_idx])
        oof[test_idx] = clf.predict_proba(X[test_idx])[:, 1]
    return oof


def _auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Plain AUC; 0.5 when undefined (single-class input)."""
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y_true)) < 2:
        return 0.5
    return float(roc_auc_score(y_true, scores))


def _analytical_auc_pvalue(y_true: np.ndarray, auc: float) -> float:
    """One-sided p-value for ``AUC > 0.5`` via the Mann-Whitney-U
    asymptotic normal approximation:

        μ_H0 = 0.5
        σ²_H0 = (n_pos + n_neg + 1) / (12 · n_pos · n_neg)

    Returns 1.0 when either class is empty (test undefined)."""
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        return 1.0
    var_null = (n_pos + n_neg + 1.0) / (12.0 * n_pos * n_neg)
    if var_null <= 0:
        return 1.0
    z = (auc - 0.5) / np.sqrt(var_null)
    # One-sided (AUC > 0.5) → upper tail.
    return float(stats.norm.sf(z))


def _columns_with_missingness(data: np.ndarray) -> np.ndarray:
    miss = np.isnan(data)
    col_any_missing = miss.any(axis=0)
    col_any_observed = (~miss).any(axis=0)
    return np.where(col_any_missing & col_any_observed)[0]


def propensity_mcar_test(
    data,
    *,
    alpha: float = 0.05,
    model: str = "hgb",
    cv_folds: int = 3,
    null: str = "analytical",
    n_permutations: int = 199,
    n_estimators: int = 50,
    n_jobs: int = -1,
    seed: int = 0,
    verbose: bool = False,
) -> NonparametricMCARResult:
    """Propensity-score MCAR test.

    For each column with missingness, fit a classifier to predict that
    column's observed-vs-missing indicator from the other columns
    (column-mean imputed + per-column missing-indicator features).
    Compute out-of-fold AUC on each column, pool predictions across
    columns, and calibrate the pooled AUC against the chosen null.

    Parameters
    ----------
    data : array-like, shape (n_observations, n_variables)
        Data matrix with ``np.nan`` marking missing entries.
    alpha : float, default 0.05
        Significance level.
    model : {'hgb', 'rf', 'gbm'}, default 'hgb'
        Classifier family. 'hgb' = HistGradientBoostingClassifier
        (fastest, default); 'rf' = RandomForestClassifier; 'gbm' =
        GradientBoostingClassifier.
    cv_folds : int, default 3
        Stratified folds for the OOF prediction.
    null : {'analytical', 'permutation'}, default 'analytical'
        'analytical' — Mann-Whitney-U asymptotic Normal p-value on the
        pooled AUC. Cheap; the right default for cache-scale use.
        'permutation' — per-column y-shuffle + refit. Expensive; use
        when asymptotic calibration is in doubt.
    n_permutations : int, default 199
        Permutation count — ignored when ``null='analytical'``.
    n_estimators : int, default 50
        Trees / boosting rounds per fit.
    n_jobs : int, default -1
        Parallelism for ``model='rf'``. Ignored for 'hgb' / 'gbm'.
    seed : int, default 0
        Seed for classifier RNG, CV splits, and permutation draws.
    verbose : bool, default False
        Print per-column OOF AUCs.

    Returns
    -------
    NonparametricMCARResult
        ``statistic`` = pooled AUC − 0.5, clipped to [0, 0.5].
        ``extra`` contains per-column observed AUCs, the pooled AUC, the
        null mode, class counts, and echoed hyperparameters.

    Raises
    ------
    ValueError
        On malformed inputs, unknown ``null``/``model``, or a data
        matrix with no column carrying both observed and missing values.
    ImportError
        If scikit-learn is not installed.
    """
    _require_sklearn()
    data = _validate_inputs(
        data, alpha, n_permutations, cv_folds, model, n_estimators, n_jobs, null,
    )

    n_obs, n_vars = data.shape
    n_missing = int(np.isnan(data).sum())

    target_cols = _columns_with_missingness(data)
    if target_cols.size == 0:
        raise ValueError(
            "propensity_mcar_test requires at least one column with BOTH "
            "missing and observed values; got none."
        )

    per_col_auc: List[Tuple[int, float]] = []
    pooled_y: List[np.ndarray] = []
    pooled_proba: List[np.ndarray] = []

    # Build the full (n, d) imputed + (n, d) indicator banks ONCE; slice
    # per target column rather than rebuilding the feature matrix d times.
    imputed_all, ind_all = _full_feature_bank(data)

    # Per-column observed OOF predictions — computed once regardless of
    # null mode. This is the expensive step per column (one RF/HGB fit
    # per fold).
    for col in target_cols:
        X = _features_excluding(imputed_all, ind_all, int(col))
        y = np.isnan(data[:, col]).astype(int)

        proba = _oof_proba(X, y, model, seed, cv_folds, n_estimators, n_jobs)
        if proba is None:
            auc_val = 0.5
        else:
            auc_val = _auc(y, proba)
            pooled_y.append(y)
            pooled_proba.append(proba)
        per_col_auc.append((int(col), auc_val))
        if verbose:
            print(f"  col {col}: observed AUC = {auc_val:.4f}")

    if not pooled_y:
        # Every column was single-class after CV shrinkage; treat as
        # strong null — no signal.
        pooled_auc = 0.5
        p_value = 1.0
        extra_null = {"pooled_auc": 0.5}
    else:
        y_all = np.concatenate(pooled_y)
        p_all = np.concatenate(pooled_proba)
        pooled_auc = _auc(y_all, p_all)

        if null == "analytical":
            p_value = _analytical_auc_pvalue(y_all, pooled_auc)
            extra_null = {
                "pooled_auc": pooled_auc,
                "pooled_n_pos": int(y_all.sum()),
                "pooled_n_neg": int(len(y_all) - y_all.sum()),
            }
        else:
            rng = np.random.default_rng(seed)
            perm_seeds = rng.integers(0, 2**31 - 1, size=n_permutations)
            null_sum = np.zeros(n_permutations, dtype=float)
            valid_col_count = 0
            for col in target_cols:
                X = _features_excluding(imputed_all, ind_all, int(col))
                y = np.isnan(data[:, col]).astype(int)
                # Skip columns that had no valid OOF above.
                proba = _oof_proba(X, y, model, seed, cv_folds, n_estimators, n_jobs)
                if proba is None:
                    continue
                valid_col_count += 1
                for p, ps in enumerate(perm_seeds):
                    perm_rng = np.random.default_rng(ps)
                    y_perm = perm_rng.permutation(y)
                    proba_perm = _oof_proba(
                        X, y_perm, model, int(ps), cv_folds, n_estimators, n_jobs,
                    )
                    if proba_perm is None:
                        continue
                    null_sum[p] += _auc(y_perm, proba_perm)
            denom = max(valid_col_count, 1)
            null_mean_auc = null_sum / denom
            n_ge = int(np.sum(null_mean_auc >= pooled_auc))
            p_value = (1 + n_ge) / (1 + n_permutations)
            extra_null = {
                "pooled_auc": pooled_auc,
                "permutation_null_mean_auc": float(np.mean(null_mean_auc)),
            }

    statistic = float(np.clip(pooled_auc - 0.5, 0.0, 0.5))

    method_tag = "analytical" if null == "analytical" else f"perm={n_permutations}"
    return NonparametricMCARResult(
        statistic=statistic,
        p_value=float(p_value),
        rejected=p_value < alpha,
        alpha=alpha,
        method=f"Propensity-AUC ({model.upper()}, cv={cv_folds}, {method_tag})",
        n_observations=n_obs,
        n_variables=n_vars,
        n_missing_cells=n_missing,
        extra={
            "per_column_auc": per_col_auc,
            "null": null,
            "model": model,
            "cv_folds": cv_folds,
            "n_permutations": n_permutations if null == "permutation" else 0,
            "n_estimators": n_estimators,
            "n_jobs": n_jobs,
            "seed": seed,
            **extra_null,
        },
    )
