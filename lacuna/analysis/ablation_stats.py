"""
lacuna.analysis.ablation_stats

Paired statistics for ablation comparisons.

Given per-seed (or per-dataset) scores from two model configurations
(baseline vs. variant), answer:
    1. Is the difference significant?           -> paired_comparison
    2. What is the CI on the mean difference?   -> bootstrap_delta_ci
    3. Exact (sign-flip) p-value for the delta? -> paired_permutation

All three functions operate on PAIRED scalar scores — the i-th entry of
`baseline` and `variant` must come from the same seed / dataset / run.
If your comparisons are unpaired, these functions are the wrong tool.

This module is OFFLINE. Do not import it from the training forward pass.
"""

from dataclasses import dataclass
from typing import Tuple
import warnings

import numpy as np

from pystatistics.hypothesis import t_test, wilcox_test
from pystatistics.montecarlo import boot, boot_ci


# =============================================================================
# Result containers
# =============================================================================


@dataclass(frozen=True)
class PairedComparison:
    """Paired-sample significance test result.

    Attributes:
        name: Human-readable label.
        n: Number of paired observations.
        mean_delta: mean(variant - baseline). Positive means variant wins.
        t_p_value: Paired t-test p-value (two-sided).
        wilcoxon_p_value: Paired Wilcoxon signed-rank test p-value (two-sided).
        cohen_dz: Paired effect size = mean_delta / std(delta). NaN if
            std(delta) == 0.
    """
    name: str
    n: int
    mean_delta: float
    t_p_value: float
    wilcoxon_p_value: float
    cohen_dz: float


@dataclass(frozen=True)
class BootstrapDeltaCI:
    """Percentile-bootstrap CI on the paired mean delta.

    Attributes:
        name: Human-readable label.
        n: Number of paired observations.
        mean_delta: mean(variant - baseline) on the original sample.
        ci_low: Lower bound of the percentile CI at `conf_level`.
        ci_high: Upper bound of the percentile CI at `conf_level`.
        conf_level: Confidence level used (e.g. 0.95).
        n_boot: Number of bootstrap resamples.
    """
    name: str
    n: int
    mean_delta: float
    ci_low: float
    ci_high: float
    conf_level: float
    n_boot: int


@dataclass(frozen=True)
class PairedPermutation:
    """Sign-flip permutation test on paired deltas.

    Attributes:
        name: Human-readable label.
        n: Number of paired observations.
        observed_delta: mean(variant - baseline) on the original sample.
        p_value: Two-sided permutation p-value, computed as the fraction
            of sign-flipped means with absolute value >= |observed_delta|.
            Uses the standard (1 + count) / (1 + R) small-sample correction.
        n_perm: Number of random sign-flip permutations used.
        alternative: "two.sided", "less", or "greater".
    """
    name: str
    n: int
    observed_delta: float
    p_value: float
    n_perm: int
    alternative: str


# =============================================================================
# Input validation
# =============================================================================


def _validate_paired(
    baseline,
    variant,
    *,
    min_n: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Coerce inputs to 1-D float64 arrays and validate length / finiteness.

    Raises:
        ValueError: On shape mismatch, insufficient samples, or non-finite
            values (fail-loud per Coding Bible).
    """
    b = np.asarray(baseline, dtype=np.float64)
    v = np.asarray(variant, dtype=np.float64)
    if b.ndim != 1 or v.ndim != 1:
        raise ValueError(
            f"Inputs must be 1-D; got baseline.ndim={b.ndim}, "
            f"variant.ndim={v.ndim}"
        )
    if b.shape != v.shape:
        raise ValueError(
            f"Shape mismatch: baseline {b.shape} vs variant {v.shape}. "
            f"Inputs must be paired per-seed/dataset."
        )
    if b.shape[0] < min_n:
        raise ValueError(
            f"Need at least {min_n} paired observations, got {b.shape[0]}"
        )
    if not (np.isfinite(b).all() and np.isfinite(v).all()):
        raise ValueError("Inputs must be finite (no NaN / Inf).")
    return b, v


# =============================================================================
# Public API
# =============================================================================


def paired_comparison(
    baseline,
    variant,
    *,
    name: str = "variant - baseline",
) -> PairedComparison:
    """Paired t-test + paired Wilcoxon + Cohen's dz on (variant - baseline).

    Both tests are two-sided. Report both because the t-test is more
    powerful under approximate normality but the Wilcoxon is robust to
    outliers — if they disagree materially, that disagreement is itself
    information about the score distribution.

    Args:
        baseline: 1-D array of per-seed (or per-dataset) scores for the baseline.
        variant: 1-D array of per-seed scores for the variant, aligned with baseline.
        name: Label propagated to the result and any reports.

    Returns:
        PairedComparison.
    """
    b, v = _validate_paired(baseline, variant, min_n=2)
    delta = v - b
    n = delta.shape[0]
    mean_delta = float(np.mean(delta))
    std_delta = float(np.std(delta, ddof=1)) if n > 1 else 0.0

    # pystatistics paired t and Wilcoxon. Suppress internal warnings about
    # exact-mode ties etc. — we report p-values as-is.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t_res = t_test(v, b, paired=True)
        w_res = wilcox_test(v, b, paired=True)

    cohen_dz = mean_delta / std_delta if std_delta > 0 else float("nan")

    return PairedComparison(
        name=name,
        n=n,
        mean_delta=mean_delta,
        t_p_value=float(t_res.p_value),
        wilcoxon_p_value=float(w_res.p_value),
        cohen_dz=cohen_dz,
    )


def bootstrap_delta_ci(
    baseline,
    variant,
    *,
    n_boot: int = 2000,
    conf_level: float = 0.95,
    seed: int = 0,
    name: str = "variant - baseline",
) -> BootstrapDeltaCI:
    """Percentile-bootstrap CI on the paired mean delta.

    Resamples pair-indices with replacement (not baseline and variant
    independently) so the pairing structure is preserved.

    Args:
        baseline, variant: Paired 1-D arrays.
        n_boot: Number of bootstrap resamples. Minimum 200.
        conf_level: Confidence level, e.g. 0.95.
        seed: Seed for the bootstrap RNG (determinism — Coding Bible rule 6).
        name: Label.

    Returns:
        BootstrapDeltaCI.
    """
    if n_boot < 200:
        raise ValueError(f"n_boot must be >= 200 for a stable CI, got {n_boot}")
    if not 0.0 < conf_level < 1.0:
        raise ValueError(f"conf_level must be in (0, 1), got {conf_level}")

    b, v = _validate_paired(baseline, variant, min_n=2)
    delta = v - b
    n = delta.shape[0]
    observed = float(np.mean(delta))

    def _mean_of_delta(data, idx):
        return float(np.mean(data[idx]))

    # Run on CPU — offline analysis, determinism and portability matter
    # more than speed here.
    boot_sol = boot(delta, _mean_of_delta, R=n_boot, seed=seed, backend="cpu")
    ci_sol = boot_ci(boot_sol, conf=conf_level, type="perc")
    # ci_sol.ci is a dict keyed by CI type; 'perc' -> shape (n_conf, 2)
    perc = ci_sol.ci["perc"]
    ci_low = float(perc[0, 0])
    ci_high = float(perc[0, 1])

    return BootstrapDeltaCI(
        name=name,
        n=n,
        mean_delta=observed,
        ci_low=ci_low,
        ci_high=ci_high,
        conf_level=conf_level,
        n_boot=n_boot,
    )


def paired_permutation(
    baseline,
    variant,
    *,
    n_perm: int = 9999,
    alternative: str = "two.sided",
    seed: int = 0,
    name: str = "variant - baseline",
) -> PairedPermutation:
    """Sign-flip permutation test on paired deltas.

    For a paired design, the relevant permutation is random sign-flipping
    of each delta (not label-shuffling between independent groups — use
    pystatistics.montecarlo.permutation_test for that). Under the null
    of "no treatment effect", the sign of each delta is exchangeable.

    p-value uses the (1 + count) / (1 + n_perm) small-sample correction so
    the reported p is never exactly 0 — which is the right thing for
    Monte-Carlo estimates of tail probabilities.

    Args:
        baseline, variant: Paired 1-D arrays.
        n_perm: Number of random sign-flip permutations. Minimum 200.
        alternative: "two.sided", "less", or "greater".
            "less" tests variant < baseline; "greater" tests variant > baseline.
        seed: Seed for the permutation RNG.
        name: Label.

    Returns:
        PairedPermutation.
    """
    if n_perm < 200:
        raise ValueError(f"n_perm must be >= 200, got {n_perm}")
    if alternative not in ("two.sided", "less", "greater"):
        raise ValueError(
            f"alternative must be 'two.sided', 'less', or 'greater'; "
            f"got {alternative!r}"
        )

    b, v = _validate_paired(baseline, variant, min_n=2)
    delta = v - b
    n = delta.shape[0]
    observed = float(np.mean(delta))

    rng = np.random.default_rng(seed)
    # signs: shape (n_perm, n), entries in {-1, +1}
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, n))
    perm_means = (signs * delta).mean(axis=1)

    if alternative == "two.sided":
        count = int(np.sum(np.abs(perm_means) >= abs(observed)))
    elif alternative == "greater":
        count = int(np.sum(perm_means >= observed))
    else:  # "less"
        count = int(np.sum(perm_means <= observed))

    p_value = (1 + count) / (1 + n_perm)

    return PairedPermutation(
        name=name,
        n=n,
        observed_delta=observed,
        p_value=p_value,
        n_perm=n_perm,
        alternative=alternative,
    )


# =============================================================================
# Reporting
# =============================================================================


def format_ablation_row(
    *,
    name: str,
    comparison: PairedComparison,
    ci: BootstrapDeltaCI,
    permutation: PairedPermutation,
) -> str:
    """Render one ablation row as a human-readable string.

    Example:
        "disable_littles  Δ=-0.003 [95% CI: -0.011, +0.005]  "
        "t.p=0.32  w.p=0.41  perm.p=0.35  dz=-0.18  n=10"
    """
    return (
        f"{name:<24}  Δ={comparison.mean_delta:+.4f} "
        f"[{int(ci.conf_level * 100)}% CI: {ci.ci_low:+.4f}, {ci.ci_high:+.4f}]  "
        f"t.p={comparison.t_p_value:.3f}  "
        f"w.p={comparison.wilcoxon_p_value:.3f}  "
        f"perm.p={permutation.p_value:.3f}  "
        f"dz={comparison.cohen_dz:+.2f}  "
        f"n={comparison.n}"
    )
