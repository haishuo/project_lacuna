"""
lacuna.analysis.generator_validation

Validate that Lacuna's synthetic missingness generators produce patterns
consistent with their declared mechanism (MCAR / MAR / MNAR).

Contract
--------
Input:  a `Generator` (or `GeneratorRegistry`) and sampling parameters.
Output: a `ValidationResult` (or list thereof) summarising Little's MCAR test
        reject-rate across N independent sampled datasets.

Interpretation
--------------
Under MCAR: reject-rate should be ~= alpha (Type-I error).
Under MAR : reject-rate should be high — Little's test typically has good
            power against MAR because missingness patterns produce
            detectable shifts in observed means.
Under MNAR: verdict is INDETERMINATE. Little's test is *underpowered*
            against self-censoring MNAR — symmetric truncation can leave
            observed means nearly equal across patterns, so the test
            stays silent even though the data is genuinely MNAR. We
            report Little's reject-rate for information but do not use
            it to pass/fail MNAR generators. Validate MNAR generators
            with mechanism-specific checks (e.g. observed-range
            truncation) in a separate utility.

A generator receives a "fail" verdict if its empirical reject-rate falls
outside the expected band for its declared class and that class is one
Little's test can decide (MCAR, MAR). "Fail" here means the generator
does not do what its name claims.

This module is OFFLINE. Do not import it from the training forward pass.
"""

from dataclasses import dataclass
from typing import List, Tuple
import warnings

import numpy as np
import torch

from pystatistics.mvnmle import little_mcar_test

from lacuna.core.rng import RNGState
from lacuna.core.types import MCAR, MAR, MNAR
from lacuna.generators.base import Generator
from lacuna.generators.registry import GeneratorRegistry


# =============================================================================
# Result container
# =============================================================================


VERDICT_PASS = "pass"
VERDICT_FAIL = "fail"
VERDICT_INDETERMINATE = "indeterminate"


@dataclass(frozen=True)
class ValidationResult:
    """Result of validating a single generator against Little's MCAR test.

    Attributes:
        generator_name: Human-readable generator name.
        generator_id: Registry ID.
        class_id: Declared mechanism (MCAR=0, MAR=1, MNAR=2).
        n_datasets: Number of independent datasets sampled.
        n_rows: Rows per sampled dataset.
        n_cols: Columns per sampled dataset.
        alpha: Significance level used in Little's test.
        reject_count: Number of sampled datasets where Little's test rejected.
        reject_rate: reject_count / n_usable.
        expected_reject_rate: alpha for MCAR, 1.0 for MAR, NaN for MNAR.
        verdict: One of "pass", "fail", "indeterminate".
            MCAR / MAR receive "pass" or "fail" from Little's reject-rate.
            MNAR always receives "indeterminate" — Little's test is
            underpowered against self-censoring MNAR, so the reject-rate
            is reported for information only.
        mean_statistic: Mean chi-squared test statistic across datasets.
        mean_p_value: Mean p-value across datasets.
        skipped_count: Datasets skipped because Little's test could not run
            (e.g. only one pattern present, or all-NaN columns). Skipped
            datasets do not count toward reject_rate.
    """

    generator_name: str
    generator_id: int
    class_id: int
    n_datasets: int
    n_rows: int
    n_cols: int
    alpha: float
    reject_count: int
    reject_rate: float
    expected_reject_rate: float
    verdict: str
    mean_statistic: float
    mean_p_value: float
    skipped_count: int


# =============================================================================
# Internal helpers
# =============================================================================


def _tensor_to_nan_array(x_full: torch.Tensor, r: torch.Tensor) -> np.ndarray:
    """Convert a Lacuna (X_full, R) pair into a numpy array with NaN for missing.

    Args:
        x_full: [n, d] float tensor of complete data.
        r: [n, d] bool tensor, True where observed.

    Returns:
        [n, d] float64 numpy array with NaN wherever r is False.

    Raises:
        ValueError: On shape or dtype mismatch (fail-loud per Coding Bible).
    """
    if x_full.dim() != 2 or r.dim() != 2:
        raise ValueError(
            f"Expected 2D tensors, got x_full.dim={x_full.dim()}, r.dim={r.dim()}"
        )
    if x_full.shape != r.shape:
        raise ValueError(
            f"Shape mismatch: x_full {tuple(x_full.shape)} vs r {tuple(r.shape)}"
        )
    if r.dtype != torch.bool:
        raise ValueError(f"r must be a bool tensor, got {r.dtype}")

    arr = x_full.detach().cpu().to(torch.float64).numpy().copy()
    mask = (~r).detach().cpu().numpy()
    arr[mask] = np.nan
    return arr


def _run_little_on_sample(x_full: torch.Tensor, r: torch.Tensor, alpha: float):
    """Run Little's MCAR test on a single (X_full, R) sample.

    Returns:
        (statistic, p_value, rejected) or None if the test could not run
        on this particular sample (e.g. only one missingness pattern).
    """
    arr = _tensor_to_nan_array(x_full, r)

    # Guard against degenerate inputs that would make Little's test fail
    # inside pystatistics: all-observed or all-missing produces zero or one
    # pattern and the chi-squared is undefined.
    n_missing = np.isnan(arr).sum()
    if n_missing == 0 or n_missing == arr.size:
        return None

    # Suppress pystatistics convergence warnings — we record them via
    # the skipped_count path if the test genuinely cannot run.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            res = little_mcar_test(arr, alpha=alpha)
        except Exception:  # noqa: BLE001 — defensive: skip unrunnable samples
            return None

    return float(res.statistic), float(res.p_value), bool(res.rejected)


# =============================================================================
# Public API
# =============================================================================


def validate_generator(
    generator: Generator,
    *,
    seed: int,
    n_datasets: int = 50,
    n_rows: int = 200,
    n_cols: int = 5,
    alpha: float = 0.05,
    mcar_upper_tolerance: float = 0.10,
    non_mcar_lower_bound: float = 0.80,
) -> ValidationResult:
    """Validate a single generator against Little's MCAR test.

    Procedure:
        1. Sample `n_datasets` independent (X_full, R) pairs from the generator.
        2. Convert each to a numpy array with NaN for missing entries.
        3. Run Little's MCAR test at level alpha on each.
        4. Record reject-rate and summary statistics.
        5. Decide verdict based on the generator's declared class:
             MCAR  : pass if reject_rate <= alpha + mcar_upper_tolerance
             MAR   : pass if reject_rate >= non_mcar_lower_bound
             MNAR  : always indeterminate (Little's is underpowered here)

    Args:
        generator: A Lacuna generator instance.
        seed: Base seed. Each dataset uses a child RNG spawned from this.
        n_datasets: Number of independent datasets to sample. Minimum 10.
        n_rows: Rows per sampled dataset. Must exceed n_cols.
        n_cols: Columns per sampled dataset. Minimum 2.
        alpha: Significance level for Little's test.
        mcar_upper_tolerance: Added to alpha to set the MCAR pass ceiling.
        non_mcar_lower_bound: Minimum reject-rate for MAR/MNAR to pass.

    Returns:
        ValidationResult.

    Raises:
        ValueError: On invalid arguments.
    """
    if n_datasets < 10:
        raise ValueError(f"n_datasets must be >= 10 for meaningful rate, got {n_datasets}")
    if n_cols < 2:
        raise ValueError(f"n_cols must be >= 2, got {n_cols}")
    if n_rows <= n_cols:
        raise ValueError(f"n_rows ({n_rows}) must exceed n_cols ({n_cols})")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    base_rng = RNGState(seed=seed)

    reject_count = 0
    skipped_count = 0
    stats: List[float] = []
    p_values: List[float] = []

    for i in range(n_datasets):
        # Deterministic per-dataset seed derived from base seed + index.
        child_rng = RNGState(seed=seed * 100_003 + i)
        x_full, r = generator.sample(child_rng, n_rows, n_cols)

        outcome = _run_little_on_sample(x_full, r, alpha)
        if outcome is None:
            skipped_count += 1
            continue

        stat, p, rejected = outcome
        stats.append(stat)
        p_values.append(p)
        if rejected:
            reject_count += 1

    # quiet: base_rng is only used if future extensions need a shared stream
    del base_rng

    n_usable = n_datasets - skipped_count
    reject_rate = reject_count / n_usable if n_usable > 0 else float("nan")
    mean_stat = float(np.mean(stats)) if stats else float("nan")
    mean_p = float(np.mean(p_values)) if p_values else float("nan")

    if generator.class_id == MCAR:
        expected = alpha
        if n_usable == 0:
            verdict = VERDICT_FAIL
        elif reject_rate <= alpha + mcar_upper_tolerance:
            verdict = VERDICT_PASS
        else:
            verdict = VERDICT_FAIL
    elif generator.class_id == MAR:
        expected = 1.0
        if n_usable == 0:
            verdict = VERDICT_FAIL
        elif reject_rate >= non_mcar_lower_bound:
            verdict = VERDICT_PASS
        else:
            verdict = VERDICT_FAIL
    elif generator.class_id == MNAR:
        # Little's test cannot reliably adjudicate MNAR. Report for
        # information only and leave the verdict indeterminate; validate
        # MNAR generators via mechanism-specific checks elsewhere.
        expected = float("nan")
        verdict = VERDICT_INDETERMINATE
    else:
        raise ValueError(f"Unknown class_id: {generator.class_id}")

    return ValidationResult(
        generator_name=generator.name,
        generator_id=generator.generator_id,
        class_id=generator.class_id,
        n_datasets=n_datasets,
        n_rows=n_rows,
        n_cols=n_cols,
        alpha=alpha,
        reject_count=reject_count,
        reject_rate=reject_rate,
        expected_reject_rate=expected,
        verdict=verdict,
        mean_statistic=mean_stat,
        mean_p_value=mean_p,
        skipped_count=skipped_count,
    )


def validate_registry(
    registry: GeneratorRegistry,
    *,
    seed: int,
    n_datasets: int = 50,
    n_rows: int = 200,
    n_cols: int = 5,
    alpha: float = 0.05,
    mcar_upper_tolerance: float = 0.10,
    non_mcar_lower_bound: float = 0.80,
) -> List[ValidationResult]:
    """Validate every generator in a registry.

    Each generator uses a distinct derived seed so their sampled datasets
    are independent.
    """
    results: List[ValidationResult] = []
    for g in registry:
        child_seed = seed * 10_007 + g.generator_id
        results.append(
            validate_generator(
                g,
                seed=child_seed,
                n_datasets=n_datasets,
                n_rows=n_rows,
                n_cols=n_cols,
                alpha=alpha,
                mcar_upper_tolerance=mcar_upper_tolerance,
                non_mcar_lower_bound=non_mcar_lower_bound,
            )
        )
    return results


# =============================================================================
# Reporting
# =============================================================================


_CLASS_NAMES = {MCAR: "MCAR", MAR: "MAR", MNAR: "MNAR"}


def format_results_table(results: List[ValidationResult]) -> str:
    """Render validation results as a fixed-width text table.

    Columns: ID, Name, Class, Reject-rate, Expected, Mean p, Skipped, Verdict.
    """
    if not results:
        return "(no results)"

    header = (
        f"{'ID':>3}  {'Name':<32}  {'Class':<5}  "
        f"{'Reject':>7}  {'Expect':>7}  {'Mean p':>7}  {'Skip':>4}  {'Verdict':<13}"
    )
    lines = [header, "-" * len(header)]
    for r in results:
        cls = _CLASS_NAMES.get(r.class_id, str(r.class_id))
        expected = (
            "  n/a  " if r.expected_reject_rate != r.expected_reject_rate  # NaN check
            else f"{r.expected_reject_rate:>7.3f}"
        )
        lines.append(
            f"{r.generator_id:>3}  {r.generator_name[:32]:<32}  {cls:<5}  "
            f"{r.reject_rate:>7.3f}  {expected}  "
            f"{r.mean_p_value:>7.3f}  {r.skipped_count:>4d}  {r.verdict:<13}"
        )
    return "\n".join(lines)


def summarize(results: List[ValidationResult]) -> Tuple[int, int, int]:
    """Return (n_pass, n_fail, n_indeterminate) across the given results."""
    n_pass = sum(1 for r in results if r.verdict == VERDICT_PASS)
    n_fail = sum(1 for r in results if r.verdict == VERDICT_FAIL)
    n_indet = sum(1 for r in results if r.verdict == VERDICT_INDETERMINATE)
    return n_pass, n_fail, n_indet
