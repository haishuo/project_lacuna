"""MNAR threshold/censoring generators."""

from typing import Tuple
import torch

from lacuna.core.rng import RNGState
from lacuna.core.types import MNAR
from lacuna.generators.base import Generator
from lacuna.generators.params import GeneratorParams
from ..base_data import sample_gaussian


class MNARThresholdLeft(Generator):
    """MNAR Threshold generator - values above threshold go missing.

    Missingness via hard threshold: values beyond tau are systematically missing.
    P(R_ij = 0 | X) = I(|X_ij| > tau) * p

    This uses a completely different functional form from sigmoid-based generators.

    Optional params:
        percentile: Threshold percentile (default: 70)
        miss_prob: Probability of missingness beyond threshold (default: 0.7)
        affected_frac: Fraction of columns with threshold (default: 0.5)
        use_absolute: If True, use |X_ij|, else use X_ij directly (default: True)
    """

    def __init__(self, generator_id: int, name: str, params: GeneratorParams):
        super().__init__(generator_id, name, MNAR, params)

    def _compute_missingness(self, X: torch.Tensor, rng: RNGState) -> torch.Tensor:
        n, d = X.shape
        R = torch.ones(n, d, dtype=torch.bool)

        affected_frac = self.params.get("affected_frac", 0.5)
        n_affected = max(1, int(d * affected_frac))
        affected_cols = rng.choice(d, size=n_affected, replace=False)

        percentile = self.params.get("percentile", 70)
        miss_prob = self.params.get("miss_prob", 0.7)
        use_absolute = self.params.get("use_absolute", True)

        for col in affected_cols:
            vals = X[:, col]
            if use_absolute:
                vals = vals.abs()

            threshold = torch.quantile(vals, percentile / 100.0)
            beyond_threshold = vals > threshold

            missing_mask = beyond_threshold & (rng.rand(n) < miss_prob)
            R[:, col] = ~missing_mask

        if R.sum() == 0:
            R[0, 0] = True

        return R

    def sample(
        self,
        rng: RNGState,
        n: int,
        d: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample synthetic data AND missingness."""
        mean = self.params.get("base_mean", 0.0)
        std = self.params.get("base_std", 1.0)

        X = sample_gaussian(rng.spawn(), n, d, mean=mean, std=std)
        R = self._compute_missingness(X, rng.spawn())

        return X, R

    def apply_to(
        self,
        X: torch.Tensor,
        rng: RNGState,
    ) -> torch.Tensor:
        """Apply threshold MNAR missingness to existing data."""
        return self._compute_missingness(X, rng)


class MNARThresholdRight(Generator):
    """MNAR Threshold generator - values BELOW threshold go missing.

    Opposite of MNARThresholdLeft: low values are censored.

    Optional params:
        percentile: Threshold percentile (default: 30)
        miss_prob: Probability of missingness below threshold (default: 0.7)
        affected_frac: Fraction of columns with threshold (default: 0.5)
    """

    def __init__(self, generator_id: int, name: str, params: GeneratorParams):
        super().__init__(generator_id, name, MNAR, params)

    def _compute_missingness(self, X: torch.Tensor, rng: RNGState) -> torch.Tensor:
        n, d = X.shape
        R = torch.ones(n, d, dtype=torch.bool)

        affected_frac = self.params.get("affected_frac", 0.5)
        n_affected = max(1, int(d * affected_frac))
        affected_cols = rng.choice(d, size=n_affected, replace=False)

        percentile = self.params.get("percentile", 30)
        miss_prob = self.params.get("miss_prob", 0.7)

        for col in affected_cols:
            vals = X[:, col]
            threshold = torch.quantile(vals, percentile / 100.0)
            below_threshold = vals < threshold

            missing_mask = below_threshold & (rng.rand(n) < miss_prob)
            R[:, col] = ~missing_mask

        if R.sum() == 0:
            R[0, 0] = True

        return R

    def sample(self, rng: RNGState, n: int, d: int) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self.params.get("base_mean", 0.0)
        std = self.params.get("base_std", 1.0)
        X = sample_gaussian(rng.spawn(), n, d, mean=mean, std=std)
        R = self._compute_missingness(X, rng.spawn())
        return X, R

    def apply_to(self, X: torch.Tensor, rng: RNGState) -> torch.Tensor:
        return self._compute_missingness(X, rng)


class MNARThresholdTwoSided(Generator):
    """MNAR Threshold generator - both extremes go missing.

    Values below lower_percentile OR above upper_percentile are censored.

    Optional params:
        lower_percentile: Lower threshold percentile (default: 20)
        upper_percentile: Upper threshold percentile (default: 80)
        miss_prob: Probability of missingness beyond thresholds (default: 0.7)
        affected_frac: Fraction of columns affected (default: 0.5)
    """

    def __init__(self, generator_id: int, name: str, params: GeneratorParams):
        super().__init__(generator_id, name, MNAR, params)

    def _compute_missingness(self, X: torch.Tensor, rng: RNGState) -> torch.Tensor:
        n, d = X.shape
        R = torch.ones(n, d, dtype=torch.bool)

        affected_frac = self.params.get("affected_frac", 0.5)
        n_affected = max(1, int(d * affected_frac))
        affected_cols = rng.choice(d, size=n_affected, replace=False)

        lower_pct = self.params.get("lower_percentile", 20)
        upper_pct = self.params.get("upper_percentile", 80)
        miss_prob = self.params.get("miss_prob", 0.7)

        for col in affected_cols:
            vals = X[:, col]
            lo = torch.quantile(vals, lower_pct / 100.0)
            hi = torch.quantile(vals, upper_pct / 100.0)
            extreme = (vals < lo) | (vals > hi)

            missing_mask = extreme & (rng.rand(n) < miss_prob)
            R[:, col] = ~missing_mask

        if R.sum() == 0:
            R[0, 0] = True

        return R

    def sample(self, rng: RNGState, n: int, d: int) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self.params.get("base_mean", 0.0)
        std = self.params.get("base_std", 1.0)
        X = sample_gaussian(rng.spawn(), n, d, mean=mean, std=std)
        R = self._compute_missingness(X, rng.spawn())
        return X, R

    def apply_to(self, X: torch.Tensor, rng: RNGState) -> torch.Tensor:
        return self._compute_missingness(X, rng)


class MNARSoftThreshold(Generator):
    """MNAR Soft Threshold generator - smooth sigmoid transition near threshold.

    Instead of a hard cutoff, uses sigmoid for smooth transition:
    P(miss) = sigmoid(steepness * (vals - threshold))

    Optional params:
        percentile: Threshold percentile (default: 70)
        steepness: Sigmoid steepness (default: 3.0)
        affected_frac: Fraction of columns affected (default: 0.5)
    """

    def __init__(self, generator_id: int, name: str, params: GeneratorParams):
        super().__init__(generator_id, name, MNAR, params)

    def _compute_missingness(self, X: torch.Tensor, rng: RNGState) -> torch.Tensor:
        n, d = X.shape
        R = torch.ones(n, d, dtype=torch.bool)

        affected_frac = self.params.get("affected_frac", 0.5)
        n_affected = max(1, int(d * affected_frac))
        affected_cols = rng.choice(d, size=n_affected, replace=False)

        percentile = self.params.get("percentile", 70)
        steepness = self.params.get("steepness", 3.0)

        for col in affected_cols:
            vals = X[:, col]
            threshold = torch.quantile(vals, percentile / 100.0)
            logits = steepness * (vals - threshold)
            p_missing = torch.sigmoid(logits)

            missing_mask = rng.rand(n) < p_missing
            R[:, col] = ~missing_mask

        if R.sum() == 0:
            R[0, 0] = True

        return R

    def sample(self, rng: RNGState, n: int, d: int) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self.params.get("base_mean", 0.0)
        std = self.params.get("base_std", 1.0)
        X = sample_gaussian(rng.spawn(), n, d, mean=mean, std=std)
        R = self._compute_missingness(X, rng.spawn())
        return X, R

    def apply_to(self, X: torch.Tensor, rng: RNGState) -> torch.Tensor:
        return self._compute_missingness(X, rng)


class MNARMultiThreshold(Generator):
    """MNAR Multi-Threshold generator - multiple thresholds per column.

    Each threshold has its own miss probability. Values beyond each threshold
    accumulate missingness probability.

    Required params:
        percentiles: List of percentile thresholds (e.g., [50, 80])
        miss_probs: List of miss probabilities per threshold (e.g., [0.3, 0.7])

    Optional params:
        affected_frac: Fraction of columns affected (default: 0.5)
    """

    def __init__(self, generator_id: int, name: str, params: GeneratorParams):
        if "percentiles" not in params:
            raise ValueError("MNARMultiThreshold requires 'percentiles' parameter")
        if "miss_probs" not in params:
            raise ValueError("MNARMultiThreshold requires 'miss_probs' parameter")
        if len(params["percentiles"]) != len(params["miss_probs"]):
            raise ValueError("percentiles and miss_probs must have the same length")
        super().__init__(generator_id, name, MNAR, params)

    def _compute_missingness(self, X: torch.Tensor, rng: RNGState) -> torch.Tensor:
        n, d = X.shape
        R = torch.ones(n, d, dtype=torch.bool)

        affected_frac = self.params.get("affected_frac", 0.5)
        n_affected = max(1, int(d * affected_frac))
        affected_cols = rng.choice(d, size=n_affected, replace=False)

        percentiles = self.params["percentiles"]
        miss_probs = self.params["miss_probs"]

        for col in affected_cols:
            vals = X[:, col]
            combined_missing = torch.zeros(n, dtype=torch.bool)

            for pct, mp in zip(percentiles, miss_probs):
                threshold = torch.quantile(vals, pct / 100.0)
                beyond = vals > threshold
                tier_missing = beyond & (rng.rand(n) < mp)
                combined_missing = combined_missing | tier_missing

            R[:, col] = ~combined_missing

        if R.sum() == 0:
            R[0, 0] = True

        return R

    def sample(self, rng: RNGState, n: int, d: int) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self.params.get("base_mean", 0.0)
        std = self.params.get("base_std", 1.0)
        X = sample_gaussian(rng.spawn(), n, d, mean=mean, std=std)
        R = self._compute_missingness(X, rng.spawn())
        return X, R

    def apply_to(self, X: torch.Tensor, rng: RNGState) -> torch.Tensor:
        return self._compute_missingness(X, rng)


class MNARQuantile70(Generator):
    """MNAR fixed 70th percentile threshold - shorthand for ThresholdLeft(percentile=70).

    Optional params:
        miss_prob: Probability of missingness beyond threshold (default: 0.7)
        affected_frac: Fraction of columns affected (default: 0.5)
    """

    def __init__(self, generator_id: int, name: str, params: GeneratorParams):
        super().__init__(generator_id, name, MNAR, params)

    def _compute_missingness(self, X: torch.Tensor, rng: RNGState) -> torch.Tensor:
        n, d = X.shape
        R = torch.ones(n, d, dtype=torch.bool)

        affected_frac = self.params.get("affected_frac", 0.5)
        n_affected = max(1, int(d * affected_frac))
        affected_cols = rng.choice(d, size=n_affected, replace=False)

        miss_prob = self.params.get("miss_prob", 0.7)

        for col in affected_cols:
            vals = X[:, col].abs()
            threshold = torch.quantile(vals, 0.70)
            beyond = vals > threshold
            missing_mask = beyond & (rng.rand(n) < miss_prob)
            R[:, col] = ~missing_mask

        if R.sum() == 0:
            R[0, 0] = True

        return R

    def sample(self, rng: RNGState, n: int, d: int) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self.params.get("base_mean", 0.0)
        std = self.params.get("base_std", 1.0)
        X = sample_gaussian(rng.spawn(), n, d, mean=mean, std=std)
        R = self._compute_missingness(X, rng.spawn())
        return X, R

    def apply_to(self, X: torch.Tensor, rng: RNGState) -> torch.Tensor:
        return self._compute_missingness(X, rng)


class MNARQuantile80(Generator):
    """MNAR fixed 80th percentile threshold.

    Optional params:
        miss_prob: Probability of missingness beyond threshold (default: 0.7)
        affected_frac: Fraction of columns affected (default: 0.5)
    """

    def __init__(self, generator_id: int, name: str, params: GeneratorParams):
        super().__init__(generator_id, name, MNAR, params)

    def _compute_missingness(self, X: torch.Tensor, rng: RNGState) -> torch.Tensor:
        n, d = X.shape
        R = torch.ones(n, d, dtype=torch.bool)

        affected_frac = self.params.get("affected_frac", 0.5)
        n_affected = max(1, int(d * affected_frac))
        affected_cols = rng.choice(d, size=n_affected, replace=False)

        miss_prob = self.params.get("miss_prob", 0.7)

        for col in affected_cols:
            vals = X[:, col].abs()
            threshold = torch.quantile(vals, 0.80)
            beyond = vals > threshold
            missing_mask = beyond & (rng.rand(n) < miss_prob)
            R[:, col] = ~missing_mask

        if R.sum() == 0:
            R[0, 0] = True

        return R

    def sample(self, rng: RNGState, n: int, d: int) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self.params.get("base_mean", 0.0)
        std = self.params.get("base_std", 1.0)
        X = sample_gaussian(rng.spawn(), n, d, mean=mean, std=std)
        R = self._compute_missingness(X, rng.spawn())
        return X, R

    def apply_to(self, X: torch.Tensor, rng: RNGState) -> torch.Tensor:
        return self._compute_missingness(X, rng)


class MNARQuantile90(Generator):
    """MNAR fixed 90th percentile threshold.

    Optional params:
        miss_prob: Probability of missingness beyond threshold (default: 0.7)
        affected_frac: Fraction of columns affected (default: 0.5)
    """

    def __init__(self, generator_id: int, name: str, params: GeneratorParams):
        super().__init__(generator_id, name, MNAR, params)

    def _compute_missingness(self, X: torch.Tensor, rng: RNGState) -> torch.Tensor:
        n, d = X.shape
        R = torch.ones(n, d, dtype=torch.bool)

        affected_frac = self.params.get("affected_frac", 0.5)
        n_affected = max(1, int(d * affected_frac))
        affected_cols = rng.choice(d, size=n_affected, replace=False)

        miss_prob = self.params.get("miss_prob", 0.7)

        for col in affected_cols:
            vals = X[:, col].abs()
            threshold = torch.quantile(vals, 0.90)
            beyond = vals > threshold
            missing_mask = beyond & (rng.rand(n) < miss_prob)
            R[:, col] = ~missing_mask

        if R.sum() == 0:
            R[0, 0] = True

        return R

    def sample(self, rng: RNGState, n: int, d: int) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self.params.get("base_mean", 0.0)
        std = self.params.get("base_std", 1.0)
        X = sample_gaussian(rng.spawn(), n, d, mean=mean, std=std)
        R = self._compute_missingness(X, rng.spawn())
        return X, R

    def apply_to(self, X: torch.Tensor, rng: RNGState) -> torch.Tensor:
        return self._compute_missingness(X, rng)


class MNARColumnSpecificThreshold(Generator):
    """MNAR Column-Specific Threshold - each column gets its own percentile.

    Each affected column's threshold percentile is drawn uniformly from
    the given range.

    Optional params:
        percentile_range: (low, high) range for per-column percentiles (default: [60, 90])
        miss_prob: Probability of missingness beyond threshold (default: 0.7)
        affected_frac: Fraction of columns affected (default: 0.5)
    """

    def __init__(self, generator_id: int, name: str, params: GeneratorParams):
        super().__init__(generator_id, name, MNAR, params)

    def _compute_missingness(self, X: torch.Tensor, rng: RNGState) -> torch.Tensor:
        n, d = X.shape
        R = torch.ones(n, d, dtype=torch.bool)

        affected_frac = self.params.get("affected_frac", 0.5)
        n_affected = max(1, int(d * affected_frac))
        affected_cols = rng.choice(d, size=n_affected, replace=False)

        pct_range = self.params.get("percentile_range", [60, 90])
        miss_prob = self.params.get("miss_prob", 0.7)

        # Draw per-column percentiles from uniform range
        col_percentiles = rng.rand(n_affected) * (pct_range[1] - pct_range[0]) + pct_range[0]

        for i, col in enumerate(affected_cols):
            vals = X[:, col].abs()
            pct = col_percentiles[i].item()
            threshold = torch.quantile(vals, pct / 100.0)
            beyond = vals > threshold

            missing_mask = beyond & (rng.rand(n) < miss_prob)
            R[:, col] = ~missing_mask

        if R.sum() == 0:
            R[0, 0] = True

        return R

    def sample(self, rng: RNGState, n: int, d: int) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self.params.get("base_mean", 0.0)
        std = self.params.get("base_std", 1.0)
        X = sample_gaussian(rng.spawn(), n, d, mean=mean, std=std)
        R = self._compute_missingness(X, rng.spawn())
        return X, R

    def apply_to(self, X: torch.Tensor, rng: RNGState) -> torch.Tensor:
        return self._compute_missingness(X, rng)
