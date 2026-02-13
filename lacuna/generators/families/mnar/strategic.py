"""MNAR strategic non-response generators."""

from typing import Tuple
import torch

from lacuna.core.rng import RNGState
from lacuna.core.types import MNAR
from lacuna.generators.base import Generator
from lacuna.generators.params import GeneratorParams
from ..base_data import sample_gaussian


class MNARGaming(Generator):
    """MNAR Gaming - values near incentive threshold go missing.

    Models strategic behavior where people hide values close to a threshold
    to game the system (e.g., income near tax brackets).

    Required params:
        incentive_threshold: The threshold value people game around
        gaming_radius: How close to threshold triggers gaming

    Optional params:
        miss_prob: Probability of missingness within radius (default: 0.8)
        affected_frac: Fraction of columns affected (default: 0.5)
    """

    def __init__(self, generator_id: int, name: str, params: GeneratorParams):
        if "incentive_threshold" not in params:
            raise ValueError("MNARGaming requires 'incentive_threshold' parameter")
        if "gaming_radius" not in params:
            raise ValueError("MNARGaming requires 'gaming_radius' parameter")
        super().__init__(generator_id, name, MNAR, params)

    def _compute_missingness(self, X: torch.Tensor, rng: RNGState) -> torch.Tensor:
        n, d = X.shape
        R = torch.ones(n, d, dtype=torch.bool)

        affected_frac = self.params.get("affected_frac", 0.5)
        n_affected = max(1, int(d * affected_frac))
        affected_cols = rng.choice(d, size=n_affected, replace=False)

        threshold = self.params["incentive_threshold"]
        radius = self.params["gaming_radius"]
        miss_prob = self.params.get("miss_prob", 0.8)

        for col in affected_cols:
            vals = X[:, col]
            near_threshold = (vals - threshold).abs() < radius
            missing_mask = near_threshold & (rng.rand(n) < miss_prob)
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


class MNARPrivacy(Generator):
    """MNAR Privacy - high values withheld for privacy.

    Models privacy concerns where individuals with high values (e.g., wealth)
    withhold information.

    Optional params:
        privacy_threshold: Percentile above which privacy kicks in (default: 80)
        miss_prob: Probability of withholding (default: 0.7)
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

        privacy_threshold = self.params.get("privacy_threshold", 80)
        miss_prob = self.params.get("miss_prob", 0.7)

        for col in affected_cols:
            vals = X[:, col]
            limit = torch.quantile(vals, privacy_threshold / 100.0)
            above = vals > limit
            missing_mask = above & (rng.rand(n) < miss_prob)
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


class MNARCompetitive(Generator):
    """MNAR Competitive - values near benchmark go missing.

    Models competitive environments where companies hide data near
    industry standards to avoid revealing competitive position.

    Required params:
        benchmark_value: The industry benchmark/standard value
        competitive_radius: How close to benchmark triggers hiding

    Optional params:
        miss_prob: Probability of hiding (default: 0.7)
        affected_frac: Fraction of columns affected (default: 0.5)
    """

    def __init__(self, generator_id: int, name: str, params: GeneratorParams):
        if "benchmark_value" not in params:
            raise ValueError("MNARCompetitive requires 'benchmark_value' parameter")
        if "competitive_radius" not in params:
            raise ValueError("MNARCompetitive requires 'competitive_radius' parameter")
        super().__init__(generator_id, name, MNAR, params)

    def _compute_missingness(self, X: torch.Tensor, rng: RNGState) -> torch.Tensor:
        n, d = X.shape
        R = torch.ones(n, d, dtype=torch.bool)

        affected_frac = self.params.get("affected_frac", 0.5)
        n_affected = max(1, int(d * affected_frac))
        affected_cols = rng.choice(d, size=n_affected, replace=False)

        benchmark = self.params["benchmark_value"]
        radius = self.params["competitive_radius"]
        miss_prob = self.params.get("miss_prob", 0.7)

        for col in affected_cols:
            vals = X[:, col]
            near_benchmark = (vals - benchmark).abs() < radius
            missing_mask = near_benchmark & (rng.rand(n) < miss_prob)
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
