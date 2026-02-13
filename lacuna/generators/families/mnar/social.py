"""MNAR social desirability generators."""

from typing import Tuple
import torch

from lacuna.core.rng import RNGState
from lacuna.core.types import MNAR
from lacuna.generators.base import Generator
from lacuna.generators.params import GeneratorParams
from ..base_data import sample_gaussian


class MNARUnderReport(Generator):
    """MNAR Under-Reporting - high values go missing (income, spending).

    Models social desirability where high values are under-reported.
    Values above threshold_percentile go missing with under_report_prob.

    Optional params:
        threshold_percentile: Percentile above which under-reporting occurs (default: 75)
        under_report_prob: Probability of under-reporting (default: 0.6)
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

        threshold_pct = self.params.get("threshold_percentile", 75)
        under_report_prob = self.params.get("under_report_prob", 0.6)

        for col in affected_cols:
            vals = X[:, col]
            threshold = torch.quantile(vals, threshold_pct / 100.0)
            above = vals > threshold
            missing_mask = above & (rng.rand(n) < under_report_prob)
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


class MNAROverReport(Generator):
    """MNAR Over-Reporting - low values go missing (exercise, healthy eating).

    Models social desirability where low values are suppressed. People
    with low exercise amounts do not report.

    Optional params:
        threshold_percentile: Percentile below which over-reporting occurs (default: 25)
        over_report_prob: Probability of suppressing low values (default: 0.6)
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

        threshold_pct = self.params.get("threshold_percentile", 25)
        over_report_prob = self.params.get("over_report_prob", 0.6)

        for col in affected_cols:
            vals = X[:, col]
            threshold = torch.quantile(vals, threshold_pct / 100.0)
            below = vals < threshold
            missing_mask = below & (rng.rand(n) < over_report_prob)
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


class MNARNonLinearSocial(Generator):
    """MNAR Non-Linear Social Desirability - both extremes go missing.

    P(miss) = sigmoid(sensitivity * (X - center)^2 - 1)
    Values far from center_value are more likely missing.

    Optional params:
        center_value: Center around which values are acceptable (default: 0.0)
        sensitivity: How strongly extremes are censored (default: 1.0)
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

        center = self.params.get("center_value", 0.0)
        sensitivity = self.params.get("sensitivity", 1.0)

        for col in affected_cols:
            vals = X[:, col]
            deviation_sq = (vals - center) ** 2
            logits = sensitivity * deviation_sq - 1.0
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
