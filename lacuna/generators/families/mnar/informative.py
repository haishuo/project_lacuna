"""MNAR informative observation generators."""

from typing import Tuple
import torch

from lacuna.core.rng import RNGState
from lacuna.core.types import MNAR
from lacuna.generators.base import Generator
from lacuna.generators.params import GeneratorParams
from ..base_data import sample_gaussian


class MNARSymptomTriggered(Generator):
    """MNAR Symptom-Triggered observation.

    Abnormal values trigger observation (INVERSE of typical MNAR):
    - Normal values (|X| <= threshold) are MORE likely missing
    - Abnormal values (|X| > threshold) are MORE likely observed

    P(miss | normal) = trigger_prob
    P(miss | abnormal) = 1 - trigger_prob

    Optional params:
        symptom_threshold: Percentile defining abnormality (default: 75)
        trigger_prob: P(miss) for normal values (default: 0.6)
        affected_frac: Fraction of columns affected (default: 0.5)
    """

    def __init__(self, generator_id: int, name: str, params: GeneratorParams):
        super().__init__(generator_id, name, MNAR, params)

    def _compute_missingness(self, X: torch.Tensor, rng: RNGState) -> torch.Tensor:
        n, d = X.shape
        R = torch.ones(n, d, dtype=torch.bool)

        symptom_threshold = self.params.get("symptom_threshold", 75)
        trigger_prob = self.params.get("trigger_prob", 0.6)
        affected_frac = self.params.get("affected_frac", 0.5)

        n_affected = max(1, int(d * affected_frac))
        affected_cols = rng.choice(d, size=n_affected, replace=False)

        for col in affected_cols:
            vals = X[:, col].abs()
            threshold = torch.quantile(vals, symptom_threshold / 100.0)
            is_abnormal = vals > threshold

            # Normal values have HIGH miss prob, abnormal have LOW miss prob
            p_missing = torch.where(
                is_abnormal,
                torch.tensor(1.0 - trigger_prob),
                torch.tensor(trigger_prob),
            )

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


class MNARRiskBasedMonitoring(Generator):
    """MNAR Risk-Based Monitoring.

    High-risk rows (sum of absolute values above threshold) get better
    monitoring and thus LESS missing data. Low-risk rows have more missing.

    Optional params:
        risk_threshold: Percentile defining high-risk rows (default: 70)
        monitoring_boost: Reduction in miss prob for monitored rows (default: 0.5)
        base_miss_prob: Baseline missingness probability (default: 0.5)
        affected_frac: Fraction of columns affected (default: 0.5)
    """

    def __init__(self, generator_id: int, name: str, params: GeneratorParams):
        super().__init__(generator_id, name, MNAR, params)

    def _compute_missingness(self, X: torch.Tensor, rng: RNGState) -> torch.Tensor:
        n, d = X.shape
        R = torch.ones(n, d, dtype=torch.bool)

        risk_threshold = self.params.get("risk_threshold", 70)
        monitoring_boost = self.params.get("monitoring_boost", 0.5)
        base_miss_prob = self.params.get("base_miss_prob", 0.5)
        affected_frac = self.params.get("affected_frac", 0.5)

        n_affected = max(1, int(d * affected_frac))
        affected_cols = rng.choice(d, size=n_affected, replace=False)

        # Risk score per row = sum of absolute values
        risk_scores = X.abs().sum(dim=1)
        risk_limit = torch.quantile(risk_scores, risk_threshold / 100.0)
        high_risk = risk_scores > risk_limit

        for col in affected_cols:
            # High-risk rows get lower miss prob (better monitoring)
            p_missing = torch.where(
                high_risk,
                torch.tensor(base_miss_prob * (1.0 - monitoring_boost)),
                torch.tensor(base_miss_prob),
            )
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


class MNARAdaptiveSampling(Generator):
    """MNAR Adaptive Sampling.

    Columns with higher variance have more observations (lower missingness).
    P(miss_col_j) = sigmoid(-adaptation_rate * std(X_j) + baseline_prob)

    Optional params:
        adaptation_rate: How strongly variance reduces missingness (default: 1.0)
        baseline_prob: Baseline logit for missingness (default: 0.0)
    """

    def __init__(self, generator_id: int, name: str, params: GeneratorParams):
        super().__init__(generator_id, name, MNAR, params)

    def _compute_missingness(self, X: torch.Tensor, rng: RNGState) -> torch.Tensor:
        n, d = X.shape
        R = torch.ones(n, d, dtype=torch.bool)

        adaptation_rate = self.params.get("adaptation_rate", 1.0)
        baseline_prob = self.params.get("baseline_prob", 0.0)

        for col in range(d):
            col_std = X[:, col].std()
            logit = -adaptation_rate * col_std + baseline_prob
            p_missing = torch.sigmoid(logit)
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


class MNAROutcomeDependent(Generator):
    """MNAR Outcome-Dependent missingness.

    A specific column (outcome) drives missingness in other columns:
    P(miss_j) = sigmoid(beta0 + beta1 * X_outcome)

    Required params:
        beta0: Intercept
        beta1: Outcome dependence strength (must be non-zero)

    Optional params:
        outcome_col: Column index of outcome variable (default: 0)
        affected_frac: Fraction of other columns affected (default: 0.5)
        base_mean: Mean for Gaussian base data (default: 0.0)
        base_std: Std for Gaussian base data (default: 1.0)
    """

    def __init__(self, generator_id: int, name: str, params: GeneratorParams):
        if "beta0" not in params:
            raise ValueError("MNAROutcomeDependent requires 'beta0' parameter")
        if "beta1" not in params:
            raise ValueError("MNAROutcomeDependent requires 'beta1' parameter")
        if params["beta1"] == 0.0:
            raise ValueError("beta1 must be non-zero for MNAR outcome dependence")
        super().__init__(generator_id, name, MNAR, params)

    def _compute_missingness(self, X: torch.Tensor, rng: RNGState) -> torch.Tensor:
        n, d = X.shape
        R = torch.ones(n, d, dtype=torch.bool)

        beta0 = self.params["beta0"]
        beta1 = self.params["beta1"]
        outcome_col = self.params.get("outcome_col", 0) % d
        affected_frac = self.params.get("affected_frac", 0.5)

        # Select affected columns (excluding the outcome column)
        other_cols = [c for c in range(d) if c != outcome_col]
        n_affected = max(1, int(len(other_cols) * affected_frac))
        if n_affected > len(other_cols):
            n_affected = len(other_cols)
        affected_cols = rng.choice(len(other_cols), size=n_affected, replace=False)
        affected_cols = [other_cols[i] for i in affected_cols]

        # Outcome drives missingness in other columns
        outcome_vals = X[:, outcome_col]
        logits = beta0 + beta1 * outcome_vals
        p_missing = torch.sigmoid(logits)

        for col in affected_cols:
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
