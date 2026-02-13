"""MNAR sample selection generators."""

from typing import Tuple
import torch

from lacuna.core.rng import RNGState
from lacuna.core.types import MNAR
from lacuna.generators.base import Generator
from lacuna.generators.params import GeneratorParams
from ..base_data import sample_gaussian


class MNARTruncation(Generator):
    """MNAR Truncation (Heckman-style sample selection).

    Rows where X[:, selection_variable] < quantile(threshold) have elevated
    missingness across ALL columns. Models sample selection bias.

    Optional params:
        selection_threshold: Percentile threshold for selection (default: 40)
        selection_variable: Column index used for selection (default: 0)
        miss_prob: Probability of missingness for non-selected rows (default: 0.7)
        affected_frac: Fraction of columns affected (default: 0.5)
    """

    def __init__(self, generator_id: int, name: str, params: GeneratorParams):
        super().__init__(generator_id, name, MNAR, params)

    def _compute_missingness(self, X: torch.Tensor, rng: RNGState) -> torch.Tensor:
        n, d = X.shape
        R = torch.ones(n, d, dtype=torch.bool)

        sel_var = self.params.get("selection_variable", 0) % d
        sel_threshold = self.params.get("selection_threshold", 40)
        miss_prob = self.params.get("miss_prob", 0.7)
        affected_frac = self.params.get("affected_frac", 0.5)

        n_affected = max(1, int(d * affected_frac))
        affected_cols = rng.choice(d, size=n_affected, replace=False)

        # Rows below threshold on selection variable
        threshold_val = torch.quantile(X[:, sel_var], sel_threshold / 100.0)
        non_selected = X[:, sel_var] < threshold_val

        for col in affected_cols:
            missing_mask = non_selected & (rng.rand(n) < miss_prob)
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


class MNARBerkson(Generator):
    """MNAR Berkson selection bias.

    Rows selected into sample based on sum of X values. Non-selected rows
    have missing values.
    P(select) = sigmoid(selection_strength * sum(X, dim=1))

    Optional params:
        selection_strength: Strength of selection (default: 0.5)
        affected_frac: Fraction of columns affected (default: 0.5)
    """

    def __init__(self, generator_id: int, name: str, params: GeneratorParams):
        super().__init__(generator_id, name, MNAR, params)

    def _compute_missingness(self, X: torch.Tensor, rng: RNGState) -> torch.Tensor:
        n, d = X.shape
        R = torch.ones(n, d, dtype=torch.bool)

        selection_strength = self.params.get("selection_strength", 0.5)
        affected_frac = self.params.get("affected_frac", 0.5)

        n_affected = max(1, int(d * affected_frac))
        affected_cols = rng.choice(d, size=n_affected, replace=False)

        # Selection probability based on row sum
        row_sums = X.sum(dim=1)
        p_select = torch.sigmoid(selection_strength * row_sums)
        not_selected = rng.rand(n) >= p_select

        for col in affected_cols:
            R[:, col] = ~not_selected

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


class MNARVolunteer(Generator):
    """MNAR Volunteer bias.

    Extreme values are more likely to be observed (U-shaped observation).
    People with extreme experiences are more motivated to respond.
    P(observe) = sigmoid(volunteer_tendency * X^2)

    Optional params:
        volunteer_tendency: Strength of volunteer effect (default: 0.5)
        affected_frac: Fraction of columns affected (default: 0.5)
    """

    def __init__(self, generator_id: int, name: str, params: GeneratorParams):
        super().__init__(generator_id, name, MNAR, params)

    def _compute_missingness(self, X: torch.Tensor, rng: RNGState) -> torch.Tensor:
        n, d = X.shape
        R = torch.ones(n, d, dtype=torch.bool)

        volunteer_tendency = self.params.get("volunteer_tendency", 0.5)
        affected_frac = self.params.get("affected_frac", 0.5)

        n_affected = max(1, int(d * affected_frac))
        affected_cols = rng.choice(d, size=n_affected, replace=False)

        for col in affected_cols:
            vals = X[:, col]
            # P(observe) = sigmoid(tendency * X^2): extremes more observed
            p_observe = torch.sigmoid(volunteer_tendency * vals ** 2)
            missing_mask = rng.rand(n) >= p_observe
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


class MNARAttrition(Generator):
    """MNAR Attrition (progressive dropout).

    Columns are observed left-to-right. Probability of dropping out at
    each column depends on current value:
    P(drop_at_j) = sigmoid(attrition_rate * j + value_dependence * X_j)
    Once dropped, all subsequent columns are missing.

    Optional params:
        attrition_rate: Rate of progressive dropout (default: 0.3)
        value_dependence: How much value affects dropout (default: 0.5)
        base_mean: Mean for Gaussian base data (default: 0.0)
        base_std: Std for Gaussian base data (default: 1.0)
    """

    def __init__(self, generator_id: int, name: str, params: GeneratorParams):
        super().__init__(generator_id, name, MNAR, params)

    def _compute_missingness(self, X: torch.Tensor, rng: RNGState) -> torch.Tensor:
        n, d = X.shape
        R = torch.ones(n, d, dtype=torch.bool)

        attrition_rate = self.params.get("attrition_rate", 0.3)
        value_dep = self.params.get("value_dependence", 0.5)

        # Track which rows have dropped out
        dropped = torch.zeros(n, dtype=torch.bool)

        for j in range(d):
            # Once dropped, stay dropped
            R[dropped, j] = False

            # Compute dropout probability for this column
            logits = attrition_rate * j + value_dep * X[:, j]
            p_drop = torch.sigmoid(logits)
            new_drops = (~dropped) & (rng.rand(n) < p_drop)
            dropped = dropped | new_drops
            R[new_drops, j] = False

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


class MNARCompetingEvents(Generator):
    """MNAR Competing Events.

    If a value exceeds a threshold, a "competing event" occurs with
    event_prob, making that value missing. Models situations where
    extreme outcomes lead to dropout (e.g., death in clinical trials).

    Optional params:
        event_threshold: Percentile above which events can occur (default: 80)
        event_prob: Probability of competing event given threshold exceeded (default: 0.6)
        affected_frac: Fraction of columns affected (default: 0.5)
    """

    def __init__(self, generator_id: int, name: str, params: GeneratorParams):
        super().__init__(generator_id, name, MNAR, params)

    def _compute_missingness(self, X: torch.Tensor, rng: RNGState) -> torch.Tensor:
        n, d = X.shape
        R = torch.ones(n, d, dtype=torch.bool)

        event_threshold = self.params.get("event_threshold", 80)
        event_prob = self.params.get("event_prob", 0.6)
        affected_frac = self.params.get("affected_frac", 0.5)

        n_affected = max(1, int(d * affected_frac))
        affected_cols = rng.choice(d, size=n_affected, replace=False)

        for col in affected_cols:
            vals = X[:, col]
            threshold = torch.quantile(vals, event_threshold / 100.0)
            exceeds = vals > threshold
            event_occurs = exceeds & (rng.rand(n) < event_prob)
            R[:, col] = ~event_occurs

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
