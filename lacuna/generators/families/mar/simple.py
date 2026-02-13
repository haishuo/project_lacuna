"""MAR simple single-predictor generators."""

from typing import List, Tuple
import torch

from lacuna.core.rng import RNGState
from lacuna.core.types import MAR
from ...base import Generator
from ...params import GeneratorParams
from ..base_data import sample_gaussian, sample_gaussian_correlated


class MARLogistic(Generator):
    """MAR mechanism via logistic model (single target column).

    Missingness in target column depends on predictor column:
    P(R_target = 0 | X) = sigmoid(alpha0 + alpha1 * X_predictor)

    Required params:
        alpha0: Intercept (controls baseline missingness).
        alpha1: Slope (controls dependence strength).

    Optional params:
        target_col_idx: Index of target column (default: -1, last column)
        predictor_col_idx: Index of predictor column (default: 0)
        base_mean: Mean for Gaussian base data (default: 0.0)
        base_std: Std for Gaussian base data (default: 1.0)
        correlation: Correlation between columns for synthetic data (default: 0.0)
    """

    def __init__(
        self,
        generator_id: int,
        name: str,
        params: GeneratorParams,
    ):
        super().__init__(generator_id, name, MAR, params)

        required = ["alpha0", "alpha1"]
        for key in required:
            if key not in params:
                raise ValueError(f"MARLogistic requires '{key}' parameter")

    def _compute_missingness(
        self,
        X: torch.Tensor,
        rng: RNGState,
    ) -> torch.Tensor:
        """Compute missingness mask R based on data X."""
        n, d = X.shape

        if d < 2:
            raise ValueError("MARLogistic requires d >= 2")

        target = self.params.get("target_col_idx", -1)
        predictor = self.params.get("predictor_col_idx", 0)

        if target < 0:
            target = d + target
        if predictor < 0:
            predictor = d + predictor

        target = target % d
        predictor = predictor % d
        if predictor == target:
            predictor = (target + 1) % d

        R = torch.ones(n, d, dtype=torch.bool)

        alpha0 = self.params["alpha0"]
        alpha1 = self.params["alpha1"]

        logits = alpha0 + alpha1 * X[:, predictor]
        p_missing = torch.sigmoid(logits)

        missing_mask = rng.rand(n) < p_missing
        R[:, target] = ~missing_mask

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
        rho = self.params.get("correlation", 0.0)

        if rho > 0:
            X = sample_gaussian_correlated(rng.spawn(), n, d, mean=mean, std=std, rho=rho)
        else:
            X = sample_gaussian(rng.spawn(), n, d, mean=mean, std=std)

        R = self._compute_missingness(X, rng.spawn())

        return X, R

    def apply_to(
        self,
        X: torch.Tensor,
        rng: RNGState,
    ) -> torch.Tensor:
        """Apply missingness mechanism to existing data."""
        return self._compute_missingness(X, rng)


class MARProbit(Generator):
    """MAR mechanism via probit (normal CDF) link.

    P(R_target = 0 | X) = Phi(alpha0 + alpha1 * X_predictor)
    where Phi is the standard normal CDF.

    Required params: alpha0, alpha1
    Optional params: target_col_idx, predictor_col_idx, base_mean, base_std
    """

    def __init__(self, generator_id: int, name: str, params: GeneratorParams):
        super().__init__(generator_id, name, MAR, params)
        for key in ["alpha0", "alpha1"]:
            if key not in params:
                raise ValueError(f"MARProbit requires '{key}' parameter")

    def _compute_missingness(self, X: torch.Tensor, rng: RNGState) -> torch.Tensor:
        n, d = X.shape
        if d < 2:
            raise ValueError("MARProbit requires d >= 2")

        target = self.params.get("target_col_idx", -1)
        predictor = self.params.get("predictor_col_idx", 0)
        if target < 0:
            target = d + target
        if predictor < 0:
            predictor = d + predictor
        target = target % d
        predictor = predictor % d
        if predictor == target:
            predictor = (target + 1) % d

        alpha0 = self.params["alpha0"]
        alpha1 = self.params["alpha1"]

        z = alpha0 + alpha1 * X[:, predictor]
        p_missing = torch.distributions.Normal(0, 1).cdf(z)

        R = torch.ones(n, d, dtype=torch.bool)
        R[:, target] = ~(rng.rand(n) < p_missing)

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


class MARThreshold(Generator):
    """MAR mechanism via hard threshold on predictor.

    If X_predictor > threshold: P(miss) = miss_prob_above
    Else: P(miss) = miss_prob_below

    Required params: threshold, miss_prob_above, miss_prob_below
    Optional params: target_col_idx, predictor_col_idx, base_mean, base_std
    """

    def __init__(self, generator_id: int, name: str, params: GeneratorParams):
        super().__init__(generator_id, name, MAR, params)
        for key in ["threshold", "miss_prob_above", "miss_prob_below"]:
            if key not in params:
                raise ValueError(f"MARThreshold requires '{key}' parameter")

    def _compute_missingness(self, X: torch.Tensor, rng: RNGState) -> torch.Tensor:
        n, d = X.shape
        if d < 2:
            raise ValueError("MARThreshold requires d >= 2")

        target = self.params.get("target_col_idx", -1)
        predictor = self.params.get("predictor_col_idx", 0)
        if target < 0:
            target = d + target
        if predictor < 0:
            predictor = d + predictor
        target = target % d
        predictor = predictor % d
        if predictor == target:
            predictor = (target + 1) % d

        threshold = self.params["threshold"]
        above = X[:, predictor] > threshold
        p_missing = torch.where(
            above,
            torch.tensor(self.params["miss_prob_above"]),
            torch.tensor(self.params["miss_prob_below"]),
        )

        R = torch.ones(n, d, dtype=torch.bool)
        R[:, target] = ~(rng.rand(n) < p_missing)

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


class MARPolynomial(Generator):
    """MAR mechanism via polynomial logistic link.

    P(R_target = 0 | X) = sigmoid(alpha0 + alpha1*X + alpha2*X^2)

    Required params: alpha0, alpha1, alpha2
    Optional params: target_col_idx, predictor_col_idx, base_mean, base_std
    """

    def __init__(self, generator_id: int, name: str, params: GeneratorParams):
        super().__init__(generator_id, name, MAR, params)
        for key in ["alpha0", "alpha1", "alpha2"]:
            if key not in params:
                raise ValueError(f"MARPolynomial requires '{key}' parameter")

    def _compute_missingness(self, X: torch.Tensor, rng: RNGState) -> torch.Tensor:
        n, d = X.shape
        if d < 2:
            raise ValueError("MARPolynomial requires d >= 2")

        target = self.params.get("target_col_idx", -1)
        predictor = self.params.get("predictor_col_idx", 0)
        if target < 0:
            target = d + target
        if predictor < 0:
            predictor = d + predictor
        target = target % d
        predictor = predictor % d
        if predictor == target:
            predictor = (target + 1) % d

        xp = X[:, predictor]
        logits = self.params["alpha0"] + self.params["alpha1"] * xp + self.params["alpha2"] * xp ** 2
        p_missing = torch.sigmoid(logits)

        R = torch.ones(n, d, dtype=torch.bool)
        R[:, target] = ~(rng.rand(n) < p_missing)

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


class MARSpline(Generator):
    """MAR mechanism via piecewise linear (spline) logistic link.

    f(x) = slopes[i] * (x - knots[i]) for x in [knots[i], knots[i+1])
    P(R_target = 0 | X) = sigmoid(f(x))

    Required params: knots (list), slopes (list, len = len(knots)-1)
    Optional params: target_col_idx, predictor_col_idx, base_mean, base_std
    """

    def __init__(self, generator_id: int, name: str, params: GeneratorParams):
        super().__init__(generator_id, name, MAR, params)
        for key in ["knots", "slopes"]:
            if key not in params:
                raise ValueError(f"MARSpline requires '{key}' parameter")
        if len(params["slopes"]) != len(params["knots"]) - 1:
            raise ValueError("slopes must have length len(knots) - 1")

    def _compute_missingness(self, X: torch.Tensor, rng: RNGState) -> torch.Tensor:
        n, d = X.shape
        if d < 2:
            raise ValueError("MARSpline requires d >= 2")

        target = self.params.get("target_col_idx", -1)
        predictor = self.params.get("predictor_col_idx", 0)
        if target < 0:
            target = d + target
        if predictor < 0:
            predictor = d + predictor
        target = target % d
        predictor = predictor % d
        if predictor == target:
            predictor = (target + 1) % d

        knots = self.params["knots"]
        slopes = self.params["slopes"]
        xp = X[:, predictor]

        # Piecewise linear evaluation
        f = torch.zeros(n)
        for i in range(len(slopes)):
            lo = knots[i]
            hi = knots[i + 1]
            mask = (xp >= lo) & (xp < hi)
            f = f + mask.float() * slopes[i] * (xp - lo)

        # Values below first knot or above last knot: clamp to boundary segments
        below = xp < knots[0]
        f = f + below.float() * slopes[0] * (xp - knots[0])
        above = xp >= knots[-1]
        f = f + above.float() * slopes[-1] * (xp - knots[-1])

        p_missing = torch.sigmoid(f)

        R = torch.ones(n, d, dtype=torch.bool)
        R[:, target] = ~(rng.rand(n) < p_missing)

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


class MARStepFunction(Generator):
    """MAR mechanism via step function on predictor.

    P(R_target = 0 | X) = rates[i] when X_pred in [thresholds[i], thresholds[i+1])

    Required params: thresholds (list), rates (list, len = len(thresholds)-1)
    Optional params: target_col_idx, predictor_col_idx, base_mean, base_std
    """

    def __init__(self, generator_id: int, name: str, params: GeneratorParams):
        super().__init__(generator_id, name, MAR, params)
        for key in ["thresholds", "rates"]:
            if key not in params:
                raise ValueError(f"MARStepFunction requires '{key}' parameter")
        if len(params["rates"]) != len(params["thresholds"]) - 1:
            raise ValueError("rates must have length len(thresholds) - 1")

    def _compute_missingness(self, X: torch.Tensor, rng: RNGState) -> torch.Tensor:
        n, d = X.shape
        if d < 2:
            raise ValueError("MARStepFunction requires d >= 2")

        target = self.params.get("target_col_idx", -1)
        predictor = self.params.get("predictor_col_idx", 0)
        if target < 0:
            target = d + target
        if predictor < 0:
            predictor = d + predictor
        target = target % d
        predictor = predictor % d
        if predictor == target:
            predictor = (target + 1) % d

        thresholds = self.params["thresholds"]
        rates = self.params["rates"]
        xp = X[:, predictor]

        # Default rate for values outside range
        p_missing = torch.full((n,), rates[0])
        for i in range(len(rates)):
            lo = thresholds[i]
            hi = thresholds[i + 1]
            mask = (xp >= lo) & (xp < hi)
            p_missing = torch.where(mask, torch.tensor(rates[i]), p_missing)

        # Clamp to last rate for values above last threshold
        above = xp >= thresholds[-1]
        p_missing = torch.where(above, torch.tensor(rates[-1]), p_missing)

        R = torch.ones(n, d, dtype=torch.bool)
        R[:, target] = ~(rng.rand(n) < p_missing)

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
