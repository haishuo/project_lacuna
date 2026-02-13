"""MAR multi-column and multi-predictor generators."""

from typing import List, Tuple
import torch

from lacuna.core.rng import RNGState
from lacuna.core.types import MAR
from ...base import Generator
from ...params import GeneratorParams
from ..base_data import sample_gaussian


class MARMultiColumn(Generator):
    """MAR mechanism affecting MULTIPLE target columns.

    Creates a strong MAR signal: multiple columns have missingness
    that depends on a shared predictor column.

    For each target column j in targets:
        P(R_j = 0 | X) = sigmoid(alpha0 + alpha1 * X_predictor)

    Required params:
        alpha0: Intercept (controls baseline missingness)
        alpha1: Slope (controls dependence strength)

    Optional params:
        n_targets: Number of target columns to affect (default: 3)
        target_frac: Alternative: fraction of columns to affect (default: None)
        predictor_col_idx: Index of predictor column (default: 0)
        base_mean: Mean for Gaussian base data (default: 0.0)
        base_std: Std for Gaussian base data (default: 1.0)
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
                raise ValueError(f"MARMultiColumn requires '{key}' parameter")

    def _compute_missingness(
        self,
        X: torch.Tensor,
        rng: RNGState,
    ) -> torch.Tensor:
        """Compute missingness mask R based on data X."""
        n, d = X.shape

        if d < 2:
            raise ValueError("MARMultiColumn requires d >= 2")

        predictor = self.params.get("predictor_col_idx", 0)
        if predictor < 0:
            predictor = d + predictor
        predictor = predictor % d

        n_targets = self.params.get("n_targets", None)
        target_frac = self.params.get("target_frac", None)

        if n_targets is None and target_frac is None:
            n_targets = max(2, int(d * 0.3))
        elif target_frac is not None:
            n_targets = max(1, int(d * target_frac))

        n_targets = min(n_targets, d - 1)

        available_cols = [j for j in range(d) if j != predictor]
        target_indices = rng.choice(len(available_cols), size=n_targets, replace=False)
        targets = [available_cols[i] for i in target_indices]

        R = torch.ones(n, d, dtype=torch.bool)

        alpha0 = self.params["alpha0"]
        alpha1 = self.params["alpha1"]

        logits = alpha0 + alpha1 * X[:, predictor]
        p_missing = torch.sigmoid(logits)

        for target in targets:
            missing_mask = rng.rand(n) < p_missing
            R[:, target] = ~missing_mask

        for col in range(d):
            if R[:, col].sum() == 0:
                rand_row = rng.randint(0, n, (1,)).item()
                R[rand_row, col] = True

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
        """Apply missingness mechanism to existing data."""
        return self._compute_missingness(X, rng)


class MARMultiPredictor(Generator):
    """MAR mechanism with multiple predictors for a single target.

    Missingness depends on multiple observed columns:
    P(R_target = 0 | X) = sigmoid(alpha0 + sum_k alpha_k * X_k)

    Required params:
        alpha0: Intercept
        alphas: List of coefficients for predictor columns

    Optional params:
        target_col_idx: Index of target column (default: -1)
        base_mean: Mean for Gaussian base data (default: 0.0)
        base_std: Std for Gaussian base data (default: 1.0)
    """

    def __init__(
        self,
        generator_id: int,
        name: str,
        params: GeneratorParams,
    ):
        super().__init__(generator_id, name, MAR, params)

        required = ["alpha0", "alphas"]
        for key in required:
            if key not in params:
                raise ValueError(f"MARMultiPredictor requires '{key}' parameter")

    def _compute_missingness(
        self,
        X: torch.Tensor,
        rng: RNGState,
    ) -> torch.Tensor:
        """Compute missingness mask R based on data X."""
        n, d = X.shape
        alphas = self.params["alphas"]
        n_predictors = len(alphas)

        if d < n_predictors + 1:
            raise ValueError(
                f"MARMultiPredictor with {n_predictors} predictors "
                f"requires d >= {n_predictors + 1}"
            )

        target = self.params.get("target_col_idx", -1)
        if target < 0:
            target = d + target
        target = target % d

        predictor_cols = [j for j in range(d) if j != target][:n_predictors]

        R = torch.ones(n, d, dtype=torch.bool)

        alpha0 = self.params["alpha0"]
        logits = torch.full((n,), alpha0)

        for k, alpha_k in enumerate(alphas):
            if k < len(predictor_cols):
                logits = logits + alpha_k * X[:, predictor_cols[k]]

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


class MARTwoPredictor(Generator):
    """MAR mechanism with exactly two predictors.

    logit = alpha0 + alpha1*X_1 + alpha2*X_2

    Required params: alpha0, alpha1, alpha2
    Optional params: target_col_idx, base_mean, base_std
    """

    def __init__(self, generator_id: int, name: str, params: GeneratorParams):
        super().__init__(generator_id, name, MAR, params)
        for key in ["alpha0", "alpha1", "alpha2"]:
            if key not in params:
                raise ValueError(f"MARTwoPredictor requires '{key}' parameter")

    def _compute_missingness(self, X: torch.Tensor, rng: RNGState) -> torch.Tensor:
        n, d = X.shape
        if d < 3:
            raise ValueError("MARTwoPredictor requires d >= 3")

        target = self.params.get("target_col_idx", -1)
        if target < 0:
            target = d + target
        target = target % d

        preds = [j for j in range(d) if j != target][:2]
        logits = (self.params["alpha0"]
                  + self.params["alpha1"] * X[:, preds[0]]
                  + self.params["alpha2"] * X[:, preds[1]])
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


class MARThreePredictor(Generator):
    """MAR mechanism with exactly three predictors.

    logit = alpha0 + sum(alphas[i]*X_i) for i in 0..2

    Required params: alpha0, alphas (list of 3)
    Optional params: target_col_idx, base_mean, base_std
    """

    def __init__(self, generator_id: int, name: str, params: GeneratorParams):
        super().__init__(generator_id, name, MAR, params)
        for key in ["alpha0", "alphas"]:
            if key not in params:
                raise ValueError(f"MARThreePredictor requires '{key}' parameter")
        if len(params["alphas"]) != 3:
            raise ValueError("MARThreePredictor requires alphas of length 3")

    def _compute_missingness(self, X: torch.Tensor, rng: RNGState) -> torch.Tensor:
        n, d = X.shape
        if d < 4:
            raise ValueError("MARThreePredictor requires d >= 4")

        target = self.params.get("target_col_idx", -1)
        if target < 0:
            target = d + target
        target = target % d

        preds = [j for j in range(d) if j != target][:3]
        logits = torch.full((n,), self.params["alpha0"])
        for k, alpha_k in enumerate(self.params["alphas"]):
            logits = logits + alpha_k * X[:, preds[k]]

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


class MARManyPredictor(Generator):
    """MAR mechanism with 4+ predictors.

    logit = alpha0 + sum(alphas[i]*X_i) using all available predictor columns.

    Required params: alpha0, alphas (list)
    Optional params: target_col_idx, base_mean, base_std
    """

    def __init__(self, generator_id: int, name: str, params: GeneratorParams):
        super().__init__(generator_id, name, MAR, params)
        for key in ["alpha0", "alphas"]:
            if key not in params:
                raise ValueError(f"MARManyPredictor requires '{key}' parameter")

    def _compute_missingness(self, X: torch.Tensor, rng: RNGState) -> torch.Tensor:
        n, d = X.shape
        alphas = self.params["alphas"]
        n_predictors = len(alphas)
        if d < n_predictors + 1:
            raise ValueError(
                f"MARManyPredictor with {n_predictors} predictors requires d >= {n_predictors + 1}"
            )

        target = self.params.get("target_col_idx", -1)
        if target < 0:
            target = d + target
        target = target % d

        preds = [j for j in range(d) if j != target][:n_predictors]
        logits = torch.full((n,), self.params["alpha0"])
        for k, alpha_k in enumerate(alphas):
            logits = logits + alpha_k * X[:, preds[k]]

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


class MARWeightedPredictor(Generator):
    """MAR mechanism with weighted predictor combination.

    logit = alpha0 + sum(weights[i]*alphas[i]*X_i)

    Required params: alpha0, alphas (list), weights (list, same length as alphas)
    Optional params: target_col_idx, base_mean, base_std
    """

    def __init__(self, generator_id: int, name: str, params: GeneratorParams):
        super().__init__(generator_id, name, MAR, params)
        for key in ["alpha0", "alphas", "weights"]:
            if key not in params:
                raise ValueError(f"MARWeightedPredictor requires '{key}' parameter")
        if len(params["alphas"]) != len(params["weights"]):
            raise ValueError("alphas and weights must have the same length")

    def _compute_missingness(self, X: torch.Tensor, rng: RNGState) -> torch.Tensor:
        n, d = X.shape
        alphas = self.params["alphas"]
        weights = self.params["weights"]
        n_predictors = len(alphas)
        if d < n_predictors + 1:
            raise ValueError(
                f"MARWeightedPredictor with {n_predictors} predictors requires d >= {n_predictors + 1}"
            )

        target = self.params.get("target_col_idx", -1)
        if target < 0:
            target = d + target
        target = target % d

        preds = [j for j in range(d) if j != target][:n_predictors]
        logits = torch.full((n,), self.params["alpha0"])
        for k in range(n_predictors):
            logits = logits + weights[k] * alphas[k] * X[:, preds[k]]

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


class MARInteractive(Generator):
    """MAR mechanism with two predictors and interaction term.

    logit = alpha0 + alpha1*X1 + alpha2*X2 + alpha12*X1*X2

    Required params: alpha0, alpha1, alpha2, alpha12
    Optional params: target_col_idx, base_mean, base_std
    """

    def __init__(self, generator_id: int, name: str, params: GeneratorParams):
        super().__init__(generator_id, name, MAR, params)
        for key in ["alpha0", "alpha1", "alpha2", "alpha12"]:
            if key not in params:
                raise ValueError(f"MARInteractive requires '{key}' parameter")

    def _compute_missingness(self, X: torch.Tensor, rng: RNGState) -> torch.Tensor:
        n, d = X.shape
        if d < 3:
            raise ValueError("MARInteractive requires d >= 3")

        target = self.params.get("target_col_idx", -1)
        if target < 0:
            target = d + target
        target = target % d

        preds = [j for j in range(d) if j != target][:2]
        x1, x2 = X[:, preds[0]], X[:, preds[1]]

        logits = (self.params["alpha0"]
                  + self.params["alpha1"] * x1
                  + self.params["alpha2"] * x2
                  + self.params["alpha12"] * x1 * x2)
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


class MARThreeWayInteraction(Generator):
    """MAR mechanism with three predictors plus all interaction terms.

    logit = alpha0 + sum(alphas[i]*X_i) + sum(alpha_interactions[k]*X_i*X_j)
            + alpha_interactions[-1]*X_0*X_1*X_2

    Required params: alpha0, alphas (list of 3), alpha_interactions (list of 4:
                     [X0*X1, X0*X2, X1*X2, X0*X1*X2])
    Optional params: target_col_idx, base_mean, base_std
    """

    def __init__(self, generator_id: int, name: str, params: GeneratorParams):
        super().__init__(generator_id, name, MAR, params)
        for key in ["alpha0", "alphas", "alpha_interactions"]:
            if key not in params:
                raise ValueError(f"MARThreeWayInteraction requires '{key}' parameter")
        if len(params["alphas"]) != 3:
            raise ValueError("alphas must have length 3")
        if len(params["alpha_interactions"]) != 4:
            raise ValueError("alpha_interactions must have length 4 (3 pairwise + 1 three-way)")

    def _compute_missingness(self, X: torch.Tensor, rng: RNGState) -> torch.Tensor:
        n, d = X.shape
        if d < 4:
            raise ValueError("MARThreeWayInteraction requires d >= 4")

        target = self.params.get("target_col_idx", -1)
        if target < 0:
            target = d + target
        target = target % d

        preds = [j for j in range(d) if j != target][:3]
        x0, x1, x2 = X[:, preds[0]], X[:, preds[1]], X[:, preds[2]]
        alphas = self.params["alphas"]
        ai = self.params["alpha_interactions"]

        logits = (self.params["alpha0"]
                  + alphas[0] * x0 + alphas[1] * x1 + alphas[2] * x2
                  + ai[0] * x0 * x1 + ai[1] * x0 * x2 + ai[2] * x1 * x2
                  + ai[3] * x0 * x1 * x2)
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


class MARConditional(Generator):
    """MAR mechanism conditionally active based on a condition predictor.

    MAR only active when condition predictor exceeds threshold;
    otherwise no missingness is applied.

    logit = alpha0 + alpha1*X_pred (only for rows where X_cond > condition_threshold)

    Required params: condition_threshold, alpha0, alpha1
    Optional params: condition_col (default 0), target_col_idx, predictor_col_idx,
                     base_mean, base_std
    """

    def __init__(self, generator_id: int, name: str, params: GeneratorParams):
        super().__init__(generator_id, name, MAR, params)
        for key in ["condition_threshold", "alpha0", "alpha1"]:
            if key not in params:
                raise ValueError(f"MARConditional requires '{key}' parameter")

    def _compute_missingness(self, X: torch.Tensor, rng: RNGState) -> torch.Tensor:
        n, d = X.shape
        if d < 3:
            raise ValueError("MARConditional requires d >= 3 (condition + predictor + target)")

        target = self.params.get("target_col_idx", -1)
        if target < 0:
            target = d + target
        target = target % d

        condition_col = self.params.get("condition_col", 0) % d
        if condition_col == target:
            condition_col = (target + 1) % d

        # Predictor is different from both condition and target
        predictor = self.params.get("predictor_col_idx", None)
        if predictor is None:
            for j in range(d):
                if j != target and j != condition_col:
                    predictor = j
                    break
        else:
            predictor = predictor % d
            if predictor == target or predictor == condition_col:
                for j in range(d):
                    if j != target and j != condition_col:
                        predictor = j
                        break

        active = X[:, condition_col] > self.params["condition_threshold"]
        logits = self.params["alpha0"] + self.params["alpha1"] * X[:, predictor]
        p_missing = torch.sigmoid(logits)

        # Only apply missingness where condition is active
        p_missing = torch.where(active, p_missing, torch.zeros_like(p_missing))

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
