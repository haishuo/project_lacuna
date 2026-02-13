"""MAR generators with different predictor types."""

from typing import Tuple
import torch

from lacuna.core.rng import RNGState
from lacuna.core.types import MAR
from ...base import Generator
from ...params import GeneratorParams
from ..base_data import sample_gaussian


class MARContinuousPredictor(Generator):
    """MAR mechanism with continuous predictor affecting multiple targets.

    Standard logistic link with a continuous predictor column,
    applied to multiple target columns.

    Required params: alpha0, alpha1
    Optional params: target_frac (default 0.3), predictor_col_idx, base_mean, base_std
    """

    def __init__(self, generator_id: int, name: str, params: GeneratorParams):
        super().__init__(generator_id, name, MAR, params)
        for key in ["alpha0", "alpha1"]:
            if key not in params:
                raise ValueError(f"MARContinuousPredictor requires '{key}' parameter")

    def _compute_missingness(self, X: torch.Tensor, rng: RNGState) -> torch.Tensor:
        n, d = X.shape
        if d < 2:
            raise ValueError("MARContinuousPredictor requires d >= 2")

        predictor = self.params.get("predictor_col_idx", 0) % d
        target_frac = self.params.get("target_frac", 0.3)
        n_targets = max(1, int(d * target_frac))
        n_targets = min(n_targets, d - 1)

        available_cols = [j for j in range(d) if j != predictor]
        target_indices = rng.choice(len(available_cols), size=n_targets, replace=False)
        targets = [available_cols[i] for i in target_indices]

        logits = self.params["alpha0"] + self.params["alpha1"] * X[:, predictor]
        p_missing = torch.sigmoid(logits)

        R = torch.ones(n, d, dtype=torch.bool)
        for target in targets:
            R[:, target] = ~(rng.rand(n) < p_missing)

        for col in range(d):
            if R[:, col].sum() == 0:
                R[rng.randint(0, n, (1,)).item(), col] = True

        return R

    def sample(self, rng: RNGState, n: int, d: int) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self.params.get("base_mean", 0.0)
        std = self.params.get("base_std", 1.0)
        X = sample_gaussian(rng.spawn(), n, d, mean=mean, std=std)
        R = self._compute_missingness(X, rng.spawn())
        return X, R

    def apply_to(self, X: torch.Tensor, rng: RNGState) -> torch.Tensor:
        return self._compute_missingness(X, rng)


class MARDiscretePredictor(Generator):
    """MAR mechanism with discretized predictor.

    Predictor column is discretized into n_levels bins (by quantile),
    and each bin has its own missingness rate.

    Required params: n_levels, alpha_per_level (list of miss rates per level)
    Optional params: target_col_idx, predictor_col_idx, base_mean, base_std
    """

    def __init__(self, generator_id: int, name: str, params: GeneratorParams):
        super().__init__(generator_id, name, MAR, params)
        for key in ["n_levels", "alpha_per_level"]:
            if key not in params:
                raise ValueError(f"MARDiscretePredictor requires '{key}' parameter")
        if len(params["alpha_per_level"]) != params["n_levels"]:
            raise ValueError("alpha_per_level must have length n_levels")

    def _compute_missingness(self, X: torch.Tensor, rng: RNGState) -> torch.Tensor:
        n, d = X.shape
        if d < 2:
            raise ValueError("MARDiscretePredictor requires d >= 2")

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

        n_levels = self.params["n_levels"]
        alpha_per_level = self.params["alpha_per_level"]
        xp = X[:, predictor]

        # Compute quantile-based bin edges
        quantiles = torch.linspace(0, 1, n_levels + 1)
        edges = torch.quantile(xp, quantiles)

        # Assign each row to a level
        p_missing = torch.zeros(n)
        for i in range(n_levels):
            if i < n_levels - 1:
                mask = (xp >= edges[i]) & (xp < edges[i + 1])
            else:
                mask = xp >= edges[i]
            p_missing = torch.where(mask, torch.tensor(alpha_per_level[i]), p_missing)

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


class MARBinaryPredictor(Generator):
    """MAR mechanism with binarized predictor.

    Predictor column is binarized at median:
    above median -> miss_rate_high, below -> miss_rate_low.

    Required params: miss_rate_high, miss_rate_low
    Optional params: target_col_idx, predictor_col_idx, base_mean, base_std
    """

    def __init__(self, generator_id: int, name: str, params: GeneratorParams):
        super().__init__(generator_id, name, MAR, params)
        for key in ["miss_rate_high", "miss_rate_low"]:
            if key not in params:
                raise ValueError(f"MARBinaryPredictor requires '{key}' parameter")

    def _compute_missingness(self, X: torch.Tensor, rng: RNGState) -> torch.Tensor:
        n, d = X.shape
        if d < 2:
            raise ValueError("MARBinaryPredictor requires d >= 2")

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
        median_val = torch.median(xp)
        above = xp >= median_val

        p_missing = torch.where(
            above,
            torch.tensor(self.params["miss_rate_high"]),
            torch.tensor(self.params["miss_rate_low"]),
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


class MARMixedPredictor(Generator):
    """MAR mechanism with both continuous and binary predictors.

    Uses two predictor columns: one continuous, one binarized at median.
    logit = alpha_continuous * X_cont + alpha_binary * X_bin

    Required params: alpha_continuous, alpha_binary
    Optional params: target_col_idx, base_mean, base_std
    """

    def __init__(self, generator_id: int, name: str, params: GeneratorParams):
        super().__init__(generator_id, name, MAR, params)
        for key in ["alpha_continuous", "alpha_binary"]:
            if key not in params:
                raise ValueError(f"MARMixedPredictor requires '{key}' parameter")

    def _compute_missingness(self, X: torch.Tensor, rng: RNGState) -> torch.Tensor:
        n, d = X.shape
        if d < 3:
            raise ValueError("MARMixedPredictor requires d >= 3 (target + 2 predictors)")

        target = self.params.get("target_col_idx", -1)
        if target < 0:
            target = d + target
        target = target % d

        # Pick two predictor columns (not the target)
        available = [j for j in range(d) if j != target]
        cont_pred = available[0]
        bin_pred = available[1]

        x_cont = X[:, cont_pred]
        x_bin = (X[:, bin_pred] >= torch.median(X[:, bin_pred])).float()

        logits = self.params["alpha_continuous"] * x_cont + self.params["alpha_binary"] * x_bin
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
