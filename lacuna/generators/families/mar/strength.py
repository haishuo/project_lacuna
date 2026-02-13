"""MAR generators with varying signal strength."""

from typing import Tuple
import torch

from lacuna.core.rng import RNGState
from lacuna.core.types import MAR
from ...base import Generator
from ...params import GeneratorParams
from ..base_data import sample_gaussian


class MARWeak(Generator):
    """MAR mechanism with weak dependence on predictor.

    Alpha1 sampled from U(alpha1_range[0], alpha1_range[1]) at generation time.
    Affects multiple target columns (target_frac=0.4).

    Required params: alpha0
    Optional params: alpha1_range (default [0.5, 1.5]), target_frac, predictor_col_idx,
                     base_mean, base_std
    """

    def __init__(self, generator_id: int, name: str, params: GeneratorParams):
        super().__init__(generator_id, name, MAR, params)
        if "alpha0" not in params:
            raise ValueError("MARWeak requires 'alpha0' parameter")

    def _compute_missingness(self, X: torch.Tensor, rng: RNGState) -> torch.Tensor:
        n, d = X.shape
        if d < 2:
            raise ValueError("MARWeak requires d >= 2")

        predictor = self.params.get("predictor_col_idx", 0) % d
        target_frac = self.params.get("target_frac", 0.4)
        n_targets = max(1, int(d * target_frac))
        n_targets = min(n_targets, d - 1)

        available_cols = [j for j in range(d) if j != predictor]
        target_indices = rng.choice(len(available_cols), size=n_targets, replace=False)
        targets = [available_cols[i] for i in target_indices]

        alpha0 = self.params["alpha0"]
        alpha1_range = self.params.get("alpha1_range", [0.5, 1.5])
        alpha1 = alpha1_range[0] + rng.rand(1).item() * (alpha1_range[1] - alpha1_range[0])

        logits = alpha0 + alpha1 * X[:, predictor]
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


class MARModerate(Generator):
    """MAR mechanism with moderate dependence on predictor.

    Alpha1 sampled from U(alpha1_range[0], alpha1_range[1]) at generation time.
    Affects multiple target columns (target_frac=0.4).

    Required params: alpha0
    Optional params: alpha1_range (default [1.5, 3.0]), target_frac, predictor_col_idx,
                     base_mean, base_std
    """

    def __init__(self, generator_id: int, name: str, params: GeneratorParams):
        super().__init__(generator_id, name, MAR, params)
        if "alpha0" not in params:
            raise ValueError("MARModerate requires 'alpha0' parameter")

    def _compute_missingness(self, X: torch.Tensor, rng: RNGState) -> torch.Tensor:
        n, d = X.shape
        if d < 2:
            raise ValueError("MARModerate requires d >= 2")

        predictor = self.params.get("predictor_col_idx", 0) % d
        target_frac = self.params.get("target_frac", 0.4)
        n_targets = max(1, int(d * target_frac))
        n_targets = min(n_targets, d - 1)

        available_cols = [j for j in range(d) if j != predictor]
        target_indices = rng.choice(len(available_cols), size=n_targets, replace=False)
        targets = [available_cols[i] for i in target_indices]

        alpha0 = self.params["alpha0"]
        alpha1_range = self.params.get("alpha1_range", [1.5, 3.0])
        alpha1 = alpha1_range[0] + rng.rand(1).item() * (alpha1_range[1] - alpha1_range[0])

        logits = alpha0 + alpha1 * X[:, predictor]
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


class MARStrong(Generator):
    """MAR mechanism with strong dependence on predictor.

    Alpha1 sampled from U(alpha1_range[0], alpha1_range[1]) at generation time.
    Affects multiple target columns (target_frac=0.4).

    Required params: alpha0
    Optional params: alpha1_range (default [3.0, 5.0]), target_frac, predictor_col_idx,
                     base_mean, base_std
    """

    def __init__(self, generator_id: int, name: str, params: GeneratorParams):
        super().__init__(generator_id, name, MAR, params)
        if "alpha0" not in params:
            raise ValueError("MARStrong requires 'alpha0' parameter")

    def _compute_missingness(self, X: torch.Tensor, rng: RNGState) -> torch.Tensor:
        n, d = X.shape
        if d < 2:
            raise ValueError("MARStrong requires d >= 2")

        predictor = self.params.get("predictor_col_idx", 0) % d
        target_frac = self.params.get("target_frac", 0.4)
        n_targets = max(1, int(d * target_frac))
        n_targets = min(n_targets, d - 1)

        available_cols = [j for j in range(d) if j != predictor]
        target_indices = rng.choice(len(available_cols), size=n_targets, replace=False)
        targets = [available_cols[i] for i in target_indices]

        alpha0 = self.params["alpha0"]
        alpha1_range = self.params.get("alpha1_range", [3.0, 5.0])
        alpha1 = alpha1_range[0] + rng.rand(1).item() * (alpha1_range[1] - alpha1_range[0])

        logits = alpha0 + alpha1 * X[:, predictor]
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
