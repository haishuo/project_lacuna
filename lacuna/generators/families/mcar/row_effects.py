"""MCAR row-effect generators."""

from typing import Tuple
import torch

from lacuna.core.rng import RNGState
from lacuna.core.types import MCAR
from ...base import Generator
from ...params import GeneratorParams
from ..base_data import sample_gaussian


class MCARRowGaussian(Generator):
    """MCAR with per-row miss rates drawn from N(mean_rate, std_rate), clipped to [0,1].

    Required params:
        mean_rate: Mean of the Gaussian for row miss rates.
        std_rate: Std of the Gaussian for row miss rates.
    """

    def __init__(self, generator_id: int, name: str, params: GeneratorParams):
        super().__init__(generator_id, name, MCAR, params)
        if "mean_rate" not in params or "std_rate" not in params:
            raise ValueError("MCARRowGaussian requires 'mean_rate' and 'std_rate'")

    def _compute_missingness(self, n: int, d: int, rng: RNGState) -> torch.Tensor:
        mean_rate = self.params["mean_rate"]
        std_rate = self.params["std_rate"]
        row_rates = (rng.randn(n) * std_rate + mean_rate).clamp(0, 1)
        R = rng.rand(n, d) >= row_rates.unsqueeze(1)
        if R.sum() == 0:
            R[0, 0] = True
        return R

    def sample(self, rng: RNGState, n: int, d: int) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self.params.get("base_mean", 0.0)
        std = self.params.get("base_std", 1.0)
        X = sample_gaussian(rng.spawn(), n, d, mean=mean, std=std)
        R = self._compute_missingness(n, d, rng.spawn())
        return X, R

    def apply_to(self, X: torch.Tensor, rng: RNGState) -> torch.Tensor:
        n, d = X.shape
        return self._compute_missingness(n, d, rng)


class MCARRowGamma(Generator):
    """MCAR with per-row miss rates drawn from Gamma(shape, scale), clipped to [0,1].

    Required params:
        shape: Shape parameter for Gamma distribution.
        scale: Scale parameter for Gamma distribution.
    """

    def __init__(self, generator_id: int, name: str, params: GeneratorParams):
        super().__init__(generator_id, name, MCAR, params)
        if "shape" not in params or "scale" not in params:
            raise ValueError("MCARRowGamma requires 'shape' and 'scale'")

    def _compute_missingness(self, n: int, d: int, rng: RNGState) -> torch.Tensor:
        shape = self.params["shape"]
        scale = self.params["scale"]
        # Use numpy for Gamma sampling with the rng's numpy generator
        row_rates_np = rng.numpy_rng.gamma(shape, scale, size=n)
        row_rates = torch.tensor(row_rates_np, dtype=torch.float32).clamp(0, 1)
        R = rng.rand(n, d) >= row_rates.unsqueeze(1)
        if R.sum() == 0:
            R[0, 0] = True
        return R

    def sample(self, rng: RNGState, n: int, d: int) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self.params.get("base_mean", 0.0)
        std = self.params.get("base_std", 1.0)
        X = sample_gaussian(rng.spawn(), n, d, mean=mean, std=std)
        R = self._compute_missingness(n, d, rng.spawn())
        return X, R

    def apply_to(self, X: torch.Tensor, rng: RNGState) -> torch.Tensor:
        n, d = X.shape
        return self._compute_missingness(n, d, rng)


class MCARRowBeta(Generator):
    """MCAR with per-row miss rates drawn from Beta(alpha, beta).

    Required params:
        alpha: Alpha parameter for Beta distribution.
        beta: Beta parameter for Beta distribution.
    """

    def __init__(self, generator_id: int, name: str, params: GeneratorParams):
        super().__init__(generator_id, name, MCAR, params)
        if "alpha" not in params or "beta" not in params:
            raise ValueError("MCARRowBeta requires 'alpha' and 'beta'")

    def _compute_missingness(self, n: int, d: int, rng: RNGState) -> torch.Tensor:
        alpha = self.params["alpha"]
        beta = self.params["beta"]
        row_rates_np = rng.numpy_rng.beta(alpha, beta, size=n)
        row_rates = torch.tensor(row_rates_np, dtype=torch.float32)
        R = rng.rand(n, d) >= row_rates.unsqueeze(1)
        if R.sum() == 0:
            R[0, 0] = True
        return R

    def sample(self, rng: RNGState, n: int, d: int) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self.params.get("base_mean", 0.0)
        std = self.params.get("base_std", 1.0)
        X = sample_gaussian(rng.spawn(), n, d, mean=mean, std=std)
        R = self._compute_missingness(n, d, rng.spawn())
        return X, R

    def apply_to(self, X: torch.Tensor, rng: RNGState) -> torch.Tensor:
        n, d = X.shape
        return self._compute_missingness(n, d, rng)


class MCARRowMixture(Generator):
    """MCAR with per-row miss rate chosen from a discrete set of rates with given weights.

    Required params:
        rates: List of possible miss rates.
        weights: List of weights (probabilities) for each rate.
    """

    def __init__(self, generator_id: int, name: str, params: GeneratorParams):
        super().__init__(generator_id, name, MCAR, params)
        if "rates" not in params or "weights" not in params:
            raise ValueError("MCARRowMixture requires 'rates' and 'weights'")

    def _compute_missingness(self, n: int, d: int, rng: RNGState) -> torch.Tensor:
        import numpy as np
        rates = list(self.params["rates"])
        weights = list(self.params["weights"])
        weights_arr = np.array(weights, dtype=np.float64)
        weights_arr /= weights_arr.sum()
        indices = rng.numpy_rng.choice(len(rates), size=n, replace=True, p=weights_arr)
        row_rates = torch.tensor([rates[i] for i in indices], dtype=torch.float32)
        R = rng.rand(n, d) >= row_rates.unsqueeze(1)
        if R.sum() == 0:
            R[0, 0] = True
        return R

    def sample(self, rng: RNGState, n: int, d: int) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self.params.get("base_mean", 0.0)
        std = self.params.get("base_std", 1.0)
        X = sample_gaussian(rng.spawn(), n, d, mean=mean, std=std)
        R = self._compute_missingness(n, d, rng.spawn())
        return X, R

    def apply_to(self, X: torch.Tensor, rng: RNGState) -> torch.Tensor:
        n, d = X.shape
        return self._compute_missingness(n, d, rng)


class MCARRowExponential(Generator):
    """MCAR with per-row miss rates drawn from Exponential(rate), clipped to [0,1].

    Required params:
        rate: Rate (lambda) parameter for Exponential distribution.
    """

    def __init__(self, generator_id: int, name: str, params: GeneratorParams):
        super().__init__(generator_id, name, MCAR, params)
        if "rate" not in params:
            raise ValueError("MCARRowExponential requires 'rate'")

    def _compute_missingness(self, n: int, d: int, rng: RNGState) -> torch.Tensor:
        rate = self.params["rate"]
        row_rates_np = rng.numpy_rng.exponential(1.0 / rate, size=n)
        row_rates = torch.tensor(row_rates_np, dtype=torch.float32).clamp(0, 1)
        R = rng.rand(n, d) >= row_rates.unsqueeze(1)
        if R.sum() == 0:
            R[0, 0] = True
        return R

    def sample(self, rng: RNGState, n: int, d: int) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self.params.get("base_mean", 0.0)
        std = self.params.get("base_std", 1.0)
        X = sample_gaussian(rng.spawn(), n, d, mean=mean, std=std)
        R = self._compute_missingness(n, d, rng.spawn())
        return X, R

    def apply_to(self, X: torch.Tensor, rng: RNGState) -> torch.Tensor:
        n, d = X.shape
        return self._compute_missingness(n, d, rng)


class MCARRowDiscrete(Generator):
    """MCAR with per-row miss rate chosen uniformly from a fixed list of levels.

    Required params:
        levels: List of possible miss rate levels (chosen uniformly).
    """

    def __init__(self, generator_id: int, name: str, params: GeneratorParams):
        super().__init__(generator_id, name, MCAR, params)
        if "levels" not in params:
            raise ValueError("MCARRowDiscrete requires 'levels'")

    def _compute_missingness(self, n: int, d: int, rng: RNGState) -> torch.Tensor:
        levels = list(self.params["levels"])
        indices = rng.numpy_rng.choice(len(levels), size=n, replace=True)
        row_rates = torch.tensor([levels[i] for i in indices], dtype=torch.float32)
        R = rng.rand(n, d) >= row_rates.unsqueeze(1)
        if R.sum() == 0:
            R[0, 0] = True
        return R

    def sample(self, rng: RNGState, n: int, d: int) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self.params.get("base_mean", 0.0)
        std = self.params.get("base_std", 1.0)
        X = sample_gaussian(rng.spawn(), n, d, mean=mean, std=std)
        R = self._compute_missingness(n, d, rng.spawn())
        return X, R

    def apply_to(self, X: torch.Tensor, rng: RNGState) -> torch.Tensor:
        n, d = X.shape
        return self._compute_missingness(n, d, rng)
