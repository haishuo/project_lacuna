"""MCAR distributional generators (heavy-tailed and mixture per-cell probabilities)."""

from typing import Tuple
import torch
import numpy as np

from lacuna.core.rng import RNGState
from lacuna.core.types import MCAR
from ...base import Generator
from ...params import GeneratorParams
from ..base_data import sample_gaussian


class MCARCauchy(Generator):
    """MCAR with per-cell miss probability from Cauchy, mapped to [0,1] via sigmoid.

    Required params:
        location: Location parameter for Cauchy distribution.
        scale: Scale parameter for Cauchy distribution.
    """

    def __init__(self, generator_id: int, name: str, params: GeneratorParams):
        super().__init__(generator_id, name, MCAR, params)
        if "location" not in params or "scale" not in params:
            raise ValueError("MCARCauchy requires 'location' and 'scale'")

    def _compute_missingness(self, n: int, d: int, rng: RNGState) -> torch.Tensor:
        location = self.params["location"]
        scale = self.params["scale"]
        # Sample Cauchy via inverse CDF: location + scale * tan(pi * (U - 0.5))
        U = rng.rand(n, d)
        cauchy_samples = location + scale * torch.tan(torch.pi * (U - 0.5))
        miss_probs = torch.sigmoid(cauchy_samples)
        R = rng.rand(n, d) >= miss_probs
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


class MCARPareto(Generator):
    """MCAR with per-cell miss probability from Pareto(alpha, scale), clipped to [0,1].

    Required params:
        alpha: Shape parameter for Pareto distribution.
        scale: Scale (x_m) parameter for Pareto distribution.
    """

    def __init__(self, generator_id: int, name: str, params: GeneratorParams):
        super().__init__(generator_id, name, MCAR, params)
        if "alpha" not in params or "scale" not in params:
            raise ValueError("MCARPareto requires 'alpha' and 'scale'")

    def _compute_missingness(self, n: int, d: int, rng: RNGState) -> torch.Tensor:
        alpha = self.params["alpha"]
        scale = self.params["scale"]
        # Pareto via inverse CDF: scale / U^(1/alpha)
        U = rng.rand(n, d).clamp(min=1e-10)
        pareto_samples = scale / U.pow(1.0 / alpha)
        miss_probs = pareto_samples.clamp(0, 1)
        R = rng.rand(n, d) >= miss_probs
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


class MCARLogNormal(Generator):
    """MCAR with per-cell miss probability from LogNormal, mapped to [0,1] via sigmoid.

    Required params:
        mu: Mean of the underlying normal distribution.
        sigma: Std of the underlying normal distribution.
    """

    def __init__(self, generator_id: int, name: str, params: GeneratorParams):
        super().__init__(generator_id, name, MCAR, params)
        if "mu" not in params or "sigma" not in params:
            raise ValueError("MCARLogNormal requires 'mu' and 'sigma'")

    def _compute_missingness(self, n: int, d: int, rng: RNGState) -> torch.Tensor:
        mu = self.params["mu"]
        sigma = self.params["sigma"]
        normal_samples = rng.randn(n, d) * sigma + mu
        lognormal_samples = torch.exp(normal_samples)
        miss_probs = torch.sigmoid(lognormal_samples)
        R = rng.rand(n, d) >= miss_probs
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


class MCARTDist(Generator):
    """MCAR with per-cell miss probability from Student-t, mapped to [0,1] via sigmoid.

    Required params:
        df: Degrees of freedom for Student-t distribution.
        scale: Scale factor applied to the t-distributed samples.
    """

    def __init__(self, generator_id: int, name: str, params: GeneratorParams):
        super().__init__(generator_id, name, MCAR, params)
        if "df" not in params or "scale" not in params:
            raise ValueError("MCARTDist requires 'df' and 'scale'")

    def _compute_missingness(self, n: int, d: int, rng: RNGState) -> torch.Tensor:
        df = self.params["df"]
        scale = self.params["scale"]
        # Student-t via numpy
        t_samples_np = rng.numpy_rng.standard_t(df, size=(n, d))
        t_samples = torch.tensor(t_samples_np, dtype=torch.float32) * scale
        miss_probs = torch.sigmoid(t_samples)
        R = rng.rand(n, d) >= miss_probs
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


class MCARMixtureGaussian(Generator):
    """MCAR with per-cell miss probability from a Gaussian mixture.

    Each cell's miss probability is drawn from one of several Gaussian components,
    then passed through sigmoid to map to [0,1].

    Required params:
        rates: List of Gaussian component means (in logit space).
        weights: List of mixture weights.
    """

    def __init__(self, generator_id: int, name: str, params: GeneratorParams):
        super().__init__(generator_id, name, MCAR, params)
        if "rates" not in params or "weights" not in params:
            raise ValueError("MCARMixtureGaussian requires 'rates' and 'weights'")

    def _compute_missingness(self, n: int, d: int, rng: RNGState) -> torch.Tensor:
        rates = list(self.params["rates"])
        weights = list(self.params["weights"])
        weights_arr = np.array(weights, dtype=np.float64)
        weights_arr /= weights_arr.sum()

        # Assign each cell to a component
        total = n * d
        component_indices = rng.numpy_rng.choice(len(rates), size=total, replace=True, p=weights_arr)
        # Sample from each component (mean = rates[k], std = 0.5)
        component_means = np.array([rates[k] for k in component_indices], dtype=np.float64)
        normals = rng.randn(total).numpy()
        logits = component_means + 0.5 * normals
        miss_probs = torch.sigmoid(torch.tensor(logits, dtype=torch.float32)).reshape(n, d)

        R = rng.rand(n, d) >= miss_probs
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


class MCARSparseMixture(Generator):
    """MCAR where most cells have zero miss rate but rare cells have a high rate.

    Required params:
        sparse_prob: Probability that a cell is in the "high missingness" group.
        high_rate: Miss rate for cells in the high group.
    """

    def __init__(self, generator_id: int, name: str, params: GeneratorParams):
        super().__init__(generator_id, name, MCAR, params)
        if "sparse_prob" not in params or "high_rate" not in params:
            raise ValueError("MCARSparseMixture requires 'sparse_prob' and 'high_rate'")

    def _compute_missingness(self, n: int, d: int, rng: RNGState) -> torch.Tensor:
        sparse_prob = self.params["sparse_prob"]
        high_rate = self.params["high_rate"]

        # Each cell: with probability sparse_prob, miss_rate = high_rate; else 0
        is_high = rng.rand(n, d) < sparse_prob
        miss_probs = is_high.float() * high_rate
        R = rng.rand(n, d) >= miss_probs
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
