"""MCAR column-effect generators."""

from typing import Tuple
import torch

from lacuna.core.rng import RNGState
from lacuna.core.types import MCAR
from ...base import Generator
from ...params import GeneratorParams
from ..base_data import sample_gaussian


class MCARColumnGaussian(Generator):
    """MCAR mechanism with different missingness rates per column.

    Each column gets a random missingness rate drawn from a range.
    Within each column, cells are independently missing.

    Required params:
        miss_rate_range: Tuple (min_rate, max_rate) for column missingness.

    Optional params:
        base_mean: Mean for Gaussian base data (default: 0.0).
        base_std: Std for Gaussian base data (default: 1.0).
    """

    def __init__(
        self,
        generator_id: int,
        name: str,
        params: GeneratorParams,
    ):
        super().__init__(generator_id, name, MCAR, params)

        if "miss_rate_range" not in params:
            raise ValueError("MCARColumnGaussian requires 'miss_rate_range' parameter")

        rate_range = params["miss_rate_range"]
        if len(rate_range) != 2:
            raise ValueError("miss_rate_range must be (min, max) tuple")
        if not (0 <= rate_range[0] <= rate_range[1] <= 1):
            raise ValueError("miss_rate_range must satisfy 0 <= min <= max <= 1")

    def _compute_missingness(
        self,
        n: int,
        d: int,
        rng: RNGState,
    ) -> torch.Tensor:
        """Compute column-wise MCAR missingness mask."""
        min_rate, max_rate = self.params["miss_rate_range"]

        col_rates = rng.rand(d) * (max_rate - min_rate) + min_rate

        R = torch.ones(n, d, dtype=torch.bool)
        for j in range(d):
            R[:, j] = rng.rand(n) >= col_rates[j]

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
        R = self._compute_missingness(n, d, rng.spawn())

        return X, R

    def apply_to(
        self,
        X: torch.Tensor,
        rng: RNGState,
    ) -> torch.Tensor:
        """Apply column-wise MCAR missingness to existing data."""
        n, d = X.shape
        return self._compute_missingness(n, d, rng)


class MCARColumnGamma(Generator):
    """MCAR with per-column miss rates drawn from Gamma(shape, scale), clipped to [0,1].

    Required params:
        shape: Shape parameter for Gamma distribution.
        scale: Scale parameter for Gamma distribution.
    """

    def __init__(self, generator_id: int, name: str, params: GeneratorParams):
        super().__init__(generator_id, name, MCAR, params)
        if "shape" not in params or "scale" not in params:
            raise ValueError("MCARColumnGamma requires 'shape' and 'scale'")

    def _compute_missingness(self, n: int, d: int, rng: RNGState) -> torch.Tensor:
        shape = self.params["shape"]
        scale = self.params["scale"]
        col_rates_np = rng.numpy_rng.gamma(shape, scale, size=d)
        col_rates = torch.tensor(col_rates_np, dtype=torch.float32).clamp(0, 1)
        R = torch.ones(n, d, dtype=torch.bool)
        for j in range(d):
            R[:, j] = rng.rand(n) >= col_rates[j]
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


class MCARColumnBeta(Generator):
    """MCAR with per-column miss rates drawn from Beta(alpha, beta).

    Required params:
        alpha: Alpha parameter for Beta distribution.
        beta: Beta parameter for Beta distribution.
    """

    def __init__(self, generator_id: int, name: str, params: GeneratorParams):
        super().__init__(generator_id, name, MCAR, params)
        if "alpha" not in params or "beta" not in params:
            raise ValueError("MCARColumnBeta requires 'alpha' and 'beta'")

    def _compute_missingness(self, n: int, d: int, rng: RNGState) -> torch.Tensor:
        alpha = self.params["alpha"]
        beta = self.params["beta"]
        col_rates_np = rng.numpy_rng.beta(alpha, beta, size=d)
        col_rates = torch.tensor(col_rates_np, dtype=torch.float32)
        R = torch.ones(n, d, dtype=torch.bool)
        for j in range(d):
            R[:, j] = rng.rand(n) >= col_rates[j]
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


class MCARColumnMixture(Generator):
    """MCAR with per-column miss rate chosen from a discrete set of rates with weights.

    Required params:
        rates: List of possible miss rates.
        weights: List of weights for each rate.
    """

    def __init__(self, generator_id: int, name: str, params: GeneratorParams):
        super().__init__(generator_id, name, MCAR, params)
        if "rates" not in params or "weights" not in params:
            raise ValueError("MCARColumnMixture requires 'rates' and 'weights'")

    def _compute_missingness(self, n: int, d: int, rng: RNGState) -> torch.Tensor:
        import numpy as np
        rates = list(self.params["rates"])
        weights = list(self.params["weights"])
        weights_arr = np.array(weights, dtype=np.float64)
        weights_arr /= weights_arr.sum()
        indices = rng.numpy_rng.choice(len(rates), size=d, replace=True, p=weights_arr)
        col_rates = torch.tensor([rates[i] for i in indices], dtype=torch.float32)
        R = torch.ones(n, d, dtype=torch.bool)
        for j in range(d):
            R[:, j] = rng.rand(n) >= col_rates[j]
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


class MCARColumnOrdered(Generator):
    """MCAR with column miss rates increasing monotonically from min_rate to max_rate.

    Required params:
        min_rate: Miss rate for the first column.
        max_rate: Miss rate for the last column.
    """

    def __init__(self, generator_id: int, name: str, params: GeneratorParams):
        super().__init__(generator_id, name, MCAR, params)
        if "min_rate" not in params or "max_rate" not in params:
            raise ValueError("MCARColumnOrdered requires 'min_rate' and 'max_rate'")

    def _compute_missingness(self, n: int, d: int, rng: RNGState) -> torch.Tensor:
        min_rate = self.params["min_rate"]
        max_rate = self.params["max_rate"]
        if d == 1:
            col_rates = torch.tensor([min_rate])
        else:
            col_rates = torch.linspace(min_rate, max_rate, d)
        R = torch.ones(n, d, dtype=torch.bool)
        for j in range(d):
            R[:, j] = rng.rand(n) >= col_rates[j]
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


class MCARColumnClustered(Generator):
    """MCAR with columns grouped into clusters, each cluster sharing a miss rate.

    Required params:
        n_clusters: Number of column clusters.
        rates_per_cluster: List of miss rates, one per cluster.
    """

    def __init__(self, generator_id: int, name: str, params: GeneratorParams):
        super().__init__(generator_id, name, MCAR, params)
        if "n_clusters" not in params or "rates_per_cluster" not in params:
            raise ValueError("MCARColumnClustered requires 'n_clusters' and 'rates_per_cluster'")

    def _compute_missingness(self, n: int, d: int, rng: RNGState) -> torch.Tensor:
        n_clusters = self.params["n_clusters"]
        rates = list(self.params["rates_per_cluster"])
        # Assign columns to clusters in contiguous blocks
        cluster_size = max(1, d // n_clusters)
        col_rates = torch.zeros(d, dtype=torch.float32)
        for j in range(d):
            cluster_idx = min(j // cluster_size, n_clusters - 1)
            col_rates[j] = rates[cluster_idx]
        R = torch.ones(n, d, dtype=torch.bool)
        for j in range(d):
            R[:, j] = rng.rand(n) >= col_rates[j]
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
