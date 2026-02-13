"""MCAR Bernoulli-based generators."""

from typing import Tuple
import torch

from lacuna.core.rng import RNGState
from lacuna.core.types import MCAR
from ...base import Generator
from ...params import GeneratorParams
from ..base_data import sample_gaussian


class MCARBernoulli(Generator):
    """MCAR mechanism with uniform missingness rate across all cells.

    Each cell is independently missing with probability miss_rate.

    Required params:
        miss_rate: Probability of missingness per cell (0 to 1).

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

        if "miss_rate" not in params:
            raise ValueError("MCARBernoulli requires 'miss_rate' parameter")

        miss_rate = params["miss_rate"]
        if not 0 <= miss_rate <= 1:
            raise ValueError(f"miss_rate must be in [0, 1], got {miss_rate}")

    def _compute_missingness(
        self,
        n: int,
        d: int,
        rng: RNGState,
    ) -> torch.Tensor:
        """Compute MCAR missingness mask.

        Args:
            n: Number of rows
            d: Number of columns
            rng: RNG state

        Returns:
            R: Boolean mask [n, d], True = observed
        """
        miss_rate = self.params["miss_rate"]
        R = rng.rand(n, d) >= miss_rate

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
        """Apply MCAR missingness to existing data."""
        n, d = X.shape
        return self._compute_missingness(n, d, rng)
