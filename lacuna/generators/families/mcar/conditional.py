"""MCAR conditional/group-based generators.

Note: These are still MCAR because missingness depends on fixed group
membership (a design variable), not on X values.
"""

from typing import Tuple
import torch

from lacuna.core.rng import RNGState
from lacuna.core.types import MCAR
from ...base import Generator
from ...params import GeneratorParams
from ..base_data import sample_gaussian


class MCARCovariateDependent(Generator):
    """MCAR with rows assigned to groups, each group having a fixed miss rate.

    Required params:
        n_groups: Number of groups.
        group_rates: List of miss rates, one per group.
    """

    def __init__(self, generator_id: int, name: str, params: GeneratorParams):
        super().__init__(generator_id, name, MCAR, params)
        if "n_groups" not in params or "group_rates" not in params:
            raise ValueError("MCARCovariateDependent requires 'n_groups' and 'group_rates'")

    def _compute_missingness(self, n: int, d: int, rng: RNGState) -> torch.Tensor:
        n_groups = self.params["n_groups"]
        group_rates = list(self.params["group_rates"])

        # Randomly assign rows to groups
        group_assignments = rng.numpy_rng.choice(n_groups, size=n, replace=True)
        row_rates = torch.tensor([group_rates[g] for g in group_assignments], dtype=torch.float32)

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


class MCARBatchEffects(Generator):
    """MCAR with rows assigned to batches, each batch getting a random miss rate.

    Required params:
        n_batches: Number of batches.
        batch_rate_range: Tuple (min_rate, max_rate) for per-batch miss rates.
    """

    def __init__(self, generator_id: int, name: str, params: GeneratorParams):
        super().__init__(generator_id, name, MCAR, params)
        if "n_batches" not in params or "batch_rate_range" not in params:
            raise ValueError("MCARBatchEffects requires 'n_batches' and 'batch_rate_range'")

    def _compute_missingness(self, n: int, d: int, rng: RNGState) -> torch.Tensor:
        n_batches = self.params["n_batches"]
        min_rate, max_rate = self.params["batch_rate_range"]

        # Draw a random rate for each batch
        batch_rates = rng.rand(n_batches) * (max_rate - min_rate) + min_rate

        # Assign rows to batches in contiguous blocks
        batch_size = max(1, n // n_batches)
        row_rates = torch.zeros(n, dtype=torch.float32)
        for i in range(n):
            batch_idx = min(i // batch_size, n_batches - 1)
            row_rates[i] = batch_rates[batch_idx]

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


class MCARSubgroupSpecific(Generator):
    """MCAR with a subgroup of rows having elevated missingness.

    A fraction of rows are designated as "subgroup" with a higher miss rate;
    the remaining rows use the background rate.

    Required params:
        subgroup_frac: Fraction of rows in the subgroup.
        subgroup_rate: Miss rate for subgroup rows.
        background_rate: Miss rate for non-subgroup rows.
    """

    def __init__(self, generator_id: int, name: str, params: GeneratorParams):
        super().__init__(generator_id, name, MCAR, params)
        if "subgroup_frac" not in params or "subgroup_rate" not in params or "background_rate" not in params:
            raise ValueError(
                "MCARSubgroupSpecific requires 'subgroup_frac', 'subgroup_rate', and 'background_rate'"
            )

    def _compute_missingness(self, n: int, d: int, rng: RNGState) -> torch.Tensor:
        subgroup_frac = self.params["subgroup_frac"]
        subgroup_rate = self.params["subgroup_rate"]
        background_rate = self.params["background_rate"]

        # Determine which rows are in the subgroup
        is_subgroup = rng.rand(n) < subgroup_frac
        row_rates = torch.where(is_subgroup, torch.tensor(subgroup_rate), torch.tensor(background_rate))

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
