"""MCAR multilevel (row + column interaction) generators."""

from typing import Tuple
import torch

from lacuna.core.rng import RNGState
from lacuna.core.types import MCAR
from ...base import Generator
from ...params import GeneratorParams
from ..base_data import sample_gaussian


class MCARRowColumnAdditive(Generator):
    """MCAR with miss rate = base_rate + row_effect + col_effect, clipped to [0,1].

    Row effects ~ N(0, row_std), column effects ~ N(0, col_std).

    Required params:
        row_std: Std of the row random effects.
        col_std: Std of the column random effects.
        base_rate: Base missingness rate before effects.
    """

    def __init__(self, generator_id: int, name: str, params: GeneratorParams):
        super().__init__(generator_id, name, MCAR, params)
        if "row_std" not in params or "col_std" not in params or "base_rate" not in params:
            raise ValueError("MCARRowColumnAdditive requires 'row_std', 'col_std', and 'base_rate'")

    def _compute_missingness(self, n: int, d: int, rng: RNGState) -> torch.Tensor:
        row_std = self.params["row_std"]
        col_std = self.params["col_std"]
        base_rate = self.params["base_rate"]

        row_effects = rng.randn(n) * row_std
        col_effects = rng.randn(d) * col_std
        miss_rates = (base_rate + row_effects.unsqueeze(1) + col_effects.unsqueeze(0)).clamp(0, 1)

        R = rng.rand(n, d) >= miss_rates
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


class MCARRowColumnInteraction(Generator):
    """MCAR with miss rate proportional to the product of row and column effects.

    Row effects ~ |N(0, row_std)|, column effects ~ |N(0, col_std)|.
    Miss rate = row_effect_i * col_effect_j, clipped to [0,1].

    Required params:
        row_std: Std of the row effects (absolute value taken).
        col_std: Std of the column effects (absolute value taken).
    """

    def __init__(self, generator_id: int, name: str, params: GeneratorParams):
        super().__init__(generator_id, name, MCAR, params)
        if "row_std" not in params or "col_std" not in params:
            raise ValueError("MCARRowColumnInteraction requires 'row_std' and 'col_std'")

    def _compute_missingness(self, n: int, d: int, rng: RNGState) -> torch.Tensor:
        row_std = self.params["row_std"]
        col_std = self.params["col_std"]

        row_effects = rng.randn(n).abs() * row_std
        col_effects = rng.randn(d).abs() * col_std
        miss_rates = (row_effects.unsqueeze(1) * col_effects.unsqueeze(0)).clamp(0, 1)

        R = rng.rand(n, d) >= miss_rates
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


class MCARNested(Generator):
    """MCAR with rows divided into groups, each group having its own miss rate.

    Each group's miss rate is drawn from N(0.2, group_rate_std), clipped to [0,1].
    Rows are assigned to groups in contiguous blocks.

    Required params:
        n_groups: Number of row groups.
        group_rate_std: Std for group-level miss rate distribution.
    """

    def __init__(self, generator_id: int, name: str, params: GeneratorParams):
        super().__init__(generator_id, name, MCAR, params)
        if "n_groups" not in params or "group_rate_std" not in params:
            raise ValueError("MCARNested requires 'n_groups' and 'group_rate_std'")

    def _compute_missingness(self, n: int, d: int, rng: RNGState) -> torch.Tensor:
        n_groups = self.params["n_groups"]
        group_rate_std = self.params["group_rate_std"]

        # Draw group-level rates
        group_rates = (rng.randn(n_groups) * group_rate_std + 0.2).clamp(0, 1)

        # Assign rows to groups in contiguous blocks
        group_size = max(1, n // n_groups)
        row_rates = torch.zeros(n, dtype=torch.float32)
        for i in range(n):
            group_idx = min(i // group_size, n_groups - 1)
            row_rates[i] = group_rates[group_idx]

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
