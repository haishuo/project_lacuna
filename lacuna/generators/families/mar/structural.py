"""MAR generators with block/group structural patterns."""

from typing import List, Tuple
import torch

from lacuna.core.rng import RNGState
from lacuna.core.types import MAR
from ...base import Generator
from ...params import GeneratorParams
from ..base_data import sample_gaussian


class MARColumnBlocks(Generator):
    """MAR mechanism with column blocks.

    Columns divided into n_blocks groups. Each block has its own alpha1
    for logistic MAR, using column 0 as predictor.

    Required params: n_blocks, alpha_per_block (list of alpha1 values)
    Optional params: alpha0 (default 0.0), predictor_col_idx, base_mean, base_std
    """

    def __init__(self, generator_id: int, name: str, params: GeneratorParams):
        super().__init__(generator_id, name, MAR, params)
        for key in ["n_blocks", "alpha_per_block"]:
            if key not in params:
                raise ValueError(f"MARColumnBlocks requires '{key}' parameter")
        if len(params["alpha_per_block"]) != params["n_blocks"]:
            raise ValueError("alpha_per_block must have length n_blocks")

    def _compute_missingness(self, X: torch.Tensor, rng: RNGState) -> torch.Tensor:
        n, d = X.shape
        if d < 2:
            raise ValueError("MARColumnBlocks requires d >= 2")

        predictor = self.params.get("predictor_col_idx", 0) % d
        alpha0 = self.params.get("alpha0", 0.0)
        n_blocks = self.params["n_blocks"]
        alpha_per_block = self.params["alpha_per_block"]

        # Assign columns (excluding predictor) to blocks
        target_cols = [j for j in range(d) if j != predictor]
        block_size = max(1, len(target_cols) // n_blocks)

        R = torch.ones(n, d, dtype=torch.bool)

        for b in range(n_blocks):
            start = b * block_size
            end = start + block_size if b < n_blocks - 1 else len(target_cols)
            block_cols = target_cols[start:end]
            if not block_cols:
                continue

            logits = alpha0 + alpha_per_block[b] * X[:, predictor]
            p_missing = torch.sigmoid(logits)

            for col in block_cols:
                R[:, col] = ~(rng.rand(n) < p_missing)

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


class MARRowBlocks(Generator):
    """MAR mechanism with row groups.

    Rows divided into n_groups groups (by quantile of predictor).
    Each group has different MAR strength.

    Required params: n_groups, alpha_per_group (list of alpha1 values)
    Optional params: alpha0 (default 0.0), predictor_col_idx, target_col_idx,
                     base_mean, base_std
    """

    def __init__(self, generator_id: int, name: str, params: GeneratorParams):
        super().__init__(generator_id, name, MAR, params)
        for key in ["n_groups", "alpha_per_group"]:
            if key not in params:
                raise ValueError(f"MARRowBlocks requires '{key}' parameter")
        if len(params["alpha_per_group"]) != params["n_groups"]:
            raise ValueError("alpha_per_group must have length n_groups")

    def _compute_missingness(self, X: torch.Tensor, rng: RNGState) -> torch.Tensor:
        n, d = X.shape
        if d < 2:
            raise ValueError("MARRowBlocks requires d >= 2")

        predictor = self.params.get("predictor_col_idx", 0) % d
        target = self.params.get("target_col_idx", -1)
        if target < 0:
            target = d + target
        target = target % d
        if predictor == target:
            predictor = (target + 1) % d

        alpha0 = self.params.get("alpha0", 0.0)
        n_groups = self.params["n_groups"]
        alpha_per_group = self.params["alpha_per_group"]

        xp = X[:, predictor]
        quantiles = torch.linspace(0, 1, n_groups + 1)
        edges = torch.quantile(xp, quantiles)

        # Assign alpha1 per row based on group
        alpha1_per_row = torch.zeros(n)
        for g in range(n_groups):
            if g < n_groups - 1:
                mask = (xp >= edges[g]) & (xp < edges[g + 1])
            else:
                mask = xp >= edges[g]
            alpha1_per_row = torch.where(mask, torch.tensor(alpha_per_group[g]), alpha1_per_row)

        logits = alpha0 + alpha1_per_row * xp
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


class MARNested(Generator):
    """MAR mechanism with nested groups.

    Rows divided into n_outer groups; within each outer group, further
    divided into n_inner subgroups. Outer groups use alpha_outer,
    inner subgroups add alpha_inner variation.

    Required params: n_outer, n_inner, alpha_outer, alpha_inner
    Optional params: alpha0 (default 0.0), predictor_col_idx, target_col_idx,
                     base_mean, base_std
    """

    def __init__(self, generator_id: int, name: str, params: GeneratorParams):
        super().__init__(generator_id, name, MAR, params)
        for key in ["n_outer", "n_inner", "alpha_outer", "alpha_inner"]:
            if key not in params:
                raise ValueError(f"MARNested requires '{key}' parameter")

    def _compute_missingness(self, X: torch.Tensor, rng: RNGState) -> torch.Tensor:
        n, d = X.shape
        if d < 2:
            raise ValueError("MARNested requires d >= 2")

        predictor = self.params.get("predictor_col_idx", 0) % d
        target = self.params.get("target_col_idx", -1)
        if target < 0:
            target = d + target
        target = target % d
        if predictor == target:
            predictor = (target + 1) % d

        alpha0 = self.params.get("alpha0", 0.0)
        n_outer = self.params["n_outer"]
        n_inner = self.params["n_inner"]
        alpha_outer = self.params["alpha_outer"]
        alpha_inner = self.params["alpha_inner"]

        xp = X[:, predictor]
        total_groups = n_outer * n_inner
        quantiles = torch.linspace(0, 1, total_groups + 1)
        edges = torch.quantile(xp, quantiles)

        # Each group gets alpha_outer + variation from alpha_inner
        alpha1_per_row = torch.zeros(n)
        for g in range(total_groups):
            outer_idx = g // n_inner
            inner_idx = g % n_inner
            # Inner variation: linearly spread alpha_inner across subgroups
            inner_factor = (inner_idx - (n_inner - 1) / 2.0) / max(n_inner - 1, 1)
            alpha1 = alpha_outer + alpha_inner * inner_factor

            if g < total_groups - 1:
                mask = (xp >= edges[g]) & (xp < edges[g + 1])
            else:
                mask = xp >= edges[g]
            alpha1_per_row = torch.where(mask, torch.tensor(alpha1), alpha1_per_row)

        logits = alpha0 + alpha1_per_row * xp
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


class MARCrossClassified(Generator):
    """MAR mechanism with cross-classified row/column groups.

    Rows divided into n_row_groups (by predictor quantile), columns
    into n_col_groups. Each (row_group, col_group) combination has a
    different MAR alpha derived from base_alpha.

    Required params: n_row_groups, n_col_groups, base_alpha
    Optional params: alpha0 (default 0.0), predictor_col_idx, base_mean, base_std
    """

    def __init__(self, generator_id: int, name: str, params: GeneratorParams):
        super().__init__(generator_id, name, MAR, params)
        for key in ["n_row_groups", "n_col_groups", "base_alpha"]:
            if key not in params:
                raise ValueError(f"MARCrossClassified requires '{key}' parameter")

    def _compute_missingness(self, X: torch.Tensor, rng: RNGState) -> torch.Tensor:
        n, d = X.shape
        if d < 2:
            raise ValueError("MARCrossClassified requires d >= 2")

        predictor = self.params.get("predictor_col_idx", 0) % d
        alpha0 = self.params.get("alpha0", 0.0)
        n_row_groups = self.params["n_row_groups"]
        n_col_groups = self.params["n_col_groups"]
        base_alpha = self.params["base_alpha"]

        # Assign rows to groups by predictor quantile
        xp = X[:, predictor]
        row_quantiles = torch.linspace(0, 1, n_row_groups + 1)
        row_edges = torch.quantile(xp, row_quantiles)

        row_group = torch.zeros(n, dtype=torch.long)
        for g in range(n_row_groups):
            if g < n_row_groups - 1:
                mask = (xp >= row_edges[g]) & (xp < row_edges[g + 1])
            else:
                mask = xp >= row_edges[g]
            row_group[mask] = g

        # Assign columns (excluding predictor) to column groups
        target_cols = [j for j in range(d) if j != predictor]
        col_block_size = max(1, len(target_cols) // n_col_groups)

        R = torch.ones(n, d, dtype=torch.bool)

        for cg in range(n_col_groups):
            start = cg * col_block_size
            end = start + col_block_size if cg < n_col_groups - 1 else len(target_cols)
            block_cols = target_cols[start:end]
            if not block_cols:
                continue

            for rg in range(n_row_groups):
                # Each (rg, cg) pair gets a unique alpha
                alpha1 = base_alpha * (1.0 + 0.5 * rg + 0.3 * cg)
                row_mask = row_group == rg

                logits = alpha0 + alpha1 * xp
                p_missing = torch.sigmoid(logits)

                for col in block_cols:
                    u = rng.rand(n)
                    col_missing = row_mask & (u < p_missing)
                    R[:, col] = R[:, col] & ~col_missing

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
