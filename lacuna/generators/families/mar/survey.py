"""MAR generators with survey-like missingness patterns."""

from typing import List, Tuple
import torch

from lacuna.core.rng import RNGState
from lacuna.core.types import MAR
from ...base import Generator
from ...params import GeneratorParams
from ..base_data import sample_gaussian


class MARSkipLogic(Generator):
    """MAR mechanism mimicking survey skip logic.

    If predictor (gate column) > threshold, subsequent columns are skipped
    (made missing) with probability conditional_rate.

    Required params: threshold, conditional_rate
    Optional params: gate_col (default 0), base_mean, base_std
    """

    def __init__(self, generator_id: int, name: str, params: GeneratorParams):
        super().__init__(generator_id, name, MAR, params)
        for key in ["threshold", "conditional_rate"]:
            if key not in params:
                raise ValueError(f"MARSkipLogic requires '{key}' parameter")

    def _compute_missingness(self, X: torch.Tensor, rng: RNGState) -> torch.Tensor:
        n, d = X.shape
        if d < 2:
            raise ValueError("MARSkipLogic requires d >= 2")

        gate_col = self.params.get("gate_col", 0) % d
        threshold = self.params["threshold"]
        cond_rate = self.params["conditional_rate"]

        gate_active = X[:, gate_col] > threshold  # [n]

        R = torch.ones(n, d, dtype=torch.bool)

        # Columns after gate_col are candidates for skipping
        skip_cols = [j for j in range(d) if j != gate_col and j > gate_col]
        if not skip_cols:
            # Wrap around if gate_col is last
            skip_cols = [j for j in range(d) if j != gate_col]

        for col in skip_cols:
            u = rng.rand(n)
            should_skip = gate_active & (u < cond_rate)
            R[:, col] = ~should_skip

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


class MARBranching(Generator):
    """MAR mechanism mimicking survey branching.

    Predictor (branch_col) assigns rows to one of n_branches branches.
    Each branch has a set of columns that are "asked" (observed);
    other columns are made missing.

    Required params: n_branches, branch_columns (list of lists: columns observed per branch)
    Optional params: branch_col (default 0), base_mean, base_std
    """

    def __init__(self, generator_id: int, name: str, params: GeneratorParams):
        super().__init__(generator_id, name, MAR, params)
        for key in ["n_branches", "branch_columns"]:
            if key not in params:
                raise ValueError(f"MARBranching requires '{key}' parameter")
        if len(params["branch_columns"]) != params["n_branches"]:
            raise ValueError("branch_columns must have length n_branches")

    def _compute_missingness(self, X: torch.Tensor, rng: RNGState) -> torch.Tensor:
        n, d = X.shape
        if d < 2:
            raise ValueError("MARBranching requires d >= 2")

        branch_col = self.params.get("branch_col", 0) % d
        n_branches = self.params["n_branches"]
        branch_columns = self.params["branch_columns"]

        xp = X[:, branch_col]
        quantiles = torch.linspace(0, 1, n_branches + 1)
        edges = torch.quantile(xp, quantiles)

        # Assign rows to branches
        branch_assignment = torch.zeros(n, dtype=torch.long)
        for b in range(n_branches):
            if b < n_branches - 1:
                mask = (xp >= edges[b]) & (xp < edges[b + 1])
            else:
                mask = xp >= edges[b]
            branch_assignment[mask] = b

        # Branch column is always observed; other columns depend on branch
        R = torch.zeros(n, d, dtype=torch.bool)
        R[:, branch_col] = True  # Predictor always observed

        for b in range(n_branches):
            row_mask = branch_assignment == b
            observed_cols = branch_columns[b]
            for col in observed_cols:
                if col < d:
                    R[row_mask, col] = True

        # Ensure at least one observed value per column
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


class MARSectionLevel(Generator):
    """MAR mechanism with section-level gating.

    Columns divided into sections. If the gateway predictor value
    is below gateway_threshold, the entire section is missing.

    Required params: n_sections, section_sizes (list), gateway_threshold
    Optional params: gateway_col (default 0), base_mean, base_std
    """

    def __init__(self, generator_id: int, name: str, params: GeneratorParams):
        super().__init__(generator_id, name, MAR, params)
        for key in ["n_sections", "section_sizes", "gateway_threshold"]:
            if key not in params:
                raise ValueError(f"MARSectionLevel requires '{key}' parameter")
        if len(params["section_sizes"]) != params["n_sections"]:
            raise ValueError("section_sizes must have length n_sections")

    def _compute_missingness(self, X: torch.Tensor, rng: RNGState) -> torch.Tensor:
        n, d = X.shape
        if d < 2:
            raise ValueError("MARSectionLevel requires d >= 2")

        gateway_col = self.params.get("gateway_col", 0) % d
        n_sections = self.params["n_sections"]
        section_sizes = self.params["section_sizes"]
        gateway_threshold = self.params["gateway_threshold"]

        # Gateway: rows where predictor is below threshold lose entire sections
        gate_open = X[:, gateway_col] >= gateway_threshold  # [n]

        # Build sections from available columns
        target_cols = [j for j in range(d) if j != gateway_col]
        sections = []
        idx = 0
        for s in range(n_sections):
            size = min(section_sizes[s], len(target_cols) - idx)
            if size <= 0:
                break
            sections.append(target_cols[idx:idx + size])
            idx += size

        R = torch.ones(n, d, dtype=torch.bool)

        # Each section uses a slightly different threshold (shifted by section index)
        for s_idx, section_cols in enumerate(sections):
            # Later sections are more likely to be skipped
            effective_threshold = gateway_threshold + 0.2 * s_idx
            section_open = X[:, gateway_col] >= effective_threshold
            for col in section_cols:
                R[:, col] = section_open

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


class MARRequiredOptional(Generator):
    """MAR mechanism with required and optional columns.

    First required_frac of columns have low miss rate (required_rate),
    remaining columns have high miss rate (optional_rate).
    Both rates depend on the predictor through logistic link.

    Required params: required_frac, required_rate, optional_rate
    Optional params: predictor_col_idx, alpha0 (default 0.0), base_mean, base_std
    """

    def __init__(self, generator_id: int, name: str, params: GeneratorParams):
        super().__init__(generator_id, name, MAR, params)
        for key in ["required_frac", "required_rate", "optional_rate"]:
            if key not in params:
                raise ValueError(f"MARRequiredOptional requires '{key}' parameter")

    def _compute_missingness(self, X: torch.Tensor, rng: RNGState) -> torch.Tensor:
        n, d = X.shape
        if d < 2:
            raise ValueError("MARRequiredOptional requires d >= 2")

        predictor = self.params.get("predictor_col_idx", 0) % d
        alpha0 = self.params.get("alpha0", 0.0)
        required_frac = self.params["required_frac"]
        required_rate = self.params["required_rate"]
        optional_rate = self.params["optional_rate"]

        target_cols = [j for j in range(d) if j != predictor]
        n_required = max(1, int(len(target_cols) * required_frac))

        required_cols = target_cols[:n_required]
        optional_cols = target_cols[n_required:]

        # Predictor modulates the base rates via logistic
        xp = X[:, predictor]
        p_required = torch.sigmoid(alpha0 + required_rate * xp)
        p_optional = torch.sigmoid(alpha0 + optional_rate * xp)

        R = torch.ones(n, d, dtype=torch.bool)

        for col in required_cols:
            R[:, col] = ~(rng.rand(n) < p_required)

        for col in optional_cols:
            R[:, col] = ~(rng.rand(n) < p_optional)

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


class MARQuotaBased(Generator):
    """MAR mechanism with quota-based missingness.

    Once the cumulative sum of the predictor column (sorted by row order)
    exceeds quota_threshold, subsequent rows have elevated missingness
    at post_quota_rate.

    Required params: quota_threshold, post_quota_rate
    Optional params: quota_col (default 0), target_col_idx, base_mean, base_std
    """

    def __init__(self, generator_id: int, name: str, params: GeneratorParams):
        super().__init__(generator_id, name, MAR, params)
        for key in ["quota_threshold", "post_quota_rate"]:
            if key not in params:
                raise ValueError(f"MARQuotaBased requires '{key}' parameter")

    def _compute_missingness(self, X: torch.Tensor, rng: RNGState) -> torch.Tensor:
        n, d = X.shape
        if d < 2:
            raise ValueError("MARQuotaBased requires d >= 2")

        quota_col = self.params.get("quota_col", 0) % d
        target = self.params.get("target_col_idx", -1)
        if target < 0:
            target = d + target
        target = target % d
        if quota_col == target:
            quota_col = (target + 1) % d

        quota_threshold = self.params["quota_threshold"]
        post_quota_rate = self.params["post_quota_rate"]

        # Cumulative sum of absolute predictor values
        xp = X[:, quota_col].abs()
        cumsum = torch.cumsum(xp, dim=0)
        quota_exceeded = cumsum > quota_threshold  # [n]

        p_missing = torch.where(
            quota_exceeded,
            torch.tensor(post_quota_rate),
            torch.tensor(0.0),
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
