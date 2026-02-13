"""MNAR self-censoring and logistic generators."""

from typing import Tuple
import torch

from lacuna.core.rng import RNGState
from lacuna.core.types import MNAR
from lacuna.generators.base import Generator
from lacuna.generators.params import GeneratorParams
from ..base_data import sample_gaussian


class MNARLogistic(Generator):
    """MNAR generator using logistic model with target dependence.

    Missingness in target column depends on:
    - The target column's own value (beta2 term - the MNAR signature)
    - Optionally, a predictor column value (beta1 term)

    P(R_target=0 | X) = sigmoid(beta0 + beta1*X_predictor + beta2*X_target)

    The key distinguishing feature from MAR is that beta2 != 0, meaning
    missingness depends on the value that would be missing.

    Required params:
        beta0: Intercept (controls baseline missingness rate)
        beta2: Coefficient for target column (must be non-zero for MNAR)

    Optional params:
        beta1: Coefficient for predictor column (default 0.0)
        target_col_idx: Column index for missingness target (default -1, last column)
        predictor_col_idx: Column index for predictor (default 0, first column)
        base_mean: Mean for Gaussian base data (default: 0.0)
        base_std: Std for Gaussian base data (default: 1.0)
    """

    def __init__(
        self,
        generator_id: int,
        name: str,
        params: GeneratorParams,
    ):
        if "beta0" not in params:
            raise ValueError("MNARLogistic requires 'beta0' parameter")
        if "beta2" not in params:
            raise ValueError("MNARLogistic requires 'beta2' parameter")

        beta2 = params["beta2"]
        if beta2 == 0.0:
            raise ValueError(
                "beta2 must be non-zero for MNAR (otherwise mechanism is MAR). "
                f"Got beta2={beta2}"
            )

        super().__init__(
            generator_id=generator_id,
            name=name,
            class_id=MNAR,
            params=params,
        )

    def _compute_missingness(
        self,
        X: torch.Tensor,
        rng: RNGState,
    ) -> torch.Tensor:
        """Compute MNAR missingness mask based on data X."""
        n, d = X.shape

        if d < 2:
            raise ValueError("MNARLogistic requires d >= 2")

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

        R = torch.ones(n, d, dtype=torch.bool)

        beta0 = self.params["beta0"]
        beta1 = self.params.get("beta1", 0.0)
        beta2 = self.params["beta2"]

        logits = beta0 + beta1 * X[:, predictor] + beta2 * X[:, target]
        p_missing = torch.sigmoid(logits)

        missing_mask = rng.rand(n) < p_missing
        R[:, target] = ~missing_mask

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
        R = self._compute_missingness(X, rng.spawn())

        return X, R

    def apply_to(
        self,
        X: torch.Tensor,
        rng: RNGState,
    ) -> torch.Tensor:
        """Apply MNAR missingness to existing data."""
        return self._compute_missingness(X, rng)


class MNARSelfCensorHigh(Generator):
    """MNAR Self-Censoring generator (high values self-censor).

    Each column's missingness depends on its own value:
    P(R_j=0 | X_j) = sigmoid(beta0 + beta1 * X_j)

    With positive beta1, high values are more likely to be missing.

    Required params:
        beta0: Intercept (controls baseline missingness rate)
        beta1: Self-censoring strength (must be non-zero)

    Optional params:
        affected_frac: Fraction of columns with self-censoring (default 0.5)
        base_mean: Mean for Gaussian base data (default: 0.0)
        base_std: Std for Gaussian base data (default: 1.0)
    """

    def __init__(
        self,
        generator_id: int,
        name: str,
        params: GeneratorParams,
    ):
        if "beta0" not in params:
            raise ValueError("MNARSelfCensorHigh requires 'beta0' parameter")
        if "beta1" not in params:
            raise ValueError("MNARSelfCensorHigh requires 'beta1' parameter")

        beta1 = params["beta1"]
        if beta1 == 0.0:
            raise ValueError(
                "beta1 must be non-zero for self-censoring MNAR. "
                f"Got beta1={beta1}"
            )

        super().__init__(
            generator_id=generator_id,
            name=name,
            class_id=MNAR,
            params=params,
        )

    def _compute_missingness(
        self,
        X: torch.Tensor,
        rng: RNGState,
    ) -> torch.Tensor:
        """Compute self-censoring MNAR missingness mask."""
        n, d = X.shape

        beta0 = self.params["beta0"]
        beta1 = self.params["beta1"]
        affected_frac = self.params.get("affected_frac", 0.5)

        n_affected = max(1, int(d * affected_frac))
        affected_cols = rng.choice(d, size=n_affected, replace=False)

        R = torch.ones(n, d, dtype=torch.bool)

        for col in affected_cols:
            logits = beta0 + beta1 * X[:, col]
            p_missing = torch.sigmoid(logits)

            missing_mask = rng.rand(n) < p_missing
            R[:, col] = ~missing_mask

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
        R = self._compute_missingness(X, rng.spawn())

        return X, R

    def apply_to(
        self,
        X: torch.Tensor,
        rng: RNGState,
    ) -> torch.Tensor:
        """Apply self-censoring MNAR missingness to existing data."""
        return self._compute_missingness(X, rng)


class MNARSelfCensorLow(Generator):
    """MNAR Self-Censoring generator (low values self-censor).

    Same as MNARSelfCensorHigh but enforces beta1 < 0, so low values
    are more likely to be missing.

    Required params:
        beta0: Intercept (controls baseline missingness rate)
        beta1: Self-censoring strength (must be negative)

    Optional params:
        affected_frac: Fraction of columns with self-censoring (default 0.5)
        base_mean: Mean for Gaussian base data (default: 0.0)
        base_std: Std for Gaussian base data (default: 1.0)
    """

    def __init__(self, generator_id: int, name: str, params: GeneratorParams):
        if "beta0" not in params:
            raise ValueError("MNARSelfCensorLow requires 'beta0' parameter")
        if "beta1" not in params:
            raise ValueError("MNARSelfCensorLow requires 'beta1' parameter")
        if params["beta1"] >= 0:
            raise ValueError(
                f"beta1 must be negative for low self-censoring. Got beta1={params['beta1']}"
            )
        super().__init__(generator_id, name, MNAR, params)

    def _compute_missingness(self, X: torch.Tensor, rng: RNGState) -> torch.Tensor:
        n, d = X.shape
        beta0 = self.params["beta0"]
        beta1 = self.params["beta1"]
        affected_frac = self.params.get("affected_frac", 0.5)

        n_affected = max(1, int(d * affected_frac))
        affected_cols = rng.choice(d, size=n_affected, replace=False)

        R = torch.ones(n, d, dtype=torch.bool)

        for col in affected_cols:
            logits = beta0 + beta1 * X[:, col]
            p_missing = torch.sigmoid(logits)
            missing_mask = rng.rand(n) < p_missing
            R[:, col] = ~missing_mask

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


class MNARSelfCensorExtreme(Generator):
    """MNAR Self-Censoring generator (extreme values self-censor).

    Both high and low extremes self-censor via quadratic term:
    P(miss) = sigmoid(beta0 + beta_quadratic * (X - mean)^2)

    Required params:
        beta0: Intercept
        beta_quadratic: Quadratic coefficient (must be positive)

    Optional params:
        affected_frac: Fraction of columns affected (default 0.5)
        base_mean: Mean for Gaussian base data (default: 0.0)
        base_std: Std for Gaussian base data (default: 1.0)
    """

    def __init__(self, generator_id: int, name: str, params: GeneratorParams):
        if "beta0" not in params:
            raise ValueError("MNARSelfCensorExtreme requires 'beta0' parameter")
        if "beta_quadratic" not in params:
            raise ValueError("MNARSelfCensorExtreme requires 'beta_quadratic' parameter")
        super().__init__(generator_id, name, MNAR, params)

    def _compute_missingness(self, X: torch.Tensor, rng: RNGState) -> torch.Tensor:
        n, d = X.shape
        beta0 = self.params["beta0"]
        beta_q = self.params["beta_quadratic"]
        affected_frac = self.params.get("affected_frac", 0.5)

        n_affected = max(1, int(d * affected_frac))
        affected_cols = rng.choice(d, size=n_affected, replace=False)

        R = torch.ones(n, d, dtype=torch.bool)

        for col in affected_cols:
            vals = X[:, col]
            col_mean = vals.mean()
            deviation_sq = (vals - col_mean) ** 2
            logits = beta0 + beta_q * deviation_sq
            p_missing = torch.sigmoid(logits)
            missing_mask = rng.rand(n) < p_missing
            R[:, col] = ~missing_mask

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


class MNARSelfCensorWeak(Generator):
    """MNAR Self-Censoring with weak dependence.

    Same as MNARSelfCensorHigh but enforces 0 < |beta1| <= 1.5.

    Required params:
        beta0: Intercept
        beta1: Self-censoring strength (0 < |beta1| <= 1.5)

    Optional params:
        affected_frac: Fraction of columns affected (default 0.5)
        base_mean: Mean for Gaussian base data (default: 0.0)
        base_std: Std for Gaussian base data (default: 1.0)
    """

    def __init__(self, generator_id: int, name: str, params: GeneratorParams):
        if "beta0" not in params:
            raise ValueError("MNARSelfCensorWeak requires 'beta0' parameter")
        if "beta1" not in params:
            raise ValueError("MNARSelfCensorWeak requires 'beta1' parameter")
        beta1 = params["beta1"]
        if abs(beta1) == 0.0 or abs(beta1) > 1.5:
            raise ValueError(
                f"For weak self-censoring, need 0 < |beta1| <= 1.5. Got |beta1|={abs(beta1)}"
            )
        super().__init__(generator_id, name, MNAR, params)

    def _compute_missingness(self, X: torch.Tensor, rng: RNGState) -> torch.Tensor:
        n, d = X.shape
        beta0 = self.params["beta0"]
        beta1 = self.params["beta1"]
        affected_frac = self.params.get("affected_frac", 0.5)

        n_affected = max(1, int(d * affected_frac))
        affected_cols = rng.choice(d, size=n_affected, replace=False)

        R = torch.ones(n, d, dtype=torch.bool)

        for col in affected_cols:
            logits = beta0 + beta1 * X[:, col]
            p_missing = torch.sigmoid(logits)
            missing_mask = rng.rand(n) < p_missing
            R[:, col] = ~missing_mask

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


class MNARSelfCensorStrong(Generator):
    """MNAR Self-Censoring with strong dependence.

    Same as MNARSelfCensorHigh but enforces |beta1| >= 2.5.

    Required params:
        beta0: Intercept
        beta1: Self-censoring strength (|beta1| >= 2.5)

    Optional params:
        affected_frac: Fraction of columns affected (default 0.5)
        base_mean: Mean for Gaussian base data (default: 0.0)
        base_std: Std for Gaussian base data (default: 1.0)
    """

    def __init__(self, generator_id: int, name: str, params: GeneratorParams):
        if "beta0" not in params:
            raise ValueError("MNARSelfCensorStrong requires 'beta0' parameter")
        if "beta1" not in params:
            raise ValueError("MNARSelfCensorStrong requires 'beta1' parameter")
        beta1 = params["beta1"]
        if abs(beta1) < 2.5:
            raise ValueError(
                f"For strong self-censoring, need |beta1| >= 2.5. Got |beta1|={abs(beta1)}"
            )
        super().__init__(generator_id, name, MNAR, params)

    def _compute_missingness(self, X: torch.Tensor, rng: RNGState) -> torch.Tensor:
        n, d = X.shape
        beta0 = self.params["beta0"]
        beta1 = self.params["beta1"]
        affected_frac = self.params.get("affected_frac", 0.5)

        n_affected = max(1, int(d * affected_frac))
        affected_cols = rng.choice(d, size=n_affected, replace=False)

        R = torch.ones(n, d, dtype=torch.bool)

        for col in affected_cols:
            logits = beta0 + beta1 * X[:, col]
            p_missing = torch.sigmoid(logits)
            missing_mask = rng.rand(n) < p_missing
            R[:, col] = ~missing_mask

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


class MNARValueDependentStrength(Generator):
    """MNAR Self-Censoring with value-dependent strength.

    Self-censoring strength varies: low values use beta1_low, high values
    use beta1_high (piecewise linear around median).

    Required params:
        beta0: Intercept
        beta1_low: Censoring strength for below-median values
        beta1_high: Censoring strength for above-median values

    Optional params:
        affected_frac: Fraction of columns affected (default 0.5)
        base_mean: Mean for Gaussian base data (default: 0.0)
        base_std: Std for Gaussian base data (default: 1.0)
    """

    def __init__(self, generator_id: int, name: str, params: GeneratorParams):
        if "beta0" not in params:
            raise ValueError("MNARValueDependentStrength requires 'beta0' parameter")
        if "beta1_low" not in params:
            raise ValueError("MNARValueDependentStrength requires 'beta1_low' parameter")
        if "beta1_high" not in params:
            raise ValueError("MNARValueDependentStrength requires 'beta1_high' parameter")
        super().__init__(generator_id, name, MNAR, params)

    def _compute_missingness(self, X: torch.Tensor, rng: RNGState) -> torch.Tensor:
        n, d = X.shape
        beta0 = self.params["beta0"]
        beta1_low = self.params["beta1_low"]
        beta1_high = self.params["beta1_high"]
        affected_frac = self.params.get("affected_frac", 0.5)

        n_affected = max(1, int(d * affected_frac))
        affected_cols = rng.choice(d, size=n_affected, replace=False)

        R = torch.ones(n, d, dtype=torch.bool)

        for col in affected_cols:
            vals = X[:, col]
            median = vals.median()
            is_low = vals <= median

            beta1 = torch.where(is_low, torch.tensor(beta1_low), torch.tensor(beta1_high))
            logits = beta0 + beta1 * vals
            p_missing = torch.sigmoid(logits)
            missing_mask = rng.rand(n) < p_missing
            R[:, col] = ~missing_mask

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


class MNARColumnSpecificCensor(Generator):
    """MNAR Column-Specific Self-Censoring.

    Each affected column gets its own beta0 and beta1 drawn uniformly
    from given ranges.

    Optional params:
        beta0_range: (low, high) range for beta0 (default: [-1.0, 1.0])
        beta1_range: (low, high) range for beta1 (default: [0.5, 3.0])
        affected_frac: Fraction of columns affected (default 0.5)
        base_mean: Mean for Gaussian base data (default: 0.0)
        base_std: Std for Gaussian base data (default: 1.0)
    """

    def __init__(self, generator_id: int, name: str, params: GeneratorParams):
        super().__init__(generator_id, name, MNAR, params)

    def _compute_missingness(self, X: torch.Tensor, rng: RNGState) -> torch.Tensor:
        n, d = X.shape
        affected_frac = self.params.get("affected_frac", 0.5)
        n_affected = max(1, int(d * affected_frac))
        affected_cols = rng.choice(d, size=n_affected, replace=False)

        b0_range = self.params.get("beta0_range", [-1.0, 1.0])
        b1_range = self.params.get("beta1_range", [0.5, 3.0])

        # Draw per-column parameters
        beta0s = rng.rand(n_affected) * (b0_range[1] - b0_range[0]) + b0_range[0]
        beta1s = rng.rand(n_affected) * (b1_range[1] - b1_range[0]) + b1_range[0]

        R = torch.ones(n, d, dtype=torch.bool)

        for i, col in enumerate(affected_cols):
            b0 = beta0s[i].item()
            b1 = beta1s[i].item()
            logits = b0 + b1 * X[:, col]
            p_missing = torch.sigmoid(logits)
            missing_mask = rng.rand(n) < p_missing
            R[:, col] = ~missing_mask

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


class MNARDemographicDependent(Generator):
    """MNAR Demographic-Dependent Self-Censoring.

    Rows are assigned to demographic groups, each with different
    self-censoring beta coefficients.

    Required params:
        n_groups: Number of demographic groups
        beta_per_group: List of (beta0, beta1) tuples per group

    Optional params:
        affected_frac: Fraction of columns affected (default 0.5)
        base_mean: Mean for Gaussian base data (default: 0.0)
        base_std: Std for Gaussian base data (default: 1.0)
    """

    def __init__(self, generator_id: int, name: str, params: GeneratorParams):
        if "n_groups" not in params:
            raise ValueError("MNARDemographicDependent requires 'n_groups' parameter")
        if "beta_per_group" not in params:
            raise ValueError("MNARDemographicDependent requires 'beta_per_group' parameter")
        if len(params["beta_per_group"]) != params["n_groups"]:
            raise ValueError("beta_per_group length must match n_groups")
        super().__init__(generator_id, name, MNAR, params)

    def _compute_missingness(self, X: torch.Tensor, rng: RNGState) -> torch.Tensor:
        n, d = X.shape
        n_groups = self.params["n_groups"]
        beta_per_group = self.params["beta_per_group"]
        affected_frac = self.params.get("affected_frac", 0.5)

        n_affected = max(1, int(d * affected_frac))
        affected_cols = rng.choice(d, size=n_affected, replace=False)

        # Assign rows to groups
        group_assignments = rng.randint(0, n_groups, (n,))

        R = torch.ones(n, d, dtype=torch.bool)

        for col in affected_cols:
            vals = X[:, col]
            p_missing = torch.zeros(n)

            for g in range(n_groups):
                mask = group_assignments == g
                b0, b1 = beta_per_group[g]
                logits = b0 + b1 * vals[mask.squeeze()]
                p_missing[mask.squeeze()] = torch.sigmoid(logits)

            missing_mask = rng.rand(n) < p_missing
            R[:, col] = ~missing_mask

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
