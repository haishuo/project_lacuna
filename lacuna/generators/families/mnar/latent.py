"""MNAR latent confounding generators.

The key pattern: a latent (unobserved) factor Z drives both the data X
and the missingness R. Because Z is unobserved, the missingness is MNAR
(it depends on values that are themselves affected by the latent).
"""

from typing import Tuple
import torch

from lacuna.core.rng import RNGState
from lacuna.core.types import MNAR
from lacuna.generators.base import Generator
from lacuna.generators.params import GeneratorParams
from ..base_data import sample_gaussian


class MNARLatentHealth(Generator):
    """MNAR Latent Health factor.

    Single latent factor (e.g., underlying health) drives both observed
    values and missingness. Sicker patients have more missing data AND
    worse values.

    Required params:
        latent_strength: Strength of latent -> missingness link
        obs_strength: Strength of latent -> observed data link

    Optional params:
        base_mean: Mean for Gaussian base data (default: 0.0)
        base_std: Std for Gaussian base data (default: 1.0)
    """

    def __init__(self, generator_id: int, name: str, params: GeneratorParams):
        if "latent_strength" not in params:
            raise ValueError("MNARLatentHealth requires 'latent_strength' parameter")
        if "obs_strength" not in params:
            raise ValueError("MNARLatentHealth requires 'obs_strength' parameter")
        super().__init__(generator_id, name, MNAR, params)

    def sample(self, rng: RNGState, n: int, d: int) -> Tuple[torch.Tensor, torch.Tensor]:
        latent_strength = self.params["latent_strength"]
        obs_strength = self.params["obs_strength"]
        mean = self.params.get("base_mean", 0.0)
        std = self.params.get("base_std", 1.0)

        # Generate single latent factor
        Z = rng.randn(n, 1)

        # Generate base data influenced by latent
        X_base = sample_gaussian(rng.spawn(), n, d, mean=mean, std=std)
        x_loadings = torch.ones(1, d) * obs_strength
        X = X_base + Z @ x_loadings

        # Missingness depends on latent (unobserved)
        miss_loadings = torch.ones(1, d) * latent_strength
        logits = Z @ miss_loadings
        P_miss = torch.sigmoid(logits)
        R = rng.rand(n, d) >= P_miss

        if R.sum() == 0:
            R[0, 0] = True

        return X, R

    def apply_to(self, X: torch.Tensor, rng: RNGState) -> torch.Tensor:
        """Apply latent MNAR. Infers latent from X (approximate)."""
        n, d = X.shape
        latent_strength = self.params["latent_strength"]

        # Approximate latent as row mean (since Z shifts all cols)
        Z_approx = X.mean(dim=1, keepdim=True)
        miss_loadings = torch.ones(1, d) * latent_strength
        logits = Z_approx @ miss_loadings
        P_miss = torch.sigmoid(logits)
        R = rng.rand(n, d) >= P_miss

        if R.sum() == 0:
            R[0, 0] = True

        return R


class MNARLatentSES(Generator):
    """MNAR Latent Socioeconomic Status factor.

    Single latent factor (SES) drives both observed values and missingness.
    Same structure as LatentHealth with different semantic interpretation.

    Required params:
        latent_strength: Strength of latent -> missingness link
        obs_strength: Strength of latent -> observed data link

    Optional params:
        base_mean: Mean for Gaussian base data (default: 0.0)
        base_std: Std for Gaussian base data (default: 1.0)
    """

    def __init__(self, generator_id: int, name: str, params: GeneratorParams):
        if "latent_strength" not in params:
            raise ValueError("MNARLatentSES requires 'latent_strength' parameter")
        if "obs_strength" not in params:
            raise ValueError("MNARLatentSES requires 'obs_strength' parameter")
        super().__init__(generator_id, name, MNAR, params)

    def sample(self, rng: RNGState, n: int, d: int) -> Tuple[torch.Tensor, torch.Tensor]:
        latent_strength = self.params["latent_strength"]
        obs_strength = self.params["obs_strength"]
        mean = self.params.get("base_mean", 0.0)
        std = self.params.get("base_std", 1.0)

        Z = rng.randn(n, 1)
        X_base = sample_gaussian(rng.spawn(), n, d, mean=mean, std=std)
        x_loadings = torch.ones(1, d) * obs_strength
        X = X_base + Z @ x_loadings

        miss_loadings = torch.ones(1, d) * latent_strength
        logits = Z @ miss_loadings
        P_miss = torch.sigmoid(logits)
        R = rng.rand(n, d) >= P_miss

        if R.sum() == 0:
            R[0, 0] = True

        return X, R

    def apply_to(self, X: torch.Tensor, rng: RNGState) -> torch.Tensor:
        n, d = X.shape
        latent_strength = self.params["latent_strength"]
        Z_approx = X.mean(dim=1, keepdim=True)
        miss_loadings = torch.ones(1, d) * latent_strength
        logits = Z_approx @ miss_loadings
        P_miss = torch.sigmoid(logits)
        R = rng.rand(n, d) >= P_miss

        if R.sum() == 0:
            R[0, 0] = True

        return R


class MNARLatentMotivation(Generator):
    """MNAR Latent Motivation factor.

    Single latent factor (motivation/engagement) drives both observed
    values and missingness. Low-motivation subjects have more missing data.

    Required params:
        latent_strength: Strength of latent -> missingness link
        obs_strength: Strength of latent -> observed data link

    Optional params:
        base_mean: Mean for Gaussian base data (default: 0.0)
        base_std: Std for Gaussian base data (default: 1.0)
    """

    def __init__(self, generator_id: int, name: str, params: GeneratorParams):
        if "latent_strength" not in params:
            raise ValueError("MNARLatentMotivation requires 'latent_strength' parameter")
        if "obs_strength" not in params:
            raise ValueError("MNARLatentMotivation requires 'obs_strength' parameter")
        super().__init__(generator_id, name, MNAR, params)

    def sample(self, rng: RNGState, n: int, d: int) -> Tuple[torch.Tensor, torch.Tensor]:
        latent_strength = self.params["latent_strength"]
        obs_strength = self.params["obs_strength"]
        mean = self.params.get("base_mean", 0.0)
        std = self.params.get("base_std", 1.0)

        Z = rng.randn(n, 1)
        X_base = sample_gaussian(rng.spawn(), n, d, mean=mean, std=std)
        x_loadings = torch.ones(1, d) * obs_strength
        X = X_base + Z @ x_loadings

        miss_loadings = torch.ones(1, d) * latent_strength
        logits = Z @ miss_loadings
        P_miss = torch.sigmoid(logits)
        R = rng.rand(n, d) >= P_miss

        if R.sum() == 0:
            R[0, 0] = True

        return X, R

    def apply_to(self, X: torch.Tensor, rng: RNGState) -> torch.Tensor:
        n, d = X.shape
        latent_strength = self.params["latent_strength"]
        Z_approx = X.mean(dim=1, keepdim=True)
        miss_loadings = torch.ones(1, d) * latent_strength
        logits = Z_approx @ miss_loadings
        P_miss = torch.sigmoid(logits)
        R = rng.rand(n, d) >= P_miss

        if R.sum() == 0:
            R[0, 0] = True

        return R


class MNARLatentMeasurementError(Generator):
    """MNAR Latent Measurement Error factor.

    Single latent factor (instrument quality) drives both observed
    values and missingness. Poor instruments produce noisier data
    AND more missing values.

    Required params:
        latent_strength: Strength of latent -> missingness link
        obs_strength: Strength of latent -> observed data link

    Optional params:
        base_mean: Mean for Gaussian base data (default: 0.0)
        base_std: Std for Gaussian base data (default: 1.0)
    """

    def __init__(self, generator_id: int, name: str, params: GeneratorParams):
        if "latent_strength" not in params:
            raise ValueError("MNARLatentMeasurementError requires 'latent_strength' parameter")
        if "obs_strength" not in params:
            raise ValueError("MNARLatentMeasurementError requires 'obs_strength' parameter")
        super().__init__(generator_id, name, MNAR, params)

    def sample(self, rng: RNGState, n: int, d: int) -> Tuple[torch.Tensor, torch.Tensor]:
        latent_strength = self.params["latent_strength"]
        obs_strength = self.params["obs_strength"]
        mean = self.params.get("base_mean", 0.0)
        std = self.params.get("base_std", 1.0)

        Z = rng.randn(n, 1)
        X_base = sample_gaussian(rng.spawn(), n, d, mean=mean, std=std)
        x_loadings = torch.ones(1, d) * obs_strength
        X = X_base + Z @ x_loadings

        miss_loadings = torch.ones(1, d) * latent_strength
        logits = Z @ miss_loadings
        P_miss = torch.sigmoid(logits)
        R = rng.rand(n, d) >= P_miss

        if R.sum() == 0:
            R[0, 0] = True

        return X, R

    def apply_to(self, X: torch.Tensor, rng: RNGState) -> torch.Tensor:
        n, d = X.shape
        latent_strength = self.params["latent_strength"]
        Z_approx = X.mean(dim=1, keepdim=True)
        miss_loadings = torch.ones(1, d) * latent_strength
        logits = Z_approx @ miss_loadings
        P_miss = torch.sigmoid(logits)
        R = rng.rand(n, d) >= P_miss

        if R.sum() == 0:
            R[0, 0] = True

        return R


class MNARLatentOrthogonal(Generator):
    """MNAR Latent with multiple orthogonal factors.

    Multiple independent latent factors, each affecting different columns.
    More realistic than single-factor models.

    Required params:
        n_factors: Number of latent factors
        strengths: List of strength values per factor (or single value for all)

    Optional params:
        base_mean: Mean for Gaussian base data (default: 0.0)
        base_std: Std for Gaussian base data (default: 1.0)
    """

    def __init__(self, generator_id: int, name: str, params: GeneratorParams):
        if "n_factors" not in params:
            raise ValueError("MNARLatentOrthogonal requires 'n_factors' parameter")
        if "strengths" not in params:
            raise ValueError("MNARLatentOrthogonal requires 'strengths' parameter")
        super().__init__(generator_id, name, MNAR, params)

    def sample(self, rng: RNGState, n: int, d: int) -> Tuple[torch.Tensor, torch.Tensor]:
        n_factors = self.params["n_factors"]
        strengths = self.params["strengths"]
        mean = self.params.get("base_mean", 0.0)
        std = self.params.get("base_std", 1.0)

        if isinstance(strengths, (int, float)):
            strengths = [strengths] * n_factors

        # Generate orthogonal latent factors
        Z = rng.randn(n, n_factors)

        # Generate base data
        X_base = sample_gaussian(rng.spawn(), n, d, mean=mean, std=std)

        # Build loadings: each factor affects a subset of columns
        x_loadings = rng.randn(n_factors, d)
        miss_loadings = rng.randn(n_factors, d)

        # Scale by per-factor strengths
        for f in range(n_factors):
            s = strengths[f] if f < len(strengths) else strengths[-1]
            x_loadings[f] *= s
            miss_loadings[f] *= s

        X = X_base + Z @ x_loadings

        logits = Z @ miss_loadings
        P_miss = torch.sigmoid(logits)
        R = rng.rand(n, d) >= P_miss

        if R.sum() == 0:
            R[0, 0] = True

        return X, R

    def apply_to(self, X: torch.Tensor, rng: RNGState) -> torch.Tensor:
        n, d = X.shape
        n_factors = self.params["n_factors"]
        strengths = self.params["strengths"]

        if isinstance(strengths, (int, float)):
            strengths = [strengths] * n_factors

        # Approximate latent from X using PCA-like projection
        Z_approx = rng.randn(n, n_factors)
        miss_loadings = rng.randn(n_factors, d)
        for f in range(n_factors):
            s = strengths[f] if f < len(strengths) else strengths[-1]
            miss_loadings[f] *= s

        logits = Z_approx @ miss_loadings
        P_miss = torch.sigmoid(logits)
        R = rng.rand(n, d) >= P_miss

        if R.sum() == 0:
            R[0, 0] = True

        return R


class MNARLatentCorrelated(Generator):
    """MNAR Latent with correlated factors.

    Multiple latent factors with AR(1) correlation structure.
    Corr(Z_i, Z_j) = correlation^|i-j|

    Required params:
        n_factors: Number of latent factors
        correlation: AR(1) correlation between adjacent factors
        strengths: List of strength values per factor (or single value for all)

    Optional params:
        base_mean: Mean for Gaussian base data (default: 0.0)
        base_std: Std for Gaussian base data (default: 1.0)
    """

    def __init__(self, generator_id: int, name: str, params: GeneratorParams):
        if "n_factors" not in params:
            raise ValueError("MNARLatentCorrelated requires 'n_factors' parameter")
        if "correlation" not in params:
            raise ValueError("MNARLatentCorrelated requires 'correlation' parameter")
        if "strengths" not in params:
            raise ValueError("MNARLatentCorrelated requires 'strengths' parameter")
        super().__init__(generator_id, name, MNAR, params)

    def sample(self, rng: RNGState, n: int, d: int) -> Tuple[torch.Tensor, torch.Tensor]:
        n_factors = self.params["n_factors"]
        correlation = self.params["correlation"]
        strengths = self.params["strengths"]
        mean = self.params.get("base_mean", 0.0)
        std = self.params.get("base_std", 1.0)

        if isinstance(strengths, (int, float)):
            strengths = [strengths] * n_factors

        # Build AR(1) correlation matrix for latent factors
        indices = torch.arange(n_factors, dtype=torch.float32)
        corr_matrix = correlation ** torch.abs(
            indices.unsqueeze(0) - indices.unsqueeze(1)
        )
        L = torch.linalg.cholesky(corr_matrix)

        # Generate correlated latent factors
        Z_indep = rng.randn(n, n_factors)
        Z = Z_indep @ L.T

        # Generate base data
        X_base = sample_gaussian(rng.spawn(), n, d, mean=mean, std=std)

        # Build loadings
        x_loadings = rng.randn(n_factors, d)
        miss_loadings = rng.randn(n_factors, d)

        for f in range(n_factors):
            s = strengths[f] if f < len(strengths) else strengths[-1]
            x_loadings[f] *= s
            miss_loadings[f] *= s

        X = X_base + Z @ x_loadings

        logits = Z @ miss_loadings
        P_miss = torch.sigmoid(logits)
        R = rng.rand(n, d) >= P_miss

        if R.sum() == 0:
            R[0, 0] = True

        return X, R

    def apply_to(self, X: torch.Tensor, rng: RNGState) -> torch.Tensor:
        n, d = X.shape
        n_factors = self.params["n_factors"]
        correlation = self.params["correlation"]
        strengths = self.params["strengths"]

        if isinstance(strengths, (int, float)):
            strengths = [strengths] * n_factors

        # Build correlated latent (approximate)
        indices = torch.arange(n_factors, dtype=torch.float32)
        corr_matrix = correlation ** torch.abs(
            indices.unsqueeze(0) - indices.unsqueeze(1)
        )
        L = torch.linalg.cholesky(corr_matrix)

        Z_indep = rng.randn(n, n_factors)
        Z = Z_indep @ L.T

        miss_loadings = rng.randn(n_factors, d)
        for f in range(n_factors):
            s = strengths[f] if f < len(strengths) else strengths[-1]
            miss_loadings[f] *= s

        logits = Z @ miss_loadings
        P_miss = torch.sigmoid(logits)
        R = rng.rand(n, d) >= P_miss

        if R.sum() == 0:
            R[0, 0] = True

        return R
