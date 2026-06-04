"""
lacuna.feasibility.xmodel_mv

Multi-column Gaussian X-model for P1R-A predictor-choice profiling.

A standard-normal-margin multivariate Gaussian over columns (default order [z_p, z_a, z_t] =
predictor, added candidate predictor, target). Supplies: joint sampling, and the conditional of the
target given a SET of observed columns (the discriminator's missing-row integral conditions z_t on
ALL observed predictors). All covariance/correlation matrices are PSD-validated by Cholesky (fail
loud — Rule 1). Computes in float64.
"""

import torch

from lacuna.core.rng import RNGState

P_IDX, A_IDX, T_IDX = 0, 1, 2  # default 3-column layout [z_p, z_a, z_t]


def build_p1ra_corr(rho_orig: float, rho_a: float) -> torch.Tensor:
    """3-column correlation matrix for [z_p, z_a, z_t].

    corr(z_p,z_t)=rho_orig (the P1 predictor), corr(z_a,z_t)=rho_a (the new knob), and
    corr(z_p,z_a)=rho_orig*rho_a — the conditional-independence value (z_p ⟂ z_a | z_t: the two
    predictors are independent noisy readings of the target). This is always PSD and interpretable.
    """
    if not (-1.0 < rho_orig < 1.0 and -1.0 < rho_a < 1.0):
        raise ValueError(f"correlations must be in (-1,1); got {rho_orig}, {rho_a}")
    rpa = rho_orig * rho_a
    return torch.tensor(
        [[1.0, rpa, rho_orig], [rpa, 1.0, rho_a], [rho_orig, rho_a, 1.0]], dtype=torch.float64
    )


class MultivariateGaussianX:
    """Standard-normal-margin MVN X-model with a PSD-validated correlation matrix."""

    def __init__(self, corr: torch.Tensor, names=None):
        corr = torch.as_tensor(corr, dtype=torch.float64)
        if corr.dim() != 2 or corr.shape[0] != corr.shape[1]:
            raise ValueError(f"corr must be square 2D; got shape {tuple(corr.shape)}")
        if not torch.allclose(corr, corr.T, atol=1e-9):
            raise ValueError("corr must be symmetric")
        if not torch.allclose(torch.diag(corr), torch.ones(corr.shape[0], dtype=torch.float64), atol=1e-9):
            raise ValueError("corr must have unit diagonal (standard-normal margins)")
        try:
            self._L = torch.linalg.cholesky(corr)
        except RuntimeError as e:
            raise ValueError(f"correlation matrix is not positive-definite: {e}")
        self.corr = corr
        self.d = corr.shape[0]
        self.names = list(names) if names is not None else [f"x{i}" for i in range(self.d)]
        self._cond_cache = {}

    def sample(self, n: int, rng: RNGState) -> torch.Tensor:
        """Draw [n, d] from MVN(0, corr)."""
        Z = rng.randn(n, self.d, dtype=torch.float64)
        return Z @ self._L.T

    def _conditioner(self, target_idx: int, obs_idx):
        key = (target_idx, tuple(obs_idx))
        if key not in self._cond_cache:
            obs = list(obs_idx)
            Soo = self.corr[obs][:, obs]
            Sto = self.corr[target_idx, obs]
            b = torch.linalg.solve(Soo, Sto)  # regression coefficients [k]
            cond_var = float((self.corr[target_idx, target_idx] - Sto @ b).item())
            if cond_var <= 0.0:
                raise ValueError(f"degenerate conditional variance {cond_var:.4g}")
            self._cond_cache[key] = (b, cond_var)
        return self._cond_cache[key]

    def conditional_mean_var(self, target_idx: int, obs_idx, z_obs: torch.Tensor):
        """E[z_target | z_obs] (tensor like z_obs[:,0]) and Var[z_target | z_obs] (scalar)."""
        b, v = self._conditioner(target_idx, obs_idx)
        z_obs = z_obs.to(torch.float64)
        if z_obs.dim() == 1:
            z_obs = z_obs.unsqueeze(1)
        return z_obs @ b, v

    @property
    def descriptor(self):
        return {
            "type": "MultivariateGaussianX",
            "d": self.d,
            "names": self.names,
            "corr": self.corr.tolist(),
            "psd": True,
        }
