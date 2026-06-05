"""
lacuna.survey.consequence_features

Fixed observed-marginal "consequence features" of the supplied target column (PROPOSAL-P2.2c;
NORTH-STAR §6). ONE job: turn the OBSERVED values of one column into a fixed-length, scale-invariant
vector of statistics that directly expose the observed-marginal distortion a censoring mechanism
leaves behind (truncated upper tail, depleted upper-tail mass, compressed spread). This tests the
§6 hypothesis that the learned channel needs explicit consequence features — statistics the encoder
does not surface — to read a signal the oracle proved present.

Computed from observed values only (what the model legitimately sees) and z-scored WITHIN the
observed values so features transfer across datasets/columns. Deterministic (quantiles + moments;
no RNG). The feature list is FROZEN (see FEATURE_NAMES) and recorded in the manifest.

Anti-leakage: `missing_rate` is matched across δ by the generator, so it carries no δ cue; the
quantile/tail features encode the legitimate observable CONSEQUENCE of δ, not the label.
"""

import torch

# Frozen feature schema (order matters; recorded in the manifest).
FEATURE_NAMES = [
    "missing_rate",
    "zq05", "zq10", "zq25", "zq50", "zq75", "zq90", "zq95", "zq99",
    "z_min", "z_max",
    "skew",
    "frac_gt_1.0", "frac_gt_1.5", "frac_gt_2.0",
    "gap_q99_q90", "gap_q95_q50",
]
N_FEATURES = len(FEATURE_NAMES)

_Qs = torch.tensor([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99], dtype=torch.float64)
_MIN_OBS = 8  # below this the column is degenerate; return a safe (mostly-zero) vector


def compute_consequence_features(x: torch.Tensor, r: torch.Tensor, target_idx: int) -> torch.Tensor:
    """Fixed [N_FEATURES] consequence vector for the observed values of column `target_idx`.

    Args:
        x: [n, d] data (missing cells may be zeroed — only observed rows are used).
        r: [n, d] bool observed mask (True = observed).
        target_idx: the supplied candidate target column.

    Returns:
        float32 tensor of shape [N_FEATURES]. Scale-invariant (z-scored within observed).

    Raises:
        ValueError: bad shapes / target index.
    """
    if x.dim() != 2:
        raise ValueError(f"x must be 2D [n, d], got {tuple(x.shape)}")
    n, d = x.shape
    if not (0 <= target_idx < d):
        raise ValueError(f"target_idx {target_idx} out of range [0, {d})")
    obs = r[:, target_idx].bool()
    n_obs = int(obs.sum().item())
    feats = torch.zeros(N_FEATURES, dtype=torch.float64)
    feats[0] = 1.0 - n_obs / n  # missing_rate (matched across δ ⇒ no δ cue)
    if n_obs < _MIN_OBS:
        return feats.to(torch.float32)

    v = x[obs, target_idx].to(torch.float64)
    std = v.std(unbiased=False)
    if float(std) == 0.0:
        return feats.to(torch.float32)  # constant observed column ⇒ shape features are 0
    z = (v - v.mean()) / std

    q = torch.quantile(z, _Qs.to(z.device))  # [8]
    feats[1:9] = q
    feats[9] = z.min()
    feats[10] = z.max()
    feats[11] = (z ** 3).mean()  # skewness (z standardized)
    feats[12] = (z > 1.0).double().mean()
    feats[13] = (z > 1.5).double().mean()
    feats[14] = (z > 2.0).double().mean()
    feats[15] = q[7] - q[5]  # q99 - q90
    feats[16] = q[6] - q[3]  # q95 - q50
    return feats.to(torch.float32)


def schema() -> dict:
    """Manifest-ready description of the frozen feature set."""
    return {"n_features": N_FEATURES, "feature_names": list(FEATURE_NAMES),
            "scale_invariant": True, "computed_on": "observed_target_values_zscored_within_observed"}
