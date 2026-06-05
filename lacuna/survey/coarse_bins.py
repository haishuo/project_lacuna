"""
lacuna.survey.coarse_bins

Coarse δ-bin schemes for the curriculum formulation (PROPOSAL-P2.2c coarse audit).

ONE job: map a non-negative δ to a COARSE ordered bin, for the curriculum rungs that match the
target to what transfers out-of-family (binary δ=0-vs-large diagnostic; 3-bin ordinal coarse prior).
The 7-bin scheme (`delta_bins`) remains the eventual destination; these are the coarse steps.

Schemes:
  "none"    -> the canonical 7 bins (delegates to delta_bins).
  "binary"  -> {δ=0} | {δ>0}                       (labeled DIAGNOSTIC, not the P2 main objective).
  "coarse3" -> {δ=0} | weak (0<δ≤τ] | strong (δ>τ)  (τ=1.0; the coarse calibrated prior — main run).

Vectorized + scalar assignment; bin-center metadata for E[δ]; fail loud on δ<0 / unknown scheme.
"""

import torch

from .delta_bins import NUM_BINS as NUM_BINS_7
from .delta_bins import assign_delta_bin, bin_edges

COARSE3_TAU = 1.0
_SCHEMES = ("none", "binary", "coarse3")


def scheme_num_bins(scheme: str) -> int:
    if scheme == "none":
        return NUM_BINS_7
    if scheme == "binary":
        return 2
    if scheme == "coarse3":
        return 3
    raise ValueError(f"unknown coarse scheme {scheme!r}; allowed {_SCHEMES}")


def assign_bins(scheme: str, delta: torch.Tensor, *, tau: float = COARSE3_TAU) -> torch.Tensor:
    """Vectorized [N] long bin labels for a [N] float δ tensor under `scheme`."""
    if (delta < 0).any():
        raise ValueError("delta must be >= 0")
    if scheme == "none":
        return torch.tensor([assign_delta_bin(float(d)) for d in delta], dtype=torch.long)
    if scheme == "binary":
        return (delta > 0).long()
    if scheme == "coarse3":
        b = torch.zeros_like(delta, dtype=torch.long)
        b[(delta > 0) & (delta <= tau)] = 1
        b[delta > tau] = 2
        return b
    raise ValueError(f"unknown coarse scheme {scheme!r}; allowed {_SCHEMES}")


def scheme_centers(scheme: str, *, tau: float = COARSE3_TAU) -> torch.Tensor:
    """Representative δ per bin (for E[δ]). Open upper bins use a recorded representative value."""
    if scheme == "none":
        from .metrics import bin_centers
        return bin_centers()
    if scheme == "binary":
        return torch.tensor([0.0, 1.75], dtype=torch.float32)  # {MAR}, {large-δ representative}
    if scheme == "coarse3":
        return torch.tensor([0.0, 0.5 * tau, tau + 0.75], dtype=torch.float32)  # MAR, weak mid, strong rep
    raise ValueError(f"unknown coarse scheme {scheme!r}; allowed {_SCHEMES}")


def scheme_spec(scheme: str, *, tau: float = COARSE3_TAU) -> dict:
    """Manifest-ready description of the active scheme."""
    spec = {"scheme": scheme, "num_bins": scheme_num_bins(scheme)}
    if scheme == "none":
        spec["edges"] = bin_edges()
    elif scheme == "binary":
        spec["labels"] = ["{0} (MAR)", "(0, inf) (any δ>0)"]
        spec["kind"] = "diagnostic"
    elif scheme == "coarse3":
        spec["tau"] = tau
        spec["labels"] = ["{0} (MAR)", f"(0, {tau:g}] weak", f"({tau:g}, inf) strong"]
        spec["kind"] = "coarse_ordinal_prior"
    return spec
