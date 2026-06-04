"""
lacuna.survey.delta_bins

The ordered δ-bin scheme for the P2 δ-prior (PROPOSAL-P2 §2).

ONE job: map a non-negative δ (own-value self-censoring strength; δ == β₂, MAR ⇔ δ=0)
to its ordered bin index, and expose the bin-edge metadata for the manifest/answer sheet.

The 7 bins (indices 0..6), half-open on the upper edge:

    bin 0 : δ == 0                 (MAR — exact)
    bin 1 : 0    < δ <= 0.25
    bin 2 : 0.25 < δ <= 0.5
    bin 3 : 0.5  < δ <= 1.0
    bin 4 : 1.0  < δ <= 1.5
    bin 5 : 1.5  < δ <= 2.0
    bin 6 : 2.0  < δ              ( (2.0, ∞) )

δ=0 gets its OWN bin (the "is this plausibly MAR?" mass) — it is never merged with the
first positive interval. Edges are interpretable and chosen to match the δ-grid the
generator samples (PROPOSAL §2, P2.0 decision).

Contract (Coding Bible Rules 1, 2): δ < 0 is rejected loudly — negative own-value
dependence is a DISTINCT directional mechanism, intentionally outside this scheme.
"""

# Upper edges of the positive bins 1..5 (bin 6 is the open tail above the last edge).
# bin k (1..5) covers (UPPER_EDGES[k-2], UPPER_EDGES[k-1]] — see assign_delta_bin.
_UPPER_EDGES = (0.25, 0.5, 1.0, 1.5, 2.0)

# Total number of ordered bins, including the exact-zero (MAR) bin 0.
NUM_BINS = 7

# The tail bin index (δ strictly greater than the last finite edge).
_TAIL_BIN = NUM_BINS - 1  # 6


def assign_delta_bin(delta: float) -> int:
    """Map δ >= 0 to its ordered bin index in [0, NUM_BINS).

    δ == 0 maps to bin 0 (MAR). Positive δ falls in the half-open interval
    (lower, upper] whose upper edge it does not exceed; δ above the last edge
    falls in the open tail bin.

    Args:
        delta: own-value self-censoring strength (β₂); must be >= 0.

    Returns:
        Integer bin index in [0, NUM_BINS).

    Raises:
        ValueError: if delta < 0 (negative dependence is out of scheme) or not finite.
    """
    d = float(delta)
    if d != d:  # NaN
        raise ValueError(f"delta must be a finite number, got {delta!r}")
    if d < 0.0:
        raise ValueError(
            f"delta must be >= 0 for the own-value self-censoring bins, got {d}; "
            "negative dependence is a distinct mechanism outside this scheme"
        )
    if d == 0.0:
        return 0
    for i, edge in enumerate(_UPPER_EDGES, start=1):
        if d <= edge:
            return i
    return _TAIL_BIN


def bin_edges() -> dict:
    """Serializable metadata describing the bin scheme (for manifest + audit).

    Returns a dict with the number of bins, the finite upper edges, and a
    human-readable label per bin. Deterministic; no state.
    """
    labels = ["{0} (MAR)"]
    lower = 0.0
    for edge in _UPPER_EDGES:
        labels.append(f"({lower:g}, {edge:g}]")
        lower = edge
    labels.append(f"({lower:g}, inf)")
    return {
        "num_bins": NUM_BINS,
        "zero_bin_index": 0,
        "finite_upper_edges": list(_UPPER_EDGES),
        "tail_bin_index": _TAIL_BIN,
        "labels": labels,
    }
