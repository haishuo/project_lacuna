"""
lacuna.survey.leakage

Matched-rate leakage diagnostics — a FORMAL, BLOCKING gate (PROPOSAL-P2 §8; audit §10;
P2.1 review carry-over #1).

ONE job: over a whole generated corpus, prove that the realized missing rate carries NO usable
cue about δ — otherwise the δ-prior could read δ off the rate instead of the footprint, and the
result is invalid. Reports (a) δ→realized-rate correlation, (b) a per-bin realized-rate table vs
the matched target, and (c) a rate-ONLY baseline that must be no better than the base rate at
predicting the δ-bin. A `main` run whose `leakage_pass` is False may not be interpreted.

Determinism (Rule 6): the rate-only baseline's train/test split uses an injected RNGState.
Pure NumPy/torch aggregation otherwise.
"""

from dataclasses import dataclass
from typing import List

import torch

from lacuna.core.rng import RNGState

from .answer_sheet import AnswerSheet

DEFAULT_CORR_TOL = 0.2
DEFAULT_RATE_ABS_TOL = 0.03
DEFAULT_BASELINE_MARGIN = 0.1


@dataclass(frozen=True)
class LeakageReport:
    delta_rate_pearson: float
    delta_rate_slope: float
    per_bin: list  # list of {bin, n, mean_rate, sd_rate, abs_dev_from_target}
    target_rate: float
    rate_only_acc: float
    base_rate_acc: float
    n_examples: int

    def to_dict(self) -> dict:
        return {
            "delta_rate_pearson": self.delta_rate_pearson,
            "delta_rate_slope": self.delta_rate_slope,
            "per_bin": self.per_bin,
            "target_rate": self.target_rate,
            "rate_only_acc": self.rate_only_acc,
            "base_rate_acc": self.base_rate_acc,
            "n_examples": self.n_examples,
        }


def _pearson_slope(x: torch.Tensor, y: torch.Tensor) -> tuple:
    xc = x - x.mean()
    yc = y - y.mean()
    denom = (xc.norm() * yc.norm()).item()
    var = float((xc * xc).sum().item())
    r = 0.0 if denom == 0 else float((xc * yc).sum().item() / denom)
    slope = 0.0 if var == 0 else float((xc * yc).sum().item() / var)
    return r, slope


def _rate_only_baseline(
    rates: torch.Tensor, bins: torch.Tensor, rng: RNGState, n_buckets: int = 10
) -> tuple:
    """Predict δ-bin from realized rate ALONE via rate-quantile-bucket majority vote.

    Train/test split (50/50, seeded). Train: bucket rates into `n_buckets` quantiles, assign each
    bucket its majority bin. Test: predict by bucket; measure accuracy. Compare to the base-rate
    (predict the global majority bin). Matched rates => rate-only ≈ base-rate (no cue).
    """
    n = rates.shape[0]
    perm = torch.from_numpy(rng.shuffle_indices(n)).long()
    half = n // 2
    tr, te = perm[:half], perm[half:]
    if len(tr) == 0 or len(te) == 0:
        return 0.0, 0.0
    tr_rates, tr_bins = rates[tr], bins[tr]
    te_rates, te_bins = rates[te], bins[te]

    # quantile bucket edges from the train rates
    qs = torch.linspace(0, 1, n_buckets + 1)
    edges = torch.quantile(tr_rates, qs)
    edges[0] = edges[0] - 1e-6
    edges[-1] = edges[-1] + 1e-6

    def bucket_of(r):
        return torch.bucketize(r, edges[1:-1], right=False)

    tr_b = bucket_of(tr_rates)
    num_classes = int(bins.max().item()) + 1
    bucket_major = {}
    for b in range(n_buckets):
        m = tr_b == b
        if m.sum() == 0:
            continue
        counts = torch.bincount(tr_bins[m], minlength=num_classes)
        bucket_major[b] = int(counts.argmax().item())
    global_major = int(torch.bincount(tr_bins, minlength=num_classes).argmax().item())

    te_b = bucket_of(te_rates)
    preds = torch.tensor([bucket_major.get(int(b.item()), global_major) for b in te_b])
    rate_only_acc = float((preds == te_bins).float().mean().item())
    base_rate_acc = float((te_bins == global_major).float().mean().item())
    return rate_only_acc, base_rate_acc


def assess_leakage(
    sheets: List[AnswerSheet],
    rng: RNGState,
    *,
    n_buckets: int = 10,
) -> LeakageReport:
    """Compute the corpus-level leakage diagnostics from a list of answer sheets."""
    if len(sheets) < 4:
        raise ValueError(f"need >= 4 examples for a leakage assessment, got {len(sheets)}")
    deltas = torch.tensor([s.delta for s in sheets], dtype=torch.float64)
    bins = torch.tensor([s.delta_bin for s in sheets], dtype=torch.long)
    rates = torch.tensor([s.realized_rate for s in sheets], dtype=torch.float64)
    target_rates = [s.target_rate for s in sheets]
    target_rate = float(sum(target_rates) / len(target_rates))

    r, slope = _pearson_slope(deltas, rates)

    per_bin = []
    for b in sorted(set(int(x) for x in bins.tolist())):
        m = bins == b
        br = rates[m]
        mean_rate = float(br.mean().item())
        sd_rate = float(br.std(unbiased=False).item()) if br.numel() > 1 else 0.0
        per_bin.append({
            "bin": b,
            "n": int(m.sum().item()),
            "mean_rate": mean_rate,
            "sd_rate": sd_rate,
            "abs_dev_from_target": abs(mean_rate - target_rate),
        })

    rate_only_acc, base_rate_acc = _rate_only_baseline(
        rates.to(torch.float32), bins, rng.spawn(), n_buckets=n_buckets
    )

    return LeakageReport(
        delta_rate_pearson=r,
        delta_rate_slope=slope,
        per_bin=per_bin,
        target_rate=target_rate,
        rate_only_acc=rate_only_acc,
        base_rate_acc=base_rate_acc,
        n_examples=len(sheets),
    )


def leakage_pass(
    report: LeakageReport,
    *,
    corr_tol: float = DEFAULT_CORR_TOL,
    rate_abs_tol: float = DEFAULT_RATE_ABS_TOL,
    baseline_margin: float = DEFAULT_BASELINE_MARGIN,
) -> bool:
    """True iff the corpus shows no usable rate cue for δ (the blocking gate).

    Passes when: |corr(δ, rate)| <= corr_tol, every per-bin mean rate is within rate_abs_tol of
    the matched target, and the rate-only baseline does not beat the base rate by > baseline_margin.
    """
    if abs(report.delta_rate_pearson) > corr_tol:
        return False
    for row in report.per_bin:
        if row["abs_dev_from_target"] > rate_abs_tol:
            return False
    if report.rate_only_acc > report.base_rate_acc + baseline_margin:
        return False
    return True
