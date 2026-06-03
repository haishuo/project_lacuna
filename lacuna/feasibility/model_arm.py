"""
lacuna.feasibility.model_arm

P1 model arm: train the full `LacunaModel` from scratch to discriminate the EXACT two
distributions the β₁′-profiled oracle compared per regime —
    H1 = MNAR(δ, β₁=1)   vs   H0 = profiled best-fit MAR(β₁′, β₂=0),
using the recorded FIXED oracle params — and measure the gap to the recorded profiled
Bayes ceiling.

This is a BINARY H0-vs-H1 test on the full Lacuna architecture, NOT a 3-class experiment.
Safety (constraint #4): the binary objective uses the RENORMALIZED MAR/MNAR ratio
    q = pMNAR / (pMAR + pMNAR),
which cancels the softmax normalizer, so the MCAR component cannot affect the loss or the
decision (both depend only on zMNAR − zMAR). The MCAR logit is thereby ignored.

Constraints honored: full LacunaModel; fresh init; NO checkpoint loaded; no frozen encoder;
all layers trainable; fixed oracle params; manifest before metrics (caller); negative gap is a
red flag, not a victory.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from lacuna.core.rng import RNGState
from lacuna.core.types import MAR, MNAR, ObservedDataset
from lacuna.data.tokenization import tokenize_and_batch
from lacuna.feasibility.delta_generator import SelfCensorParams
from lacuna.feasibility.xmodel import ConditionalGaussian

_CLASS_MAP = {0: MAR, 1: MNAR}  # generator-id 0 -> MAR (H0), 1 -> MNAR (H1)


# ---------------- data generation (fixed oracle params) ----------------

def sample_observed_fixed(
    rho: float, params: SelfCensorParams, n: int, rng: RNGState
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Draw one synthetic 2-col dataset (corr ρ) and censor col1 with FIXED params.

    Returns (x [n,2] missing-zeroed, r [n,2] bool observed, realized_missing_rate).
    """
    xm = ConditionalGaussian.synthetic(rho)
    z_p = xm.sample_predictor(n, rng.spawn())
    z_t = xm.sample_conditional(z_p, rng.spawn())
    X = torch.stack([z_p, z_t], dim=1).float()
    eta = params.beta0 + params.beta1 * z_p + params.beta2 * z_t
    missing = rng.rand(n) < torch.sigmoid(eta)
    r = torch.ones(n, 2, dtype=torch.bool)
    r[:, 1] = ~missing
    return X * r.float(), r, float(missing.float().mean().item())


def _batch_sample(
    rho: float, params: SelfCensorParams, count: int, n: int, rng: RNGState
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorized: draw `count` synthetic n-row datasets, censor col1 with FIXED params.

    Returns (x [count,n,2] missing-zeroed, r [count,n,2] bool, per-dataset realized rate [count]).
    Equivalent in distribution to `count` independent `sample_observed_fixed` calls.
    """
    sd = math.sqrt(1.0 - rho * rho)
    z_p = rng.randn(count, n)
    z_t = rho * z_p + sd * rng.randn(count, n)
    eta = params.beta0 + params.beta1 * z_p + params.beta2 * z_t
    missing = rng.rand(count, n) < torch.sigmoid(eta)  # [count, n]
    r = torch.ones(count, n, 2, dtype=torch.bool)
    r[:, :, 1] = ~missing
    X = torch.stack([z_p, z_t], dim=2)  # [count, n, 2]
    x = X * r.float()
    rates = missing.float().mean(dim=1)  # [count]
    return x, r, rates


def regime_pool(
    rho: float, h1: SelfCensorParams, h0: SelfCensorParams, n: int, n_datasets: int, rng: RNGState
) -> Tuple[List[ObservedDataset], List[int], Dict[str, float]]:
    """Build a balanced pool of datasets labelled 0=MAR(H0) / 1=MNAR(H1). Vectorized generation."""
    n1 = n_datasets // 2
    n0 = n_datasets - n1
    x0, r0, rate0 = _batch_sample(rho, h0, n0, n, rng.spawn())
    x1, r1, rate1 = _batch_sample(rho, h1, n1, n, rng.spawn())

    datasets: List[ObservedDataset] = []
    labels: List[int] = []
    for k in range(n0):
        datasets.append(ObservedDataset(x=x0[k], r=r0[k], n=n, d=2,
                                        feature_names=("predictor", "target"),
                                        dataset_id=f"0_{k}", meta={}))
        labels.append(0)
    for k in range(n1):
        datasets.append(ObservedDataset(x=x1[k], r=r1[k], n=n, d=2,
                                        feature_names=("predictor", "target"),
                                        dataset_id=f"1_{k}", meta={}))
        labels.append(1)
    # (order is H0-block then H1-block; the training loop shuffles indices each epoch)
    stats = {
        "rate_h0_mean": float(rate0.mean().item()),
        "rate_h1_mean": float(rate1.mean().item()),
        "n_h0": n0, "n_h1": n1,
    }
    return datasets, labels, stats


# ---------------- binary objective (MCAR-invariant) ----------------

def binary_q(p_class: torch.Tensor) -> torch.Tensor:
    """q = P(MNAR) / (P(MAR)+P(MNAR)). EXACTLY independent of the MCAR component.

    No additive epsilon (an eps in the denominator would re-introduce a tiny MCAR dependence
    when MAR+MNAR mass is small). The only guard is the degenerate denom==0 case ⇒ 0.5 (chance).
    """
    p_mar = p_class[:, MAR]
    p_mnar = p_class[:, MNAR]
    denom = p_mar + p_mnar
    safe = denom.clone()
    safe[denom == 0] = 1.0
    return torch.where(denom > 0, p_mnar / safe, torch.full_like(denom, 0.5))


def binary_loss(p_class: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    q = binary_q(p_class).clamp(1e-6, 1.0 - 1e-6)
    return F.binary_cross_entropy(q, y.float())


def binary_pred(p_class: torch.Tensor) -> torch.Tensor:
    return (binary_q(p_class) > 0.5).long()


# ---------------- training / evaluation ----------------

def _tokenize(datasets: List[ObservedDataset], labels: List[int], idx, n: int):
    ds = [datasets[i] for i in idx]
    gids = [labels[i] for i in idx]
    return tokenize_and_batch(ds, max_rows=n, max_cols=2, generator_ids=gids, class_mapping=_CLASS_MAP)


@torch.no_grad()
def evaluate(model, datasets, labels, n, batch_size, device) -> Tuple[float, torch.Tensor, torch.Tensor]:
    model.eval()
    preds, ys, qs = [], [], []
    for s in range(0, len(datasets), batch_size):
        idx = list(range(s, min(s + batch_size, len(datasets))))
        batch = _tokenize(datasets, labels, idx, n)
        out = model(batch, compute_reconstruction=True, compute_decision=False)
        p = out.posterior.p_class
        y = (batch.class_ids == MNAR).long().to(p.device)
        preds.append(binary_pred(p).cpu())
        qs.append(binary_q(p).cpu())
        ys.append(y.cpu())
    pred = torch.cat(preds); q = torch.cat(qs); ytrue = torch.cat(ys)
    error = float((pred != ytrue).float().mean().item())
    return error, q, ytrue


def _ece(q: torch.Tensor, y: torch.Tensor, n_bins: int = 10) -> float:
    """Expected calibration error of q (=P(MNAR)) vs outcome y."""
    edges = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for b in range(n_bins):
        m = (q >= edges[b]) & (q < edges[b + 1] if b < n_bins - 1 else q <= edges[b + 1])
        if m.sum() == 0:
            continue
        ece += (m.float().mean() * (q[m].mean() - y[m].float().mean()).abs()).item()
    return ece


def assert_fresh_and_trainable(model) -> int:
    n_param = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if n_param != n_train:
        raise ValueError(f"not all layers trainable: {n_train}/{n_param}")
    return n_param


def train_regime(
    name: str,
    rho: float,
    delta: float,
    n: int,
    h1: SelfCensorParams,
    h0: SelfCensorParams,
    ceiling: float,
    ceiling_se: float,
    model_factory,
    rng: RNGState,
    device: str,
    *,
    n_train: int = 8000,
    n_val: int = 2000,
    n_test: int = 4000,
    batch_size: int = 32,
    lr: float = 3e-4,
    max_epochs: int = 40,
    patience: int = 10,
    grad_clip: float = 1.0,
) -> Dict:
    """Train one fresh full LacunaModel on a regime and report gap-to-ceiling + leakage stats."""
    train_ds, train_y, st_tr = regime_pool(rho, h1, h0, n, n_train, rng.spawn())
    val_ds, val_y, _ = regime_pool(rho, h1, h0, n, n_val, rng.spawn())
    test_ds, test_y, st_te = regime_pool(rho, h1, h0, n, n_test, rng.spawn())

    model = model_factory().to(device)
    n_param = assert_fresh_and_trainable(model)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_val, best_state, since = math.inf, None, 0
    epochs_run = 0
    for epoch in range(max_epochs):
        epochs_run = epoch + 1
        model.train()
        perm = rng.shuffle_indices(len(train_ds))
        for s in range(0, len(train_ds), batch_size):
            idx = [int(j) for j in perm[s:s + batch_size]]
            batch = _tokenize(train_ds, train_y, idx, n)
            out = model(batch, compute_reconstruction=True, compute_decision=False)
            y = (batch.class_ids == MNAR).long().to(out.posterior.p_class.device)
            loss = binary_loss(out.posterior.p_class, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
        val_err, _, _ = evaluate(model, val_ds, val_y, n, batch_size, device)
        if val_err < best_val - 1e-4:
            best_val, since = val_err, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            since += 1
            if since >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)  # best-val weights (in-memory; NOT a loaded checkpoint)
    test_err, q, ytrue = evaluate(model, test_ds, test_y, n, batch_size, device)

    se = math.sqrt(max(test_err * (1 - test_err), 1e-12) / len(test_ds))
    gap = test_err - ceiling
    gap_se = math.sqrt(se ** 2 + ceiling_se ** 2)
    suspicious = gap < -2 * gap_se
    return {
        "regime": name, "rho": rho, "delta": delta, "n": n,
        "ceiling": ceiling, "ceiling_se": ceiling_se,
        "model_error": test_err, "model_se": se,
        "gap": gap, "gap_se": gap_se, "gap_ci": [gap - 1.96 * gap_se, gap + 1.96 * gap_se],
        "suspicious_negative_gap": bool(suspicious),
        "best_val_error": best_val, "epochs_run": epochs_run,
        "rate_h0_mean": st_te["rate_h0_mean"], "rate_h1_mean": st_te["rate_h1_mean"],
        "ece": _ece(q, ytrue),
        "trainable_param_count": n_param, "checkpoint_loaded": False,
        "binary_head": "renormalized MAR/MNAR ratio; MCAR ignored (cannot affect loss/prediction)",
        "best_state": best_state,  # caller may save for provenance; pop before serializing metrics
    }
