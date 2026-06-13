"""
scripts/illustrative_pooling_probe.py  —  ILLUSTRATIVE ONLY (NOT a Lacuna training run).

Supports docs/architecture/ARCHITECTURE-FITNESS-delta-estimation.md §3. Question: can the encoder's actual
pooling primitive (lacuna.models.encoder.AttentionPooling) — a content-weighted MEAN over set
elements — learn to output an ORDER STATISTIC of a set (the 90th percentile / the max), the kind of
function the δ footprint lives in, versus the MEAN (the function pooling is biased toward)?

Three set→scalar regressors over sets of n=64 scalars whose mean and shape vary independently:
  A. phi-MLP -> AttentionPooling (the project's layer) -> linear head      [the encoder's primitive]
  B. phi-MLP -> masked MEAN pool          -> linear head                   [pure-average baseline]
  C. sorted/ECDF front-end (11 fixed quantiles) -> linear head            [order-stat front-end]

For each, three targets: mean (control, easy for pooling), p90, max. We report test R^2.
Deterministic (fixed seeds; injected generator). A few seconds on CPU. Throwaway — numbers are
quoted inline in the report; this script is not wired into the suite.
"""

import torch
import torch.nn as nn

from lacuna.models.encoder import AttentionPooling

N_SET = 64
N_TRAIN, N_TEST = 6000, 2000
STEPS = 400
HID = 32
DEVICE = "cpu"


def make_sets(n_sets, gen):
    """n_sets sets of N_SET scalars; mean (loc) and shape (scale, skew) vary independently so the
    p90/max targets are NOT recoverable from the mean alone."""
    loc = (torch.rand(n_sets, 1, generator=gen) * 6.0 - 3.0)
    scale = (torch.rand(n_sets, 1, generator=gen) * 1.5 + 0.3)
    skew = (torch.rand(n_sets, 1, generator=gen) * 2.0)  # exponentiate a Gaussian by skew amount
    base = torch.randn(n_sets, N_SET, generator=gen)
    x = loc + scale * (base + skew * (torch.exp(0.5 * base) - 1.3))
    targets = {
        "mean": x.mean(dim=1, keepdim=True),
        "p90": torch.quantile(x, 0.90, dim=1, keepdim=True),
        "max": x.max(dim=1, keepdim=True).values,
    }
    return x, targets


class AttnPoolReg(nn.Module):
    def __init__(self):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(1, HID), nn.GELU(), nn.Linear(HID, HID), nn.GELU())
        self.pool = AttentionPooling(HID, dropout=0.0)
        self.head = nn.Linear(HID, 1)

    def forward(self, x):  # x: [B, N_SET]
        h = self.phi(x.unsqueeze(-1))            # [B, N, HID]
        pooled = self.pool(h)                    # [B, HID]  (content-weighted mean)
        return self.head(pooled)


class MeanPoolReg(nn.Module):
    def __init__(self):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(1, HID), nn.GELU(), nn.Linear(HID, HID), nn.GELU())
        self.head = nn.Linear(HID, 1)

    def forward(self, x):
        h = self.phi(x.unsqueeze(-1)).mean(dim=1)
        return self.head(h)


class ECDFReg(nn.Module):
    """Order-statistic front-end: 11 fixed quantiles of the set, then a linear head."""
    QS = torch.linspace(0, 1, 11)

    def __init__(self):
        super().__init__()
        self.head = nn.Linear(11, 1)

    def forward(self, x):
        q = torch.quantile(x, self.QS.to(x.device), dim=1).t()  # [B, 11]
        return self.head(q)


def r2(pred, y):
    ss_res = ((pred - y) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    return float(1.0 - ss_res / ss_tot)


def train_eval(model_cls, target_key, gen):
    torch.manual_seed(0)
    xtr, ttr = make_sets(N_TRAIN, gen)
    xte, tte = make_sets(N_TEST, gen)
    ytr, yte = ttr[target_key], tte[target_key]
    model = model_cls().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    lossf = nn.MSELoss()
    model.train()
    for _ in range(STEPS):
        opt.zero_grad()
        loss = lossf(model(xtr), ytr)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        return r2(model(xte), yte)


def main():
    gen = torch.Generator().manual_seed(1234)
    rows = [("AttentionPooling (encoder primitive)", AttnPoolReg),
            ("Mean pool", MeanPoolReg),
            ("ECDF/quantile front-end", ECDFReg)]
    targets = ["mean", "p90", "max"]
    print(f"Illustrative set->scalar probe (n_set={N_SET}, {N_TRAIN} train / {N_TEST} test, "
          f"{STEPS} steps). Test R^2:\n")
    print(f"{'model':40s} " + "  ".join(f"{t:>8s}" for t in targets))
    for name, cls in rows:
        r2s = [train_eval(cls, t, gen) for t in targets]
        print(f"{name:40s} " + "  ".join(f"{v:8.3f}" for v in r2s))


if __name__ == "__main__":
    main()
