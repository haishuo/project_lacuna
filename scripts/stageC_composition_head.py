#!/usr/bin/env python3
"""
Stage C (ADR-0007): a Dirichlet COMPOSITION head + can't-tell mass on the encoder.

Trains `CompositionHead` on the dataset-level evidence vector of the existing encoder, supervised by
the Stage-B generator's REALISED by-cell composition (the shared-denominator ground truth). The head
emits a Dirichlet over the composition simplex; the can't-tell mass and simplex-region statements are
queries on it. Default = FROZEN-encoder probe (does the evidence the encoder already learned carry the
composition signal?); `--fine-tune` trains encoder+head end-to-end.

A deep ensemble (`--n-models`) is pooled by evidence-summing (`ensemble_alpha`). Evaluation (held-out
val datasets, broad composition prior) reports, as DISTRIBUTIONS where stochastic:
  - composition L1 (Dirichlet mean vs realised) vs a constant prior-only baseline (the ADR "tanked" bar);
  - the honest seam: error on the random-vs-structured axis (f_MCAR) vs the MAR-vs-MNAR split;
  - can't-tell behaviour: correlation of vacuity (K/alpha0) with the composition error;
  - a calibration PREVIEW: reliability/ECE over simplex-region queries (the Stage-D headline in miniature).

Deterministic via explicit seeds. Usage:
    python scripts/stageC_composition_head.py --freeze-encoder            # probe first
    python scripts/stageC_composition_head.py --freeze-encoder --n-models 3
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lacuna.core.rng import RNGState
from lacuna.config import load_config
from lacuna.models.encoder import create_encoder
from lacuna.models.composition_head import (
    CompositionHead, composition_mean, cant_tell_mass, ensemble_alpha, prob_region,
)
from lacuna.training.composition_loss import dirichlet_edl_loss
from lacuna.data.catalog import create_default_catalog
from lacuna.data.composition_batch import build_composition_batch, N_FOOTPRINT_FEATURES

BASELINE = "/mnt/artifacts/project_lacuna/runs/stage0_general_baseline"
_QUERY_THRESHOLDS = (0.33, 0.5, 0.66)   # simplex-region queries: P(f_c >= t) per class c


def load_raws(names, max_cols):
    cat = create_default_catalog()
    out = []
    for n in names:
        try:
            raw = cat.load(n)
        except Exception as e:  # noqa: BLE001
            print(f"  warn: skip '{n}': {e}"); continue
        if 4 <= raw.d <= max_cols and np.isfinite(raw.data).all():
            out.append(raw)
    return out


def init_encoder(ckpt, dims, device):
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)["model_state"]
    enc_keys = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
    if not enc_keys:
        raise RuntimeError(f"No 'encoder.*' weights in {ckpt}")
    enc = create_encoder(**dims)
    enc.load_state_dict(enc_keys, strict=True)
    return enc.to(device)


def forward_alpha(encoder, head, b, extra=None):
    evidence = encoder(b.tokens, b.row_mask, b.col_mask)   # [B, evidence_dim]
    return head(evidence, extra)


def train_head(encoder, head, raws, *, freeze, epochs, batches_per_epoch, batch_size, max_rows,
               max_cols, lr, kl_max, device, seed, block_rate_share, use_footprints):
    if freeze:
        for p in encoder.parameters():
            p.requires_grad = False
        encoder.eval()
        params = list(head.parameters())
    else:
        params = list(encoder.parameters()) + list(head.parameters())
    opt = torch.optim.Adam(params, lr=lr, weight_decay=0.01)
    rng = RNGState(seed=seed)
    anneal = max(1, epochs // 2)
    for epoch in range(epochs):
        if not freeze:
            encoder.train()
        head.train()
        kl_weight = kl_max * min(1.0, (epoch + 1) / anneal)   # EDL anneal: fit first, calibrate later
        ep_loss = 0.0
        for _ in range(batches_per_epoch):
            mb = build_composition_batch(raws, rng.spawn(), max_rows=max_rows, max_cols=max_cols,
                                         batch_size=batch_size, block_rate_share=block_rate_share,
                                         with_footprints=use_footprints)
            b = mb.batch.to(device)
            extra = mb.footprints.to(device) if use_footprints else None
            alpha = forward_alpha(encoder, head, b, extra)
            loss = dirichlet_edl_loss(alpha, mb.composition.to(device), kl_weight=kl_weight)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            ep_loss += loss.item()
        print(f"  epoch {epoch+1:2d}/{epochs}  loss={ep_loss/batches_per_epoch:.4f}  kl_w={kl_weight:.3f}",
              flush=True)


def _make_eval_set(raws, *, n_batches, batch_size, max_rows, max_cols, device, seed,
                   block_rate_share, use_footprints):
    """Fixed held-out eval batches (same data for every ensemble member)."""
    rng = RNGState(seed=seed)
    out = []
    for _ in range(n_batches):
        mb = build_composition_batch(raws, rng.spawn(), max_rows=max_rows, max_cols=max_cols,
                                     batch_size=batch_size, block_rate_share=block_rate_share,
                                     with_footprints=use_footprints)
        extra = mb.footprints.to(device) if use_footprints else None
        out.append((mb.batch.to(device), extra, mb.composition.clone()))
    return out


def _collect_alpha(encoder, heads, eval_set):
    """Return (alpha_per_model [M,N,3], realised [N,3]) over the fixed eval set."""
    encoder.eval()
    for h in heads:
        h.eval()
    per_model, realised = [], []
    with torch.no_grad():
        for m, head in enumerate(heads):
            alphas = []
            for b, extra, comp in eval_set:
                alphas.append(forward_alpha(encoder, head, b, extra).cpu())
                if m == 0:
                    realised.append(comp)
            per_model.append(torch.cat(alphas, dim=0))
    return torch.stack(per_model, dim=0), torch.cat(realised, dim=0)


def _query_ece(alpha, realised, rng, n_bins=10):
    """Reliability over simplex-region queries P(f_c >= t): pool (pred_prob, indicator) and bin."""
    preds, inds = [], []
    for c in range(3):
        for t in _QUERY_THRESHOLDS:
            p = prob_region(alpha, (lambda s, cc=c, tt=t: s[:, cc] >= tt), rng.spawn(), n_samples=2000)
            preds.append(p.numpy())
            inds.append((realised[:, c] >= t).float().numpy())
    preds = np.concatenate(preds); inds = np.concatenate(inds)
    ece, n = 0.0, len(preds)
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        m = (preds >= lo) & (preds < hi if b < n_bins - 1 else preds <= hi)
        if m.any():
            ece += m.sum() / n * abs(preds[m].mean() - inds[m].mean())
    return float(ece)


def _metrics(alpha, realised, rng, *, label):
    mean = composition_mean(alpha)                                   # [N,3]
    l1 = (mean - realised).abs().sum(-1)                             # [N]
    prior_const = realised.mean(0, keepdim=True)                     # best constant predictor
    prior_l1 = (prior_const - realised).abs().sum(-1)
    vac = cant_tell_mass(alpha)                                      # [N]
    # honest seam: random-vs-structured (f_MCAR) error vs MAR-vs-MNAR split error
    mcar_mae = (mean[:, 0] - realised[:, 0]).abs().mean()
    struct = realised[:, 1] + realised[:, 2]
    has_struct = struct > 0.05
    pred_struct = mean[:, 1] + mean[:, 2]
    pred_mar_frac = mean[:, 1] / pred_struct.clamp(min=1e-6)
    true_mar_frac = realised[:, 1] / struct.clamp(min=1e-6)
    split_mae = (pred_mar_frac[has_struct] - true_mar_frac[has_struct]).abs().mean()
    # can't-tell behaviour: does vacuity rise with error?
    if l1.std() > 1e-6 and vac.std() > 1e-6:
        vac_err_corr = float(torch.corrcoef(torch.stack([vac, l1]))[0, 1])
    else:
        vac_err_corr = None
    return {
        "label": label,
        "composition_l1_mean": round(float(l1.mean()), 4),
        "composition_l1_prior_only": round(float(prior_l1.mean()), 4),
        "per_class_mae": [round(float((mean[:, k] - realised[:, k]).abs().mean()), 4) for k in range(3)],
        "mcar_axis_mae": round(float(mcar_mae), 4),
        "mar_vs_mnar_split_mae": round(float(split_mae), 4),
        "mean_cant_tell": round(float(vac.mean()), 4),
        "vacuity_error_corr": round(vac_err_corr, 4) if vac_err_corr is not None else None,
        "query_ece": round(_query_ece(alpha, realised, rng), 4),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline-checkpoint", default=f"{BASELINE}/checkpoints/best_model.pt")
    ap.add_argument("--baseline-config", default=f"{BASELINE}/config.yaml")
    ap.add_argument("--freeze-encoder", action="store_true")
    ap.add_argument("--n-models", type=int, default=3, help="deep ensemble size")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batches-per-epoch", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--kl-max", type=float, default=0.5, help="max EDL KL weight (annealed)")
    ap.add_argument("--eval-batches", type=int, default=40)
    ap.add_argument("--head-hidden", type=int, default=64)
    ap.add_argument("--use-footprint-features", action="store_true",
                    help="concatenate the 20-D observable footprint to the encoder evidence "
                         "(deployable; Stage-C attribution: the encoder under-represents it)")
    ap.add_argument("--block-rate-share", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=20260530)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    cfg = load_config(args.baseline_config)
    dims = dict(hidden_dim=cfg.model.hidden_dim, evidence_dim=cfg.model.evidence_dim,
                n_layers=cfg.model.n_layers, n_heads=cfg.model.n_heads,
                max_cols=cfg.data.max_cols, dropout=cfg.model.dropout)
    max_rows, max_cols = cfg.data.max_rows, cfg.data.max_cols
    mode = "frozen-probe" if args.freeze_encoder else "fine-tune"
    if args.use_footprint_features:
        mode += "+footprint"
    n_extra = N_FOOTPRINT_FEATURES if args.use_footprint_features else 0
    print(f"Mode: {mode} | ensemble {args.n_models} | device {args.device}")

    train_raws = load_raws(cfg.data.train_datasets, max_cols)
    val_raws = load_raws(cfg.data.val_datasets, max_cols)
    print(f"Train datasets: {len(train_raws)} | Val datasets: {len(val_raws)}")

    encoder = init_encoder(args.baseline_checkpoint, dims, args.device)
    heads = []
    for m in range(args.n_models):
        print(f"--- training head {m+1}/{args.n_models} ---", flush=True)
        head = CompositionHead(cfg.model.evidence_dim, hidden_dim=args.head_hidden,
                               dropout=cfg.model.dropout, n_extra_features=n_extra).to(args.device)
        # Each ensemble member: own encoder copy only if fine-tuning (frozen shares the one encoder).
        enc_m = encoder
        if not args.freeze_encoder and m > 0:
            enc_m = init_encoder(args.baseline_checkpoint, dims, args.device)
        train_head(enc_m, head, train_raws, freeze=args.freeze_encoder, epochs=args.epochs,
                   batches_per_epoch=args.batches_per_epoch, batch_size=args.batch_size,
                   max_rows=max_rows, max_cols=max_cols, lr=args.lr, kl_max=args.kl_max,
                   device=args.device, seed=args.seed + 101 * m, block_rate_share=args.block_rate_share,
                   use_footprints=args.use_footprint_features)
        heads.append(head)

    eval_set = _make_eval_set(val_raws, n_batches=args.eval_batches, batch_size=args.batch_size,
                              max_rows=max_rows, max_cols=max_cols, device=args.device,
                              seed=args.seed + 7, block_rate_share=args.block_rate_share,
                              use_footprints=args.use_footprint_features)
    alpha_per_model, realised = _collect_alpha(encoder, heads, eval_set)
    eval_rng = RNGState(seed=args.seed + 13)

    single = _metrics(alpha_per_model[0], realised, eval_rng.spawn(), label="single_head")
    ens = _metrics(ensemble_alpha(alpha_per_model), realised, eval_rng.spawn(),
                   label=f"ensemble_{args.n_models}")

    print("\n" + "=" * 74)
    print(f"STAGE C ({mode}) — composition head on held-out mixtures (n={realised.shape[0]})")
    print("=" * 74)
    for mt in (single, ens):
        print(f"[{mt['label']}]")
        print(f"  composition L1      : {mt['composition_l1_mean']}  (prior-only {mt['composition_l1_prior_only']})")
        print(f"  per-class MAE       : {mt['per_class_mae']}  (MCAR/MAR/MNAR)")
        print(f"  honest seam         : random-vs-structured MAE {mt['mcar_axis_mae']} | MAR-vs-MNAR split MAE {mt['mar_vs_mnar_split_mae']}")
        print(f"  can't-tell mass     : mean {mt['mean_cant_tell']} | corr(vacuity, error) {mt['vacuity_error_corr']}")
        print(f"  query ECE (preview) : {mt['query_ece']}")
    print("=" * 74)

    out = args.output or Path(f"{BASELINE}/stageC_composition_{mode}.json")
    out.write_text(json.dumps({
        "mode": mode, "n_models": args.n_models, "epochs": args.epochs,
        "batches_per_epoch": args.batches_per_epoch, "batch_size": args.batch_size,
        "lr": args.lr, "kl_max": args.kl_max, "block_rate_share": args.block_rate_share,
        "n_eval": int(realised.shape[0]), "single_head": single, "ensemble": ens,
    }, indent=2))
    ckpt = Path(f"{BASELINE}/checkpoints/stageC_composition_{mode}.pt")
    torch.save({"head_states": [h.state_dict() for h in heads]}, ckpt)
    print(f"\nWrote metrics -> {out}\nWrote heads   -> {ckpt}")


if __name__ == "__main__":
    main()
