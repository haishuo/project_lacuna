#!/usr/bin/env python3
"""
Stage 1 (ADR-0006): can the architecture read out the mechanism PER COLUMN?

Adds a per-column readout head (lacuna.models.column_head) on top of the encoder and trains
it on mixed-mechanism datasets with per-column labels (lacuna.data.mixed_batch). Two modes:

  --freeze-encoder : probe — freeze the baseline encoder, train only the head. Tests whether
                     the per-column mechanism signal is ALREADY present in the baseline's
                     learned token representations.
  (default)        : fine-tune — train encoder (from baseline init) + head end-to-end.

Evaluation (held-out datasets, mixed compositions): per-column accuracy (overall + per-class
recall/precision), confusion, per-column ECE, and composition-estimate error — the quantities
the dataset-level model could not represent (Stage 0). Deterministic via explicit seeds.

Usage:
    python scripts/stage1_column_head.py --freeze-encoder      # probe first
    python scripts/stage1_column_head.py                       # then fine-tune
"""

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lacuna.core.rng import RNGState
from lacuna.core.types import CLASS_NAMES
from lacuna.config import load_config
from lacuna.models.encoder import create_encoder
from lacuna.models.column_head import ColumnReadoutHead, masked_per_column_ce, per_column_posterior
from lacuna.models.column_recon_features import (
    per_column_recon_features, load_reconstruction_heads_from_baseline,
)
from lacuna.models.column_deployable_features import (
    per_column_deployable_features, N_DEPLOYABLE_FEATURES,
)
from lacuna.data.catalog import create_default_catalog
from lacuna.data.mixed_batch import build_mixed_batch

BASELINE = "/mnt/artifacts/project_lacuna/runs/stage0_general_baseline"


def load_raws(names, max_cols):
    catalog = create_default_catalog()
    out = []
    for n in names:
        try:
            raw = catalog.load(n)
        except Exception as e:  # noqa: BLE001
            print(f"  warn: skip '{n}': {e}"); continue
        if raw.d <= max_cols:
            out.append(raw)
    return out


def init_encoder(ckpt, dims, device):
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)["model_state"]
    enc_keys = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
    if not enc_keys:
        raise RuntimeError(f"No 'encoder.*' weights in {ckpt}")
    enc = create_encoder(**dims)
    enc.load_state_dict(enc_keys, strict=True)  # fail loud on any mismatch
    return enc.to(device)


def _forward_logits(encoder, head, b, *, recon_heads=None, recon_target=None, deployable=False):
    """Per-column logits, optionally augmenting the head input with per-column extra features:
    reconstruction-error features (recon_heads; target = recon_target, else zeroed
    original_values) and/or deployable distributional features. Sources are concatenated."""
    tr = encoder.get_token_representations(b.tokens, b.row_mask, b.col_mask)
    feats = []
    if recon_heads is not None:
        target = recon_target if recon_target is not None else b.original_values
        feats.append(per_column_recon_features(
            recon_heads, tr, b.tokens, b.row_mask, b.col_mask, target))
    if deployable:
        feats.append(per_column_deployable_features(b.tokens, b.row_mask, b.col_mask))
    extra = torch.cat(feats, dim=-1) if feats else None
    return head(tr, b.row_mask, b.col_mask, extra)


def evaluate(encoder, head, raws, *, n_batches, batch_size, max_rows, max_cols, device, seed,
             mixture_kwargs, recon_heads=None, use_true_target=False, use_deployable=False):
    """Per-column metrics over fixed eval batches (supervised columns only)."""
    encoder.eval(); head.eval()
    K = 3
    confusion = torch.zeros(K, K, dtype=torch.long)          # [true, pred]
    conf_bins = torch.zeros(10); acc_bins = torch.zeros(10); cnt_bins = torch.zeros(10)
    comp_l1 = []  # per-item composition L1 error (soft/expected vs true fraction)

    rng = RNGState(seed=seed)
    with torch.no_grad():
        for _ in range(n_batches):
            mb = build_mixed_batch(raws, rng.spawn(), max_rows=max_rows, max_cols=max_cols,
                                   batch_size=batch_size, **mixture_kwargs)
            b = mb.batch.to(device)
            recon_target = mb.complete_values.to(device) if use_true_target else None
            probs = per_column_posterior(_forward_logits(
                encoder, head, b, recon_heads=recon_heads, recon_target=recon_target,
                deployable=use_deployable)).cpu()  # [B,C,K]
            preds = probs.argmax(-1)                                              # [B,C]
            labels, mask = mb.labels, mb.supervision_mask

            # Confusion + calibration over supervised cells
            for t in range(K):
                for p in range(K):
                    confusion[t, p] += int(((labels == t) & (preds == p) & mask).sum())
            conf = probs.max(-1).values                                          # [B,C]
            correct = (preds == labels)
            sel = mask.reshape(-1)
            cv = conf.reshape(-1)[sel]; cc = correct.reshape(-1)[sel].float()
            idx = (cv * 10).clamp(max=9).long()
            for bnum in range(10):
                m = idx == bnum
                if m.any():
                    conf_bins[bnum] += cv[m].sum(); acc_bins[bnum] += cc[m].sum(); cnt_bins[bnum] += m.sum()

            # Composition estimate: expected fraction per class vs true fraction, per item
            for i in range(mb.labels.shape[0]):
                mi = mask[i]
                if not mi.any():
                    continue
                true_frac = torch.bincount(labels[i][mi], minlength=K).float()
                true_frac /= true_frac.sum()
                pred_frac = probs[i][mi].mean(0)  # expected composition (soft)
                comp_l1.append(float((pred_frac - true_frac).abs().sum()))

    total = int(confusion.sum())
    correct_total = int(confusion.diag().sum())
    overall_acc = correct_total / total if total else 0.0
    recall = {CLASS_NAMES[k]: (confusion[k, k].item() / confusion[k].sum().item()
                               if confusion[k].sum() else None) for k in range(K)}
    precision = {CLASS_NAMES[k]: (confusion[k, k].item() / confusion[:, k].sum().item()
                                  if confusion[:, k].sum() else None) for k in range(K)}
    ece = float((cnt_bins * (acc_bins / cnt_bins.clamp(min=1) - conf_bins / cnt_bins.clamp(min=1)).abs()
                 ).sum() / cnt_bins.sum()) if cnt_bins.sum() else None
    return {
        "n_supervised_cols": total,
        "per_column_accuracy": round(overall_acc, 4),
        "per_class_recall": {k: (round(v, 4) if v is not None else None) for k, v in recall.items()},
        "per_class_precision": {k: (round(v, 4) if v is not None else None) for k, v in precision.items()},
        "confusion_true_by_pred": confusion.tolist(),
        "per_column_ece": round(ece, 4) if ece is not None else None,
        "composition_l1_mean": round(sum(comp_l1) / len(comp_l1), 4) if comp_l1 else None,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline-checkpoint", default=f"{BASELINE}/checkpoints/best_model.pt")
    ap.add_argument("--baseline-config", default=f"{BASELINE}/config.yaml")
    ap.add_argument("--freeze-encoder", action="store_true")
    ap.add_argument("--use-recon-features", action="store_true",
                    help="Stage 2: feed per-column reconstruction-error features into the head")
    ap.add_argument("--true-recon-target", action="store_true",
                    help="Stage 2b: compute recon features vs the TRUE complete values "
                         "(implies --use-recon-features)")
    ap.add_argument("--deployable-features", action="store_true",
                    help="Stage 3: add deployable per-column distributional features (no oracle)")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batches-per-epoch", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-batches", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20260529)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--p-observed", type=float, default=0.25)
    ap.add_argument("--target-miss-rate", type=float, default=0.25)
    args = ap.parse_args()
    if args.true_recon_target:
        args.use_recon_features = True  # true-target recon implies recon features

    torch.manual_seed(args.seed)
    cfg = load_config(args.baseline_config)
    dims = dict(hidden_dim=cfg.model.hidden_dim, evidence_dim=cfg.model.evidence_dim,
                n_layers=cfg.model.n_layers, n_heads=cfg.model.n_heads,
                max_cols=cfg.data.max_cols, dropout=cfg.model.dropout)
    max_rows, max_cols = cfg.data.max_rows, cfg.data.max_cols
    mixture_kwargs = dict(p_observed=args.p_observed, target_miss_rate=args.target_miss_rate)

    encoder = init_encoder(args.baseline_checkpoint, dims, args.device)

    # Stage 2: load the baseline reconstruction heads (kept FROZEN — they are the fixed signal
    # source, like a feature extractor) and size the column head to take their per-column errors.
    recon_heads = None
    n_extra = 0
    if args.use_recon_features:
        recon_heads = load_reconstruction_heads_from_baseline(
            args.baseline_checkpoint, hidden_dim=cfg.model.hidden_dim,
            dropout=cfg.model.dropout, device=args.device)
        for p in recon_heads.parameters():
            p.requires_grad = False
        recon_heads.eval()
        n_extra += recon_heads.n_heads
    if args.deployable_features:
        n_extra += N_DEPLOYABLE_FEATURES

    head = ColumnReadoutHead(hidden_dim=cfg.model.hidden_dim, dropout=cfg.model.dropout,
                             n_extra_features=n_extra).to(args.device)

    if args.freeze_encoder:
        for p in encoder.parameters():
            p.requires_grad = False
        encoder.eval()
        params = list(head.parameters())
        mode = "frozen-probe"
    else:
        params = list(encoder.parameters()) + list(head.parameters())
        mode = "fine-tune"
    if args.use_recon_features:
        mode += "+recon-true" if args.true_recon_target else "+recon"
    if args.deployable_features:
        mode += "+deploy"
    print(f"Mode: {mode} | trainable params: {sum(p.numel() for p in params):,} | device: {args.device}")

    train_raws = load_raws(cfg.data.train_datasets, max_cols)
    val_raws = load_raws(cfg.data.val_datasets, max_cols)
    print(f"Train datasets: {len(train_raws)} | Val datasets: {len(val_raws)}")

    opt = torch.optim.Adam(params, lr=args.lr, weight_decay=0.01)
    rng = RNGState(seed=args.seed)

    for epoch in range(args.epochs):
        if not args.freeze_encoder:
            encoder.train()
        head.train()
        ep_loss = 0.0
        for _ in range(args.batches_per_epoch):
            mb = build_mixed_batch(train_raws, rng.spawn(), max_rows=max_rows, max_cols=max_cols,
                                   batch_size=args.batch_size, **mixture_kwargs)
            b = mb.batch.to(args.device)
            recon_target = mb.complete_values.to(args.device) if args.true_recon_target else None
            logits = _forward_logits(encoder, head, b, recon_heads=recon_heads,
                                     recon_target=recon_target, deployable=args.deployable_features)
            loss = masked_per_column_ce(logits, mb.labels.to(args.device),
                                        mb.supervision_mask.to(args.device))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            ep_loss += loss.item()
        print(f"  epoch {epoch+1:2d}/{args.epochs}  train_ce={ep_loss/args.batches_per_epoch:.4f}")

    metrics = evaluate(encoder, head, val_raws, n_batches=args.eval_batches,
                       batch_size=args.batch_size, max_rows=max_rows, max_cols=max_cols,
                       device=args.device, seed=args.seed + 7, mixture_kwargs=mixture_kwargs,
                       recon_heads=recon_heads, use_true_target=args.true_recon_target,
                       use_deployable=args.deployable_features)

    print("\n" + "=" * 70)
    print(f"STAGE 1/2 ({mode}) — per-column evaluation on held-out mixtures")
    print("=" * 70)
    print(f"  per-column accuracy : {metrics['per_column_accuracy']}")
    print(f"  per-class recall    : {metrics['per_class_recall']}")
    print(f"  per-class precision : {metrics['per_class_precision']}")
    print(f"  per-column ECE      : {metrics['per_column_ece']}")
    print(f"  composition L1 (mean): {metrics['composition_l1_mean']}")
    print(f"  confusion [true×pred]: {metrics['confusion_true_by_pred']}")
    print("=" * 70)

    out = args.output or Path(f"{BASELINE}/stage1_column_head_{mode}.json")
    payload = {"mode": mode, "baseline_checkpoint": args.baseline_checkpoint,
               "epochs": args.epochs, "batches_per_epoch": args.batches_per_epoch,
               "batch_size": args.batch_size, "lr": args.lr,
               "mixture_kwargs": mixture_kwargs, "metrics": metrics}
    out.write_text(json.dumps(payload, indent=2))
    ckpt = Path(f"{BASELINE}/checkpoints/stage1_head_{mode}.pt")
    torch.save({"head_state": head.state_dict(),
                "encoder_finetuned": None if args.freeze_encoder else encoder.state_dict()},
               ckpt)
    print(f"\nWrote metrics → {out}\nWrote head checkpoint → {ckpt}")


if __name__ == "__main__":
    main()
