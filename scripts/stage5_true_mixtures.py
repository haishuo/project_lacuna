#!/usr/bin/env python3
"""
Stage 5 (ADR-0006): does the deployable per-column classifier hold up at FULL subtype diversity
on TRUE mixtures?

Stage 4b showed that the Stage 0-3 "per-column MNAR is hard" / "MAR<->MNAR seed-instability"
results were largely artefacts of a self-censoring MNAR monoculture: with 8 diverse MNAR subtypes
(but still a single MARLogistic family) the frozen-encoder probe + deployable features gave stable,
deployable per-column results. Stage 5 pushes to a deployment-grade estimate by running TRUE
mixtures at FULL subtype diversity — diverse MAR subtypes AND diverse MNAR subtypes coexisting
across columns of one dataset — and comparing three arms, changing ONE diversity axis at a time:

  - monoculture     : MNAR = self-censoring only,  MAR = logistic only      (Stage 0-3 control)
  - diverse_mnar    : MNAR = 25-subtype pool,       MAR = logistic only      (Stage 4b, expanded)
  - full_diversity  : MNAR = 25-subtype pool,       MAR = 8-subtype pool      (the Stage 5 headline)

The lens is the one the arc established as deployable (Stage 3a/4): a FROZEN baseline encoder +
per-column readout head + deployable distributional features (no fine-tune — fine-tuning collapses
seed-dependently per Stage 4 — and no oracle). Train AND eval on the SAME arm distribution
(matched regime) so the number is a fair deployment estimate, not a ceiling. Report per-class
recall/precision, per-column ECE, composition-L1, realised per-class miss rate (confound check),
and a per-subtype recall breakdown, as mean +/- sd across >= 5 seeds.

Determinism (Coding Bible Rule 6): all randomness flows through explicit RNGState seeds.

Usage:
    python scripts/stage5_true_mixtures.py --seeds 1 2 3 4 5
    python scripts/stage5_true_mixtures.py --quick            # fast smoke test (1 seed, few epochs)
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lacuna.core.rng import RNGState
from lacuna.core.types import CLASS_NAMES, MCAR, MAR, MNAR
from lacuna.config import load_config
from lacuna.models.encoder import create_encoder
from lacuna.models.column_head import ColumnReadoutHead, masked_per_column_ce, per_column_posterior
from lacuna.models.column_deployable_features import (
    per_column_deployable_features, N_DEPLOYABLE_FEATURES,
)
from lacuna.data.catalog import create_default_catalog
from lacuna.data.mixed_batch import build_mixed_batch
from lacuna.data.tokenization import IDX_OBSERVED

BASELINE = "/mnt/artifacts/project_lacuna/runs/stage0_general_baseline"

# Three arms, one diversity axis added at a time (the only variable changed between them).
ARMS = {
    "monoculture":    dict(mnar_diverse=False, mar_diverse=False),
    "diverse_mnar":   dict(mnar_diverse=True,  mar_diverse=False),
    "full_diversity": dict(mnar_diverse=True,  mar_diverse=True),
}


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
    for p in enc.parameters():
        p.requires_grad = False
    enc.eval()
    return enc.to(device)


def _forward_logits(encoder, head, b):
    """Frozen-encoder probe forward: token reps + deployable distributional features -> logits."""
    tr = encoder.get_token_representations(b.tokens, b.row_mask, b.col_mask)
    feats = per_column_deployable_features(b.tokens, b.row_mask, b.col_mask)
    return head(tr, b.row_mask, b.col_mask, feats)


def _col_miss_rates(b):
    """Per-column realised miss rate from the batch tokens: 1 - observed/valid. [B, C]."""
    is_obs = b.tokens[..., IDX_OBSERVED] > 0.5
    valid = b.row_mask.unsqueeze(-1) & b.col_mask.unsqueeze(1)
    n_valid = valid.sum(dim=1).clamp(min=1).float()
    n_obs = (is_obs & valid).sum(dim=1).float()
    return (1.0 - n_obs / n_valid)


def train_head(encoder, raws, *, arm, mixture, epochs, bpe, bs, lr, max_rows, max_cols,
               hidden, dropout, device, seed):
    """Train ONLY the per-column head (frozen encoder + deployable features) on the arm dist."""
    head = ColumnReadoutHead(hidden_dim=hidden, dropout=dropout,
                             n_extra_features=N_DEPLOYABLE_FEATURES).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=0.01)
    rng = RNGState(seed=seed)
    for epoch in range(epochs):
        head.train()
        ep = 0.0
        for _ in range(bpe):
            mb = build_mixed_batch(raws, rng.spawn(), max_rows=max_rows, max_cols=max_cols,
                                   batch_size=bs, **mixture, **arm)
            b = mb.batch.to(device)
            logits = _forward_logits(encoder, head, b)
            loss = masked_per_column_ce(logits, mb.labels.to(device), mb.supervision_mask.to(device))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            ep += loss.item()
        print(f"    epoch {epoch+1:2d}/{epochs}  train_ce={ep/bpe:.4f}")
    return head


def evaluate(encoder, head, raws, *, arm, mixture, n_batches, bs, max_rows, max_cols, device, seed):
    """Per-column metrics on the matched arm distribution, with a per-subtype recall breakdown."""
    head.eval()
    K = 3
    confusion = torch.zeros(K, K, dtype=torch.long)               # [true, pred]
    conf_bins = torch.zeros(10); acc_bins = torch.zeros(10); cnt_bins = torch.zeros(10)
    comp_l1 = []
    sub_stats = defaultdict(lambda: [0, 0])                       # subtype -> [correct, total]
    miss_by_class = defaultdict(list)                            # true class -> [realised rates]

    rng = RNGState(seed=seed)
    with torch.no_grad():
        for _ in range(n_batches):
            mb = build_mixed_batch(raws, rng.spawn(), max_rows=max_rows, max_cols=max_cols,
                                   batch_size=bs, **mixture, **arm)
            b = mb.batch.to(device)
            probs = per_column_posterior(_forward_logits(encoder, head, b)).cpu()  # [B,C,K]
            preds = probs.argmax(-1)
            labels, mask = mb.labels, mb.supervision_mask
            miss = _col_miss_rates(b).cpu()

            for t in range(K):
                for p in range(K):
                    confusion[t, p] += int(((labels == t) & (preds == p) & mask).sum())
            conf = probs.max(-1).values
            correct = (preds == labels)
            sel = mask.reshape(-1)
            cv = conf.reshape(-1)[sel]; cc = correct.reshape(-1)[sel].float()
            idx = (cv * 10).clamp(max=9).long()
            for bnum in range(10):
                m = idx == bnum
                if m.any():
                    conf_bins[bnum] += cv[m].sum(); acc_bins[bnum] += cc[m].sum(); cnt_bins[bnum] += m.sum()

            for i in range(labels.shape[0]):
                mi = mask[i]
                if mi.any():
                    true_frac = torch.bincount(labels[i][mi], minlength=K).float()
                    true_frac /= true_frac.sum()
                    pred_frac = probs[i][mi].mean(0)
                    comp_l1.append(float((pred_frac - true_frac).abs().sum()))
                for col in mi.nonzero().flatten().tolist():
                    lab = int(labels[i, col].item())
                    miss_by_class[CLASS_NAMES[lab]].append(float(miss[i, col]))
                    if lab == MNAR:
                        sub = mb.mnar_subtypes[i].get(col, "self_censoring")
                    elif lab == MAR:
                        sub = mb.mar_subtypes[i].get(col, "logistic")
                    else:
                        sub = "mcar"
                    key = f"{CLASS_NAMES[lab]}:{sub}"
                    st = sub_stats[key]
                    st[0] += int(preds[i, col].item() == lab); st[1] += 1

    total = int(confusion.sum())
    overall_acc = int(confusion.diag().sum()) / total if total else 0.0
    recall = {CLASS_NAMES[k]: (confusion[k, k].item() / confusion[k].sum().item()
                               if confusion[k].sum() else None) for k in range(K)}
    precision = {CLASS_NAMES[k]: (confusion[k, k].item() / confusion[:, k].sum().item()
                                  if confusion[:, k].sum() else None) for k in range(K)}
    ece = float((cnt_bins * (acc_bins / cnt_bins.clamp(min=1) - conf_bins / cnt_bins.clamp(min=1)).abs()
                 ).sum() / cnt_bins.sum()) if cnt_bins.sum() else None
    per_subtype = {k: {"recall": round(c / t, 4), "n": t} for k, (c, t) in sorted(sub_stats.items())}
    miss_rate = {c: round(sum(v) / len(v), 4) for c, v in sorted(miss_by_class.items()) if v}
    return {
        "n_supervised_cols": total,
        "per_column_accuracy": round(overall_acc, 4),
        "per_class_recall": {k: (round(v, 4) if v is not None else None) for k, v in recall.items()},
        "per_class_precision": {k: (round(v, 4) if v is not None else None) for k, v in precision.items()},
        "confusion_true_by_pred": confusion.tolist(),
        "per_column_ece": round(ece, 4) if ece is not None else None,
        "composition_l1_mean": round(sum(comp_l1) / len(comp_l1), 4) if comp_l1 else None,
        "realised_miss_rate_by_class": miss_rate,
        "per_subtype_recall": per_subtype,
    }


def _mean_sd(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return [None, None]
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs)) if len(xs) > 1 else 0.0
    return [round(m, 4), round(sd, 4)]


def aggregate(per_seed_metrics):
    """Aggregate a list of per-seed metric dicts -> mean+/-sd scalars + per-subtype mean+/-sd."""
    agg = {}
    agg["per_column_accuracy"] = _mean_sd([m["per_column_accuracy"] for m in per_seed_metrics])
    agg["per_column_ece"] = _mean_sd([m["per_column_ece"] for m in per_seed_metrics])
    agg["composition_l1_mean"] = _mean_sd([m["composition_l1_mean"] for m in per_seed_metrics])
    for cls in CLASS_NAMES:
        agg[f"recall_{cls}"] = _mean_sd([m["per_class_recall"].get(cls) for m in per_seed_metrics])
        agg[f"precision_{cls}"] = _mean_sd([m["per_class_precision"].get(cls) for m in per_seed_metrics])
        agg[f"recall_{cls}_per_seed"] = [m["per_class_recall"].get(cls) for m in per_seed_metrics]
        agg[f"missrate_{cls}"] = _mean_sd(
            [m["realised_miss_rate_by_class"].get(cls) for m in per_seed_metrics])
    # per-subtype: mean+/-sd of per-seed recall over seeds where the subtype had n>=20
    sub_keys = sorted({k for m in per_seed_metrics for k in m["per_subtype_recall"]})
    per_subtype = {}
    for k in sub_keys:
        vals = [m["per_subtype_recall"][k]["recall"] for m in per_seed_metrics
                if k in m["per_subtype_recall"] and m["per_subtype_recall"][k]["n"] >= 20]
        n_tot = sum(m["per_subtype_recall"][k]["n"] for m in per_seed_metrics
                    if k in m["per_subtype_recall"])
        ms = _mean_sd(vals)
        per_subtype[k] = {"recall_mean": ms[0], "recall_sd": ms[1], "n_seeds": len(vals), "n_total": n_tot}
    agg["per_subtype_recall"] = per_subtype
    return agg


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline-checkpoint", default=f"{BASELINE}/checkpoints/best_model.pt")
    ap.add_argument("--baseline-config", default=f"{BASELINE}/config.yaml")
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batches-per-epoch", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-batches", type=int, default=60)
    ap.add_argument("--p-observed", type=float, default=0.25)
    ap.add_argument("--target-miss-rate", type=float, default=0.25)
    ap.add_argument("--legacy-rate", action="store_true",
                    help="use the uncompensated composer intercept (direct MAR/MNAR overshoot to "
                         "~0.30); default compensates so every mechanism hits ~target_miss_rate "
                         "(confound control: the ONLY variable across arms is diversity)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output-dir", type=Path, default=Path(f"{BASELINE}/stage5_true_mixtures"))
    ap.add_argument("--quick", action="store_true", help="fast smoke test (1 seed, 2 epochs, few batches)")
    args = ap.parse_args()

    if args.quick:
        args.seeds = [1]; args.epochs = 2; args.batches_per_epoch = 10; args.eval_batches = 10

    cfg = load_config(args.baseline_config)
    dims = dict(hidden_dim=cfg.model.hidden_dim, evidence_dim=cfg.model.evidence_dim,
                n_layers=cfg.model.n_layers, n_heads=cfg.model.n_heads,
                max_cols=cfg.data.max_cols, dropout=cfg.model.dropout)
    max_rows, max_cols = cfg.data.max_rows, cfg.data.max_cols
    mixture = dict(p_observed=args.p_observed, target_miss_rate=args.target_miss_rate,
                   compensate_rate=not args.legacy_rate)

    encoder = init_encoder(args.baseline_checkpoint, dims, args.device)
    train_raws = load_raws(cfg.data.train_datasets, max_cols)
    val_raws = load_raws(cfg.data.val_datasets, max_cols)
    print(f"Frozen-probe Stage 5 | train datasets {len(train_raws)} | val {len(val_raws)} | "
          f"device {args.device} | arms {args.arms} | seeds {args.seeds}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    aggregate_out = {}
    for arm in args.arms:
        per_seed = []
        for seed in args.seeds:
            print(f"\n=== arm={arm} seed={seed} ===")
            head = train_head(encoder, train_raws, arm=ARMS[arm], mixture=mixture,
                              epochs=args.epochs, bpe=args.batches_per_epoch, bs=args.batch_size,
                              lr=args.lr, max_rows=max_rows, max_cols=max_cols,
                              hidden=cfg.model.hidden_dim, dropout=cfg.model.dropout,
                              device=args.device, seed=seed)
            metrics = evaluate(encoder, head, val_raws, arm=ARMS[arm], mixture=mixture,
                               n_batches=args.eval_batches, bs=args.batch_size,
                               max_rows=max_rows, max_cols=max_cols, device=args.device,
                               seed=seed + 7)
            metrics["seed"] = seed
            per_seed.append(metrics)
            print(f"  acc={metrics['per_column_accuracy']} recall={metrics['per_class_recall']} "
                  f"ece={metrics['per_column_ece']} compL1={metrics['composition_l1_mean']}")
            (args.output_dir / f"{arm}_s{seed}.json").write_text(json.dumps(metrics, indent=2))
        aggregate_out[arm] = {"seeds": args.seeds, **aggregate(per_seed)}

    (args.output_dir / "aggregate.json").write_text(json.dumps(aggregate_out, indent=2))
    print("\n" + "=" * 78)
    print("STAGE 5 — frozen-probe per-column, mean +/- sd across seeds")
    print("=" * 78)
    for arm, a in aggregate_out.items():
        print(f"\n[{arm}]  acc {a['per_column_accuracy']}  "
              f"MCAR {a['recall_MCAR']}  MAR {a['recall_MAR']}  MNAR {a['recall_MNAR']}  "
              f"ECE {a['per_column_ece']}  compL1 {a['composition_l1_mean']}")
    print(f"\nWrote per-run + aggregate -> {args.output_dir}")


if __name__ == "__main__":
    main()
