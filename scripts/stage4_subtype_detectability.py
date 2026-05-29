#!/usr/bin/env python3
"""
Stage 4 (ADR-0006): does per-column detectability STRATIFY by generator subtype?

Stages 0-3 used a single MNAR subtype (logistic self-censoring) — plausibly the hardest, and
the only MNAR family that supports per-column targeting. This experiment uses the FULL registry
(~113 generators) the WRONG way for mixtures but the RIGHT way for this question: apply one
registry generator per dataset (full-diversity, single-mechanism), label each column the
generator made missing by that generator's class, and break down per-column recall BY GENERATOR.
No per-column targeting needed. The decisive output: is recall uniform, or stratified (some
subtypes ~reliably detected, others ~0)?

Fast frozen-probe check (head-only). The full fine-tune/multi-seed version is a separate run.

Usage:
    python scripts/stage4_subtype_detectability.py            # frozen probe (default)
    python scripts/stage4_subtype_detectability.py --fine-tune --epochs 40   # full version
"""
import argparse, json, sys
from collections import defaultdict
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lacuna.core.rng import RNGState
from lacuna.core.types import CLASS_NAMES
from lacuna.config import load_config
from lacuna.models.encoder import create_encoder
from lacuna.models.column_head import ColumnReadoutHead, masked_per_column_ce, per_column_posterior
from lacuna.models.column_deployable_features import per_column_deployable_features, N_DEPLOYABLE_FEATURES
from lacuna.data.catalog import create_default_catalog
from lacuna.data.tokenization import tokenize_and_batch
from lacuna.data.semisynthetic import apply_missingness, subsample_raw
from lacuna.generators.families.registry_builder import load_registry_from_config
from lacuna.generators.priors import GeneratorPrior

BASELINE = "/mnt/artifacts/project_lacuna/runs/stage0_general_baseline"


def load_raws(names, max_cols):
    cat = create_default_catalog(); out = []
    for n in names:
        try:
            raw = cat.load(n)
        except Exception as e:  # noqa: BLE001
            print(f"  skip {n}: {e}"); continue
        if raw.d <= max_cols:
            out.append(raw)
    return out


def init_encoder(ckpt, dims, device):
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)["model_state"]
    keys = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
    enc = create_encoder(**dims); enc.load_state_dict(keys, strict=True)
    return enc.to(device)


def registry_batch(raws, registry, prior, rng, *, max_rows, max_cols, batch_size):
    """One registry generator per dataset; label each missing-bearing column by its class."""
    observed, gen_names, gen_classes = [], [], []
    for _ in range(batch_size):
        irng = rng.spawn()
        raw = raws[irng.randint(0, len(raws), (1,)).item()]
        raw_sub = subsample_raw(raw, max_rows=max_rows, rng=irng.spawn())
        for attempt in range(25):
            gen = registry[prior.sample(irng.spawn())]
            try:
                ss = apply_missingness(raw_sub, gen, irng.spawn()); break
            except ValueError:
                if attempt == 24:
                    raise RuntimeError(f"no compatible generator for {raw.name} (d={raw_sub.d})")
        observed.append(ss.observed); gen_names.append(gen.name); gen_classes.append(gen.class_id)
    batch = tokenize_and_batch(observed, max_rows=max_rows, max_cols=max_cols)
    labels = torch.zeros(batch_size, max_cols, dtype=torch.long)
    sup = torch.zeros(batch_size, max_cols, dtype=torch.bool)
    for i, obs in enumerate(observed):
        col_missing = ~obs.r.all(dim=0)  # [d]
        for j in range(obs.d):
            if bool(col_missing[j]):
                labels[i, j] = gen_classes[i]; sup[i, j] = True
    return batch, labels, sup, gen_names


def fwd(encoder, head, b, device):
    tr = encoder.get_token_representations(b.tokens, b.row_mask, b.col_mask)
    extra = per_column_deployable_features(b.tokens, b.row_mask, b.col_mask)
    return head(tr, b.row_mask, b.col_mask, extra)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=f"{BASELINE}/checkpoints/best_model.pt")
    ap.add_argument("--config", default=f"{BASELINE}/config.yaml")
    ap.add_argument("--generators", default="lacuna_tabular_110")
    ap.add_argument("--fine-tune", action="store_true")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batches-per-epoch", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--eval-batches", type=int, default=150)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=20260529)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output", type=Path, default=Path(f"{BASELINE}/stage4_subtype.json"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    cfg = load_config(args.config)
    dims = dict(hidden_dim=cfg.model.hidden_dim, evidence_dim=cfg.model.evidence_dim,
                n_layers=cfg.model.n_layers, n_heads=cfg.model.n_heads,
                max_cols=cfg.data.max_cols, dropout=cfg.model.dropout)
    mr, mc = cfg.data.max_rows, cfg.data.max_cols
    enc = init_encoder(args.checkpoint, dims, args.device)
    head = ColumnReadoutHead(hidden_dim=cfg.model.hidden_dim, dropout=cfg.model.dropout,
                             n_extra_features=N_DEPLOYABLE_FEATURES).to(args.device)
    registry = load_registry_from_config(args.generators)
    prior = GeneratorPrior.uniform(registry)
    print(f"Registry {args.generators}: {registry.K} gens {registry.class_counts()} | "
          f"mode={'fine-tune' if args.fine_tune else 'frozen-probe'}")

    if args.fine_tune:
        params = list(enc.parameters()) + list(head.parameters())
    else:
        for p in enc.parameters():
            p.requires_grad = False
        enc.eval(); params = list(head.parameters())

    train_raws = load_raws(cfg.data.train_datasets, mc)
    val_raws = load_raws(cfg.data.val_datasets, mc)
    opt = torch.optim.Adam(params, lr=args.lr, weight_decay=0.01)
    rng = RNGState(seed=args.seed)
    for ep in range(args.epochs):
        if args.fine_tune:
            enc.train()
        head.train(); tot = 0.0
        for _ in range(args.batches_per_epoch):
            b, lab, sup, _ = registry_batch(train_raws, registry, prior, rng.spawn(),
                                            max_rows=mr, max_cols=mc, batch_size=args.batch_size)
            b = b.to(args.device)
            loss = masked_per_column_ce(fwd(enc, head, b, args.device),
                                        lab.to(args.device), sup.to(args.device))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); tot += loss.item()
        print(f"  epoch {ep+1}/{args.epochs} ce={tot/args.batches_per_epoch:.4f}")

    # Eval: per-column recall grouped by generator name
    enc.eval(); head.eval()
    correct = defaultdict(int); total = defaultdict(int); gclass = {}
    erng = RNGState(seed=args.seed + 7)
    with torch.no_grad():
        for _ in range(args.eval_batches):
            b, lab, sup, gnames = registry_batch(val_raws, registry, prior, erng.spawn(),
                                                 max_rows=mr, max_cols=mc, batch_size=args.batch_size)
            preds = per_column_posterior(fwd(enc, head, b.to(args.device), args.device)).argmax(-1).cpu()
            for i, g in enumerate(gnames):
                m = sup[i]
                if not m.any():
                    continue
                total[g] += int(m.sum()); correct[g] += int(((preds[i] == lab[i]) & m).sum())
                gclass[g] = CLASS_NAMES[int(lab[i][m][0])]
    rows = sorted(((g, correct[g]/total[g], total[g], gclass[g]) for g in total if total[g] >= 20),
                  key=lambda r: r[1])
    print(f"\n{'='*70}\nPER-GENERATOR per-column recall (n>=20 supervised cols), sorted\n{'='*70}")
    for g, rec, n, c in rows:
        print(f"  {rec:5.2f}  {c:5s} {g:42s} (n={n})")
    # by class summary
    byc = defaultdict(lambda: [0, 0])
    for g in total:
        byc[gclass[g]][0] += correct[g]; byc[gclass[g]][1] += total[g]
    print("  --- by class ---")
    for c in CLASS_NAMES:
        if byc[c][1]:
            print(f"  {c}: recall={byc[c][0]/byc[c][1]:.3f} (n={byc[c][1]})")
    args.output.write_text(json.dumps(
        {"mode": "fine-tune" if args.fine_tune else "frozen-probe",
         "per_generator": {g: {"recall": correct[g]/total[g], "n": total[g], "class": gclass[g]}
                           for g in total if total[g] >= 20}}, indent=2))
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
