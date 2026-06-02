#!/usr/bin/env python3
"""
Stage T — train the REAL full Lacuna MoE on the IMPROVED (mixed / heterogeneous) generators.

The fix for the Stage-S mistake: Stage S retrained the MoE but on the OLD v1.0 single-mechanism
registry. Here the hole-punching is the improved realistic process (`compose_mixed_missingness`:
heterogeneous per-column mechanisms, diverse subtype pools, matched rate), and the per-column ANSWER
SHEET is collapsed to a single dataset-level label for the existing single-label MoE — which is trained
END-TO-END FROM SCRATCH through the unchanged `create_lacuna_model` / `Trainer` / `LacunaLoss` pipeline.

Audit (in the commit message / the conversation): the existing MoE pipeline needs NO internal change.
The only new code is the wrapper loader below: mixed generator -> answer sheet -> composition + collapsed
label -> `tokenize_and_batch` -> `dataclasses.replace(class_ids=collapsed)`. `original_values`/
`reconstruction_mask` come from `tokenize_and_batch` (self-supervised recon, mechanism-agnostic); the MoE
gate's missingness features are computed inside the model; `variant_ids`/`generator_ids` are optional.

The generator emits BOTH a single MoE label AND the full composition vector (saved in the manifest).
Collapse rule is configurable: dominant / primary / thresholded (`lacuna.data.label_collapse`).

`--smoke` runs a few REAL train steps (forward + loss + backward through the full MoE) to CONFIRM the
mixed generator feeds the pipeline, then stops — NO full training. Drop `--smoke` for the real run.
"""

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lacuna.core.rng import RNGState
from lacuna.core.types import MCAR, MAR, MNAR
from lacuna.config import load_config
from lacuna.models import create_lacuna_model
from lacuna.training import Trainer, TrainerConfig
from lacuna.data.catalog import create_default_catalog
from lacuna.data.semisynthetic import subsample_raw
from lacuna.data.mixed_missingness import compose_mixed_missingness, OBSERVED
from lacuna.data.mixed_batch import sample_column_classes
from lacuna.data.tokenization import tokenize_and_batch
from lacuna.data.label_collapse import composition_vector, collapse_label

_MECHS = (MCAR, MAR, MNAR)


def _sample_primary_classes(d, primary, rng, p_observed, primary_frac=0.7):
    """Column assignment biased toward a PRIMARY mechanism (rest = observed + minority secondaries)."""
    classes = []
    for _ in range(d):
        u = rng.rand(1).item()
        if u < p_observed:
            classes.append(OBSERVED)
        elif u < p_observed + primary_frac * (1 - p_observed):
            classes.append(primary)
        else:
            classes.append(_MECHS[rng.randint(0, 3, (1,)).item()])
    if not any(c in (OBSERVED, MCAR) for c in classes):
        classes[rng.randint(0, d, (1,)).item()] = MCAR      # clean predictor for any MAR
    if not any(c in _MECHS for c in classes):
        classes[rng.randint(0, d, (1,)).item()] = primary   # >=1 missing-bearing
    return tuple(classes)


class MixedMoEDataLoader:
    """Yields MoE-compatible TokenBatches whose holes are punched by the improved MIXED generator and
    whose label is the collapsed dataset-level mechanism. Records every dataset's composition vector."""

    def __init__(self, raw_datasets, *, max_rows, max_cols, batch_size, batches_per_epoch, seed,
                 collapse_rule="dominant", mnar_threshold=0.5, p_observed=0.25,
                 mnar_diverse=True, mar_diverse=True, compensate_rate=True, primary_frac=0.7):
        self.raws = raw_datasets
        self.max_rows, self.max_cols = max_rows, max_cols
        self.batch_size, self.batches_per_epoch = batch_size, batches_per_epoch
        self.seed = seed
        self.rule, self.mnar_threshold, self.p_observed = collapse_rule, mnar_threshold, p_observed
        self.mnar_diverse, self.mar_diverse, self.compensate_rate = mnar_diverse, mar_diverse, compensate_rate
        self.primary_frac = primary_frac
        self._epoch = 0
        self.compositions = []   # manifest: [p_mcar, p_mar, p_mnar] per generated dataset
        self.label_counts = [0, 0, 0]

    def __len__(self):
        return self.batches_per_epoch

    def _make_item(self, rng):
        for _ in range(20):
            it = rng.spawn()
            raw = self.raws[it.randint(0, len(self.raws), (1,)).item()]
            sub = subsample_raw(raw, max_rows=self.max_rows, rng=it.spawn())
            if sub.d < 4:
                continue
            primary = _MECHS[it.randint(0, 3, (1,)).item()]
            classes = (_sample_primary_classes(sub.d, primary, it.spawn(), self.p_observed, self.primary_frac)
                       if self.rule == "primary"
                       else sample_column_classes(sub.d, it.spawn(), p_observed=self.p_observed))
            try:
                res = compose_mixed_missingness(
                    sub, classes, it.spawn(), target_miss_rate=0.25,
                    mnar_diverse=self.mnar_diverse, mar_diverse=self.mar_diverse,
                    compensate_rate=self.compensate_rate)
                comp = composition_vector(res.column_classes, res.per_column_miss_rate, sub.n)
                label = collapse_label(comp, self.rule, primary=primary, mnar_threshold=self.mnar_threshold)
            except ValueError:
                continue
            return res.observed, label, comp
        raise RuntimeError("could not generate a valid mixed dataset after 20 tries")

    def __iter__(self):
        rng = RNGState(seed=self.seed + self._epoch * 1_000_000)
        self._epoch += 1
        for _ in range(self.batches_per_epoch):
            brng = rng.spawn()
            datasets, labels = [], []
            for _ in range(self.batch_size):
                obs, label, comp = self._make_item(brng)
                datasets.append(obs); labels.append(label)
                self.compositions.append([round(float(x), 4) for x in comp]); self.label_counts[label] += 1
            batch = tokenize_and_batch(datasets=datasets, max_rows=self.max_rows, max_cols=self.max_cols)
            batch = dataclasses.replace(batch, class_ids=torch.tensor(labels, dtype=torch.long))
            yield batch


def build_model(cfg, mnar_variants):
    return create_lacuna_model(
        hidden_dim=cfg.model.hidden_dim, evidence_dim=cfg.model.evidence_dim,
        n_layers=cfg.model.n_layers, n_heads=cfg.model.n_heads, max_cols=cfg.data.max_cols,
        dropout=cfg.model.dropout, mnar_variants=mnar_variants,
        learn_evidence_attenuation=cfg.model.learn_evidence_attenuation,
        evidence_attenuation_init=cfg.model.evidence_attenuation_init)


def load_raws(cfg, names):
    cat = create_default_catalog()
    out = []
    for n in names:
        try:
            raw = cat.load(n)
        except Exception:  # noqa: BLE001
            continue
        if 4 <= raw.d <= cfg.data.max_cols and np.isfinite(raw.data).all():
            out.append(raw)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/training/survey.yaml")
    ap.add_argument("--collapse-rule", choices=["dominant", "primary", "thresholded"], default="dominant")
    ap.add_argument("--mnar-threshold", type=float, default=0.5)
    ap.add_argument("--smoke", action="store_true", help="run a few real train steps then stop (no full train)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=20260601)
    ap.add_argument("--name", default="stageT_mixed_moe")
    args = ap.parse_args()

    cfg = load_config(args.config)
    torch.manual_seed(args.seed)
    train_raws = load_raws(cfg, cfg.data.train_datasets)
    val_raws = load_raws(cfg, cfg.data.val_datasets)
    mnar_variants = ["self_censoring"]
    mixarg = dict(max_rows=cfg.data.max_rows, max_cols=cfg.data.max_cols, batch_size=cfg.training.batch_size,
                  collapse_rule=args.collapse_rule, mnar_threshold=args.mnar_threshold,
                  mnar_diverse=True, mar_diverse=True, compensate_rate=True)
    train_loader = MixedMoEDataLoader(train_raws, batches_per_epoch=cfg.training.batches_per_epoch,
                                      seed=args.seed, **mixarg)
    val_loader = MixedMoEDataLoader(val_raws, batches_per_epoch=cfg.training.val_batches,
                                    seed=args.seed + 7, **mixarg)
    model = build_model(cfg, mnar_variants)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Stage T | MIXED generators -> collapsed '{args.collapse_rule}' label -> full MoE "
          f"({n_params:,} params) | train {len(train_raws)} / val {len(val_raws)} sources | {args.device}")

    out_dir = Path(f"/mnt/artifacts/project_lacuna/runs/{args.name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "generator_path": "lacuna.data.mixed_missingness.compose_mixed_missingness (improved mixed pools)",
        "mixed_heterogeneous_punching": True,
        "mnar_diverse": True, "mar_diverse": True, "compensate_rate_matched": True,
        "label_collapse_rule": args.collapse_rule, "mnar_threshold": args.mnar_threshold,
        "full_moe_trained_end_to_end": None, "encoder_frozen": False, "checkpoint_loaded": None,
        "wall_clock_seconds": None, "n_model_params": int(n_params),
        "train_generator_config": "mixed(compose_mixed_missingness, compensate_rate=True)",
        "val_generator_config": "mixed(compose_mixed_missingness, compensate_rate=True)",
        "test_generator_config": "mixed(compose_mixed_missingness, compensate_rate=True)",
        "train_sources": [r.name for r in train_raws], "val_sources": [r.name for r in val_raws],
        "config": args.config, "seed": args.seed,
    }

    tcfg = TrainerConfig(lr=cfg.training.lr, weight_decay=cfg.training.weight_decay,
                         grad_clip=cfg.training.grad_clip, epochs=cfg.training.epochs,
                         warmup_steps=cfg.training.warmup_steps, patience=cfg.training.patience,
                         min_delta=cfg.training.min_delta, checkpoint_dir=str(out_dir / "checkpoints"),
                         save_best_only=True, quiet=False)
    trainer = Trainer(model, tcfg, device=args.device)

    if args.smoke:
        print("\n=== SMOKE: confirming the MIXED generator feeds the real MoE train step ===")
        n_steps = 5
        it = iter(train_loader)
        for s in range(n_steps):
            batch = next(it)
            metrics = trainer.train_step(batch)
            loss = metrics.get("loss") or metrics.get("total_loss")
            assert batch.class_ids is not None and batch.class_ids.shape[0] == cfg.training.batch_size
            assert batch.original_values is not None and batch.reconstruction_mask is not None
            print(f"  step {s+1}: loss={loss:.4f} | class_ids(sample)={batch.class_ids[:8].tolist()} "
                  f"| tokens{tuple(batch.tokens.shape)} | finite={np.isfinite(loss)}")
        comp = np.array(train_loader.compositions)
        manifest["full_moe_trained_end_to_end"] = False
        manifest["mode"] = "smoke"
        manifest["smoke_steps"] = n_steps
        manifest["composition_mean_over_generated"] = [round(float(x), 4) for x in comp.mean(0)]
        manifest["collapsed_label_counts_MCAR_MAR_MNAR"] = train_loader.label_counts
        manifest["composition_examples"] = train_loader.compositions[:10]
        (out_dir / "manifest_smoke.json").write_text(json.dumps(manifest, indent=2))
        print(f"\nSMOKE CONFIRMED: mixed generator -> collapsed label -> full MoE forward+loss+backward OK.")
        print(f"  mean composition over {len(comp)} generated datasets [MCAR/MAR/MNAR]: "
              f"{manifest['composition_mean_over_generated']}")
        print(f"  collapsed label counts [MCAR/MAR/MNAR]: {train_loader.label_counts}")
        print(f"  manifest -> {out_dir / 'manifest_smoke.json'}")
        print("  (NO full training run — drop --smoke to train for real.)")
        return

    # full training (only when NOT --smoke)
    start = time.time()
    result = trainer.fit(train_loader, val_loader)
    wall = time.time() - start
    manifest.update({"full_moe_trained_end_to_end": True, "mode": "full", "wall_clock_seconds": round(wall, 1),
                     "best_val_acc": result["best_val_acc"], "best_val_loss": result["best_val_loss"],
                     "best_epoch": result["best_epoch"], "epochs_completed": result["final_epoch"] + 1,
                     "composition_mean_over_generated": [round(float(x), 4)
                                                         for x in np.array(train_loader.compositions).mean(0)],
                     "collapsed_label_counts_MCAR_MAR_MNAR": train_loader.label_counts})
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nDONE | wall-clock {wall:.1f}s ({wall/60:.1f}m) | best val acc {result['best_val_acc']*100:.1f}% "
          f"@ epoch {result['best_epoch']} | manifest -> {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
