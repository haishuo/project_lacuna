"""
lacuna.survey.train

Minimal from-scratch training loop for the P2.2 δ-prior (PROPOSAL-P2 §11; audit §3, §8, §13).

ONE job: train a fresh `DeltaPriorModel` on P2.1 generator output with the RPS objective, select
on val RPS (early stop, in-memory best-val restore — NOT a loaded checkpoint), fit the post-hoc
temperature on val, evaluate calibration-first metrics on a held-out (leave-datasets-out) test
pool before/after temperature, run the blocking leakage diagnostic, and emit a validated run
manifest. No checkpoint is ever loaded; all layers stay trainable (asserted). Determinism via an
injected RNGState (Rule 6).

This module is orchestration only — the objective (`loss`), metrics (`metrics`), batching
(`batching`), leakage gate (`leakage`), and manifest (`run_manifest`) live in their own modules.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from lacuna.core.rng import RNGState

from . import metrics as M
from .batching import DeltaExample, collate
from .conditioned_head import CONDITIONING_METHOD, create_target_conditioned_model
from .delta_head import assert_fresh_and_trainable, create_delta_prior_model
from .example_source import ExampleSource
from .leakage import assess_leakage, leakage_pass
from .loss import fit_temperature, log_score, rps_loss, uniform_rps
from . import run_manifest


@dataclass
class TrainConfig:
    """All knobs for a δ-prior run (recorded into the manifest's model_arch/grid)."""

    delta_grid: List[float] = field(default_factory=lambda: [0.0, 0.15, 0.4, 0.75, 1.25, 1.75, 2.5])
    beta1_range: tuple = (0.0, 2.0)
    target_rate: float = 0.3
    max_rows: int = 256
    max_cols: int = 32
    batch_size: int = 16
    train_batches_per_epoch: int = 20
    max_epochs: int = 15
    patience: int = 4
    lr: float = 3e-4
    grad_clip: float = 1.0
    val_size: int = 96
    test_size: int = 96
    # model
    hidden_dim: int = 128
    evidence_dim: int = 64
    n_layers: int = 4
    n_heads: int = 4
    dropout: float = 0.1
    head_hidden_dim: Optional[int] = None
    target_conditioned: bool = False  # rung 3: head conditions on the supplied target column

    def __post_init__(self):
        if not any(float(g) == 0.0 for g in self.delta_grid):
            raise ValueError("delta_grid must include 0.0 (MAR) so P(δ=0) is learnable")
        if self.beta1_range[0] < 0.0 or self.beta1_range[1] < self.beta1_range[0]:
            raise ValueError(f"beta1_range must satisfy 0 <= lo <= hi, got {self.beta1_range}")


def _make_examples(source: ExampleSource, cfg, n, rng, *, stratify=False) -> List[DeltaExample]:
    """Draw n examples from a source. If stratify, cycle the δ-grid deterministically for balance."""
    out = []
    for i in range(n):
        if stratify:
            delta = float(cfg.delta_grid[i % len(cfg.delta_grid)])
        else:
            delta = float(cfg.delta_grid[rng.randint(0, len(cfg.delta_grid), (1,)).item()])
        lo, hi = cfg.beta1_range
        beta1 = float(lo + (hi - lo) * rng.rand(1).item())
        out.append(source.make_one(cfg, rng.spawn(), delta=delta, beta1=beta1))
    return out


@torch.no_grad()
def _forward_examples(model, examples: List[DeltaExample], cfg: TrainConfig):
    """Forward a fixed example list in chunks; return (logits, labels, deltas, sheets) on CPU."""
    model.eval()
    logits, labels, deltas, sheets = [], [], [], []
    for s in range(0, len(examples), cfg.batch_size):
        chunk = examples[s:s + cfg.batch_size]
        db = collate(chunk, max_rows=cfg.max_rows, max_cols=cfg.max_cols)
        logits.append(model(db.tokens, db.target_idx).cpu())
        labels.append(db.delta_bin)
        deltas.append(db.delta)
        sheets.extend(db.sheets)
    return torch.cat(logits), torch.cat(labels), torch.cat(deltas), sheets


def _metrics_block(logits, labels, deltas, temperature: float) -> Dict:
    """Calibration-first metric block at a given temperature."""
    scaled = logits / temperature
    probs = torch.softmax(scaled, dim=-1)
    return {
        "rps": float(rps_loss(scaled, labels).item()),
        "log_score": float(log_score(scaled, labels).item()),
        "e_delta": M.e_delta_error(probs, deltas),
        "bin_accuracy": M.bin_accuracy(probs, labels),
        "adjacent_accuracy": M.adjacent_accuracy(probs, labels),
        "p_delta0": M.p_delta_zero(probs),
        "ece": M.ece(probs, labels)["ece"],
        "coverage": M.coverage_table(probs, labels),
    }


def train_delta_prior(
    train_source: ExampleSource,
    val_source: ExampleSource,
    test_source: ExampleSource,
    cfg: TrainConfig,
    rng: RNGState,
    *,
    kind: str,
    run_id: str,
    git_commit: str,
    timestamp: str,
    device: str = "cpu",
    wall_clock_seconds: float = 0.0,
) -> Dict:
    """Train + evaluate a δ-prior from injected example sources; return {model, results, manifest}.

    The data source is pluggable (real survey X or synthetic 2-col — see `example_source`); the
    head / loss / metrics / leakage / manifest harness is identical across sources, which is what
    makes a ladder rung-to-rung difference attributable. Manifest is validated before return.
    """
    num_bins = train_source.num_bins
    if not (val_source.num_bins == test_source.num_bins == num_bins):
        raise ValueError("train/val/test sources must agree on num_bins")

    builder = create_target_conditioned_model if cfg.target_conditioned else create_delta_prior_model
    model = builder(
        hidden_dim=cfg.hidden_dim, evidence_dim=cfg.evidence_dim, n_layers=cfg.n_layers,
        n_heads=cfg.n_heads, max_cols=cfg.max_cols, dropout=cfg.dropout,
        head_hidden_dim=cfg.head_hidden_dim, num_bins=num_bins, rng=rng.spawn(),
    ).to(device)
    n_param = assert_fresh_and_trainable(model)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # Fixed, seeded val/test sets (val also fits temperature; never regenerated).
    val_ex = _make_examples(val_source, cfg, cfg.val_size, rng.spawn(), stratify=True)
    test_ex = _make_examples(test_source, cfg, cfg.test_size, rng.spawn(), stratify=True)

    best_val, best_state, since, epochs_run = math.inf, None, 0, 0
    train_rng = rng.spawn()
    for epoch in range(cfg.max_epochs):
        epochs_run = epoch + 1
        model.train()
        # NON-DETERMINISTIC: two torch-level effects make the TRAINED weights not bit-reproducible
        # even with a fixed injected RNGState: (1) train-time Dropout draws from torch's global RNG
        # (nn.Dropout takes no generator), and (2) nn.Embedding / scatter-add BACKWARD uses a
        # non-deterministic CPU reduction unless torch.use_deterministic_algorithms(True) is set
        # globally by the caller. The injected-RNG contract therefore covers model construction/init,
        # eval-mode forward, and the seeded DATA/leakage generation — all deterministic. For a
        # fully bit-reproducible training run the caller sets dropout=0 AND the global determinism
        # flag; the library does not toggle global torch state itself (Rule 6 / Rule 5).
        for _ in range(cfg.train_batches_per_epoch):
            ex = _make_examples(train_source, cfg, cfg.batch_size, train_rng.spawn())
            db = collate(ex, max_rows=cfg.max_rows, max_cols=cfg.max_cols)
            logits = model(db.tokens, db.target_idx)
            loss = rps_loss(logits, db.delta_bin.to(logits.device))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
        v_logits, v_labels, _, _ = _forward_examples(model, val_ex, cfg)
        val_rps = float(rps_loss(v_logits, v_labels).item())
        if val_rps < best_val - 1e-5:
            best_val, since = val_rps, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            since += 1
            if since >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)  # in-memory best-val (NOT a loaded checkpoint)

    # Calibration: fit temperature on val, then evaluate test before/after.
    v_logits, v_labels, _, _ = _forward_examples(model, val_ex, cfg)
    temperature = float(fit_temperature(v_logits, v_labels))
    model.set_temperature(temperature)

    t_logits, t_labels, t_deltas, t_sheets = _forward_examples(model, test_ex, cfg)
    _, _, _, v_sheets = _forward_examples(model, val_ex, cfg)
    before = _metrics_block(t_logits, t_labels, t_deltas, temperature=1.0)
    after = _metrics_block(t_logits, t_labels, t_deltas, temperature=temperature)

    # Leakage gate over the generated corpus (val + test sheets).
    report = assess_leakage(v_sheets + t_sheets, rng.spawn())
    lk_pass = leakage_pass(report)

    model_arch = {
        "hidden_dim": cfg.hidden_dim, "evidence_dim": cfg.evidence_dim,
        "n_layers": cfg.n_layers, "n_heads": cfg.n_heads, "max_cols": cfg.max_cols,
        "dropout": cfg.dropout, "head_hidden_dim": cfg.head_hidden_dim,
        "target_conditioned": cfg.target_conditioned,
        "conditioning": CONDITIONING_METHOD if cfg.target_conditioned else "global_evidence_only",
        "head": "MLP([evidence;pooled_target]->hidden->num_bins)" if cfg.target_conditioned
        else "MLP(evidence->hidden->num_bins)",
    }
    split_scheme = {
        "train": train_source.describe(),
        "val": val_source.describe(),
        "test": test_source.describe(),
        "val_size": cfg.val_size, "test_size": cfg.test_size,
    }
    metrics_block = {
        "best_val_rps": best_val, "epochs_run": epochs_run,
        "uniform_rps": uniform_rps(model.num_bins),
        "test_before_temperature": before, "test_after_temperature": after,
    }
    calibration_block = {
        "temperature": temperature,
        "ece_before": before["ece"], "ece_after": after["ece"],
        "coverage_before": before["coverage"], "coverage_after": after["coverage"],
    }

    manifest = run_manifest.build_manifest(
        run_id=run_id, git_commit=git_commit, timestamp=timestamp, kind=kind,
        delta_grid=cfg.delta_grid, beta1_range=list(cfg.beta1_range), target_rate=cfg.target_rate,
        seed=rng.seed, model_arch=model_arch, trainable_param_count=n_param,
        checkpoint_loaded=False, all_layers_trainable=True, temperature=temperature,
        split_scheme=split_scheme, metrics=metrics_block, calibration=calibration_block,
        leakage=report.to_dict(), leakage_pass=lk_pass, wall_clock_seconds=wall_clock_seconds,
    )
    run_manifest.validate_manifest(manifest)

    return {"model": model, "results": metrics_block, "leakage": report,
            "leakage_pass": lk_pass, "manifest": manifest}
