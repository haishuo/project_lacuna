"""
lacuna.survey.level1_train

From-scratch training loop for the Level-1 φ-spine δ-prior (Stage-1 spec §4; MASTER §5).

ONE job: train a fresh `Level1Model` (column-primary φ → δ-bin head) on role-B examples with the RPS
objective, select on val RPS, fit the post-hoc temperature, evaluate calibration-first metrics on a
held-out (leave-datasets-out) pool, run the blocking leakage gate, and emit a validated manifest that
records the explicit `named_prior` (P_prior). No checkpoint is ever loaded; all layers stay trainable
(asserted). Orchestration only — objective/metrics/leakage/manifest live in their own modules.
Determinism via an injected RNGState (Rule 6).
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from lacuna.core.rng import RNGState

from . import metrics as M
from . import run_manifest
from . import survey_catalog as SC
from .answer_sheet import GENERATOR_FAMILY, LOD_FAMILY
from .coarse_bins import assign_bins, scheme_centers, scheme_num_bins, scheme_spec
from .column_batching import collate_columns
from .column_phi import QUANTILE_LEVELS
from .delta_bins import NUM_BINS, assign_delta_bin
from .delta_head import assert_fresh_and_trainable
from .example_source import ExampleSource
from .leakage import assess_leakage, leakage_pass
from .level1_model import create_level1_model
from .loss import fit_temperature, log_score, rps_loss, uniform_rps
from .train import _make_examples  # shared example-draw helper (duck-typed on cfg)

MODEL_PATH = "lacuna.survey.level1_model.Level1Model"
# Survey-realistic idiom names for the named-prior vocabulary (MASTER §6; assay-LOD is out of scope).
_IDIOM_NAME = {GENERATOR_FAMILY: "own_value_self_censoring", LOD_FAMILY: "top_coding"}


@dataclass
class Level1Config:
    """Knobs for a Level-1 φ-spine run (recorded into the manifest)."""

    delta_grid: List[float] = field(default_factory=lambda: [0.0, 0.15, 0.4, 0.75, 1.25, 1.75, 2.5])
    beta1_range: tuple = (0.0, 2.0)
    target_rate: float = 0.3
    max_rows: int = 384
    max_cols: int = 32  # unused by φ (column-major); kept for source compatibility
    batch_size: int = 16
    train_batches_per_epoch: int = 30
    max_epochs: int = 20
    patience: int = 5
    lr: float = 2e-3
    grad_clip: float = 1.0
    val_size: int = 120
    test_size: int = 120
    # model
    m: int = 16
    e_col: int = 32
    phi_hidden: int = 48
    head_hidden_dim: Optional[int] = None
    dropout: float = 0.1
    coarse_scheme: str = "none"  # "none"(7-bin) | "binary" | "coarse3" — evaluation/curriculum target

    def __post_init__(self):
        if not any(float(g) == 0.0 for g in self.delta_grid):
            raise ValueError("delta_grid must include 0.0 (MAR)")


def _labels(cb, scheme: str) -> torch.Tensor:
    return cb.delta_bin if scheme == "none" else assign_bins(scheme, cb.delta)


@torch.no_grad()
def _forward(model, examples, cfg: Level1Config, scheme: str):
    """Forward a fixed example list in chunks; return (logits, labels, deltas, sheets) on CPU."""
    model.eval()
    logits, labels, deltas, sheets = [], [], [], []
    for s in range(0, len(examples), cfg.batch_size):
        cb = collate_columns(examples[s:s + cfg.batch_size], max_rows=cfg.max_rows)
        logits.append(model(cb).cpu())
        labels.append(_labels(cb, scheme))
        deltas.append(cb.delta)
        sheets.extend(cb.sheets)
    return torch.cat(logits), torch.cat(labels), torch.cat(deltas), sheets


def _metrics_block(logits, labels, deltas, temperature: float, centers=None) -> Dict:
    scaled = logits / temperature
    probs = torch.softmax(scaled, dim=-1)
    return {
        "rps": float(rps_loss(scaled, labels).item()),
        "log_score": float(log_score(scaled, labels).item()),
        "e_delta": M.e_delta_error(probs, deltas, centers=centers),
        "bin_accuracy": M.bin_accuracy(probs, labels),
        "adjacent_accuracy": M.adjacent_accuracy(probs, labels),
        "p_delta0": M.p_delta_zero(probs),
        "entropy_bits": M.mean_predictive_entropy(probs),
        "ece": M.ece(probs, labels)["ece"],
        "coverage": M.coverage_table(probs, labels),
    }


def _prior_marginal(grid: List[float], scheme: str, num_bins: int) -> Dict:
    """δ-bin marginal under uniform grid sampling — the flat-footprint output target (MASTER §4)."""
    counts = [0] * num_bins
    for d in grid:
        b = assign_delta_bin(d) if scheme == "none" else int(assign_bins(scheme, torch.tensor([d]))[0])
        counts[b] += 1
    tot = float(sum(counts))
    return {str(i): counts[i] / tot for i in range(num_bins)}


def _named_prior(train_source: ExampleSource, cfg: Level1Config, scheme: str, num_bins: int,
                 phi_schema: Dict) -> Dict:
    fam = train_source.generator_family
    return {
        "dataset_catalog": train_source.describe(),
        "contaminants_excluded": sorted(SC.CONTAMINANTS),
        "idiom_vocabulary": [_IDIOM_NAME.get(fam, fam)],
        "delta_grid": list(cfg.delta_grid),
        "delta_grid_weights": [1.0 / len(cfg.delta_grid)] * len(cfg.delta_grid),
        "prior_marginal": _prior_marginal(cfg.delta_grid, scheme, num_bins),
        "rate_regime": {"target_rate": cfg.target_rate, "matched": True},
        "phi_config": phi_schema,
        "data_role": "B_complete_projection",
        "quantile_levels": list(QUANTILE_LEVELS),
    }


def train_level1(
    train_source: ExampleSource, val_source: ExampleSource, test_source: ExampleSource,
    cfg: Level1Config, rng: RNGState, *, kind: str, run_id: str, git_commit: str, timestamp: str,
    device: str = "cpu", wall_clock_seconds: float = 0.0,
) -> Dict:
    """Train + evaluate a Level-1 φ-spine δ-prior; return {model, results, leakage, manifest}."""
    scheme = cfg.coarse_scheme
    num_bins = scheme_num_bins(scheme)
    centers = scheme_centers(scheme)

    model = create_level1_model(
        m=cfg.m, e_col=cfg.e_col, phi_hidden=cfg.phi_hidden, head_hidden_dim=cfg.head_hidden_dim,
        dropout=cfg.dropout, num_bins=num_bins, rng=rng.spawn(),
    ).to(device)
    n_param = assert_fresh_and_trainable(model)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    val_ex = _make_examples(val_source, cfg, cfg.val_size, rng.spawn(), stratify=True)
    test_ex = _make_examples(test_source, cfg, cfg.test_size, rng.spawn(), stratify=True)

    best_val, best_state, since, epochs_run = math.inf, None, 0, 0
    train_rng = rng.spawn()
    for epoch in range(cfg.max_epochs):
        epochs_run = epoch + 1
        model.train()
        # NON-DETERMINISTIC: train-time Dropout draws from torch's global RNG (nn.Dropout takes no
        # generator). The injected-RNG contract covers init, eval-forward, and seeded data/leakage.
        for _ in range(cfg.train_batches_per_epoch):
            ex = _make_examples(train_source, cfg, cfg.batch_size, train_rng.spawn())
            cb = collate_columns(ex, max_rows=cfg.max_rows)
            logits = model(cb)
            loss = rps_loss(logits, _labels(cb, scheme).to(logits.device))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip); opt.step()
        v_logits, v_labels, _, _ = _forward(model, val_ex, cfg, scheme)
        val_rps = float(rps_loss(v_logits, v_labels).item())
        if val_rps < best_val - 1e-5:
            best_val, since = val_rps, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            since += 1
            if since >= cfg.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    v_logits, v_labels, _, _ = _forward(model, val_ex, cfg, scheme)
    temperature = float(fit_temperature(v_logits, v_labels))
    model.set_temperature(temperature)

    t_logits, t_labels, t_deltas, t_sheets = _forward(model, test_ex, cfg, scheme)
    _, _, _, v_sheets = _forward(model, val_ex, cfg, scheme)
    before = _metrics_block(t_logits, t_labels, t_deltas, temperature=1.0, centers=centers)
    after = _metrics_block(t_logits, t_labels, t_deltas, temperature=temperature, centers=centers)

    report = assess_leakage(v_sheets + t_sheets, rng.spawn())
    lk_pass = leakage_pass(report)

    phi_schema = model.phi.schema()
    model_arch = {
        "spine": "column_phi", "encoder": None, "m": cfg.m, "e_col": cfg.e_col,
        "phi_hidden": cfg.phi_hidden, "head_hidden_dim": cfg.head_hidden_dim, "dropout": cfg.dropout,
        "phi_schema": phi_schema, "coarse_scheme": scheme_spec(scheme),
        "head": "DeltaBinHead(e_col->hidden->num_bins)",
    }
    split_scheme = {"train": train_source.describe(), "val": val_source.describe(),
                    "test": test_source.describe(), "val_size": cfg.val_size, "test_size": cfg.test_size}
    metrics_block = {"best_val_rps": best_val, "epochs_run": epochs_run,
                     "uniform_rps": uniform_rps(num_bins),
                     "test_before_temperature": before, "test_after_temperature": after}
    calibration_block = {"temperature": temperature, "ece_before": before["ece"],
                         "ece_after": after["ece"], "coverage_before": before["coverage"],
                         "coverage_after": after["coverage"]}

    named_prior = _named_prior(train_source, cfg, scheme, num_bins, phi_schema)
    run_manifest.validate_named_prior(named_prior)
    manifest = run_manifest.build_manifest(
        run_id=run_id, git_commit=git_commit, timestamp=timestamp, kind=kind,
        delta_grid=cfg.delta_grid, beta1_range=list(cfg.beta1_range), target_rate=cfg.target_rate,
        seed=rng.seed, model_arch=model_arch, trainable_param_count=n_param, checkpoint_loaded=False,
        all_layers_trainable=True, temperature=temperature, split_scheme=split_scheme,
        metrics=metrics_block, calibration=calibration_block, leakage=report.to_dict(),
        leakage_pass=lk_pass, wall_clock_seconds=wall_clock_seconds,
        generator_family=train_source.generator_family, num_bins=num_bins,
        model_path=MODEL_PATH, named_prior=named_prior,
    )
    run_manifest.validate_manifest(manifest)
    return {"model": model, "results": metrics_block, "leakage": report,
            "leakage_pass": lk_pass, "manifest": manifest}
