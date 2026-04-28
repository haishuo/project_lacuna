"""Pure-Python helpers for the Lacuna demo (no Streamlit imports).

Kept separate from `app.py` so the pipeline can be unit-tested and the
Streamlit script stays focused on UI. All functions here are deterministic
given their inputs.
"""

from __future__ import annotations

import io
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from lacuna.core.types import ObservedDataset
from lacuna.data.tokenization import tokenize_and_batch
from lacuna.models import create_lacuna_model
from lacuna.training import load_model_weights


CLASS_NAMES = ("MCAR", "MAR", "MNAR")
ACTION_LABELS = {0: "GREEN", 1: "YELLOW", 2: "RED"}

# Architecture defaults — must match configs/training/semisynthetic_full.yaml.
MODEL_DEFAULTS = dict(
    hidden_dim=128,
    evidence_dim=64,
    n_layers=4,
    n_heads=4,
    max_cols=48,
    max_rows=128,
    dropout=0.1,
)

CHECKPOINT_ENV = "LACUNA_DEMO_CHECKPOINT"


@dataclass
class CSVLoadResult:
    dataset: ObservedDataset
    dataframe: pd.DataFrame
    dropped_columns: list
    subsampled_from: Optional[int]


def resolve_checkpoint(default: Path) -> Path:
    env = os.environ.get(CHECKPOINT_ENV)
    return Path(env) if env else default


def load_metadata(checkpoint_path: Path) -> dict:
    """Read sidecar JSON next to the checkpoint, if present."""
    meta_path = checkpoint_path.with_suffix(".json")
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text())
        except Exception:
            return {}
    return {}


def build_model(checkpoint_path: str):
    """Build the architecture and load weights. CPU-only."""
    model = create_lacuna_model(
        hidden_dim=MODEL_DEFAULTS["hidden_dim"],
        evidence_dim=MODEL_DEFAULTS["evidence_dim"],
        n_layers=MODEL_DEFAULTS["n_layers"],
        n_heads=MODEL_DEFAULTS["n_heads"],
        max_cols=MODEL_DEFAULTS["max_cols"],
        dropout=MODEL_DEFAULTS["dropout"],
    )
    load_model_weights(model, checkpoint_path, device="cpu")
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    return model, n_params


def csv_to_dataset(uploaded_bytes: bytes, name: str, max_rows: int) -> CSVLoadResult:
    """Parse CSV bytes into an ObservedDataset.

    Non-numeric columns are dropped; NaN cells become missing. If the file
    has more than `max_rows` rows, it is randomly subsampled (seed=42).
    """
    df = pd.read_csv(io.BytesIO(uploaded_bytes))
    df_numeric = df.select_dtypes(include=[np.number])
    dropped = sorted(set(df.columns) - set(df_numeric.columns))

    if df_numeric.shape[1] == 0:
        raise ValueError("No numeric columns found.")

    original_n = df_numeric.shape[0]
    subsampled_from = None
    if original_n > max_rows:
        df_numeric = df_numeric.sample(n=max_rows, random_state=42).reset_index(drop=True)
        subsampled_from = original_n

    values = df_numeric.values.astype(np.float32)
    observed = ~np.isnan(values)
    values = np.nan_to_num(values, nan=0.0)

    dataset = ObservedDataset(
        x=torch.from_numpy(values),
        r=torch.from_numpy(observed),
        n=df_numeric.shape[0],
        d=df_numeric.shape[1],
        feature_names=tuple(df_numeric.columns.tolist()),
        dataset_id=Path(name).stem,
    )
    return CSVLoadResult(dataset, df_numeric, dropped, subsampled_from)


def missingness_overview(dataset: ObservedDataset) -> dict:
    is_missing = (~dataset.r).numpy()
    n_total = dataset.n * dataset.d
    n_missing = int(is_missing.sum())
    return {
        "n_missing": n_missing,
        "n_total": n_total,
        "pct_missing": n_missing / n_total * 100 if n_total else 0.0,
        "per_col_rate": is_missing.mean(axis=0),
        "feature_names": list(dataset.feature_names or []),
    }


def cross_column_corr(dataset: ObservedDataset) -> np.ndarray:
    miss = (~dataset.r).numpy().astype(np.float32)
    if miss.shape[1] < 2:
        return np.zeros((miss.shape[1], miss.shape[1]))
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(miss, rowvar=False)
    return np.nan_to_num(corr, nan=0.0)


_CALIBRATION_PATH = Path(__file__).parent.parent / "lacuna_survey" / "deployment" / "calibration.json"
_OOD_PATH = Path(__file__).parent.parent / "lacuna_survey" / "deployment" / "ood_detector.json"


def _apply_vector_scaling(p: np.ndarray, T: float, bias) -> np.ndarray:
    logits = np.log(np.clip(p, 1e-8, 1.0))
    cal = (logits - np.asarray(bias)) / T
    e = np.exp(cal - cal.max())
    return e / e.sum()


def run_model(model, dataset: ObservedDataset) -> dict:
    batch = tokenize_and_batch(
        datasets=[dataset],
        max_rows=MODEL_DEFAULTS["max_rows"],
        max_cols=MODEL_DEFAULTS["max_cols"],
    )
    with torch.no_grad():
        out = model.forward(batch, compute_reconstruction=True, compute_decision=True)

    p_class_raw = out.posterior.p_class[0].cpu().numpy()
    entropy = float(out.posterior.entropy_class[0].cpu().item())
    risks = out.decision.all_risks[0].cpu().numpy()
    action_id = int(out.decision.action_ids[0].cpu().item())

    # Per-mechanism reconstruction error (likelihood-of-data evidence).
    # Lower = the data is highly self-consistent under that mechanism's
    # reconstruction model. The MAR head fitting near-zero on data the
    # gate routes to MNAR is the most informative single signal.
    recon = {}
    if out.posterior.reconstruction_errors is not None:
        for k, v in out.posterior.reconstruction_errors.items():
            recon[k] = float(v[0].mean().item())

    # Apply post-hoc calibration if present.
    p_class_cal = p_class_raw
    calibration_applied = False
    calibration_error: Optional[str] = None
    if not _CALIBRATION_PATH.exists():
        calibration_error = f"calibration file not found at {_CALIBRATION_PATH}"
    else:
        try:
            cal_payload = json.loads(_CALIBRATION_PATH.read_text())
            p_class_cal = _apply_vector_scaling(
                p_class_raw, cal_payload["temperature"], cal_payload["bias"]
            )
            calibration_applied = True
        except Exception as e:
            calibration_error = f"{type(e).__name__}: {e}"

    # Compute OOD score if a detector is present.
    p_ood = None
    ood_error: Optional[str] = None
    if not _OOD_PATH.exists():
        ood_error = f"OOD detector not found at {_OOD_PATH}"
    else:
        try:
            from lacuna_survey.ood import features_for_observed, ood_score
            from lacuna.data.missingness_features import MissingnessFeatureConfig
            cfg = MissingnessFeatureConfig()
            f = features_for_observed(dataset, cfg)
            d = json.loads(_OOD_PATH.read_text())
            p_ood = float(ood_score(
                f, np.asarray(d["mean"]), np.asarray(d["std"]),
                np.asarray(d["weights"]), d["bias"],
            ))
        except Exception as e:
            ood_error = f"{type(e).__name__}: {e}"

    p_class = p_class_cal
    return {
        "p_class": p_class.tolist(),
        "p_class_raw": p_class_raw.tolist(),
        "calibration_applied": calibration_applied,
        "predicted_class": int(np.argmax(p_class)),
        "predicted_class_name": CLASS_NAMES[int(np.argmax(p_class))],
        "confidence": float(np.max(p_class)),
        "entropy": entropy,
        "max_entropy": float(np.log(3)),
        "normalized_entropy": float(entropy / np.log(3)),
        "expected_risks": risks.tolist(),
        "action_id": action_id,
        "action_label": ACTION_LABELS[action_id],
        "recon_errors": recon,
        "p_ood": p_ood,
        "calibration_error": calibration_error,
        "ood_error": ood_error,
    }
