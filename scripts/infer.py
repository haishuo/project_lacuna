#!/usr/bin/env python3
"""
Classify the missing data mechanism of a dataset.

Given a CSV file with missing values (NaN), this script determines whether
the missingness is MCAR, MAR, or MNAR, and recommends the appropriate
statistical analysis strategy.

Usage:
    python scripts/infer.py --input data.csv --checkpoint model.pt
    python scripts/infer.py --input data.csv --checkpoint model.pt --device cuda
    python scripts/infer.py --input data.csv --checkpoint model.pt --json output.json

Example output:
    ════════════════════════════════════════════════════════════
    LACUNA — Missing Data Mechanism Classifier
    ════════════════════════════════════════════════════════════
    Dataset: patient_records.csv (482 rows × 12 columns)
    Missing values: 1,247 / 5,784 cells (21.6%)

    Classification:
      MCAR    12.3%
      MAR     73.6%  ← most likely
      MNAR    14.1%

    Confidence: 73.6% (moderate)
    Entropy: 0.42 / 1.10 (normalized: 0.38)

    Recommended action: YELLOW — assume MAR
      → Use multiple imputation (e.g., MICE, Amelia) or
        likelihood-based methods (e.g., EM algorithm, full
        information maximum likelihood).

    Bayes risk analysis:
      Green  (assume MCAR): risk = 0.363
      Yellow (assume MAR):  risk = 0.077  ← minimum
      Red    (assume MNAR): risk = 0.344
    ════════════════════════════════════════════════════════════
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lacuna.core.types import ObservedDataset
from lacuna.models import create_lacuna_model
from lacuna.data.tokenization import tokenize_and_batch
from lacuna.training import load_model_weights


# =========================================================================
# Constants
# =========================================================================

CLASS_NAMES = ["MCAR", "MAR", "MNAR"]

ACTION_NAMES = ["GREEN", "YELLOW", "RED"]

ACTION_DESCRIPTIONS = {
    0: (
        "GREEN — assume MCAR",
        "The missingness appears random and unrelated to any variables.\n"
        "  → Complete-case analysis is valid, or use simple imputation\n"
        "    (e.g., mean/median imputation, listwise deletion)."
    ),
    1: (
        "YELLOW — assume MAR",
        "The missingness depends on observed variables but not on the\n"
        "  missing values themselves.\n"
        "  → Use multiple imputation (e.g., MICE, Amelia II) or\n"
        "    likelihood-based methods (e.g., EM algorithm, full\n"
        "    information maximum likelihood / FIML)."
    ),
    2: (
        "RED — assume MNAR",
        "The missingness likely depends on the unobserved values.\n"
        "  → Use sensitivity analysis, selection models (Heckman),\n"
        "    or pattern-mixture models. Results from standard\n"
        "    imputation may be biased."
    ),
}

# Default model hyperparameters (matching semisynthetic_full.yaml)
DEFAULT_HIDDEN_DIM = 128
DEFAULT_EVIDENCE_DIM = 64
DEFAULT_N_LAYERS = 4
DEFAULT_N_HEADS = 4
DEFAULT_MAX_COLS = 48
DEFAULT_MAX_ROWS = 128
DEFAULT_DROPOUT = 0.1


# =========================================================================
# Data loading
# =========================================================================

def load_csv_as_observed(csv_path: str, max_rows: int = None) -> ObservedDataset:
    """Load a CSV file and convert to ObservedDataset.

    Numeric columns are extracted. NaN values become missing.
    Non-numeric columns are silently dropped.
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required for CSV loading: pip install pandas")

    df = pd.read_csv(csv_path)

    # Keep only numeric columns
    df_numeric = df.select_dtypes(include=[np.number])
    dropped = set(df.columns) - set(df_numeric.columns)
    if dropped:
        print(f"  Note: dropped {len(dropped)} non-numeric columns: {sorted(dropped)}")

    if df_numeric.shape[1] == 0:
        raise ValueError("No numeric columns found in the CSV file.")

    # Subsample rows if needed
    if max_rows and df_numeric.shape[0] > max_rows:
        df_numeric = df_numeric.sample(n=max_rows, random_state=42)
        print(f"  Note: subsampled to {max_rows} rows (from {df.shape[0]})")

    # Convert to tensors
    values = df_numeric.values.astype(np.float32)
    observed = ~np.isnan(values)

    # Replace NaN with 0 for the value tensor
    values = np.nan_to_num(values, nan=0.0)

    x = torch.from_numpy(values)
    r = torch.from_numpy(observed)
    n, d = x.shape

    feature_names = tuple(df_numeric.columns.tolist())

    return ObservedDataset(
        x=x, r=r, n=n, d=d,
        feature_names=feature_names,
        dataset_id=Path(csv_path).stem,
    )


# =========================================================================
# Inference
# =========================================================================

def run_inference(model, dataset: ObservedDataset, max_rows: int, max_cols: int,
                  device: str = "cpu"):
    """Run model inference on a single dataset."""
    model.eval()
    model.to(device)

    # Tokenize
    batch = tokenize_and_batch(
        datasets=[dataset],
        max_rows=max_rows,
        max_cols=max_cols,
    )

    # Move batch to device
    batch = batch.to(device)

    # Forward pass
    with torch.no_grad():
        output = model.forward(batch, compute_reconstruction=True, compute_decision=True)

    # Extract results
    posterior = output.posterior
    decision = output.decision

    p_class = posterior.p_class[0].cpu().numpy()       # [3]
    entropy = posterior.entropy_class[0].cpu().item()
    action_id = decision.action_ids[0].cpu().item()
    risks = decision.all_risks[0].cpu().numpy()        # [3]

    return {
        "p_class": p_class,
        "predicted_class": int(np.argmax(p_class)),
        "predicted_class_name": CLASS_NAMES[int(np.argmax(p_class))],
        "confidence": float(np.max(p_class)),
        "entropy": float(entropy),
        "max_entropy": float(np.log(3)),
        "normalized_entropy": float(entropy / np.log(3)),
        "action_id": int(action_id),
        "action_name": ACTION_NAMES[int(action_id)],
        "expected_risks": risks.tolist(),
    }


# =========================================================================
# Output formatting
# =========================================================================

def format_report(result: dict, csv_path: str, dataset: ObservedDataset) -> str:
    """Format inference results as a human-readable report."""
    lines = []
    sep = "═" * 60

    lines.append(sep)
    lines.append("LACUNA — Missing Data Mechanism Classifier")
    lines.append(sep)

    # Dataset info
    n_missing = int((~dataset.r).sum().item())
    n_total = dataset.n * dataset.d
    pct_missing = n_missing / n_total * 100
    lines.append(f"Dataset: {Path(csv_path).name} ({dataset.n} rows × {dataset.d} columns)")
    lines.append(f"Missing values: {n_missing:,} / {n_total:,} cells ({pct_missing:.1f}%)")

    # Per-column missingness
    col_missing = (~dataset.r).float().mean(dim=0)
    cols_with_missing = (col_missing > 0).sum().item()
    lines.append(f"Columns with missing data: {cols_with_missing} / {dataset.d}")

    # Classification
    lines.append("")
    lines.append("Classification:")
    p_class = result["p_class"]
    predicted = result["predicted_class"]
    for i, (name, prob) in enumerate(zip(CLASS_NAMES, p_class)):
        marker = "  ← most likely" if i == predicted else ""
        bar_len = int(prob * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        lines.append(f"  {name:5s} {prob*100:5.1f}%  {bar}{marker}")

    # Confidence and entropy
    lines.append("")
    conf = result["confidence"]
    entropy = result["entropy"]
    max_ent = result["max_entropy"]
    norm_ent = result["normalized_entropy"]

    conf_label = "high" if conf > 0.80 else "moderate" if conf > 0.60 else "low"
    lines.append(f"Confidence: {conf*100:.1f}% ({conf_label})")
    lines.append(f"Entropy: {entropy:.2f} / {max_ent:.2f} (normalized: {norm_ent:.2f})")

    if conf < 0.50:
        lines.append("")
        lines.append("⚠  LOW CONFIDENCE: The model is uncertain about this classification.")
        lines.append("   Consider collecting more data or examining missingness patterns manually.")

    # Recommended action
    lines.append("")
    action_id = result["action_id"]
    action_title, action_desc = ACTION_DESCRIPTIONS[action_id]
    lines.append(f"Recommended action: {action_title}")
    lines.append(action_desc)

    # Bayes risk analysis
    lines.append("")
    lines.append("Bayes risk analysis:")
    risks = result["expected_risks"]
    min_risk = min(risks)
    for i, (name, risk) in enumerate(zip(["Green  (assume MCAR)", "Yellow (assume MAR) ", "Red    (assume MNAR)"], risks)):
        marker = "  ← minimum" if risk == min_risk else ""
        lines.append(f"  {name}: risk = {risk:.3f}{marker}")

    lines.append(sep)
    return "\n".join(lines)


# =========================================================================
# CLI
# =========================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Classify missing data mechanism of a CSV dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/infer.py --input data.csv --checkpoint best_model.pt
  python scripts/infer.py --input data.csv --checkpoint calibrated.pt --device cuda
  python scripts/infer.py --input data.csv --checkpoint model.pt --json result.json
        """,
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to CSV file with missing values (NaN)",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to trained model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device to run on (default: cpu)",
    )
    parser.add_argument(
        "--json", type=str, default=None,
        help="Save results as JSON to this path",
    )
    parser.add_argument(
        "--max-rows", type=int, default=DEFAULT_MAX_ROWS,
        help=f"Max rows to use from dataset (default: {DEFAULT_MAX_ROWS})",
    )
    parser.add_argument(
        "--max-cols", type=int, default=DEFAULT_MAX_COLS,
        help=f"Max columns (default: {DEFAULT_MAX_COLS})",
    )
    parser.add_argument(
        "--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM,
        help=f"Model hidden dimension (default: {DEFAULT_HIDDEN_DIM})",
    )
    parser.add_argument(
        "--evidence-dim", type=int, default=DEFAULT_EVIDENCE_DIM,
        help=f"Model evidence dimension (default: {DEFAULT_EVIDENCE_DIM})",
    )
    parser.add_argument(
        "--n-layers", type=int, default=DEFAULT_N_LAYERS,
        help=f"Number of transformer layers (default: {DEFAULT_N_LAYERS})",
    )
    parser.add_argument(
        "--n-heads", type=int, default=DEFAULT_N_HEADS,
        help=f"Number of attention heads (default: {DEFAULT_N_HEADS})",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Only output JSON (implies --json /dev/stdout)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    csv_path = args.input
    if not Path(csv_path).exists():
        print(f"Error: File not found: {csv_path}")
        sys.exit(1)

    if not args.quiet:
        print(f"Loading: {csv_path}")

    # Load data
    dataset = load_csv_as_observed(csv_path, max_rows=args.max_rows)

    if dataset.d > args.max_cols:
        print(f"Warning: Dataset has {dataset.d} columns, but model supports {args.max_cols}. "
              f"Only the first {args.max_cols} columns will be used.")

    n_missing = int((~dataset.r).sum().item())
    if n_missing == 0:
        print("Warning: No missing values found in the dataset. "
              "Lacuna classifies missingness mechanisms — the data must have missing values.")
        print("If your data uses a sentinel value (e.g., -999, 'NA'), convert these to NaN first.")
        sys.exit(1)

    if not args.quiet:
        print(f"Dataset: {dataset.n} rows × {dataset.d} columns, "
              f"{n_missing} missing values ({n_missing / (dataset.n * dataset.d) * 100:.1f}%)")

    # Load model
    if not args.quiet:
        print(f"Loading model: {args.checkpoint}")

    model = create_lacuna_model(
        hidden_dim=args.hidden_dim,
        evidence_dim=args.evidence_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        max_cols=args.max_cols,
        dropout=DEFAULT_DROPOUT,
    )
    load_model_weights(model, args.checkpoint, device=args.device)

    if not args.quiet:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Model: {n_params:,} parameters")

    # Run inference
    if not args.quiet:
        print("Running inference...")

    result = run_inference(model, dataset, args.max_rows, args.max_cols, args.device)

    # Output
    if args.quiet:
        json.dump(result, sys.stdout, indent=2)
        print()
    else:
        print()
        print(format_report(result, csv_path, dataset))

    # Save JSON if requested
    if args.json and not args.quiet:
        result_with_meta = {
            "input_file": csv_path,
            "n_rows": dataset.n,
            "n_cols": dataset.d,
            "n_missing": n_missing,
            "pct_missing": round(n_missing / (dataset.n * dataset.d) * 100, 2),
            "feature_names": list(dataset.feature_names) if dataset.feature_names else None,
            "checkpoint": args.checkpoint,
            **result,
            "p_class": {name: float(p) for name, p in zip(CLASS_NAMES, result["p_class"])},
            "expected_risks": {name: float(r) for name, r in zip(ACTION_NAMES, result["expected_risks"])},
        }
        with open(args.json, "w") as f:
            json.dump(result_with_meta, f, indent=2)
        print(f"\nResults saved to: {args.json}")


if __name__ == "__main__":
    main()
