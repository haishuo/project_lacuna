"""
lacuna.visualization.curves

Loss, accuracy, and calibration curve plotting utilities.
"""

import json
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.figure


def _read_jsonl(metrics_path: Path) -> List[dict]:
    """Read a JSONL file and return list of dicts.

    Args:
        metrics_path: Path to the JSONL file.

    Returns:
        List of parsed JSON objects.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is empty or contains no valid entries.
    """
    metrics_path = Path(metrics_path)
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    entries = []
    with open(metrics_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    if not entries:
        raise ValueError(f"No entries found in {metrics_path}")

    return entries


def plot_loss_curves(metrics_path: Path) -> matplotlib.figure.Figure:
    """Plot training and validation loss curves from a JSONL metrics file.

    Reads entries with type "epoch" or "validation" and plots loss values
    over epochs.

    Args:
        metrics_path: Path to the JSONL metrics file.

    Returns:
        matplotlib Figure with loss curves.

    Raises:
        FileNotFoundError: If metrics_path does not exist.
        ValueError: If file is empty or contains no loss data.
    """
    entries = _read_jsonl(metrics_path)

    train_epochs = []
    train_losses = []
    val_epochs = []
    val_losses = []

    for entry in entries:
        entry_type = entry.get("type", "")
        if entry_type == "epoch" and "loss" in entry:
            train_epochs.append(entry.get("epoch", len(train_epochs)))
            train_losses.append(entry["loss"])
        elif entry_type == "validation" and "val_loss" in entry:
            val_epochs.append(entry.get("epoch", len(val_epochs)))
            val_losses.append(entry["val_loss"])

    if not train_losses and not val_losses:
        raise ValueError(f"No loss data found in {metrics_path}")

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    if train_losses:
        ax.plot(train_epochs, train_losses, label="Train Loss", marker="o", markersize=3)
    if val_losses:
        ax.plot(val_epochs, val_losses, label="Val Loss", marker="s", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return fig


def plot_accuracy_curves(metrics_path: Path) -> matplotlib.figure.Figure:
    """Plot training and validation accuracy curves from a JSONL metrics file.

    Reads entries with type "epoch" or "validation" and plots accuracy
    values over epochs.

    Args:
        metrics_path: Path to the JSONL metrics file.

    Returns:
        matplotlib Figure with accuracy curves.

    Raises:
        FileNotFoundError: If metrics_path does not exist.
        ValueError: If file is empty or contains no accuracy data.
    """
    entries = _read_jsonl(metrics_path)

    train_epochs = []
    train_accs = []
    val_epochs = []
    val_accs = []

    for entry in entries:
        entry_type = entry.get("type", "")
        if entry_type == "epoch" and "accuracy" in entry:
            train_epochs.append(entry.get("epoch", len(train_epochs)))
            train_accs.append(entry["accuracy"])
        elif entry_type == "validation" and "val_acc" in entry:
            val_epochs.append(entry.get("epoch", len(val_epochs)))
            val_accs.append(entry["val_acc"])

    if not train_accs and not val_accs:
        raise ValueError(f"No accuracy data found in {metrics_path}")

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    if train_accs:
        ax.plot(train_epochs, train_accs, label="Train Accuracy", marker="o", markersize=3)
    if val_accs:
        ax.plot(val_epochs, val_accs, label="Val Accuracy", marker="s", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return fig


def plot_calibration_curve(ece_data: dict) -> matplotlib.figure.Figure:
    """Plot a calibration curve from ECE data.

    Args:
        ece_data: Dict from compute_ece() containing "ece" and "bins" keys.

    Returns:
        matplotlib Figure with calibration diagram.

    Raises:
        ValueError: If ece_data is missing required keys.
        ValueError: If ece_data has no bins.
    """
    if not isinstance(ece_data, dict):
        raise ValueError(f"ece_data must be a dict, got {type(ece_data).__name__}")
    if "bins" not in ece_data:
        raise ValueError("ece_data must contain 'bins' key")
    if "ece" not in ece_data:
        raise ValueError("ece_data must contain 'ece' key")

    bins = ece_data["bins"]
    if not bins:
        raise ValueError("ece_data contains no bins")

    confidences = []
    accuracies = []
    counts = []
    for b in bins:
        if b["count"] > 0:
            confidences.append(b["mean_confidence"])
            accuracies.append(b["accuracy"])
            counts.append(b["count"])

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")

    if confidences:
        ax.bar(
            confidences,
            accuracies,
            width=0.08,
            alpha=0.7,
            label="Model",
            edgecolor="black",
            linewidth=0.5,
        )

    ax.set_xlabel("Mean Predicted Confidence")
    ax.set_ylabel("Fraction of Positives (Accuracy)")
    ax.set_title(f"Calibration Curve (ECE={ece_data['ece']:.4f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return fig
