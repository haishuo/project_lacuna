"""Probabilistic diagnostic for Lacuna-Survey.

Reframes the standard diagnostic from "did argmax match consensus?"
to "is the posterior calibrated and interpretable?". Lacuna is a
posterior estimator, not a classifier — its job is to assign
calibrated probabilities to mechanisms, not categorical answers.
This view exposes the actual signal underneath the argmax mask.

For each anchor we report:
  - p_class: posterior class probabilities [P(MCAR), P(MAR), P(MNAR)]
  - reconstruction errors per mechanism head
    (low error = data is highly self-consistent under that mechanism)
  - entropy of the class posterior (uncertainty measure)
  - the literature consensus reading (for context only)

The interpretive frame:
  - When P(MNAR) is non-trivial (≥ 0.20) but argmax says MAR,
    the model is reporting "MAR is more likely but MNAR is
    plausible" — this is the calibrated answer for a Molenberghs-
    underdetermined case, not a failure.
  - When the recon error of the argmax mechanism is very low
    relative to the others, the model has strong evidence; when
    they're close, the model has weak evidence and the posterior
    should reflect that.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lacuna.core.types import ObservedDataset
from lacuna.data.tokenization import tokenize_and_batch
from demo.pipeline import build_model, MODEL_DEFAULTS

DEFAULT_CHECKPOINT = PROJECT_ROOT / "demo" / "model.pt"
EVAL_DIR = PROJECT_ROOT / "lacuna_survey" / "evaluation_data"

# Hardcoded literature consensus for reference (NOT used to compute
# accuracy — this is interpretive context only).
CONSENSUS = {
    "survey_bfi_real": "MAR",
    "survey_chile_real": "MAR",
    "survey_cars93_real": "MAR",
    "survey_survey_real": "MAR",
    "survey_yrbss_real": "MAR",
    "survey_gssvocab_real": "MAR",
    "survey_ucla_textbooks_real": "MAR",
    "survey_nhanes_demographics_real": "MNAR (Allison) / MAR-cond (van Buuren)",
    "nhanes_dpq_phq9_real": "MNAR (Allison) / MAR-cond (van Buuren)",
    "pisa2018_gbr_rotation_real": "MCAR",
    "pisa2022_deu_rotation_real": "MCAR",
}


def df_to_observed(df: pd.DataFrame, dataset_id: str = "x") -> ObservedDataset:
    df = df.select_dtypes(include=[np.number])
    if len(df) > MODEL_DEFAULTS["max_rows"]:
        df = df.sample(n=MODEL_DEFAULTS["max_rows"], random_state=42).reset_index(drop=True)
    v = df.values.astype(np.float32)
    r = ~np.isnan(v)
    v = np.nan_to_num(v, nan=0.0)
    return ObservedDataset(
        x=torch.from_numpy(v), r=torch.from_numpy(r),
        n=df.shape[0], d=df.shape[1],
        feature_names=tuple(df.columns.tolist()),
        dataset_id=dataset_id,
    )


def run_one(model, csv_path: Path) -> dict:
    df = pd.read_csv(csv_path)
    obs = df_to_observed(df, dataset_id=csv_path.stem)
    batch = tokenize_and_batch(
        datasets=[obs],
        max_rows=MODEL_DEFAULTS["max_rows"],
        max_cols=MODEL_DEFAULTS["max_cols"],
    )
    with torch.no_grad():
        out = model.forward(batch, compute_reconstruction=True, compute_decision=False)
    p_class = out.posterior.p_class[0].cpu().numpy()
    recon = {k: float(v[0].mean().item()) for k, v in out.posterior.reconstruction_errors.items()}
    entropy = float(out.posterior.entropy_class[0].item())
    return {"p_class": p_class, "recon": recon, "entropy": entropy}


def fmt_bar(p: float, width: int = 16) -> str:
    n = int(round(p * width))
    return "█" * n + "·" * (width - n)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    args = ap.parse_args()

    model, _ = build_model(str(args.checkpoint))
    anchors = sorted(EVAL_DIR.glob("*_real.csv"))

    print("Lacuna-Survey probabilistic diagnostic — posteriors over mechanisms")
    print("Recon error: mean per-head reconstruction error (lower = data self-consistent under that mechanism)")
    print()
    print(f"{'anchor':<34s} {'P(MCAR)':>9s} {'P(MAR)':>9s} {'P(MNAR)':>9s}  H  "
          f"{'recon[MCAR]':>13s} {'recon[MAR]':>13s} {'recon[MNAR]':>13s}  consensus")
    print("-" * 150)
    for path in anchors:
        slug = path.stem
        try:
            r = run_one(model, path)
        except Exception as e:
            print(f"{slug:<34s} ERR: {e}")
            continue
        p = r["p_class"]
        rec = r["recon"]
        cons = CONSENSUS.get(slug, "?")
        # MNAR probability bar visualises uncertainty
        bars = (
            f"{p[0]:.3f} {fmt_bar(p[0], 6)}",
            f"{p[1]:.3f} {fmt_bar(p[1], 6)}",
            f"{p[2]:.3f} {fmt_bar(p[2], 6)}",
        )
        print(f"{slug:<34s} {bars[0]:>15s} {bars[1]:>15s} {bars[2]:>15s}  "
              f"{r['entropy']:.2f}  "
              f"{rec.get('mcar', 0):>13.4f} {rec.get('mar', 0):>13.4f} "
              f"{rec.get('self_censoring', 0):>13.4f}  {cons}")

    print()
    print("Interpretation:")
    print("  - P(MNAR) ≥ 0.30 with low recon[MAR]: data prefers MAR but MNAR is plausible")
    print("    (Molenberghs unidentifiability — calibrated answer, not failure)")
    print("  - High entropy H ≥ 1.0: model is genuinely uncertain across classes")
    print("  - recon errors near 0 for one head + much higher for others: strong evidence")


if __name__ == "__main__":
    main()
