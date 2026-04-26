"""Fit Lacuna-Survey calibration parameters from real-survey anchors.

Calibration model: vector scaling.
    logits' = (logits - bias) / T
    p_calibrated = softmax(logits')

Fit data: combined NLL over a small set of LABELED real surveys (the
within-domain anchors with textbook-consensus mechanism) and a large
synthetic-eval set drawn from the survey-trained generator registry.
The real-data anchors get up-weighted relative to synthetic so the
calibration is pulled toward the actual distribution we deploy on,
without losing MCAR/MNAR coverage (which only synthetic provides).

LIMITATIONS:
  - Only five real-survey anchors are available, all consensus-MAR.
    This means calibration can only adjust the MAR boundary; we have
    no real data to constrain MCAR or MNAR boundaries beyond what
    synthetic provides. As more labeled real survey data becomes
    available (whether through active learning, expert annotation,
    or PISA-style planned-missing examples for MCAR), re-fit with
    the new corpus.
  - Survey_yrbss is included as MAR but its consensus is contested
    (phone-survey nonresponse can be defensibly MCAR). Fitting it as
    MAR injects a systematic anchor toward MAR; if the anchor turns
    out to be wrong, the calibration will be biased.

Usage:
    python -m lacuna_survey.calibrate --output deployment/calibration.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lacuna.core.rng import RNGState
from lacuna.core.types import ObservedDataset
from lacuna.data.catalog import create_default_catalog
from lacuna.data.semisynthetic import apply_missingness
from lacuna.data.tokenization import tokenize_and_batch
from lacuna.generators.families.registry_builder import load_registry_from_config
from demo.pipeline import build_model, csv_to_dataset, MODEL_DEFAULTS

from lacuna_survey.anchors import ANCHORS, class_balance

CLASS_NAMES = ("MCAR", "MAR", "MNAR")

# Real-survey anchors come from the declarative registry in
# lacuna_survey/anchors.py. Each anchor is (slug, label) — the slug
# resolves to evaluation_data/<slug>_real.csv. Adding new anchors is
# a one-row edit there + dropping the CSV in evaluation_data/.
REAL_ANCHORS: List[Tuple[str, int]] = [
    (f"{a.slug}_real", a.label) for a in ANCHORS
]

# How many synthetic samples per class to draw for calibration.
SYNTH_PER_CLASS = 100

# Weight on each real anchor relative to a single synthetic sample.
# Set so the total real-data weight is comparable to the total
# synthetic weight (~1/3 of fit signal from real anchors).
REAL_WEIGHT = (3 * SYNTH_PER_CLASS / len(REAL_ANCHORS)) * 0.5


def get_logits_for_observed(model, observed: ObservedDataset) -> np.ndarray:
    """Run the model on a single ObservedDataset; return [3] logits."""
    batch = tokenize_and_batch(
        datasets=[observed],
        max_rows=MODEL_DEFAULTS["max_rows"],
        max_cols=MODEL_DEFAULTS["max_cols"],
    )
    with torch.no_grad():
        out = model.forward(batch, compute_reconstruction=True, compute_decision=False)
    p = out.posterior.p_class[0].cpu().numpy()
    # Convert posterior probabilities back to logits via log; +tiny
    # for numerical stability.
    return np.log(np.clip(p, 1e-8, 1.0))


def collect_real_logits(model) -> Tuple[np.ndarray, np.ndarray]:
    """Get [N, 3] logits and [N] labels for the real-survey anchors."""
    eval_dir = HERE / "evaluation_data"
    logits_list, labels = [], []
    for slug, label in REAL_ANCHORS:
        path = eval_dir / f"{slug}.csv"
        with path.open("rb") as f:
            b = f.read()
        loaded = csv_to_dataset(b, slug, max_rows=MODEL_DEFAULTS["max_rows"])
        logits_list.append(get_logits_for_observed(model, loaded.dataset))
        labels.append(label)
    return np.array(logits_list, dtype=np.float32), np.array(labels, dtype=np.int64)


def collect_synth_logits(model, n_per_class: int = SYNTH_PER_CLASS) -> Tuple[np.ndarray, np.ndarray]:
    """Apply each survey-registry generator to several X-bases; collect
    logits and class labels. Sample n_per_class items per class."""
    reg = load_registry_from_config("lacuna_survey")
    cat = create_default_catalog()
    x_bases = [cat.load(n) for n in
               ("survey_bfi", "survey_hmda", "survey_psid7682", "survey_yrbss",
                "survey_workinghours", "survey_chile")]

    by_class: dict = {0: [], 1: [], 2: []}
    rng_seed = 1337
    for g in reg:
        cls = g.class_id
        if len(by_class[cls]) >= n_per_class:
            continue
        for x_base in x_bases:
            if len(by_class[cls]) >= n_per_class:
                break
            try:
                semi = apply_missingness(x_base, g, RNGState(seed=rng_seed))
                rng_seed += 1
                obs = ObservedDataset(
                    x=semi.complete * semi.observed.r.float(),
                    r=semi.observed.r,
                    n=semi.observed.n,
                    d=semi.observed.d,
                    feature_names=semi.observed.feature_names,
                    dataset_id=g.name,
                )
                by_class[cls].append(get_logits_for_observed(model, obs))
            except Exception:
                pass

    logits_list, labels = [], []
    for cls in (0, 1, 2):
        for l in by_class[cls][:n_per_class]:
            logits_list.append(l)
            labels.append(cls)
    return np.array(logits_list, dtype=np.float32), np.array(labels, dtype=np.int64)


def fit_calibration(real_logits, real_labels, synth_logits, synth_labels):
    """Fit (T, bias_3) by minimising weighted NLL on real + synthetic.

    Vector scaling: logits' = (logits - bias) / T. T initialised to 1,
    bias to zero, optimised with Adam. Real anchors weighted REAL_WEIGHT
    times more than synthetic samples.
    """
    R = torch.from_numpy(real_logits)
    R_lab = torch.from_numpy(real_labels)
    S = torch.from_numpy(synth_logits)
    S_lab = torch.from_numpy(synth_labels)

    log_T = torch.zeros((), requires_grad=True)  # T = exp(log_T)
    bias = torch.zeros(3, requires_grad=True)
    opt = torch.optim.Adam([log_T, bias], lr=0.05)

    # Weights
    real_w = torch.full((len(R),), REAL_WEIGHT)
    synth_w = torch.ones(len(S))
    total_real_w = real_w.sum().item()
    total_synth_w = synth_w.sum().item()

    print(f"Calibration set: {len(R)} real (total weight {total_real_w:.1f}) + "
          f"{len(S)} synthetic (total weight {total_synth_w:.1f})")

    history = []
    for step in range(400):
        opt.zero_grad()
        T = torch.exp(log_T)
        # Calibrated logits and NLL
        R_cal = (R - bias) / T
        S_cal = (S - bias) / T
        nll_real = F.cross_entropy(R_cal, R_lab, reduction="none")
        nll_synth = F.cross_entropy(S_cal, S_lab, reduction="none")
        loss = (nll_real * real_w).sum() / total_real_w + (nll_synth * synth_w).sum() / total_synth_w
        loss.backward()
        opt.step()
        if step % 50 == 0 or step == 399:
            with torch.no_grad():
                R_acc = (R_cal.argmax(-1) == R_lab).float().mean().item()
                S_acc = (S_cal.argmax(-1) == S_lab).float().mean().item()
            history.append({"step": step, "loss": loss.item(),
                            "T": T.item(), "bias": bias.tolist(),
                            "real_acc": R_acc, "synth_acc": S_acc})
            print(f"  step {step:>3d}  loss {loss.item():.4f}  "
                  f"T {T.item():.3f}  bias {bias.tolist()}  "
                  f"real_acc {R_acc:.2f}  synth_acc {S_acc:.2f}")

    return float(torch.exp(log_T).item()), bias.detach().tolist(), history


def apply_calibration(logits: np.ndarray, T: float, bias: List[float]) -> np.ndarray:
    """Return calibrated probabilities given raw logits, T, and bias."""
    cal = (logits - np.asarray(bias)) / T
    e = np.exp(cal - cal.max())
    return e / e.sum()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default=str(PROJECT_ROOT / "demo" / "model.pt"))
    ap.add_argument("--output", default=str(HERE / "deployment" / "calibration.json"))
    args = ap.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    model, _ = build_model(args.checkpoint)

    print(f"\nAnchor registry composition: {class_balance()}")
    if class_balance()["MNAR"] == 0:
        print("  WARNING: zero real-survey MNAR anchors. Calibration will be")
        print("  structurally biased toward MAR. See lacuna_survey/anchors.py")
        print("  for candidate sources to expand the corpus.")
    if class_balance()["MCAR"] == 0:
        print("  WARNING: zero real-survey MCAR anchors. Same caveat.")

    print("\nCollecting real-survey logits...")
    R, R_lab = collect_real_logits(model)
    for (slug, lab), l in zip(REAL_ANCHORS, R):
        p = np.exp(l - l.max()); p = p / p.sum()
        print(f"  {slug:24s}  consensus={CLASS_NAMES[lab]}  raw_p={p.round(3).tolist()}")

    print("\nCollecting synthetic logits...")
    S, S_lab = collect_synth_logits(model)
    print(f"  total: {len(S)} samples ({np.bincount(S_lab).tolist()} per class)")

    print("\nFitting calibration...")
    T, bias, history = fit_calibration(R, R_lab, S, S_lab)

    print(f"\nFinal calibration: T={T:.3f}  bias={[round(b, 4) for b in bias]}")

    # Show calibrated predictions on real anchors
    print("\nCalibrated predictions on real anchors:")
    print(f"  {'dataset':24s}  {'consensus':10s}  {'raw_pred':>8s}  {'cal_pred':>8s}  {'cal_p':>22s}")
    for (slug, lab), l in zip(REAL_ANCHORS, R):
        raw_p = np.exp(l - l.max()); raw_p = raw_p / raw_p.sum()
        cal_p = apply_calibration(l, T, bias)
        raw_pred = CLASS_NAMES[int(raw_p.argmax())]
        cal_pred = CLASS_NAMES[int(cal_p.argmax())]
        match_raw = "✓" if int(raw_p.argmax()) == lab else "✗"
        match_cal = "✓" if int(cal_p.argmax()) == lab else "✗"
        print(f"  {slug:24s}  {CLASS_NAMES[lab]:10s}  "
              f"{raw_pred:>5s} {match_raw}  {cal_pred:>5s} {match_cal}  "
              f"{cal_p.round(3).tolist()}")

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": "vector_scaling",
        "temperature": T,
        "bias": bias,
        "n_real_anchors": len(R),
        "n_synth_samples": len(S),
        "real_weight": REAL_WEIGHT,
        "fit_history_summary": history[-1],
        "checkpoint": args.checkpoint,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote calibration → {out_path}")


if __name__ == "__main__":
    main()
