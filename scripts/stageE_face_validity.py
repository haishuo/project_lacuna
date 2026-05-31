#!/usr/bin/env python3
"""
Stage E (ADR-0007): FACE VALIDITY of the calibrated composition posterior on real survey anchors.

Runs the full calibrated instrument — encoder + Stage-C frozen+footprint composition heads (deep
ensemble) + the Stage-D temperature — on the 14 real survey anchors (`lacuna_survey/anchors.py`),
loaded mask-preserving. Real data has NO mechanism ground truth (the manifold caveat, in neon below),
so this is FACE VALIDITY ONLY: does the posterior's read agree, in DIRECTION, with the textbook
consensus for each anchor?

Checks, in order of how defensible they are:
  1. Random-vs-structured (the IDENTIFIABLE axis): real survey missingness should read as STRUCTURED
     (low f_MCAR) for the item-/module-nonresponse anchors. The strong check.
  2. MNAR-lean (the NON-identifiable axis): do the NHANES sensitive-item anchors (income/weight/drug/
     PHQ-9) lean MNAR more than the CRAN item-nonresponse anchors? Reported as a SOFT, wide-band
     directional signal — Molenberghs bounds this; a wide MAR/MNAR band is the correct report.
  3. The can't-tell mass: substantial on every anchor (honest uncertainty on the manifold).

NEON CAVEAT. The anchor labels are TEXTBOOK CONSENSUS, not ground truth — mechanism accuracy on real
data is impossible by construction (you cannot observe why a value is missing). The model is trained on
the survey *manifold* (the generator registry's operationalization); face validity is NECESSARY, not
SUFFICIENT. In particular the PISA anchors are MCAR-BY-DESIGN (random rotated booklets) yet have heavy
BLOCK co-missingness, so our "MCAR = value-independent AND unstructured" head is expected to read them
as structured — a documented blind spot of the random-vs-structured framing, not a generic failure.

Deterministic via explicit seeds. Usage:
    python scripts/stageE_face_validity.py
"""

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lacuna.core.rng import RNGState
from lacuna.core.types import ObservedDataset
from lacuna.data.missingness_footprint import missingness_footprint, FOOTPRINT_FEATURES
from lacuna.data.tokenization import tokenize_and_batch
from lacuna.config import load_config
from lacuna.models.composition_head import (
    CompositionHead, ensemble_alpha, composition_mean, cant_tell_mass,
)
from lacuna.training.composition_calibration import apply_temperature, region_prob_ge
from lacuna.data.composition_batch import N_FOOTPRINT_FEATURES
from lacuna_survey.anchors import ANCHORS
from scripts.stageC_composition_head import init_encoder, forward_alpha

BASELINE = "/mnt/artifacts/project_lacuna/runs/stage0_general_baseline"
ANCHOR_DIR = PROJECT_ROOT / "lacuna_survey" / "evaluation_data"
CLASS_NAMES = ("MCAR", "MAR", "MNAR")


def _load_anchor(slug: str):
    """Mask-preserving numeric load of an anchor CSV -> (values[n,d] with NaN, feature_names)."""
    df = pd.read_csv(ANCHOR_DIR / f"{slug}_real.csv")
    num = df.select_dtypes(include=[np.number])
    drop = [c for c in num.columns if c.upper() in ("SEQN", "ID", "RESPONDENT_ID")]
    num = num.drop(columns=drop)
    return num.values.astype(np.float32), tuple(num.columns.tolist())


def _read_anchor(encoder, heads, values, fnames, slug, *, rng, max_rows, max_cols, device, tau,
                 n_draws):
    """Calibrated composition read for one anchor, averaged over row-subsample draws.

    Returns (mean_composition[3], cant_tell, n_used). For n>max_rows we average the posterior over
    `n_draws` random row-subsets (report stochastic quantities as distributions, not single draws)."""
    n = values.shape[0]
    draws = n_draws if n > max_rows else 1
    means, vacs = [], []
    for _ in range(draws):
        if n > max_rows:
            idx = np.sort(rng.choice(n, size=max_rows, replace=False))
            v = values[idx]
        else:
            v = values
        mask = ~np.isnan(v)
        x = np.nan_to_num(v, nan=0.0)
        obs = ObservedDataset(x=torch.from_numpy(x), r=torch.from_numpy(mask),
                              n=x.shape[0], d=x.shape[1], feature_names=fnames, dataset_id=slug)
        fp = missingness_footprint(obs.x, obs.r)
        extra = torch.tensor([[fp[k] for k in FOOTPRINT_FEATURES]], dtype=torch.float32).to(device)
        b = tokenize_and_batch([obs], max_rows=max_rows, max_cols=max_cols).to(device)
        with torch.no_grad():
            per_model = torch.stack([forward_alpha(encoder, h, b, extra).cpu() for h in heads], 0)
        alpha = apply_temperature(ensemble_alpha(per_model).numpy(), tau)   # [1,3]
        means.append(composition_mean(torch.tensor(alpha)).numpy()[0])
        vacs.append(float(cant_tell_mass(torch.tensor(alpha))[0]))
    return np.mean(means, axis=0), float(np.mean(vacs)), min(n, max_rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline-checkpoint", default=f"{BASELINE}/checkpoints/best_model.pt")
    ap.add_argument("--baseline-config", default=f"{BASELINE}/config.yaml")
    ap.add_argument("--heads-checkpoint",
                    default=f"{BASELINE}/checkpoints/stageC_composition_frozen-probe+footprint.pt")
    ap.add_argument("--calibration-report", default=f"{BASELINE}/stageD_calibration.json",
                    help="read the fitted Stage-D temperature from here")
    ap.add_argument("--tau", type=float, default=None, help="override temperature (else read from report)")
    ap.add_argument("--n-draws", type=int, default=8)
    ap.add_argument("--head-hidden", type=int, default=64)
    ap.add_argument("--seed", type=int, default=20260530)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output", type=Path, default=Path(f"{BASELINE}/stageE_face_validity.json"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    cfg = load_config(args.baseline_config)
    dims = dict(hidden_dim=cfg.model.hidden_dim, evidence_dim=cfg.model.evidence_dim,
                n_layers=cfg.model.n_layers, n_heads=cfg.model.n_heads,
                max_cols=cfg.data.max_cols, dropout=cfg.model.dropout)
    max_rows, max_cols = cfg.data.max_rows, cfg.data.max_cols
    tau = args.tau
    if tau is None:
        tau = float(json.loads(Path(args.calibration_report).read_text())["tau"])
    print(f"Calibrated instrument: encoder + frozen+footprint ensemble + temperature tau={tau:.3f}")

    encoder = init_encoder(args.baseline_checkpoint, dims, args.device)
    state = torch.load(args.heads_checkpoint, map_location="cpu", weights_only=False)["head_states"]
    heads = []
    for hs in state:
        h = CompositionHead(cfg.model.evidence_dim, hidden_dim=args.head_hidden,
                            dropout=cfg.model.dropout, n_extra_features=N_FOOTPRINT_FEATURES)
        h.load_state_dict(hs)
        heads.append(h.to(args.device).eval())

    rng = RNGState(seed=args.seed)
    rows = []
    for a in ANCHORS:
        values, fnames = _load_anchor(a.slug)
        comp, vac, n_used = _read_anchor(encoder, heads, values, fnames, a.slug, rng=rng.spawn(),
                                         max_rows=max_rows, max_cols=max_cols, device=args.device,
                                         tau=tau, n_draws=args.n_draws)
        struct = comp[1] + comp[2]
        rows.append({
            "slug": a.slug, "consensus": a.label_name, "n": int(values.shape[0]), "d": int(values.shape[1]),
            "composition": [round(float(x), 3) for x in comp],     # MCAR / MAR / MNAR
            "f_structured": round(float(struct), 3),
            "mnar_within_structured": round(float(comp[2] / struct), 3) if struct > 1e-6 else None,
            "argmax": CLASS_NAMES[int(np.argmax(comp))],
            "cant_tell": round(vac, 3),
        })

    print("\n" + "=" * 96)
    print(f"STAGE E — face validity on {len(rows)} real survey anchors (composition = MCAR/MAR/MNAR)")
    print("=" * 96)
    print(f"  {'anchor':28s} {'consensus':9s} {'n':>5s} {'composition (M/MAR/MNAR)':27s} {'struct':>6s} "
          f"{'MNAR|str':>8s} {'argmax':>6s} {'cant':>5s}")
    for r in sorted(rows, key=lambda x: (x["consensus"], x["slug"])):
        print(f"  {r['slug']:28s} {r['consensus']:9s} {r['n']:5d} {str(r['composition']):27s} "
              f"{r['f_structured']:6.3f} {str(r['mnar_within_structured']):>8s} {r['argmax']:>6s} {r['cant_tell']:5.3f}")

    # --- aggregate face-validity checks ---
    def grp(name):
        return [r for r in rows if r["consensus"] == name]
    agg = {}
    for name in CLASS_NAMES:
        g = grp(name)
        if g:
            agg[name] = {
                "n_anchors": len(g),
                "mean_f_mcar": round(float(np.mean([r["composition"][0] for r in g])), 3),
                "mean_f_structured": round(float(np.mean([r["f_structured"] for r in g])), 3),
                "mean_mnar_within_structured": round(float(np.mean(
                    [r["mnar_within_structured"] for r in g if r["mnar_within_structured"] is not None])), 3),
                "mean_cant_tell": round(float(np.mean([r["cant_tell"] for r in g])), 3),
            }
    print("\n  --- aggregate by consensus class ---")
    for name in CLASS_NAMES:
        if name in agg:
            x = agg[name]
            print(f"  {name:5s} (n={x['n_anchors']}): mean f_MCAR {x['mean_f_mcar']} | f_structured "
                  f"{x['mean_f_structured']} | MNAR|structured {x['mean_mnar_within_structured']} | "
                  f"cant-tell {x['mean_cant_tell']}")
    print("\n  CHECK 1 (identifiable): MAR+MNAR anchors read structured (low f_MCAR)?")
    print("  CHECK 2 (non-identifiable, soft): MNAR anchors lean MNAR-within-structured more than MAR anchors?")
    print("  CAVEAT: PISA MCAR anchors are MCAR-by-DESIGN but block-structured -> expected to read structured.")
    print("=" * 96)

    args.output.write_text(json.dumps({"tau": tau, "anchors": rows, "aggregate": agg}, indent=2))
    print(f"\nWrote face-validity report -> {args.output}")


if __name__ == "__main__":
    main()
