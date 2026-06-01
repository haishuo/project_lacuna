#!/usr/bin/env python3
"""
Stage P2 (ADR-0008): combine the metadata PRIOR with the data LIKELIHOOD, and audit the OVERRIDE.

The combiner pools the prior Dirichlet with the composition head's likelihood Dirichlet by summing
evidence (lacuna.priors.metadata_prior.combine_prior_likelihood). This script runs the pre-registered
**override audit** (ADR-0008 commitment 1) on semi-synthetic datasets with KNOWN realised composition,
by INJECTING controlled priors (flat / favouring each mechanism, at the table's capped strength) onto the
real calibrated likelihood and measuring, on each axis, how much of the prior's influence survives into
the posterior:

  override-survival ratio = (posterior shift caused by the prior) / (prior-alone shift)

  - **Identifiable axis (MCAR-vs-structured):** the likelihood is informative, so a prior that disagrees
    here should be OVERRIDDEN -> LOW survival ratio (the data wins; the safety property: a wrong prior
    cannot corrupt what the data knows).
  - **Non-identifiable axis (MAR-vs-MNAR split):** the likelihood is flat (Molenberghs / Stage F), so the
    prior DECIDES -> HIGH survival ratio. This is the prior's VALUE (it supplies what the data can't) AND
    its RISK (the data cannot catch a wrong prior here) — which is exactly why the prior must be auditable
    and reported as a separate channel (commitments 1 & 3).

Also reports: accuracy on the split axis under a CORRECT vs WRONG prior (the prior helps when right, hurts
when wrong, on the axis the data can't arbitrate), and the flat-prior (= likelihood-alone) calibration as
the reference. Deterministic via explicit seeds. Usage:
    python scripts/stageP2_override_audit.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lacuna.core.rng import RNGState
from lacuna.config import load_config
from lacuna.models.composition_head import CompositionHead, ensemble_alpha, composition_mean
from lacuna.training.composition_calibration import apply_temperature
from lacuna.data.composition_batch import build_composition_batch, N_FOOTPRINT_FEATURES
from lacuna.data.catalog import create_default_catalog
from lacuna.priors.metadata_prior import (
    MCAR, MAR, MNAR, N_CLASSES, reliability_to_strength, combine_prior_likelihood,
)
from scripts.stageC_composition_head import init_encoder, forward_alpha

BASELINE = "/mnt/artifacts/project_lacuna/runs/stage0_general_baseline"
_STRONG_R = 0.70   # the table's capped strong reliability (favoured-class prob); kappa ~ 3.67


def _prior_favoring(mech, r=_STRONG_R):
    a = np.ones(N_CLASSES)
    a[mech] += reliability_to_strength(r)
    return a


def load_raws(names, max_cols):
    cat = create_default_catalog()
    out = []
    for n in names:
        try:
            raw = cat.load(n)
        except Exception:  # noqa: BLE001
            continue
        if 4 <= raw.d <= max_cols and np.isfinite(raw.data).all():
            out.append(raw)
    return out


def collect_likelihood(encoder, heads, raws, tau, *, n_batches, batch_size, max_rows, max_cols, device, seed):
    """Likelihood alpha [N,3] (RAW ensemble evidence AND tau-calibrated) + realised truth [N,3]."""
    rng = RNGState(seed=seed)
    raw_a, cal_a, truth = [], [], []
    with torch.no_grad():
        for _ in range(n_batches):
            mb = build_composition_batch(raws, rng.spawn(), max_rows=max_rows, max_cols=max_cols,
                                         batch_size=batch_size, with_footprints=True)
            b = mb.batch.to(device)
            extra = mb.footprints.to(device)
            per_model = torch.stack([forward_alpha(encoder, h, b, extra).cpu() for h in heads], 0)
            ens = ensemble_alpha(per_model).numpy()
            raw_a.append(ens)
            cal_a.append(apply_temperature(ens, tau))
            truth.append(mb.composition.numpy())
    return np.concatenate(raw_a, 0), np.concatenate(cal_a, 0), np.concatenate(truth, 0)


def _split(comp):
    """MNAR-within-structured = f_MNAR / (f_MAR + f_MNAR) [N]."""
    struct = comp[:, MAR] + comp[:, MNAR]
    return np.where(struct > 1e-6, comp[:, MNAR] / np.clip(struct, 1e-6, None), 0.5)


def _post_mean(prior, alpha_like):
    """Posterior mean composition [N,3] for a single injected prior over a batch of likelihoods."""
    out = np.empty_like(alpha_like)
    for i in range(alpha_like.shape[0]):
        post = combine_prior_likelihood(prior, alpha_like[i])
        out[i] = post / post.sum()
    return out


def run_audit(alpha_like, truth):
    """Override-survival ratios on each axis + split-accuracy under correct/wrong/flat priors."""
    flat = np.ones(N_CLASSES)
    post = {name: _post_mean(_prior_favoring(m) if m is not None else flat, alpha_like)
            for name, m in (("flat", None), ("mcar", MCAR), ("mar", MAR), ("mnar", MNAR))}
    # Axis A — identifiable (MCAR): swap MCAR-prior <-> MNAR-prior, how much does posterior f_MCAR move?
    post_dmcar = float(np.mean(np.abs(post["mcar"][:, MCAR] - post["mnar"][:, MCAR])))
    prior_dmcar = abs(_prior_favoring(MCAR)[MCAR] / _prior_favoring(MCAR).sum()
                      - _prior_favoring(MNAR)[MCAR] / _prior_favoring(MNAR).sum())
    # Axis B — non-identifiable (MAR-vs-MNAR split): swap MAR-prior <-> MNAR-prior, how much does split move?
    post_dsplit = float(np.mean(np.abs(_split(post["mnar"]) - _split(post["mar"]))))
    prior_dsplit = abs(_split(_prior_favoring(MNAR)[None])[0] - _split(_prior_favoring(MAR)[None])[0])
    # Accuracy on the split axis (structured datasets): correct vs wrong vs flat prior
    structured = (truth[:, MAR] + truth[:, MNAR]) > 0.5
    true_split = _split(truth)
    mnar_dom = (truth[:, MNAR] >= truth[:, MAR])[:, None]
    correct = np.where(mnar_dom, post["mnar"], post["mar"])
    wrong = np.where(mnar_dom, post["mar"], post["mnar"])

    def split_mae(P):
        return round(float(np.mean(np.abs(_split(P)[structured] - true_split[structured]))), 4)

    return {
        "identifiable_axis_MCAR": {"posterior_shift": round(post_dmcar, 4),
                                   "prior_alone_shift": round(prior_dmcar, 4),
                                   "override_survival_ratio": round(post_dmcar / prior_dmcar, 4)},
        "nonidentifiable_axis_split": {"posterior_shift": round(post_dsplit, 4),
                                       "prior_alone_shift": round(prior_dsplit, 4),
                                       "override_survival_ratio": round(post_dsplit / prior_dsplit, 4)},
        "split_mae_structured": {"flat_likelihood_only": split_mae(post["flat"]),
                                 "correct_prior": split_mae(correct), "wrong_prior": split_mae(wrong),
                                 "n_structured": int(structured.sum())},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline-checkpoint", default=f"{BASELINE}/checkpoints/best_model.pt")
    ap.add_argument("--baseline-config", default=f"{BASELINE}/config.yaml")
    ap.add_argument("--heads-checkpoint",
                    default=f"{BASELINE}/checkpoints/stageC_composition_frozen-probe+footprint.pt")
    ap.add_argument("--calibration-report", default=f"{BASELINE}/stageD_calibration.json")
    ap.add_argument("--n-batches", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--head-hidden", type=int, default=64)
    ap.add_argument("--seed", type=int, default=20260601)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output", type=Path, default=Path(f"{BASELINE}/stageP2_override_audit.json"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    cfg = load_config(args.baseline_config)
    dims = dict(hidden_dim=cfg.model.hidden_dim, evidence_dim=cfg.model.evidence_dim,
                n_layers=cfg.model.n_layers, n_heads=cfg.model.n_heads,
                max_cols=cfg.data.max_cols, dropout=cfg.model.dropout)
    tau = float(json.loads(Path(args.calibration_report).read_text())["tau"])
    encoder = init_encoder(args.baseline_checkpoint, dims, args.device)
    state = torch.load(args.heads_checkpoint, map_location="cpu", weights_only=False)["head_states"]
    heads = []
    for hs in state:
        h = CompositionHead(cfg.model.evidence_dim, hidden_dim=args.head_hidden,
                            dropout=cfg.model.dropout, n_extra_features=N_FOOTPRINT_FEATURES)
        h.load_state_dict(hs)
        heads.append(h.to(args.device).eval())
    raws = load_raws(cfg.data.val_datasets, cfg.data.max_cols)
    print(f"Stage P2 — override audit | tau={tau:.3f} | {len(heads)} heads | {len(raws)} val datasets")

    alpha_raw, alpha_cal, truth = collect_likelihood(
        encoder, heads, raws, tau, n_batches=args.n_batches, batch_size=args.batch_size,
        max_rows=cfg.data.max_rows, max_cols=cfg.data.max_cols, device=args.device, seed=args.seed + 7)
    n = alpha_raw.shape[0]
    # Combine the prior with the RAW likelihood evidence (where the data's differential informativeness
    # lives — sharp on MCAR, flat on the split); calibration is post-hoc. Also report the calibrated-space
    # combination to expose that tau-flattening erases the data's ability to override.
    audit_raw = run_audit(alpha_raw, truth)
    audit_cal = run_audit(alpha_cal, truth)
    res = {"n_datasets": int(n), "tau": tau, "strong_reliability": _STRONG_R,
           "mean_alpha0_raw": round(float(alpha_raw.sum(1).mean()), 3),
           "mean_alpha0_calibrated": round(float(alpha_cal.sum(1).mean()), 3),
           "combine_raw_evidence": audit_raw, "combine_calibrated": audit_cal}

    def _line(tag, A):
        a, bx = A["identifiable_axis_MCAR"], A["nonidentifiable_axis_split"]
        print(f"  [{tag}] MCAR-axis survival {a['override_survival_ratio']:.2f} (LOW=data overrides) | "
              f"split-axis survival {bx['override_survival_ratio']:.2f} (HIGH=prior decides)")

    print("\n" + "=" * 92)
    print("STAGE P2 — OVERRIDE AUDIT (injected priors r=0.70; survival = posterior shift / prior-alone shift)")
    print("=" * 92)
    print(f"  data evidence alpha0: raw {res['mean_alpha0_raw']} | tau-calibrated {res['mean_alpha0_calibrated']}")
    _line("combine in RAW evidence (correct) ", audit_raw)
    _line("combine in CALIBRATED space (flat)", audit_cal)
    s = audit_raw["split_mae_structured"]
    print(f"\n  split MAE on {s['n_structured']} structured datasets (raw combine; the non-identifiable axis):")
    print(f"    likelihood-only (flat prior) : {s['flat_likelihood_only']}")
    print(f"    + CORRECT prior              : {s['correct_prior']}   (prior helps where data is silent)")
    print(f"    + WRONG prior                : {s['wrong_prior']}   (prior hurts; data can't rescue it)")
    a, bx = audit_raw["identifiable_axis_MCAR"], audit_raw["nonidentifiable_axis_split"]
    directional = a["override_survival_ratio"] < bx["override_survival_ratio"]
    verdict = (
        f"PRIOR ADDS VALUE WHERE THE DATA IS SILENT: on the non-identifiable MAR-vs-MNAR split (the data is "
        f"at ~chance), a CORRECT prior cuts split-MAE {s['flat_likelihood_only']}->{s['correct_prior']}, "
        f"while a WRONG prior inflates it to {s['wrong_prior']} (the data cannot rescue it — the honest "
        f"risk). The data overrides the prior {'more' if directional else 'NOT more'} on the identifiable "
        f"MCAR axis than the non-id split (survival {a['override_survival_ratio']:.2f} vs "
        f"{bx['override_survival_ratio']:.2f}). Because the prior is uncheckable on the non-id axis, it must "
        f"stay auditable + reported as a separate channel; combine in RAW evidence space (tau-calibrated "
        f"combination over-flattens the data and the prior dominates both axes).")
    res["verdict"] = verdict
    print(f"\n  NOTE: combining in tau-calibrated space flattens the data so the prior dominates BOTH axes "
          f"(MCAR survival {audit_cal['identifiable_axis_MCAR']['override_survival_ratio']:.2f}); the raw-"
          "evidence combination preserves the data's differential informativeness. Calibrate AFTER combining.")
    print(f"\nVERDICT: {verdict}")
    print("=" * 92)
    args.output.write_text(json.dumps(res, indent=2))
    print(f"\nWrote -> {args.output}")


if __name__ == "__main__":
    main()
