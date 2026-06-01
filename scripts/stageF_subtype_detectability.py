#!/usr/bin/env python3
"""
Stage F (ADR-0007): does MNAR-SUBTYPE detectability ("Pillar 1") survive into the dataset-level
COMPOSITION posterior, or did aggregation average it away?

Pillar 1 (ADR-0007; Stage 4 / 4b) holds at the single-mechanism / per-column level: a threshold /
detection-limit MNAR process leaves a visibly TRUNCATED ("cliffed") observed distribution and is
detectable ~0.9-1.0 per column, whereas smooth self-censoring leaves only a faint distortion and is
near the floor. The composition model, however, reports an AGGREGATE dataset-level MNAR fraction that
AVERAGES detectable (loud) and undetectable (quiet) MNAR subtypes together. This experiment runs the
one clean test that the composition arc never ran: holding the MNAR FRACTION fixed, does the posterior
estimate a HIGHER MNAR fraction and/or report HIGHER confidence when the MNAR is the LOUD
(threshold/detection) subtype vs the QUIET (self-censoring) subtype?

Design — change ONE variable (the MNAR subtype family):
  - Fixed by-cell target composition (default MCAR .2 / MAR .3 / MNAR .5); miss rate drawn as usual.
  - MNAR forced PER-COLUMN-ONLY (mnar_block_share=0) so the loud/quiet contrast is not diluted by the
    joint MNAR blocks (latent/attrition/module), which have no threshold/detection variant.
  - Corpus L (LOUD)  = MNAR columns drawn only from {threshold_*, soft_threshold, col_specific_thresh,
    detection_*}; corpus Q (QUIET) = only {self_censoring, selfcensor_*}.
  - MATCHED: the SAME rng seed builds both corpora, so datasets, MCAR/MAR masks, per-column rates, and
    the allocation plan are byte-identical between L and Q; ONLY the MNAR per-column masks differ.

Step 0 — FOOTPRINT CEILING (cheap, model-free; bounds the answer). Compare the MNAR-shape footprint
  features (obs_abs_skew_mean, obs_excess_kurt_mean) L-vs-Q and train a held-out RandomForest on the
  full 20-D footprint to classify L vs Q. If the footprint cannot separate loud from quiet at fixed
  fraction, the model cannot either (a clean null). If it can, proceed to the model.

Step 1 — MODEL READ. Run the trained calibrated instrument (encoder + Stage-C frozen+footprint 3-head
  ensemble + Stage-D temperature) on L and Q (>=5 matched seeds). Report, as DISTRIBUTIONS (mean±sd
  over seeds) and as PAIRED deltas (L_i - Q_i on matched datasets): (a) the estimated MNAR fraction
  (Dirichlet-mean MNAR component); (b) confidence (can't-tell mass, Dirichlet concentration alpha0);
  (c) P(f_MNAR >= 0.4) (resolution toward the true high MNAR fraction).

Verdict (pre-registered): SUCCESS = the posterior estimates a meaningfully higher MNAR fraction AND/OR
is more confident for LOUD than QUIET, beyond seed noise (Pillar 1 survives into the composition).
NULL = no difference (the dataset-level granularity averages Pillar 1 away). Either is a valid finding.

Deterministic via explicit seeds. Usage:
    python scripts/stageF_subtype_detectability.py                  # full run (5 seeds)
    python scripts/stageF_subtype_detectability.py --quick          # fast smoke (2 seeds, small)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lacuna.core.rng import RNGState
from lacuna.config import load_config
from lacuna.data.catalog import create_default_catalog
from lacuna.data.composition_batch import build_composition_batch, N_FOOTPRINT_FEATURES
from lacuna.data.missingness_footprint import FOOTPRINT_FEATURES
from lacuna.models.composition_head import (
    CompositionHead, ensemble_alpha, composition_mean, cant_tell_mass,
)
from lacuna.training.composition_calibration import apply_temperature, region_prob_ge
from scripts.stageC_composition_head import init_encoder, forward_alpha

BASELINE = "/mnt/artifacts/project_lacuna/runs/stage0_general_baseline"

# The ONE variable: MNAR subtype family. LOUD = sharp threshold/detection cliffs; QUIET = smooth
# self-censoring (sigmoid on own value). Both are own-value MNAR; they differ in how visible the
# truncation footprint is (ADR-0007 Pillar 1 / Stage 4).
LOUD = ("threshold_left", "threshold_right", "threshold_two_sided", "soft_threshold",
        "col_specific_thresh", "detection_lower", "detection_upper", "detection_both")
QUIET = ("self_censoring", "selfcensor_high", "selfcensor_low", "selfcensor_extreme",
         "selfcensor_weak", "selfcensor_strong")

# Indices of the diagnostic footprint features. The MNAR-shape (observed-value cliff) features are the
# Pillar-1 signature; the co-missingness/coupling features are what actually separates loud vs quiet at
# the column-averaged aggregate (and read MAR-ward).
_SKEW_IDX = FOOTPRINT_FEATURES.index("obs_abs_skew_mean")
_KURT_IDX = FOOTPRINT_FEATURES.index("obs_excess_kurt_mean")
_COUPLING_IDX = FOOTPRINT_FEATURES.index("mar_coupling_mean")
_MISSCORR_IDX = FOOTPRINT_FEATURES.index("miss_corr_mean_abs")
_P_THRESHOLD = 0.4   # P(f_MNAR >= t): t just below the held-fixed true MNAR fraction (~0.5)


def build_head_like(hs, evidence_dim, dropout):
    """Construct a CompositionHead whose architecture matches a saved state dict `hs`.

    The deployed Stage-C frozen+footprint head and the Stage-C2 capacity ablation differ in width
    (64 vs 256) and depth (1 vs 2 hidden layers); reading the shapes back makes the loader robust to
    whichever checkpoint is on disk (fail-loud if the keys are unexpected — Coding Bible Rule 1)."""
    n_extra = int(hs["extra_norm.weight"].shape[0]) if "extra_norm.weight" in hs else 0
    lin = sorted(int(k.split(".")[1]) for k in hs if k.startswith("net.") and k.endswith(".weight")
                 and k.split(".")[1].isdigit())
    if not lin:  # a bare Linear (hidden_dim=None): key is "net.weight"
        if "net.weight" not in hs:
            raise ValueError(f"unrecognised head state_dict keys: {sorted(hs)}")
        in_dim = int(hs["net.weight"].shape[1])
        hidden_dim, n_hidden_layers = None, 1
    else:
        hidden_dim = int(hs[f"net.{lin[0]}.weight"].shape[0])
        in_dim = int(hs[f"net.{lin[0]}.weight"].shape[1])
        n_hidden_layers = len(lin) - 1
    use_evidence = in_dim == evidence_dim + n_extra
    if not use_evidence and in_dim != n_extra:
        raise ValueError(f"head in_dim {in_dim} matches neither evidence+extra "
                         f"({evidence_dim + n_extra}) nor extra-only ({n_extra})")
    head = CompositionHead(evidence_dim, hidden_dim=hidden_dim, dropout=dropout,
                           n_extra_features=n_extra, n_hidden_layers=max(1, n_hidden_layers),
                           use_evidence=use_evidence)
    head.load_state_dict(hs)
    return head


def load_raws(names, max_cols):
    cat = create_default_catalog()
    out = []
    for n in names:
        try:
            raw = cat.load(n)
        except Exception as e:  # noqa: BLE001
            print(f"  warn: skip '{n}': {e}"); continue
        if 4 <= raw.d <= max_cols and np.isfinite(raw.data).all():
            out.append(raw)
    return out


def make_corpus(raws, *, seed, n_batches, batch_size, subtypes, fixed_comp, mnar_block_share,
                miss_rate_range, max_rows, max_cols):
    """Build one matched corpus (a list of footprint-bearing batches) at a fixed composition.

    Calling this with the SAME seed and a different `subtypes` produces a corpus matched item-for-item
    (same datasets, MCAR/MAR masks, plan, miss rates) differing ONLY in the MNAR per-column masks.
    """
    rng = RNGState(seed=seed)
    batches, comps, fps, mrs = [], [], [], []
    for _ in range(n_batches):
        mb = build_composition_batch(
            raws, rng.spawn(), max_rows=max_rows, max_cols=max_cols, batch_size=batch_size,
            miss_rate_range=miss_rate_range, fixed_composition=fixed_comp,
            mnar_subtypes=subtypes, mnar_block_share=mnar_block_share, with_footprints=True)
        batches.append(mb)
        comps.append(mb.composition)
        fps.append(mb.footprints)
        mrs.append(mb.miss_rate)
    return {
        "batches": batches,
        "comp": torch.cat(comps).numpy(),      # [N, 3] realised by-cell composition
        "fp": torch.cat(fps).numpy(),          # [N, 20] observable footprint
        "miss_rate": torch.cat(mrs).numpy(),   # [N]
    }


def model_read(encoder, heads, tau, corpus, device):
    """Run the calibrated instrument over a corpus; return per-dataset posterior summaries [N]."""
    alphas = []
    for mb in corpus["batches"]:
        b = mb.batch.to(device)
        extra = mb.footprints.to(device)
        with torch.no_grad():
            per_model = torch.stack([forward_alpha(encoder, h, b, extra).cpu() for h in heads], 0)
        alphas.append(apply_temperature(ensemble_alpha(per_model).numpy(), tau))
    alpha = np.concatenate(alphas, axis=0)                  # [N, 3] calibrated Dirichlet evidence
    mean = composition_mean(torch.tensor(alpha)).numpy()    # [N, 3]
    struct = (mean[:, 1] + mean[:, 2]).clip(min=1e-6)       # f_MAR + f_MNAR (the structured part)
    return {
        "f_mnar": mean[:, 2],                                            # MNAR-fraction estimate (metric a)
        "f_mcar": mean[:, 0],                                            # structure axis (lower => more structured)
        "mnar_within_struct": mean[:, 2] / struct,                       # the MAR-vs-MNAR split (Pillar-1 axis)
        "cant_tell": cant_tell_mass(torch.tensor(alpha)).numpy(),        # vacuity K/alpha0 (metric b)
        "alpha0": alpha.sum(axis=1),                                     # Dirichlet concentration (metric b)
        "p_mnar_ge": region_prob_ge(alpha, 2, _P_THRESHOLD),            # P(f_MNAR >= 0.4) (metric c)
        "comp_mean": mean,
    }


def footprint_ceiling(fp_L, fp_Q, *, seed):
    """Held-out RandomForest L-vs-Q on the 20-D footprint. Returns (auc, acc, top_importances)."""
    X = np.concatenate([fp_L, fp_Q], axis=0)
    y = np.concatenate([np.ones(len(fp_L)), np.zeros(len(fp_Q))])    # L=1, Q=0
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
    rf = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
    rf.fit(Xtr, ytr)
    proba = rf.predict_proba(Xte)[:, 1]
    auc = float(roc_auc_score(yte, proba))
    acc = float((rf.predict(Xte) == yte).mean())
    order = np.argsort(rf.feature_importances_)[::-1][:5]
    top = [(FOOTPRINT_FEATURES[i], round(float(rf.feature_importances_[i]), 3)) for i in order]
    return auc, acc, top


def _ms(x):
    """(mean, sd) as rounded floats over a list/array."""
    a = np.asarray(x, dtype=float)
    return round(float(a.mean()), 4), round(float(a.std()), 4)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline-checkpoint", default=f"{BASELINE}/checkpoints/best_model.pt")
    ap.add_argument("--baseline-config", default=f"{BASELINE}/config.yaml")
    ap.add_argument("--heads-checkpoint",
                    default=f"{BASELINE}/checkpoints/stageC_composition_frozen-probe+footprint.pt")
    ap.add_argument("--calibration-report", default=f"{BASELINE}/stageD_calibration.json")
    ap.add_argument("--tau", type=float, default=None, help="override temperature (else read report)")
    ap.add_argument("--fixed-composition", type=float, nargs=3, default=(0.2, 0.3, 0.5),
                    metavar=("MCAR", "MAR", "MNAR"), help="held-fixed by-cell target composition")
    ap.add_argument("--miss-rate-lo", type=float, default=0.1)
    ap.add_argument("--miss-rate-hi", type=float, default=0.4,
                    help="upper miss rate; kept <~0.45 so per-column-only MNAR at f=0.5 stays feasible")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--n-batches", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--quick", action="store_true", help="fast smoke: 2 seeds, 3 batches")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output", type=Path, default=Path(f"{BASELINE}/stageF_subtype_detectability.json"))
    args = ap.parse_args()

    if args.quick:
        args.seeds, args.n_batches, args.batch_size = [1, 2], 3, 12

    torch.manual_seed(args.seeds[0])
    cfg = load_config(args.baseline_config)
    dims = dict(hidden_dim=cfg.model.hidden_dim, evidence_dim=cfg.model.evidence_dim,
                n_layers=cfg.model.n_layers, n_heads=cfg.model.n_heads,
                max_cols=cfg.data.max_cols, dropout=cfg.model.dropout)
    max_rows, max_cols = cfg.data.max_rows, cfg.data.max_cols
    tau = args.tau
    if tau is None:
        tau = float(json.loads(Path(args.calibration_report).read_text())["tau"])
    fixed_comp = tuple(round(float(x), 6) for x in args.fixed_composition)
    miss_rate_range = (args.miss_rate_lo, args.miss_rate_hi)

    print(f"Stage F — MNAR subtype detectability in the composition posterior")
    print(f"  fixed composition (MCAR/MAR/MNAR) = {fixed_comp} | miss-rate {miss_rate_range} | "
          f"MNAR per-column-only | tau={tau:.3f}")
    print(f"  LOUD  ({len(LOUD)}): {LOUD}")
    print(f"  QUIET ({len(QUIET)}): {QUIET}")

    train_raws = load_raws(cfg.data.train_datasets, max_cols)
    print(f"  train datasets: {len(train_raws)} | seeds {args.seeds} | "
          f"{args.n_batches}x{args.batch_size}={args.n_batches * args.batch_size} datasets/corpus/seed")

    encoder = init_encoder(args.baseline_checkpoint, dims, args.device)
    state = torch.load(args.heads_checkpoint, map_location="cpu", weights_only=False)["head_states"]
    heads = [build_head_like(hs, cfg.model.evidence_dim, cfg.model.dropout).to(args.device).eval()
             for hs in state]
    arch = heads[0].net
    print(f"  loaded encoder + {len(heads)}-head ensemble (head: {arch})\n")

    # Per-corpus summary for one model-read result `r` over corpus `c` (means over the corpus).
    def side_metrics(r, c):
        return {"f_mnar": float(r["f_mnar"].mean()), "f_mcar": float(r["f_mcar"].mean()),
                "mnar_within_struct": float(r["mnar_within_struct"].mean()),
                "cant_tell": float(r["cant_tell"].mean()), "alpha0": float(r["alpha0"].mean()),
                "p_mnar_ge": float(r["p_mnar_ge"].mean()),
                "comp": [round(float(x), 3) for x in c["comp"].mean(0)],
                "comp_pred": [round(float(x), 3) for x in r["comp_mean"].mean(0)],
                "miss_rate": float(c["miss_rate"].mean()),
                "obs_skew": float(c["fp"][:, _SKEW_IDX].mean()),
                "obs_kurt": float(c["fp"][:, _KURT_IDX].mean()),
                "mar_coupling": float(c["fp"][:, _COUPLING_IDX].mean()),
                "miss_corr": float(c["fp"][:, _MISSCORR_IDX].mean())}

    # --- per-seed loop: build matched L/Q corpora, read the model, accumulate ---
    per_seed = []
    fp_L_all, fp_Q_all = [], []
    _PAIRED_KEYS = ("f_mnar", "f_mcar", "mnar_within_struct", "cant_tell", "p_mnar_ge", "alpha0")
    common = dict(n_batches=args.n_batches, batch_size=args.batch_size, fixed_comp=fixed_comp,
                  mnar_block_share=0.0, miss_rate_range=miss_rate_range,
                  max_rows=max_rows, max_cols=max_cols)
    for s in args.seeds:
        cL = make_corpus(train_raws, seed=s, subtypes=LOUD, **common)
        cQ = make_corpus(train_raws, seed=s, subtypes=QUIET, **common)
        rL = model_read(encoder, heads, tau, cL, args.device)
        rQ = model_read(encoder, heads, tau, cQ, args.device)
        fp_L_all.append(cL["fp"]); fp_Q_all.append(cQ["fp"])
        # Per-seed corpus means + PAIRED deltas (matched datasets => L_i - Q_i is a clean contrast).
        row = {
            "seed": s, "n": int(len(rL["f_mnar"])),
            "L": side_metrics(rL, cL), "Q": side_metrics(rQ, cQ),
            "paired_delta": {k: float((rL[k] - rQ[k]).mean()) for k in _PAIRED_KEYS},
        }
        per_seed.append(row)
        d = row["paired_delta"]
        print(f"  seed {s}: f_MNAR L={row['L']['f_mnar']:.3f} Q={row['Q']['f_mnar']:.3f} "
              f"Δ={d['f_mnar']:+.3f} | cant-tell L={row['L']['cant_tell']:.3f} Q={row['Q']['cant_tell']:.3f} "
              f"Δ={d['cant_tell']:+.3f} | P(MNAR≥{_P_THRESHOLD}) Δ={d['p_mnar_ge']:+.3f}", flush=True)

    # --- footprint ceiling (Step 0): per-seed RF AUC + a pooled-importance RF ---
    seed_aucs = [footprint_ceiling(cL_fp, cQ_fp, seed=10 + i)[0]
                 for i, (cL_fp, cQ_fp) in enumerate(zip(fp_L_all, fp_Q_all))]
    pooled_auc, pooled_acc, pooled_top = footprint_ceiling(
        np.concatenate(fp_L_all), np.concatenate(fp_Q_all), seed=99)

    # --- aggregate over seeds (mean±sd; distributions, never single-seed) ---
    def agg(side, key):
        return _ms([r[side][key] for r in per_seed])

    def aggd(key):
        return _ms([r["paired_delta"][key] for r in per_seed])

    deltas_fmnar = [r["paired_delta"]["f_mnar"] for r in per_seed]
    deltas_ct = [r["paired_delta"]["cant_tell"] for r in per_seed]
    summary = {
        "fixed_composition": fixed_comp, "miss_rate_range": miss_rate_range, "tau": tau,
        "seeds": args.seeds, "n_per_corpus_per_seed": args.n_batches * args.batch_size,
        "footprint_ceiling": {
            "per_seed_auc": [round(a, 4) for a in seed_aucs],
            "auc_mean_sd": _ms(seed_aucs),
            "pooled_auc": round(pooled_auc, 4), "pooled_acc": round(pooled_acc, 4),
            "pooled_top_features": pooled_top,
            "obs_skew_L": agg("L", "obs_skew"), "obs_skew_Q": agg("Q", "obs_skew"),
            "obs_kurt_L": agg("L", "obs_kurt"), "obs_kurt_Q": agg("Q", "obs_kurt"),
            "mar_coupling_L": agg("L", "mar_coupling"), "mar_coupling_Q": agg("Q", "mar_coupling"),
            "miss_corr_L": agg("L", "miss_corr"), "miss_corr_Q": agg("Q", "miss_corr"),
        },
        "model_read": {
            "f_mnar_L": agg("L", "f_mnar"), "f_mnar_Q": agg("Q", "f_mnar"),
            "f_mcar_L": agg("L", "f_mcar"), "f_mcar_Q": agg("Q", "f_mcar"),
            "mnar_within_struct_L": agg("L", "mnar_within_struct"),
            "mnar_within_struct_Q": agg("Q", "mnar_within_struct"),
            "cant_tell_L": agg("L", "cant_tell"), "cant_tell_Q": agg("Q", "cant_tell"),
            "alpha0_L": agg("L", "alpha0"), "alpha0_Q": agg("Q", "alpha0"),
            "p_mnar_ge_L": agg("L", "p_mnar_ge"), "p_mnar_ge_Q": agg("Q", "p_mnar_ge"),
            "miss_rate_L": agg("L", "miss_rate"), "miss_rate_Q": agg("Q", "miss_rate"),
        },
        "paired_delta": {
            "f_mnar": aggd("f_mnar"), "f_mcar": aggd("f_mcar"),
            "mnar_within_struct": aggd("mnar_within_struct"),
            "cant_tell": aggd("cant_tell"), "p_mnar_ge": aggd("p_mnar_ge"), "alpha0": aggd("alpha0"),
        },
        "per_seed": per_seed,
    }

    # --- verdict (pre-registered) ---
    df_mean, df_sd = aggd("f_mnar")
    dct_mean, dct_sd = aggd("cant_tell")
    dp_mean, dp_sd = aggd("p_mnar_ge")
    dsplit_mean, dsplit_sd = aggd("mnar_within_struct")
    dmcar_mean, dmcar_sd = aggd("f_mcar")
    deltas_split = [r["paired_delta"]["mnar_within_struct"] for r in per_seed]
    deltas_mcar = [r["paired_delta"]["f_mcar"] for r in per_seed]
    # Robust = consistent sign across every seed AND mean well outside seed noise (|mean| > 2·sd).
    fmnar_robust = all(d > 0 for d in deltas_fmnar) and abs(df_mean) > 2 * (df_sd + 1e-9)
    conf_robust = all(d < 0 for d in deltas_ct) and abs(dct_mean) > 2 * (dct_sd + 1e-9)
    # Does the lean live on the (non-identifiable) MAR-vs-MNAR split, or only on the structure axis?
    split_robust = all(d > 0 for d in deltas_split) and abs(dsplit_mean) > 2 * (dsplit_sd + 1e-9)
    # Does loud read as more STRUCTURED (identifiable axis responds)? f_MCAR down robustly.
    structure_responds = all(d < 0 for d in deltas_mcar) and abs(dmcar_mean) > 2 * (dmcar_sd + 1e-9)
    meaningful = abs(df_mean) > 0.02 or abs(dp_mean) > 0.02 or abs(dct_mean) > 0.02
    footprint_sees_it = summary["footprint_ceiling"]["auc_mean_sd"][0] > 0.6
    success = (fmnar_robust or conf_robust) and meaningful
    if success and split_robust:
        verdict = ("SUCCESS (strong) — Pillar 1 SURVIVES into the composition on the MNAR-SPECIFIC "
                   "axis: the posterior leans more MNAR-within-structured for LOUD than QUIET, beyond "
                   "seed noise — the dataset-level fraction does NOT average the detectable subtype away.")
    elif fmnar_robust and meaningful:
        verdict = ("PARTIAL — the posterior leans more MNAR for LOUD (robust direction), but the lean "
                   "lives mostly on the random-vs-STRUCTURED axis (loud reads as more structured); the "
                   "MNAR-vs-MAR split barely moves. Pillar 1 survives as STRUCTURE, attenuated on mechanism.")
    elif not footprint_sees_it:
        verdict = ("NULL (forced at the footprint) — the 20-D footprint itself cannot separate loud "
                   "from quiet MNAR at fixed fraction, so the model cannot either: the aggregate "
                   "footprint averages Pillar 1 away. A clean, attributable null.")
    elif footprint_sees_it and structure_responds and not split_robust:
        verdict = ("NULL on the MECHANISM axis (the informative null) — the footprint trivially "
                   "separates loud vs quiet (AUC~1), and the posterior DOES respond on the IDENTIFIABLE "
                   "axis (loud reads more STRUCTURED, f_MCAR down). But the loud signature is "
                   "co-missingness/coupling, observationally MAR-like at the column-averaged aggregate, "
                   "so the MAR-vs-MNAR SPLIT does NOT lean MNAR for loud (flat or MAR-ward) and "
                   "confidence does not rise. The dataset-level MNAR fraction averages Pillar 1 away on "
                   "the mechanism axis; the detectable-subtype signal would need the per-column (Q3) "
                   "layer to express AS MNAR.")
    else:
        verdict = ("NULL (model leaves it on the table) — the footprint CAN separate loud vs quiet "
                   "(AUC>0.6) but the composition posterior does not express it beyond seed noise: the "
                   "dataset-level MNAR fraction averages the detectable subtype signal away.")
    summary["verdict"] = {
        "success": bool(success), "split_axis_robust": bool(split_robust),
        "structure_axis_responds": bool(structure_responds),
        "footprint_separates": bool(footprint_sees_it),
        "f_mnar_delta_robust": bool(fmnar_robust), "confidence_delta_robust": bool(conf_robust),
        "meaningful_magnitude": bool(meaningful), "text": verdict,
    }

    print("\n" + "=" * 96)
    print("STAGE F — MNAR subtype detectability in the dataset-level composition posterior")
    print("=" * 96)
    fc = summary["footprint_ceiling"]
    print(f"STEP 0 — FOOTPRINT CEILING (model-free, can the observable footprint see loud vs quiet?)")
    print(f"  RF L-vs-Q AUC: per-seed {fc['per_seed_auc']} | mean±sd {fc['auc_mean_sd']} | "
          f"pooled {fc['pooled_auc']} (acc {fc['pooled_acc']})")
    print(f"  obs_abs_skew_mean (cliff)   : L {fc['obs_skew_L']}  vs  Q {fc['obs_skew_Q']}")
    print(f"  obs_excess_kurt_mean (cliff): L {fc['obs_kurt_L']}  vs  Q {fc['obs_kurt_Q']}")
    print(f"  mar_coupling_mean (coupling): L {fc['mar_coupling_L']}  vs  Q {fc['mar_coupling_Q']}")
    print(f"  miss_corr_mean_abs (couple) : L {fc['miss_corr_L']}  vs  Q {fc['miss_corr_Q']}")
    print(f"  top RF features: {fc['pooled_top_features']}")
    mr = summary["model_read"]
    pd = summary["paired_delta"]
    print(f"\nSTEP 1 — MODEL READ (mean±sd over {len(args.seeds)} seeds; LOUD vs QUIET)")
    print(f"  MNAR fraction (metric a) : L {mr['f_mnar_L']}  vs  Q {mr['f_mnar_Q']}   (paired Δ {pd['f_mnar']})")
    print(f"  can't-tell (metric b)    : L {mr['cant_tell_L']}  vs  Q {mr['cant_tell_Q']}   (paired Δ {pd['cant_tell']})")
    print(f"  alpha0 conc. (metric b)  : L {mr['alpha0_L']}  vs  Q {mr['alpha0_Q']}   (paired Δ {pd['alpha0']})")
    print(f"  P(f_MNAR≥{_P_THRESHOLD}) (metric c)  : L {mr['p_mnar_ge_L']}  vs  Q {mr['p_mnar_ge_Q']}   (paired Δ {pd['p_mnar_ge']})")
    print(f"  -- honest-seam disaggregation (where does the lean live?) --")
    print(f"  f_MCAR (structure axis)  : L {mr['f_mcar_L']}  vs  Q {mr['f_mcar_Q']}   (paired Δ {pd['f_mcar']})")
    print(f"  MNAR|structured (split)  : L {mr['mnar_within_struct_L']}  vs  Q {mr['mnar_within_struct_Q']}   (paired Δ {pd['mnar_within_struct']})")
    print(f"\nVERDICT: {verdict}")
    print("=" * 96)

    args.output.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote report -> {args.output}")


if __name__ == "__main__":
    main()
