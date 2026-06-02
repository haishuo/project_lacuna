#!/usr/bin/env python3
"""
Stage Q (ADR-0008 commitment 6): the per-column SUBTYPE layer — the restored Q3.

Extends the ADR-0008 prior x likelihood instrument from MECHANISM (MCAR/MAR/MNAR) to SUBTYPE
granularity, where property 3's literal output lives ("fairly certain 20% is THRESHOLD MNAR ...
unable to determine for the rest"). Two separable channels, fused per column, aggregated to a
dataset subtype-composition by missing cell, with a calibrated abstain mass:

  LIKELIHOOD (data): a deployable-feature gradient-boosted readout over L = {threshold, detection,
    reject} (lacuna.models.subtype_likelihood). It detects the LOUD MNAR fingerprints and REJECTS the
    non-identifiable region (quiet self-censoring MNAR, MAR, MCAR) into one class — NOT the Stage-5
    forced 3-way MCAR/MAR/MNAR softmax that collapses at matched rate. The frozen-encoder reps were
    measured to DILUTE the loud signal (scripts/stageQ_feature_attribution.py), so they are excluded:
    the deployable distributional features carry it (no encoder, no oracle).
  PRIOR (metadata): the subtype prior (lacuna.priors.subtype_prior); on semi-synthetic, simulated at
    a known reliability rho (Stage P5/P6 method) to separate calibration from correctness, plus a
    benchmark-consistency check of the real frozen spec.

Reports, over >=5 seeds (mean +/- sd): loud-vs-reject DETECTION (AUC/AP + recall-at-precision +
argmax recall + STABILITY); the fused subtype-composition calibration (data-only vs reliable vs
near-chance prior); the per-column ABSTENTION (coverage at a target committed-accuracy, abstain mass);
and the OVERRIDE (data overrides a wrong prior on the loud axis). Matched miss rate throughout
(diverse pools => compensate_rate is a no-op; the Stage-5 cue-free regime).

Deterministic via explicit seeds. Usage:
    python scripts/stageQ_subtype_layer.py --seeds 5
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

from lacuna.core.rng import RNGState
from lacuna.config import load_config
from lacuna.models.column_deployable_features import per_column_deployable_features, N_DEPLOYABLE_FEATURES
from lacuna.models.subtype_likelihood import SubtypeLikelihoodDetector
from lacuna.data.catalog import create_default_catalog
from lacuna.data.mixed_batch import build_mixed_batch
from lacuna.data.subtype_targets import subtype_targets
from lacuna.data.tokenization import IDX_OBSERVED
from lacuna.priors.dirichlet_evidence import aggregate_evidence, evidence_mean, reliability_to_strength
from lacuna.priors.subtype_ontology import (
    SUBTYPES, N_SUBTYPES, DETECTABLE_SUBTYPES, embed_likelihood_evidence,
    LIKELIHOOD_LABELS, LIKE_THRESHOLD, LIKE_DETECTION, LIKE_INDETERMINATE,
    SELF_CENSORING_MNAR, MCAR_SUB, MAR_SUB,
)
from lacuna.priors.subtype_prior import subtype_prior_alpha
from lacuna.training.composition_calibration import (
    apply_temperature, fit_temperature, default_query_set, query_ece, collect_query_pairs, brier_score,
)

BASELINE = "/mnt/artifacts/project_lacuna/runs/stage0_general_baseline"
# matched-rate full-diversity regime (Stage-5 cue-free control): diverse MNAR + diverse MAR.
MIXTURE_KWARGS = dict(p_observed=0.3, target_miss_rate=0.25,
                      mnar_diverse=True, mar_diverse=True, compensate_rate=True)


def load_raws(names, max_cols):
    cat = create_default_catalog()
    out = []
    for n in names:
        try:
            raw = cat.load(n)
        except Exception as e:  # noqa: BLE001
            print(f"  warn: skip '{n}': {e}"); continue
        if raw.d <= max_cols:
            out.append(raw)
    return out


def collect(raws, *, seed, n_batches, batch_size, max_rows, max_cols):
    """Per supervised column: deployable features, L-truth, O-truth, missing-cell weight, dataset id."""
    rng = RNGState(seed=seed)
    F, L, O, W, D = [], [], [], [], []
    dset_base = 0
    for _ in range(n_batches):
        mb = build_mixed_batch(raws, rng.spawn(), max_rows=max_rows, max_cols=max_cols,
                               batch_size=batch_size, **MIXTURE_KWARGS)
        b = mb.batch
        feats = per_column_deployable_features(b.tokens, b.row_mask, b.col_mask).numpy()      # [B,C,5]
        like, onto, sup = subtype_targets(mb)
        is_obs = b.tokens[..., IDX_OBSERVED] > 0.5
        valid = b.row_mask.unsqueeze(-1) & b.col_mask.unsqueeze(1)
        n_missing = (valid & ~is_obs).sum(dim=1).numpy()                                       # [B,C]
        B, C = sup.shape
        for i in range(B):
            for j in range(C):
                if not bool(sup[i, j]):
                    continue
                F.append(feats[i, j]); L.append(int(like[i, j])); O.append(int(onto[i, j]))
                W.append(float(max(n_missing[i, j], 1.0))); D.append(dset_base + i)
        dset_base += B
    return dict(feat=np.array(F), like=np.array(L), onto=np.array(O), w=np.array(W), dset=np.array(D))


def _recall_at_precision(p_loud, y_loud, min_prec):
    prec, rec, _ = precision_recall_curve(y_loud, p_loud)
    ok = prec >= min_prec
    return float(rec[ok].max()) if ok.any() else 0.0


def likelihood_metrics(p, like):
    """Loud-vs-reject AUC/AP + recall-at-precision + argmax recall + threshold<->detection confusion."""
    pred = p.argmax(1)
    def recall(label):
        m = like == label
        return float((pred[m] == label).mean()) if m.any() else None
    def among(tl, pl):
        m = like == tl
        return float((pred[m] == pl).mean()) if m.any() else None
    p_loud = p[:, LIKE_THRESHOLD] + p[:, LIKE_DETECTION]
    y_loud = (like != LIKE_INDETERMINATE).astype(int)
    auc = float(roc_auc_score(y_loud, p_loud)) if 0 < y_loud.mean() < 1 else None
    ap = float(average_precision_score(y_loud, p_loud)) if y_loud.any() else None
    return {
        "loud_vs_reject_auc": round(auc, 4) if auc is not None else None,
        "loud_vs_reject_ap": round(ap, 4) if ap is not None else None,
        "loud_base_rate": round(float(y_loud.mean()), 4),
        "loud_recall_at_prec60": round(_recall_at_precision(p_loud, y_loud, 0.6), 4),
        "loud_recall_at_prec80": round(_recall_at_precision(p_loud, y_loud, 0.8), 4),
        "argmax_recall_threshold": recall(LIKE_THRESHOLD),
        "argmax_recall_detection": recall(LIKE_DETECTION),
        "reject_correct_on_silent": recall(LIKE_INDETERMINATE),
        "threshold_to_detection": among(LIKE_THRESHOLD, LIKE_DETECTION),
        "detection_to_threshold": among(LIKE_DETECTION, LIKE_THRESHOLD),
    }


def simulated_prior_alphas(onto, rho, fav_prob, rng):
    """Per-column prior of known reliability rho: favours the TRUE O subtype w.p. rho, else a random
    wrong one, at favoured-prob `fav_prob` (Stage P5/P6 method — separates calibration from correctness)."""
    kappa = reliability_to_strength(fav_prob, N_SUBTYPES)
    n = len(onto)
    u = rng.rand(n).numpy()
    alphas = np.ones((n, N_SUBTYPES))
    for i in range(n):
        if u[i] < rho:
            fav = int(onto[i])
        else:
            others = [c for c in range(N_SUBTYPES) if c != int(onto[i])]
            fav = others[rng.randint(0, len(others), (1,)).item()]
        alphas[i, fav] += kappa
    return alphas


def fuse(prior_alphas, p_like, kappa_like):
    """Per-column posterior = combine(prior, embed(readout)). Vectorised evidence sum."""
    like_alpha = embed_likelihood_evidence(p_like, kappa_like)        # [N,5]
    return 1.0 + (prior_alphas - 1.0) + (like_alpha - 1.0)


def per_dataset_composition(post, onto, w, dset):
    """Aggregate per-column posteriors to a dataset Dirichlet (by missing cell) + realised composition."""
    ids = sorted(set(int(d) for d in dset))
    alpha_d, realised_d = [], []
    for d in ids:
        m = dset == d
        alpha_d.append(aggregate_evidence(post[m], w[m]))
        rc = np.zeros(N_SUBTYPES)
        for o, wi in zip(onto[m], w[m]):
            rc[int(o)] += wi
        realised_d.append(rc / rc.sum())
    return np.array(ids), np.array(alpha_d), np.array(realised_d)


def composition_calibration(alpha_d, realised_d, cal_idx, test_idx):
    """Fit temperature on cal datasets; report query-ECE/Brier/L1 on test vs the prior-only baseline."""
    queries = default_query_set(n_classes=N_SUBTYPES)
    tau = fit_temperature(alpha_d[cal_idx], realised_d[cal_idx], queries)
    a_test = apply_temperature(alpha_d[test_idx], tau)
    ece = query_ece(a_test, realised_d[test_idx], queries)
    preds, inds = collect_query_pairs(a_test, realised_d[test_idx], queries)
    const = realised_d[cal_idx].mean(0, keepdims=True).repeat(len(test_idx), 0)
    cp, ci = [], []
    for c, t in queries:
        cp.append((const[:, c] >= t).astype(float)); ci.append((realised_d[test_idx][:, c] >= t).astype(float))
    comp_l1 = float(np.abs(np.array([evidence_mean(a) for a in a_test]) - realised_d[test_idx]).sum(1).mean())
    return dict(tau=round(tau, 3), query_ece=round(ece, 4), brier=round(brier_score(preds, inds), 4),
                brier_prior_only=round(brier_score(np.concatenate(cp), np.concatenate(ci)), 4),
                composition_l1=round(comp_l1, 4))


def risk_coverage_threshold(p_max, correct, target_acc):
    """Smallest threshold (max coverage) whose committed argmax-accuracy >= target_acc (Stage P6)."""
    best = 1.0
    for t in np.unique(np.concatenate([p_max, [1.0]])):
        m = p_max >= t
        if m.any() and correct[m].mean() >= target_acc:
            best = min(best, float(t))
    return best


def abstention_metrics(post, onto, w, dset, cal_ds, test_ds, primary_target):
    """Per-column selective subtype decision (Stage P6 at subtype granularity).

    Reports the coverage CURVE at committed-accuracy targets {0.6, 0.7, 0.8} (each t* fit on cal,
    coverage measured on test) so the data-only operating point is visible rather than a single,
    possibly-degenerate, point; plus committed-acc / abstain-mass / loud-vs-silent split at the
    primary target. The abstain mass is by missing cell (the property-3 'unable to determine' fraction)."""
    mean = np.array([evidence_mean(a) for a in post])
    pred = mean.argmax(1); p_max = mean.max(1); correct = (pred == onto).astype(float)
    cal = np.isin(dset, cal_ds); test = np.isin(dset, test_ds)
    cov = {}
    for tgt in (0.6, 0.7, 0.8):
        t = risk_coverage_threshold(p_max[cal], correct[cal], tgt)
        cov[tgt] = float((p_max[test] >= t).mean()) if test.any() else None
    t_star = risk_coverage_threshold(p_max[cal], correct[cal], primary_target)
    committed = p_max[test] >= t_star
    comm_acc = float(correct[test][committed].mean()) if committed.any() else None
    wt = w[test]; abstain_mass = float(wt[~committed].sum() / wt.sum()) if wt.sum() > 0 else None
    ot = onto[test]
    def split_acc(classes):
        s = np.isin(ot, classes) & committed
        return float(correct[test][s].mean()) if s.any() else None
    return dict(commit_threshold=round(t_star, 4),
                cov_at_60=round(cov[0.6], 4) if cov[0.6] is not None else None,
                cov_at_70=round(cov[0.7], 4) if cov[0.7] is not None else None,
                cov_at_80=round(cov[0.8], 4) if cov[0.8] is not None else None,
                committed_acc=round(comm_acc, 4) if comm_acc is not None else None,
                abstain_mass=round(abstain_mass, 4) if abstain_mass is not None else None,
                acc_loud=split_acc(list(DETECTABLE_SUBTYPES)),
                acc_silent=split_acc([SELF_CENSORING_MNAR, MCAR_SUB, MAR_SUB]))


def override_metrics(prior_alphas, post, onto):
    """On LOUD columns where the (simulated) prior favoured a WRONG class, does the data still win?"""
    pri = prior_alphas.argmax(1); pos = np.array([evidence_mean(a) for a in post]).argmax(1)
    loud = np.isin(onto, list(DETECTABLE_SUBTYPES))
    wrong = loud & (pri != onto)
    if not wrong.any():
        return dict(n_wrong_prior_loud=0, override_rate=None)
    return dict(n_wrong_prior_loud=int(wrong.sum()),
                override_rate=round(float((pos[wrong] == onto[wrong]).mean()), 4))


def benchmark_consistency():
    """P1-style check: the frozen subtype-prior spec faithfully encodes the benchmark's gold subtype."""
    bench = json.loads((PROJECT_ROOT / "scripts/metadata_prior/benchmark.json").read_text())
    gold_to_o = {"detection_limit": 1, "self_censoring": SELF_CENSORING_MNAR, "skip_logic": MAR_SUB,
                 "by_design_random": MCAR_SUB, "administrative": MCAR_SUB, "covariate_driven": MAR_SUB}
    ok = tot = 0
    for col in bench["columns"]:
        gsub = col["gold_subtype"]
        if gsub not in gold_to_o:  # 'none' (abstain) cols: prior is flat, not an argmax target
            continue
        tot += 1
        if int(subtype_prior_alpha(col["gold_semantic"]).argmax()) == gold_to_o[gsub]:
            ok += 1
    return dict(consistency=round(ok / tot, 4), n=tot)


def _agg(per_seed, key):
    vals = [s[key] for s in per_seed if s.get(key) is not None]
    if not vals:
        return None
    return [round(float(np.mean(vals)), 4), round(float(np.std(vals)), 4)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline-config", default=f"{BASELINE}/config.yaml")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--train-batches", type=int, default=120)
    ap.add_argument("--eval-batches", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--kappa-like", type=float, default=20.0, help="data-evidence concentration (embed)")
    ap.add_argument("--prior-fav-prob", type=float, default=0.70,
                    help="simulated prior favoured prob (<=0.70 = overridable nudge, the commitment-1 cap)")
    ap.add_argument("--rho-reliable", type=float, default=0.85)
    ap.add_argument("--rho-chance", type=float, default=0.45)
    ap.add_argument("--target-acc", type=float, default=0.80, help="abstention committed-accuracy target")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    cfg = load_config(args.baseline_config)
    max_rows, max_cols = cfg.data.max_rows, cfg.data.max_cols
    train_raws = load_raws(cfg.data.train_datasets, max_cols)
    val_raws = load_raws(cfg.data.val_datasets, max_cols)
    print(f"Stage Q | deployable-feature subtype detector | train {len(train_raws)} / val {len(val_raws)} datasets")
    print(f"Ontology O={SUBTYPES} | L={LIKELIHOOD_LABELS} | kappa_like={args.kappa_like}")

    per_seed = []
    for s in range(args.seeds):
        seed = 20260601 + 101 * s
        print(f"--- seed {s+1}/{args.seeds} (={seed}) ---", flush=True)
        tr = collect(train_raws, seed=seed, n_batches=args.train_batches, batch_size=args.batch_size,
                     max_rows=max_rows, max_cols=max_cols)
        ev = collect(val_raws, seed=seed + 7, n_batches=args.eval_batches, batch_size=args.batch_size,
                     max_rows=max_rows, max_cols=max_cols)
        # HONEST detector: drop missing_rate (feature 0) — a per-subtype rate is a non-transferable
        # confound (rate audit below: loud realises a LOWER rate than reject even at "matched" rate,
        # a generator residual the detector would otherwise exploit; the Stage-5 confound). Fusion uses
        # this. A rate-INCLUDED detector is fit alongside ONLY to quantify the artifact.
        nf = ev["feat"].shape[1]
        det = SubtypeLikelihoodDetector(seed=seed, n_features=nf - 1).fit(tr["feat"][:, 1:], tr["like"])
        p_like = det.predict_proba(ev["feat"][:, 1:])
        det_wr = SubtypeLikelihoodDetector(seed=seed, n_features=nf).fit(tr["feat"], tr["like"])
        p_like_wr = det_wr.predict_proba(ev["feat"])

        lik = likelihood_metrics(p_like, ev["like"])
        lik["loud_auc_WITH_rate"] = likelihood_metrics(p_like_wr, ev["like"])["loud_vs_reject_auc"]
        yl = ev["like"] != LIKE_INDETERMINATE
        lik["realized_rate_loud"] = round(float(ev["feat"][yl, 0].mean()), 4)
        lik["realized_rate_reject"] = round(float(ev["feat"][~yl, 0].mean()), 4)
        ids = np.array(sorted(set(int(d) for d in ev["dset"])))
        cut = len(ids) // 2
        cal_ds, test_ds = ids[:cut], ids[cut:]
        prng = RNGState(seed=seed + 31)

        fusion = {}
        for name, rho in (("data_only", None), ("reliable", args.rho_reliable), ("chance", args.rho_chance)):
            prior_a = (np.ones((len(ev["onto"]), N_SUBTYPES)) if rho is None
                       else simulated_prior_alphas(ev["onto"], rho, args.prior_fav_prob, prng.spawn()))
            post = fuse(prior_a, p_like, args.kappa_like)
            ids2, alpha_d, realised_d = per_dataset_composition(post, ev["onto"], ev["w"], ev["dset"])
            pos = {int(v): k for k, v in enumerate(ids2)}
            cal_i = np.array([pos[d] for d in cal_ds]); test_i = np.array([pos[d] for d in test_ds])
            entry = {**{f"cal_{k}": v for k, v in composition_calibration(alpha_d, realised_d, cal_i, test_i).items()},
                     **{f"ab_{k}": v for k, v in abstention_metrics(post, ev["onto"], ev["w"], ev["dset"],
                                                                    cal_ds, test_ds, args.target_acc).items()}}
            if rho is not None:
                entry.update({f"ovr_{k}": v for k, v in override_metrics(prior_a, post, ev["onto"]).items()})
            fusion[name] = entry
        per_seed.append({"likelihood": lik, "fusion": fusion})
        print(f"    loud AUC={lik['loud_vs_reject_auc']} AP={lik['loud_vs_reject_ap']} "
              f"R@P60={lik['loud_recall_at_prec60']} reject_ok={lik['reject_correct_on_silent']}", flush=True)

    lik_keys = ["loud_vs_reject_auc", "loud_auc_WITH_rate", "realized_rate_loud", "realized_rate_reject",
                "loud_vs_reject_ap", "loud_base_rate", "loud_recall_at_prec60",
                "loud_recall_at_prec80", "argmax_recall_threshold", "argmax_recall_detection",
                "reject_correct_on_silent", "threshold_to_detection", "detection_to_threshold"]
    lik_agg = {k: _agg([p["likelihood"] for p in per_seed], k) for k in lik_keys}
    fusion_agg = {}
    for cond in ("data_only", "reliable", "chance"):
        seeds_c = [p["fusion"][cond] for p in per_seed]
        fusion_agg[cond] = {k: _agg(seeds_c, k) for k in sorted(seeds_c[0].keys())}
    bench = benchmark_consistency()

    report = {
        "config": {"seeds": args.seeds, "train_batches": args.train_batches, "eval_batches": args.eval_batches,
                   "kappa_like": args.kappa_like, "prior_fav_prob": args.prior_fav_prob,
                   "rho_reliable": args.rho_reliable, "rho_chance": args.rho_chance,
                   "target_acc": args.target_acc, "mixture": MIXTURE_KWARGS},
        "ontology": {"O": list(SUBTYPES), "L": list(LIKELIHOOD_LABELS),
                     "detectable": [SUBTYPES[i] for i in DETECTABLE_SUBTYPES]},
        "likelihood_mean_sd": lik_agg, "fusion_mean_sd": fusion_agg,
        "real_prior_spec_benchmark_consistency": bench, "per_seed": per_seed,
    }
    out = args.output or Path(f"{BASELINE}/stageQ_subtype_layer.json")
    out.write_text(json.dumps(report, indent=2))

    print("\n" + "=" * 80)
    print(f"STAGE Q — per-column subtype layer ({args.seeds} seeds, mean+/-sd)")
    print("=" * 80)
    print("LIKELIHOOD (deployable-feature detector, data channel; miss-rate EXCLUDED = honest):")
    print("  [rate audit] loud_vs_reject_auc is rate-FREE; loud_auc_WITH_rate is the rate-contaminated")
    print("  upper bound; realized_rate_loud<reject shows the non-transferable generator residual.")
    for k in lik_keys:
        print(f"  {k:26s}: {lik_agg[k]}")
    print("FUSION (subtype-composition calibration + per-column abstention):")
    for cond in ("data_only", "reliable", "chance"):
        f = fusion_agg[cond]
        print(f"  [{cond}] comp_L1 {f['cal_composition_l1']} | query_ECE {f['cal_query_ece']} | "
              f"Brier {f['cal_brier']} (prior-only {f['cal_brier_prior_only']})")
        print(f"            coverage @acc.6/.7/.8 {f['ab_cov_at_60']}/{f['ab_cov_at_70']}/{f['ab_cov_at_80']} | "
              f"committed_acc@{args.target_acc} {f['ab_committed_acc']} | abstain_mass {f['ab_abstain_mass']}")
        print(f"            committed subtype-acc: loud {f['ab_acc_loud']} | silent {f['ab_acc_silent']}")
        if "ovr_override_rate" in f:
            print(f"            override(data beats wrong prior on loud): {f['ovr_override_rate']}")
    print(f"real subtype-prior spec vs benchmark gold_subtype: {bench}")
    print("=" * 80)
    print(f"Wrote -> {out}")


if __name__ == "__main__":
    main()
