"""
scripts/run_p2p2c_coarse_ab.py

P2.2c — Coarse / curriculum δ A/B (PROPOSAL-P2.2c coarse audit). Tests whether the learned channel
can recover the coarse, transferable LOD signal out-of-family, using the EXISTING 17 marginal
features (no new features, no transfer set).

- Step 1 BINARY diagnostic (δ=0 vs δ=2.5): features-only model, LOD vs own-value.
- Step 2 3-BIN main A/B ({δ=0}|weak(0,1]|strong(>1)): features-only model (MAIN) + encoder-features
  (DIAGNOSTIC), LOD vs own-value.
- Every coarse result reported vs base-rate AND the LR-on-17-features baseline (in-dist + OOF).

Same held-out leave-datasets-out split, same leakage gate, same manifest. Run:
python -u scripts/run_p2p2c_coarse_ab.py
"""

import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from lacuna.core.rng import RNGState
from lacuna.data.catalog import create_default_catalog
from lacuna.survey.coarse_bins import assign_bins, scheme_num_bins
from lacuna.survey.consequence_features import compute_consequence_features
from lacuna.survey.example_source import LODSurveyExampleSource, SurveyExampleSource
from lacuna.survey.loss import rps_loss, uniform_rps
from lacuna.survey import metrics as M
from lacuna.survey.run_manifest import write_manifest
from lacuna.survey.train import TrainConfig, _forward_examples, _make_examples, train_delta_prior

TRAIN = ["survey_cps1988", "survey_yrbss", "survey_computers", "survey_psid7682"]
VAL = ["survey_chile", "survey_hmda"]
TEST = ["survey_cps1985", "survey_workinghours"]
BINARY_GRID = [0.0, 2.5]
COARSE3_GRID = [0.0, 0.4, 0.75, 1.25, 1.75, 2.5]


def _src(family, pool):
    return LODSurveyExampleSource(pool, tau_quantile=0.70) if family == "lod" else SurveyExampleSource(pool)


def _cfg(scheme, model_kind):
    grid = BINARY_GRID if scheme == "binary" else COARSE3_GRID
    fo = model_kind == "features_only"
    return TrainConfig(
        delta_grid=grid, beta1_range=(0.0, 2.0), target_rate=0.3,
        max_rows=512, max_cols=12, batch_size=16,
        train_batches_per_epoch=(30 if fo else 40), max_epochs=(25 if fo else 14),
        patience=(6 if fo else 5), val_size=140, test_size=140,
        hidden_dim=96, evidence_dim=48, n_layers=2, n_heads=4,
        target_conditioned=(not fo), consequence_features=(not fo),
        coarse_scheme=scheme, model_kind=model_kind, head_hidden_dim=(32 if fo else None),
    )


def _base_rate_rps(labels, num_bins):
    marg = torch.bincount(labels, minlength=num_bins).float()
    marg = (marg / marg.sum()).clamp(min=1e-9)
    logits = torch.log(marg).unsqueeze(0).repeat(len(labels), 1)
    return float(rps_loss(logits, labels).item())


def _arm(cat, scheme, model_kind, family, git):
    cfg = _cfg(scheme, model_kind)
    tr, va, te = (_src(family, [cat.load(n) for n in P]) for P in (TRAIN, VAL, TEST))
    t0 = time.time()
    out = train_delta_prior(tr, va, te, cfg, RNGState(seed=2026),
                            kind=("ablation" if scheme == "binary" else "main"),
                            run_id=f"p2p2c-coarse-{scheme}-{model_kind}-{family}",
                            git_commit=git, timestamp="2026-06-04T12:00:00Z")
    out["manifest"]["wall_clock_seconds"] = round(time.time() - t0, 1)
    num_bins = scheme_num_bins(scheme)
    # fresh held-out eval for per-example SE / AUC
    test_ex = _make_examples(_src(family, [cat.load(n) for n in TEST]), cfg, 140,
                             RNGState(seed=777), stratify=True)
    logits, labels, _, _ = _forward_examples(out["model"], test_ex, cfg, scheme)
    T = out["manifest"]["temperature"]
    probs = torch.softmax(logits / T, dim=-1)
    per = rps_loss(logits / T, labels, reduction="none")
    rps_m, se = float(per.mean()), float(per.std(unbiased=True) / (len(per) ** 0.5))
    row = {
        "scheme": scheme, "model_kind": model_kind, "family": family,
        "num_bins": num_bins, "rps": rps_m, "rps_se": se,
        "uniform_rps": uniform_rps(num_bins), "base_rate_rps": _base_rate_rps(labels, num_bins),
        "bin_acc": M.bin_accuracy(probs, labels), "adj_acc": M.adjacent_accuracy(probs, labels),
        "entropy_bits": M.mean_predictive_entropy(probs), "ece": M.ece(probs, labels)["ece"],
        "leakage_pass": out["leakage_pass"], "wall": out["manifest"]["wall_clock_seconds"],
    }
    row["uni_minus_rps_over_se"] = (row["uniform_rps"] - rps_m) / se if se > 0 else 0.0
    row["base_minus_rps_over_se"] = (row["base_rate_rps"] - rps_m) / se if se > 0 else 0.0
    if num_bins == 2:
        row["auc"] = float(roc_auc_score(labels.numpy(), probs[:, 1].numpy()))
    write_manifest(Path(f"runs/{out['manifest']['run_id']}.json"), out["manifest"])
    return row


# ---- LR-on-features baseline (decoupled dataset/δ) ----
def _lr_build(pool, grid, rng, scheme, n=300):
    X, y, ds = [], [], []
    ds_rng = rng.spawn()
    for i in range(n):
        d = grid[i % len(grid)]
        raw = pool[int(ds_rng.randint(0, len(pool), (1,)).item())]
        from lacuna.survey.lod_generator import generate_lod_example
        res = generate_lod_example(raw, beta1=1.0, delta=d, target_rate=0.3,
                                   tau_quantile=0.70, rng=rng.spawn())
        X.append(compute_consequence_features(res.x_observed, res.mask, res.answer_sheet.target_col_idx).numpy())
        y.append(int(assign_bins(scheme, torch.tensor([d]))[0]))
        ds.append(raw.name)
    return np.asarray(X), np.asarray(y), np.asarray(ds)


def _lr_baseline(cat, scheme):
    grid = BINARY_GRID if scheme == "binary" else COARSE3_GRID
    tr = [cat.load(n) for n in TRAIN]; te = [cat.load(n) for n in TEST]
    Xtr, ytr, _ = _lr_build(tr, grid, RNGState(seed=31), scheme)
    Xte, yte, _ = _lr_build(te, grid, RNGState(seed=32), scheme)
    Xi, yi, _ = _lr_build(te, grid, RNGState(seed=33), scheme)
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=5000).fit(sc.transform(Xtr), ytr)
    sci = StandardScaler().fit(Xi)
    clfi = LogisticRegression(max_iter=5000).fit(sci.transform(Xi), yi)
    out = {"scheme": scheme, "oof_acc": float(clf.score(sc.transform(Xte), yte)),
           "indist_acc": float(clfi.score(sci.transform(Xte), yte))}
    if scheme == "binary":
        out["oof_auc"] = float(roc_auc_score(yte, clf.predict_proba(sc.transform(Xte))[:, 1]))
    return out


def main():
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    Path("runs").mkdir(exist_ok=True)
    cat = create_default_catalog()
    print("=" * 100)
    print("P2.2c COARSE / CURRICULUM δ A/B  (held-out; train=%s test=%s)" % (TRAIN, TEST))
    print("=" * 100)

    lr_bin = _lr_baseline(cat, "binary")
    lr_c3 = _lr_baseline(cat, "coarse3")
    print(f"LR-on-17-features baseline: binary OOF AUC={lr_bin['oof_auc']:.3f} (in-dist acc {lr_bin['indist_acc']:.3f}); "
          f"coarse3 OOF acc={lr_c3['oof_acc']:.3f} (in-dist {lr_c3['indist_acc']:.3f}, chance 0.333)")

    rows = []
    plan = [
        ("binary", "features_only", "lod"), ("binary", "features_only", "ownvalue"),
        ("coarse3", "features_only", "lod"), ("coarse3", "features_only", "ownvalue"),
        ("coarse3", "auto", "lod"), ("coarse3", "auto", "ownvalue"),
    ]
    for scheme, mk, fam in plan:
        r = _arm(cat, scheme, mk, fam, git)
        rows.append(r)
        extra = f" AUC={r['auc']:.3f}" if "auc" in r else ""
        tag = "MAIN" if mk == "features_only" else "diag"
        print(f"  [{tag}] {scheme:8} {mk:14} {fam:9} | rps={r['rps']:.4f} "
              f"(uni-rps)/SE={r['uni_minus_rps_over_se']:+.2f} (base-rps)/SE={r['base_minus_rps_over_se']:+.2f} "
              f"bin_acc={r['bin_acc']:.3f} adj={r['adj_acc']:.3f} H={r['entropy_bits']:.2f}{extra} "
              f"leak={r['leakage_pass']} ({r['wall']}s)")

    def g(scheme, mk, fam):
        return next(r for r in rows if r["scheme"] == scheme and r["model_kind"] == mk and r["family"] == fam)

    print("\n" + "=" * 100); print("VERDICT (coarse spectrum in the held-out learned channel)"); print("=" * 100)
    b_lod, b_ov = g("binary", "features_only", "lod"), g("binary", "features_only", "ownvalue")
    c_lod_fo, c_ov_fo = g("coarse3", "features_only", "lod"), g("coarse3", "features_only", "ownvalue")
    c_lod_en = g("coarse3", "auto", "lod")
    print(f"  BINARY features-only: LOD AUC={b_lod['auc']:.3f} vs own-value AUC={b_ov['auc']:.3f} "
          f"(LR OOF AUC {lr_bin['oof_auc']:.3f})")
    lod_learns = c_lod_fo["uni_minus_rps_over_se"] > 2 and c_lod_fo["base_minus_rps_over_se"] > 2
    ov_flat = c_ov_fo["uni_minus_rps_over_se"] < 2
    print(f"  3-BIN features-only (MAIN): LOD beats uniform&base by >2SE? {lod_learns}  "
          f"own-value flat? {ov_flat}")
    print(f"  3-BIN encoder-features (diag): LOD (uni-rps)/SE={c_lod_en['uni_minus_rps_over_se']:+.2f} "
          f"(integration check vs features-only {c_lod_fo['uni_minus_rps_over_se']:+.2f})")
    if lod_learns and ov_flat:
        verdict = ("SPECTRUM DEMONSTRATED IN THE HELD-OUT LEARNED CHANNEL AT COARSE RESOLUTION: "
                   "features-only 3-bin LOD learns out-of-family while own-value stays flat.")
    elif b_lod["auc"] > 0.65 and not lod_learns:
        verdict = ("Binary coarse signal recovered but 3-bin not beating refs -> presence/absence of "
                   "strong truncation, not yet calibrated 3-bin magnitude.")
    else:
        verdict = "Coarse LOD not recovered by the learned channel -> revisit gate/integration."
    print(f"  VERDICT: {verdict}")

    Path("runs/p2p2c-coarse-summary.json").write_text(json.dumps(
        {"git": git, "lr_binary": lr_bin, "lr_coarse3": lr_c3, "rows": rows, "verdict": verdict}, indent=2))
    print("\nsaved: runs/p2p2c-coarse-*.json")


if __name__ == "__main__":
    main()
