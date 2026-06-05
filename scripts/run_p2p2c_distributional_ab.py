"""
scripts/run_p2p2c_distributional_ab.py

Distributional-consequence stream A/B (PROPOSAL-distributional-consequence-stream-audit §8).
Tests the PI-approved bounded architecture experiment: does a learned ECDF / order-statistic pooling
branch (Proposal B, replacing the across-row MEAN in the target-conditioned head) move the HELD-OUT
LEARNED channel off the floor for the DETECTABLE idiom (LOD) while the FLAT idiom (own-value) stays
flat?

The ONLY moving part between the two arms is `rep_ecdf_pooling`:
  - off  = legacy masked-MEAN pool  → reproduces the current learned-channel baseline (the floor).
  - on   = rep-ECDF order-statistic pool (m=4 probes, Q=6 quantiles + max).
Both arms keep the encoder evidence AND the fixed 17-dim value-ECDF (`consequence_features`) — the
strongest reasonable version (PI decision 3). Same held-out leave-datasets-out split, same δ-bins /
RPS / calibration / leakage gate / manifest, same LR-on-features ceiling. 1B conditional-reference is
EXCLUDED (PI decision 2).

Pre-registered success (NOT "beats uniform"): LOD-on improves MATERIALLY vs LOD-off AND reaches the
LR feature-level OOF ceiling, WHILE own-value stays flat. Run: python -u scripts/run_p2p2c_distributional_ab.py
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
from lacuna.survey import metrics as M
from lacuna.survey.coarse_bins import assign_bins, scheme_num_bins
from lacuna.survey.consequence_features import compute_consequence_features
from lacuna.survey.example_source import LODSurveyExampleSource, SurveyExampleSource
from lacuna.survey.loss import rps_loss, uniform_rps
from lacuna.survey.lod_generator import generate_lod_example
from lacuna.survey.run_manifest import write_manifest
from lacuna.survey.train import TrainConfig, _forward_examples, _make_examples, train_delta_prior

TRAIN = ["survey_cps1988", "survey_yrbss", "survey_computers", "survey_psid7682"]
VAL = ["survey_chile", "survey_hmda"]
TEST = ["survey_cps1985", "survey_workinghours"]
BINARY_GRID = [0.0, 2.5]
COARSE3_GRID = [0.0, 0.4, 0.75, 1.25, 1.75, 2.5]


def _src(family, pool):
    return LODSurveyExampleSource(pool, tau_quantile=0.70) if family == "lod" else SurveyExampleSource(pool)


def _cfg(scheme, stream_on):
    grid = BINARY_GRID if scheme == "binary" else COARSE3_GRID
    return TrainConfig(
        delta_grid=grid, beta1_range=(0.0, 2.0), target_rate=0.3,
        max_rows=384, max_cols=12, batch_size=16,
        train_batches_per_epoch=30, max_epochs=14, patience=5, val_size=120, test_size=120,
        hidden_dim=80, evidence_dim=40, n_layers=2, n_heads=4,
        target_conditioned=True, consequence_features=True,
        rep_ecdf_pooling=stream_on, n_shape_probes=4,
        coarse_scheme=scheme, model_kind="auto",
    )


def _base_rate_rps(labels, num_bins):
    marg = torch.bincount(labels, minlength=num_bins).float()
    marg = (marg / marg.sum()).clamp(min=1e-9)
    logits = torch.log(marg).unsqueeze(0).repeat(len(labels), 1)
    return float(rps_loss(logits, labels).item())


def _arm(cat, scheme, stream_on, family, git):
    cfg = _cfg(scheme, stream_on)
    tr, va, te = (_src(family, [cat.load(n) for n in P]) for P in (TRAIN, VAL, TEST))
    stream = "ecdf" if stream_on else "mean"
    t0 = time.time()
    out = train_delta_prior(tr, va, te, cfg, RNGState(seed=2026),
                            kind=("ablation" if scheme == "binary" else "main"),
                            run_id=f"p2p2c-dist-{scheme}-{stream}-{family}",
                            git_commit=git, timestamp="2026-06-05T12:00:00Z")
    out["manifest"]["wall_clock_seconds"] = round(time.time() - t0, 1)
    num_bins = scheme_num_bins(scheme)
    test_ex = _make_examples(_src(family, [cat.load(n) for n in TEST]), cfg, 140,
                             RNGState(seed=777), stratify=True)
    logits, labels, _, _ = _forward_examples(out["model"], test_ex, cfg, scheme)
    T = out["manifest"]["temperature"]
    probs = torch.softmax(logits / T, dim=-1)
    per = rps_loss(logits / T, labels, reduction="none")
    rps_m, se = float(per.mean()), float(per.std(unbiased=True) / (len(per) ** 0.5))
    row = {
        "scheme": scheme, "stream": stream, "family": family, "num_bins": num_bins,
        "rps": rps_m, "rps_se": se, "uniform_rps": uniform_rps(num_bins),
        "base_rate_rps": _base_rate_rps(labels, num_bins),
        "bin_acc": M.bin_accuracy(probs, labels), "adj_acc": M.adjacent_accuracy(probs, labels),
        "entropy_bits": M.mean_predictive_entropy(probs), "ece": M.ece(probs, labels)["ece"],
        "leakage_pass": out["leakage_pass"], "wall": out["manifest"]["wall_clock_seconds"],
    }
    row["base_minus_rps_over_se"] = (row["base_rate_rps"] - rps_m) / se if se > 0 else 0.0
    if num_bins == 2:
        row["auc"] = float(roc_auc_score(labels.numpy(), probs[:, 1].numpy()))
    write_manifest(Path(f"runs/{out['manifest']['run_id']}.json"), out["manifest"])
    return row


# ---- LR-on-features baseline (decoupled dataset/δ; the feature-level ceiling) ----
def _lr_build(pool, grid, rng, scheme, n=300):
    X, y = [], []
    ds_rng = rng.spawn()
    for i in range(n):
        d = grid[i % len(grid)]
        raw = pool[int(ds_rng.randint(0, len(pool), (1,)).item())]
        res = generate_lod_example(raw, beta1=1.0, delta=d, target_rate=0.3,
                                   tau_quantile=0.70, rng=rng.spawn())
        X.append(compute_consequence_features(res.x_observed, res.mask, res.answer_sheet.target_col_idx).numpy())
        y.append(int(assign_bins(scheme, torch.tensor([d]))[0]))
    return np.asarray(X), np.asarray(y)


def _lr_baseline(cat, scheme):
    grid = BINARY_GRID if scheme == "binary" else COARSE3_GRID
    tr = [cat.load(n) for n in TRAIN]; te = [cat.load(n) for n in TEST]
    Xtr, ytr = _lr_build(tr, grid, RNGState(seed=31), scheme)
    Xte, yte = _lr_build(te, grid, RNGState(seed=32), scheme)
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=5000).fit(sc.transform(Xtr), ytr)
    out = {"scheme": scheme, "oof_acc": float(clf.score(sc.transform(Xte), yte))}
    if scheme == "binary":
        out["oof_auc"] = float(roc_auc_score(yte, clf.predict_proba(sc.transform(Xte))[:, 1]))
    return out


def main():
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    Path("runs").mkdir(exist_ok=True)
    cat = create_default_catalog()
    print("=" * 100)
    print("DISTRIBUTIONAL-STREAM A/B (held-out; train=%s test=%s)" % (TRAIN, TEST))
    print("rep-ECDF pooling ON vs OFF; both keep evidence + fixed value-ECDF; 1B excluded")
    print("=" * 100)

    lr_bin = _lr_baseline(cat, "binary")
    lr_c3 = _lr_baseline(cat, "coarse3")
    print(f"LR feature-level ceiling: binary OOF AUC={lr_bin['oof_auc']:.3f}; "
          f"coarse3 OOF acc={lr_c3['oof_acc']:.3f} (chance 0.333)")

    rows = []
    plan = [
        ("binary", False, "lod"), ("binary", True, "lod"),
        ("binary", False, "ownvalue"), ("binary", True, "ownvalue"),
        ("coarse3", False, "lod"), ("coarse3", True, "lod"),
        ("coarse3", False, "ownvalue"), ("coarse3", True, "ownvalue"),
    ]
    for scheme, stream_on, fam in plan:
        r = _arm(cat, scheme, stream_on, fam, git)
        rows.append(r)
        extra = f" AUC={r['auc']:.3f}" if "auc" in r else ""
        print(f"  {scheme:8} stream={'ECDF' if stream_on else 'mean':4} {fam:9} | rps={r['rps']:.4f} "
              f"(base-rps)/SE={r['base_minus_rps_over_se']:+.2f} bin={r['bin_acc']:.3f} "
              f"adj={r['adj_acc']:.3f} H={r['entropy_bits']:.2f}{extra} "
              f"leak={r['leakage_pass']} ({r['wall']}s)")

    def g(scheme, stream, fam):
        return next(r for r in rows if r["scheme"] == scheme and r["stream"] == stream and r["family"] == fam)

    print("\n" + "=" * 100)
    print("VERDICT (does the stream move the held-out LEARNED channel; spectrum LOD vs own-value?)")
    print("=" * 100)
    b_lod_off, b_lod_on = g("binary", "mean", "lod"), g("binary", "ecdf", "lod")
    b_ov_off, b_ov_on = g("binary", "mean", "ownvalue"), g("binary", "ecdf", "ownvalue")
    c_lod_on, c_ov_on = g("coarse3", "ecdf", "lod"), g("coarse3", "ecdf", "ownvalue")
    d_auc = b_lod_on["auc"] - b_lod_off["auc"]
    print(f"  BINARY LOD AUC: mean-pool {b_lod_off['auc']:.3f} -> ECDF {b_lod_on['auc']:.3f} "
          f"(Δ={d_auc:+.3f}; LR ceiling {lr_bin['oof_auc']:.3f})")
    print(f"  BINARY own-value AUC: mean-pool {b_ov_off['auc']:.3f} -> ECDF {b_ov_on['auc']:.3f} "
          f"(should stay ~chance)")
    print(f"  COARSE3 ECDF: LOD (base-rps)/SE={c_lod_on['base_minus_rps_over_se']:+.2f}  "
          f"own-value (base-rps)/SE={c_ov_on['base_minus_rps_over_se']:+.2f}")
    material = d_auc >= 0.05 and b_lod_on["auc"] >= lr_bin["oof_auc"] - 0.03
    ov_flat = b_ov_on["auc"] <= 0.60
    if material and ov_flat:
        verdict = ("STREAM MOVES THE HELD-OUT LEARNED CHANNEL: LOD improves materially and reaches the "
                   "feature-level ceiling while own-value stays flat -> distributional-consequence "
                   "inductive bias represents a detectable consequence when one exists.")
    elif d_auc >= 0.05 and not ov_flat:
        verdict = ("LOD improved but own-value also moved -> stream may be reading an artifact; "
                   "inspect leakage / generality before any claim.")
    else:
        verdict = ("Stream does NOT move the held-out learned channel materially -> stronger evidence "
                   "the bottleneck is the low transfer ceiling / regime, NOT representation. Redirect "
                   "to domain-randomization / detectability-map (DECISION-MEMO §9).")
    print(f"  VERDICT: {verdict}")

    Path("runs/p2p2c-distributional-summary.json").write_text(json.dumps(
        {"git": git, "lr_binary": lr_bin, "lr_coarse3": lr_c3, "rows": rows, "verdict": verdict}, indent=2))
    print("\nsaved: runs/p2p2c-distributional-*.json")


if __name__ == "__main__":
    main()
