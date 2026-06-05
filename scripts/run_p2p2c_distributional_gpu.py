"""
scripts/run_p2p2c_distributional_gpu.py

GATE 2 — full-scale GPU distributional-stream A/B (PI-mandated; only after Gate-1 stream-live PASS).
The bounded CPU run is a smoke test, NOT evidence about the architecture; THIS is the real test.

Full/approved model size (hidden 128 / evidence 64 / 4 layers), 8 TRAIN survey datasets (broader
than the CPU run's 4), held-out leave-datasets-out split, MULTIPLE SEEDS, more epochs/patience. The
only moving part within an A/B cell is `rep_ecdf_pooling` (off = legacy mean pool = current learned
baseline; on = learned ECDF order-statistic pool). Both keep encoder evidence + the fixed 17-dim
value-ECDF. 1B conditional-reference excluded. Same δ-bins / RPS / calibration / leakage gate /
manifest. LR-on-features is reported as a BASELINE (not a ceiling — PI reframe).

Interpretation bar (PI): if stream-on closes the gap toward / beats the LR baseline on the DETECTABLE
idiom (LOD) while own-value stays flat, the architecture idea has legs; if Gate-1 passed and full-
scale still fails ACROSS SEEDS, that is much stronger evidence the current stream is insufficient.

Run: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u scripts/run_p2p2c_distributional_gpu.py
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
from lacuna.survey.lod_generator import generate_lod_example
from lacuna.survey.loss import rps_loss, uniform_rps
from lacuna.survey.run_manifest import write_manifest
from lacuna.survey.train import TrainConfig, _forward_examples, _make_examples, train_delta_prior

# Leave-datasets-out: 8 train / 2 val / 2 test (test held identical to prior A/Bs for comparability).
TRAIN = ["survey_bfi", "survey_cars93", "survey_computers", "survey_cps1988",
         "survey_psid1976", "survey_psid7682", "survey_survey", "survey_yrbss"]
VAL = ["survey_chile", "survey_hmda"]
TEST = ["survey_cps1985", "survey_workinghours"]
BINARY_GRID = [0.0, 2.5]
COARSE3_GRID = [0.0, 0.4, 0.75, 1.25, 1.75, 2.5]
SEEDS = [2026, 7, 99]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EVAL_SEED = 777


def _src(family, pool):
    return LODSurveyExampleSource(pool, tau_quantile=0.70) if family == "lod" else SurveyExampleSource(pool)


def _cfg(scheme, stream_on):
    grid = BINARY_GRID if scheme == "binary" else COARSE3_GRID
    return TrainConfig(
        delta_grid=grid, beta1_range=(0.0, 2.0), target_rate=0.3,
        max_rows=384, max_cols=32, batch_size=16,
        train_batches_per_epoch=40, max_epochs=22, patience=6, val_size=160, test_size=160,
        hidden_dim=128, evidence_dim=64, n_layers=4, n_heads=4, dropout=0.1,
        target_conditioned=True, consequence_features=True,
        rep_ecdf_pooling=stream_on, n_shape_probes=4, coarse_scheme=scheme, model_kind="auto",
    )


def _base_rate_rps(labels, num_bins):
    marg = torch.bincount(labels, minlength=num_bins).float()
    marg = (marg / marg.sum()).clamp(min=1e-9)
    logits = torch.log(marg).unsqueeze(0).repeat(len(labels), 1)
    return float(rps_loss(logits, labels).item())


def _arm(cat, scheme, family, stream_on, seed, git):
    cfg = _cfg(scheme, stream_on)
    tr, va, te = (_src(family, [cat.load(n) for n in P]) for P in (TRAIN, VAL, TEST))
    stream = "ecdf" if stream_on else "mean"
    torch.manual_seed(seed)
    kind = "main" if (scheme == "coarse3" and family == "lod") else "ablation"
    t0 = time.time()
    out = train_delta_prior(tr, va, te, cfg, RNGState(seed=seed), kind=kind,
                            run_id=f"p2p2c-gpu-{scheme}-{stream}-{family}-s{seed}",
                            git_commit=git, timestamp="2026-06-05T12:00:00Z", device=DEVICE)
    out["manifest"]["wall_clock_seconds"] = round(time.time() - t0, 1)
    num_bins = scheme_num_bins(scheme)
    # fixed held-out eval set (same examples across all arms for comparability)
    test_ex = _make_examples(_src(family, [cat.load(n) for n in TEST]), cfg, 200,
                             RNGState(seed=EVAL_SEED), stratify=True)
    logits, labels, _, _ = _forward_examples(out["model"], test_ex, cfg, scheme)
    T = out["manifest"]["temperature"]
    probs = torch.softmax(logits / T, dim=-1)
    per = rps_loss(logits / T, labels, reduction="none")
    rps_m, se = float(per.mean()), float(per.std(unbiased=True) / (len(per) ** 0.5))
    row = {"scheme": scheme, "stream": stream, "family": family, "seed": seed, "num_bins": num_bins,
           "rps": rps_m, "rps_se": se, "uniform_rps": uniform_rps(num_bins),
           "base_rate_rps": _base_rate_rps(labels, num_bins), "bin_acc": M.bin_accuracy(probs, labels),
           "entropy_bits": M.mean_predictive_entropy(probs), "leakage_pass": out["leakage_pass"],
           "wall": out["manifest"]["wall_clock_seconds"], "epochs": out["results"]["epochs_run"]}
    row["base_minus_rps_over_se"] = (row["base_rate_rps"] - rps_m) / se if se > 0 else 0.0
    if num_bins == 2:
        row["auc"] = float(roc_auc_score(labels.numpy(), probs[:, 1].numpy()))
    write_manifest(Path(f"runs/{out['manifest']['run_id']}.json"), out["manifest"])
    del out
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return row


# ---- LR-on-features BASELINE (decoupled dataset/δ) ----
def _lr_build(pool, grid, rng, scheme, n=360):
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


def _agg(rows, key):
    v = np.array([r[key] for r in rows])
    return float(v.mean()), float(v.std())


def main():
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    Path("runs").mkdir(exist_ok=True)
    cat = create_default_catalog()
    print("=" * 104)
    print(f"GATE 2 — FULL-SCALE GPU DISTRIBUTIONAL-STREAM A/B  device={DEVICE} seeds={SEEDS}")
    print(f"model=128/64/4L  train={len(TRAIN)} datasets  held-out test={TEST}")
    print("=" * 104)
    lr_bin = _lr_baseline(cat, "binary"); lr_c3 = _lr_baseline(cat, "coarse3")
    print(f"LR-on-features BASELINE: binary OOF AUC={lr_bin['oof_auc']:.3f}; "
          f"coarse3 OOF acc={lr_c3['oof_acc']:.3f} (chance 0.333)\n")

    rows = []
    plan = [(s, f, st) for s in ("binary", "coarse3") for f in ("lod", "ownvalue") for st in (False, True)]
    for scheme, fam, st in plan:
        cell = []
        for seed in SEEDS:
            r = _arm(cat, scheme, fam, st, seed, git)
            cell.append(r); rows.append(r)
        stream = "ecdf" if st else "mean"
        if scheme == "binary":
            am, asd = _agg(cell, "auc")
            print(f"  binary  {fam:9} stream={stream:4} | AUC {am:.3f}±{asd:.3f}  "
                  f"(seeds {[round(c['auc'],3) for c in cell]})  leak={all(c['leakage_pass'] for c in cell)}  "
                  f"{[c['epochs'] for c in cell]}ep")
        else:
            bm, bsd = _agg(cell, "base_minus_rps_over_se")
            rm, _ = _agg(cell, "rps")
            print(f"  coarse3 {fam:9} stream={stream:4} | rps {rm:.4f} (base-rps)/SE {bm:+.2f}±{bsd:.2f}  "
                  f"leak={all(c['leakage_pass'] for c in cell)}")

    def cell(scheme, fam, stream):
        return [r for r in rows if r["scheme"] == scheme and r["family"] == fam and r["stream"] == stream]

    print("\n" + "=" * 104)
    print("VERDICT (full-scale, multi-seed; does the stream move the held-out LEARNED channel?)")
    print("=" * 104)
    lod_off, lod_on = _agg(cell("binary", "lod", "mean"), "auc"), _agg(cell("binary", "lod", "ecdf"), "auc")
    ov_off, ov_on = _agg(cell("binary", "ownvalue", "mean"), "auc"), _agg(cell("binary", "ownvalue", "ecdf"), "auc")
    d = lod_on[0] - lod_off[0]
    print(f"  BINARY LOD AUC: mean-pool {lod_off[0]:.3f}±{lod_off[1]:.3f} -> ECDF {lod_on[0]:.3f}±{lod_on[1]:.3f} "
          f"(Δ={d:+.3f}); LR baseline {lr_bin['oof_auc']:.3f}")
    print(f"  BINARY own-value AUC: mean-pool {ov_off[0]:.3f} -> ECDF {ov_on[0]:.3f} (should stay ~chance)")
    closes_gap = d >= 0.03 and lod_on[0] >= lr_bin["oof_auc"] - 0.03
    ov_flat = ov_on[0] <= 0.60
    if closes_gap and ov_flat:
        verdict = ("LEGS: full-scale stream-on closes the gap toward / reaches the LR baseline on LOD "
                   "while own-value stays flat. The distributional-consequence inductive bias helps.")
    elif d >= 0.03 and not ov_flat:
        verdict = "LOD up but own-value also moved -> inspect leakage/generality before any claim."
    else:
        verdict = ("INSUFFICIENT: Gate-1 passed (stream live+used) yet full-scale multi-seed stream-on "
                   "does NOT materially beat mean-pool or reach the LR baseline on LOD. Stronger evidence "
                   "the current distributional stream is insufficient -> domain-randomization / "
                   "detectability-map (DECISION-MEMO §9), not more pooling variants.")
    print(f"  VERDICT: {verdict}")
    Path("runs/p2p2c-distributional-gpu-summary.json").write_text(json.dumps(
        {"git": git, "device": DEVICE, "seeds": SEEDS, "lr_binary": lr_bin, "lr_coarse3": lr_c3,
         "rows": rows, "verdict": verdict}, indent=2))
    print("\nsaved: runs/p2p2c-distributional-gpu-summary.json")


if __name__ == "__main__":
    main()
