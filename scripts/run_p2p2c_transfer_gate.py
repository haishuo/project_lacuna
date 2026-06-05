"""
scripts/run_p2p2c_transfer_gate.py

P2.2c — NO-TRAINING LR TRANSFER GATE (PROPOSAL transfer-features-note). Run BEFORE any model
training. Tests whether the frozen transfer-robust consequence features TRANSFER leave-datasets-out
(the property the previous 17 features lacked: OOF AUC 0.43). A logistic regression on the features
alone is the arbiter: proceed to the neural A/B ONLY if the LR transfers out-of-family
(OOF binary δ0-vs-δ2.5 AUC >= GATE_AUC). Reports in-distribution AND leave-datasets-out, per-dataset
OOF, coefficient signs, and direction-flip — for BOTH the old 17 features and the new transfer set.
Run: python -u scripts/run_p2p2c_transfer_gate.py
"""

import json
import subprocess
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from lacuna.core.rng import RNGState
from lacuna.data.catalog import create_default_catalog
from lacuna.survey.consequence_features import compute_consequence_features
from lacuna.survey.delta_bins import assign_delta_bin
from lacuna.survey.lod_generator import generate_lod_example
from lacuna.survey.transfer_features import FEATURE_NAMES, compute_transfer_features

TRAIN = ["survey_cps1988", "survey_yrbss", "survey_computers", "survey_psid7682"]
TEST = ["survey_cps1985", "survey_workinghours"]
GRID = [0.0, 0.15, 0.4, 0.75, 1.25, 1.75, 2.5]
GATE_AUC = 0.65


def _build(pool, deltas, rng, feat_fn, n=300):
    X, yb, y7, ds = [], [], [], []
    for i in range(n):
        d = deltas[i % len(deltas)]
        raw = pool[i % len(pool)]
        res = generate_lod_example(raw, beta1=1.0, delta=d, target_rate=0.3,
                                   tau_quantile=0.70, rng=rng.spawn())
        X.append(feat_fn(res.x_observed, res.mask, res.answer_sheet.target_col_idx).numpy())
        yb.append(0 if d == 0.0 else 1)
        y7.append(assign_delta_bin(d))
        ds.append(raw.name)
    return np.asarray(X), np.asarray(yb), np.asarray(y7), np.asarray(ds)


def _auc(Xtr, ytr, Xte, yte):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=5000).fit(sc.transform(Xtr), ytr)
    return float(roc_auc_score(yte, clf.predict_proba(sc.transform(Xte))[:, 1])), sc, clf


def _evaluate(name, feat_fn, cat):
    tr = [cat.load(n) for n in TRAIN]
    te = [cat.load(n) for n in TEST]
    # binary δ0-vs-δ2.5
    Xtr, ybtr, _, _ = _build(tr, [0.0, 2.5], RNGState(seed=11), feat_fn)
    Xte, ybte, _, dste = _build(te, [0.0, 2.5], RNGState(seed=12), feat_fn)
    Xi, ybi, _, _ = _build(te, [0.0, 2.5], RNGState(seed=13), feat_fn)
    Xj, ybj, _, _ = _build(te, [0.0, 2.5], RNGState(seed=14), feat_fn)
    auc_oof, sc, clf = _auc(Xtr, ybtr, Xte, ybte)
    auc_id, _, _ = _auc(Xi, ybi, Xj, ybj)
    # per held-out dataset OOF
    per_ds = {}
    for nm in TEST:
        m = dste == nm
        if m.sum() > 5 and len(set(ybte[m])) == 2:
            per_ds[nm] = float(roc_auc_score(ybte[m], clf.predict_proba(sc.transform(Xte[m]))[:, 1]))
    # 7-bin OOF accuracy
    X7, _, y7tr, _ = _build(tr, GRID, RNGState(seed=21), feat_fn)
    Xt7, _, y7te, _ = _build(te, GRID, RNGState(seed=22), feat_fn)
    sc7 = StandardScaler().fit(X7)
    acc7 = float(LogisticRegression(max_iter=5000).fit(sc7.transform(X7), y7tr).score(sc7.transform(Xt7), y7te))
    return {
        "name": name, "auc_in_distribution": auc_id, "auc_oof": auc_oof,
        "direction_flips": auc_oof < 0.5, "per_dataset_oof": per_ds, "acc7_oof": acc7,
        "binary_coef_signs": [int(np.sign(c)) for c in clf.coef_[0]],
    }


def main():
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    cat = create_default_catalog()
    print("=" * 92)
    print("P2.2c NO-TRAINING LR TRANSFER GATE  (binary δ=0 vs δ=2.5; train=%s test=%s)" % (TRAIN, TEST))
    print(f"  GATE: proceed to neural A/B only if transfer-feature OOF AUC >= {GATE_AUC}")
    print("=" * 92)
    results = []
    for name, fn in (("old_17_marginal", compute_consequence_features),
                     ("new_transfer", compute_transfer_features)):
        r = _evaluate(name, fn, cat)
        results.append(r)
        print(f"\n[{name}]")
        print(f"  binary AUC: in-distribution={r['auc_in_distribution']:.3f}   "
              f"LEAVE-DATASETS-OUT={r['auc_oof']:.3f}   direction_flips={r['direction_flips']}")
        print(f"  per-held-out-dataset OOF AUC: {{{', '.join(f'{k}:{v:.3f}' for k,v in r['per_dataset_oof'].items())}}}")
        print(f"  7-bin OOF acc={r['acc7_oof']:.3f} (chance {1/7:.3f})")

    new = next(r for r in results if r["name"] == "new_transfer")
    old = next(r for r in results if r["name"] == "old_17_marginal")
    gate_pass = new["auc_oof"] >= GATE_AUC
    print("\n" + "=" * 92)
    print("GATE VERDICT")
    print("=" * 92)
    print(f"  old-17 OOF AUC = {old['auc_oof']:.3f}  ->  new-transfer OOF AUC = {new['auc_oof']:.3f}  "
          f"(gate {GATE_AUC})")
    if gate_pass:
        verdict = f"PASS — transfer features generalize OOF (AUC {new['auc_oof']:.3f} >= {GATE_AUC}). Proceed to the neural A/B."
    else:
        verdict = (f"FAIL — transfer features do NOT generalize OOF (AUC {new['auc_oof']:.3f} < {GATE_AUC}). "
                   f"STOP; do NOT run the neural A/B; do NOT blame the encoder.")
    print(f"  VERDICT: {verdict}")
    print(f"  coefficient signs (transfer, binary): "
          f"{dict(zip(FEATURE_NAMES, new['binary_coef_signs']))}")

    Path("runs").mkdir(exist_ok=True)
    Path("runs/p2p2c-transfer-gate.json").write_text(json.dumps(
        {"git": git, "gate_auc": GATE_AUC, "gate_pass": gate_pass, "results": results,
         "verdict": verdict}, indent=2))
    print("\nsaved: runs/p2p2c-transfer-gate.json")


if __name__ == "__main__":
    main()
