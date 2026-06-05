"""
scripts/run_p2p2c_lod_oracle.py

P2.2c — ORACLE-FIRST gate (PROPOSAL-P2.2c §3). Run BEFORE any model training.

Computes the Bayes-optimal n-sample error of distinguishing LOD/top-coding(δ) from a matched-rate
MAR null (point AND β₁′-profiled) under the KNOWN generative model, on (a) synthetic Gaussian X and
(b) ConditionalGaussian X-models FIT to real survey (target, predictor) columns. By Neyman–Pearson
no estimator can beat this, so:
  - Bayes error << 0.5 at moderate/large δ on real-fitted X  => signal EXISTS => proceed to the
    model ladder (next cycle).
  - Bayes error ≈ 0.5 even at large δ                          => LOD also flat on real-fitted X =>
    a deeper finding; STOP, do not train-and-blame.
The fitted Gaussian is the simplest stated real-column X-model (explicit assumption); the oracle is a
GATE, not a final product claim. Run: python -u scripts/run_p2p2c_lod_oracle.py
"""

import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from lacuna.core.rng import RNGState
from lacuna.data.catalog import create_default_catalog
from lacuna.data.semisynthetic import _zscore_columns
from lacuna.feasibility.xmodel import ConditionalGaussian
from lacuna.survey.lod_oracle import lod_oracle_cell

POOL = ["survey_cps1988", "survey_psid7682", "survey_computers",
        "survey_hmda", "survey_cps1985", "survey_workinghours"]
DELTAS = [0.0, 1.0, 2.0, 3.0]
TAU_Q = 0.70
RATE = 0.30
N = 1024
N_MC = 1200
N_NODES = 96


def _fit_xmodel(raw):
    """Pick a continuous target (highest cardinality) + its most-correlated predictor; fit Gaussian."""
    X = torch.from_numpy(raw.data.astype("float32"))
    Z = _zscore_columns(X)
    d = X.shape[1]
    std = X.std(dim=0, unbiased=False)
    nonconst = [c for c in range(d) if float(std[c]) > 0]
    card = {c: int(np.unique(raw.data[:, c]).size) for c in nonconst}
    t = max(nonconst, key=lambda c: card[c])  # continuous-ish target
    best_p, best_abs = None, -1.0
    zt = Z[:, t]
    for c in nonconst:
        if c == t:
            continue
        zc = Z[:, c]
        corr = float(((zt - zt.mean()) * (zc - zc.mean())).sum() / (zt.std(unbiased=False) * zc.std(unbiased=False) * len(zt)))
        if abs(corr) > best_abs:
            best_abs, best_p = abs(corr), c
    xm = ConditionalGaussian.fit(Z[:, best_p], Z[:, t])
    return xm, {"target": raw.feature_names[t], "target_card": card[t],
                "predictor": raw.feature_names[best_p], "abs_corr": round(best_abs, 3)}


def _sweep(label, xmodel, meta, rng):
    rows = []
    for i, delta in enumerate(DELTAS):
        c = lod_oracle_cell(xmodel, delta=delta, beta1=1.0, tau_quantile=TAU_Q, target_rate=RATE,
                            n=N, rng=rng.spawn(), n_mc=N_MC, n_nodes=N_NODES)
        rows.append(c)
        print(f"  [{label:22}] δ={delta:<4} point_BE={c['point_bayes_error']:.3f}±{c['point_se']:.3f}  "
              f"profiled_BE={c['profiled_bayes_error']:.3f}±{c['profiled_se']:.3f}")
    return {"label": label, "meta": meta, "cells": rows}


def main():
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    rng = RNGState(seed=2026)
    t0 = time.time()
    results = []

    print("=" * 92)
    print(f"P2.2c ORACLE GATE — LOD/top-coding Bayes error vs MAR (matched rate={RATE}, τ_q={TAU_Q}, n={N})")
    print("=" * 92)
    print("SYNTHETIC (controlled ρ):")
    for rho in (0.0, 0.3, 0.6):
        xm = ConditionalGaussian.synthetic(rho)
        results.append(_sweep(f"synthetic ρ={rho}", xm, {"rho": rho}, rng.spawn()))

    print("\nREAL-FITTED ConditionalGaussian X-models (continuous target, most-correlated predictor):")
    cat = create_default_catalog()
    for name in POOL:
        raw = cat.load(name)
        xm, meta = _fit_xmodel(raw)
        print(f"  ({name}: target={meta['target']} card={meta['target_card']} "
              f"pred={meta['predictor']} |corr|={meta['abs_corr']})")
        results.append(_sweep(name, xm, meta, rng.spawn()))

    # ---- gate verdict ----
    def max_delta_be(res):  # profiled BE at the largest δ
        return res["cells"][-1]["profiled_bayes_error"]
    real = [r for r in results if not r["label"].startswith("synthetic")]
    real_detectable = [r["label"] for r in real if max_delta_be(r) < 0.40]
    print("\n" + "=" * 92)
    print("GATE VERDICT")
    print("=" * 92)
    print(f"  synthetic large-δ profiled BE: "
          f"{[round(max_delta_be(r), 3) for r in results if r['label'].startswith('synthetic')]}")
    print(f"  real-fitted large-δ profiled BE: {[(r['label'], round(max_delta_be(r), 3)) for r in real]}")
    n_detect = len(real_detectable)
    if n_detect >= max(1, len(real) // 2):
        verdict = (f"SIGNAL EXISTS on real-fitted X ({n_detect}/{len(real)} datasets with profiled BE<0.40 "
                   f"at δ={DELTAS[-1]}) -> PROCEED to the model ladder (next cycle).")
    elif n_detect >= 1:
        verdict = (f"PARTIAL ({n_detect}/{len(real)} detectable) -> report; proceed cautiously on the "
                   f"detectable subset.")
    else:
        verdict = ("FLAT even at the oracle on real-fitted X -> LOD non-identifiable under the fitted "
                   "Gaussian model at matched rate. STOP; do not train-and-blame. A deeper result.")
    print(f"  VERDICT: {verdict}")
    print(f"  (oracle is under the fitted-Gaussian X-model — explicit assumption; a GATE, not a final claim.)")
    print(f"  wall={time.time()-t0:.0f}s")

    Path("runs").mkdir(exist_ok=True)
    Path("runs/p2p2c-lod-oracle.json").write_text(json.dumps(
        {"git": git, "config": {"deltas": DELTAS, "tau_q": TAU_Q, "rate": RATE, "n": N, "n_mc": N_MC},
         "results": results, "verdict": verdict}, indent=2))
    print("saved: runs/p2p2c-lod-oracle.json")


if __name__ == "__main__":
    main()
