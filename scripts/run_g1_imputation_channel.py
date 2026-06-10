"""
scripts/run_g1_imputation_channel.py

G1 — truth-level information test for the MAR-imputation counterfactual channel, executed exactly per
the LOCKED pre-registration (`PREREGISTRATION-G1-imputation-channel.md`, commit 6f1bff9). EVALUATION
INSTRUMENTATION ONLY: LR-level (no neural model training), no proxies, no runtime pathway, no
architecture. Deterministic.

Protocol (locked): 4 domains (labor / nhanes / hmda / wealth, continuous bases), all continuous targets;
24 examples per (dataset, target, idiom, delta), delta in {0.0, 2.5}, matched rate 0.3, beta1=1.0,
max_rows 384; per example a PAIRED matched-rate MCAR mask; 5 fixed imputers; channel = 6 paired
differences per imputer; baseline = frozen 17 consequence features (mech mask). Block-aware
leave-one-domain-out LR, pooled OOF AUC.

Criteria (locked): G1-a own_value channel-only >= 0.65 | G1-b own_value baseline+channel - baseline
>= +0.05 | G1-c idiom separation at delta=2.5 >= 0.65; a,b for the SAME >=2 imputers; PASS = a&b&c.

Run: /home/haishuo/miniconda3/envs/lacuna/bin/python -u scripts/run_g1_imputation_channel.py
"""

import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from lacuna.core.rng import RNGState
from lacuna.data.catalog import create_default_catalog
from lacuna.data.ingestion import RawDataset
from lacuna.data.semisynthetic import subsample_raw
from lacuna.feasibility.delta_generator import apply_self_censor
from lacuna.survey.consequence_features import compute_consequence_features
from lacuna.survey.delta_generator import select_target_predictor
from lacuna.survey.imputation_channel import IMPUTERS, FEATURE_NAMES, imputer_channel_features, mcar_pair_mask
from lacuna.survey.lod_generator import generate_lod_example

RB = Path("/mnt/data/lacuna/role_b")
OUT = Path("runs/g1_imputation_channel.json")
N_EX, MAX_ROWS, RATE, BETA1, TAU_Q = 24, 384, 0.3, 1.0, 0.70
DELTAS = (0.0, 2.5)
IDIOMS = ("top_coding", "own_value")
MIN_CARD = 30
DOMAINS = {
    "labor": ["survey_cps1985", "survey_cps1988", "survey_psid1976", "survey_psid7682", "survey_workinghours"],
    "nhanes": ["rb_nhanes_weight", "rb_nhanes_poverty", "rb_nhanes_income"],
    "hmda": ["survey_hmda"],
    "wealth": ["rb_scf2022_wealth_cont"],
}
cat = create_default_catalog()


def cont_base(name):
    if name.startswith("rb_"):
        df = pd.read_csv(RB / f"{name}.csv")
        mat, feats = df.to_numpy(np.float32), list(df.columns)
    else:
        raw = cat.load(name)
        mat, feats = np.asarray(raw.data, np.float32), list(raw.feature_names)
    idx = [j for j in range(mat.shape[1])
           if np.unique(mat[:, j][np.isfinite(mat[:, j])]).size >= MIN_CARD]
    if len(idx) < 2:
        raise ValueError(f"{name}: <2 continuous columns")
    return RawDataset(data=mat[:, idx], feature_names=tuple(feats[j] for j in idx),
                      source="g1", name=name)


def mech_mask(X_t: torch.Tensor, t_idx: int, idiom: str, delta: float, rng) -> np.ndarray:
    """Observed-mask for the target column under the mechanism (complete X retained by caller)."""
    if idiom == "own_value":
        _, p_idx, _ = select_target_predictor(X_t, rng.spawn(), target_idx=t_idx)
        res = apply_self_censor(X_complete=X_t, target_idx=t_idx, predictor_idx=p_idx,
                                beta1=BETA1, delta=delta, target_rate=RATE, rng=rng.spawn())
        return res.mask[:, t_idx].numpy().astype(bool)
    raw = RawDataset(data=X_t.numpy(), feature_names=tuple(f"c{j}" for j in range(X_t.shape[1])),
                     source="g1", name="tmp")
    res = generate_lod_example(raw, beta1=BETA1, delta=delta, target_rate=RATE,
                               tau_quantile=TAU_Q, rng=rng.spawn(), target_idx=t_idx)
    return res.mask[:, t_idx].numpy().astype(bool)


def build_rows():
    rows, cell_i = [], 0
    for dom in sorted(DOMAINS):
        for name in DOMAINS[dom]:
            base = cont_base(name)
            for t_idx, t_name in enumerate(base.feature_names):
                for idiom in IDIOMS:
                    for delta in DELTAS:
                        rng = RNGState(seed=31000 + cell_i)
                        cell_i += 1
                        for e in range(N_EX):
                            sub = subsample_raw(base, max_rows=MAX_ROWS, rng=rng.spawn())
                            X_t = torch.from_numpy(np.asarray(sub.data, np.float32))
                            try:
                                m_mech = mech_mask(X_t, t_idx, idiom, delta, rng)
                                m_mcar = mcar_pair_mask(m_mech, rng.spawn())
                                X = X_t.numpy().astype(np.float64)
                                feats = {}
                                for imp in IMPUTERS:
                                    feats[imp] = imputer_channel_features(
                                        X, t_idx, m_mech, m_mcar, imputer=imp, seed=1000 + e)
                                R = torch.ones_like(X_t, dtype=torch.bool)
                                R[:, t_idx] = torch.from_numpy(m_mech)
                                base17 = compute_consequence_features(
                                    X_t * R.float(), R, t_idx).numpy().tolist()
                            except ValueError as err:  # degenerate draw (logged, skipped)
                                print(f"  skip {name}.{t_name} {idiom} d={delta} ex{e}: {err}")
                                continue
                            rows.append({"domain": dom, "dataset": name, "target": t_name,
                                         "idiom": idiom, "delta": delta, "base17": base17,
                                         **{f"{imp}.{k}": v for imp in IMPUTERS
                                            for k, v in feats[imp].items()}})
        print(f"[{time.strftime('%H:%M:%S')}] domain {dom} done ({len(rows)} rows)")
    return rows


def pooled_oof_auc(rows, sel_fn, label_fn, feat_fn):
    """Block-aware leave-one-domain-out LR; pooled OOF AUC."""
    sel = [r for r in rows if sel_fn(r)]
    y_all, p_all = [], []
    for held in sorted(DOMAINS):
        tr = [r for r in sel if r["domain"] != held]
        te = [r for r in sel if r["domain"] == held]
        if not te or len({label_fn(r) for r in tr}) < 2:
            continue
        Xtr = np.array([feat_fn(r) for r in tr]); Xte = np.array([feat_fn(r) for r in te])
        sc = StandardScaler().fit(Xtr)
        lr = LogisticRegression(max_iter=2000).fit(sc.transform(Xtr), [label_fn(r) for r in tr])
        p_all.extend(lr.predict_proba(sc.transform(Xte))[:, 1])
        y_all.extend(label_fn(r) for r in te)
    return float(roc_auc_score(y_all, p_all))


def main():
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    print("=" * 100)
    print("G1 — MAR-imputation counterfactual channel (locked pre-registration 6f1bff9)")
    print("=" * 100)
    rows = build_rows()
    print(f"total example rows: {len(rows)}")

    chan = lambda imp: (lambda r: [r[f"{imp}.{k}"] for k in FEATURE_NAMES])
    base = lambda r: r["base17"]
    both = lambda imp: (lambda r: r["base17"] + [r[f"{imp}.{k}"] for k in FEATURE_NAMES])
    is_ov = lambda r: r["idiom"] == "own_value"
    is_tc = lambda r: r["idiom"] == "top_coding"
    y_delta = lambda r: int(r["delta"] > 0)

    res = {"git": git, "n_rows": len(rows), "per_imputer": {}, "baseline": {}}
    res["baseline"]["own_value"] = pooled_oof_auc(rows, is_ov, y_delta, base)
    res["baseline"]["top_coding"] = pooled_oof_auc(rows, is_tc, y_delta, base)
    print(f"\nbaseline (17 consequence feats): own_value OOF AUC {res['baseline']['own_value']:.3f} | "
          f"top_coding {res['baseline']['top_coding']:.3f}")

    print(f"\n{'imputer':10} {'a:ov chan':>9} {'b:ov base+ch':>12} {'b:increment':>11} "
          f"{'c:idiom sep':>11} {'tc chan':>8} {'tc incr':>8}")
    for imp in IMPUTERS:
        a = pooled_oof_auc(rows, is_ov, y_delta, chan(imp))
        b_full = pooled_oof_auc(rows, is_ov, y_delta, both(imp))
        b_inc = b_full - res["baseline"]["own_value"]
        c = pooled_oof_auc(rows, lambda r: r["delta"] > 0, lambda r: int(is_tc(r)), chan(imp))
        tc_a = pooled_oof_auc(rows, is_tc, y_delta, chan(imp))
        tc_inc = pooled_oof_auc(rows, is_tc, y_delta, both(imp)) - res["baseline"]["top_coding"]
        res["per_imputer"][imp] = {"a_ov_chan": a, "b_ov_full": b_full, "b_increment": b_inc,
                                   "c_idiom_sep": c, "tc_chan": tc_a, "tc_increment": tc_inc,
                                   "a_pass": a >= 0.65, "b_pass": b_inc >= 0.05, "c_pass": c >= 0.65}
        print(f"{imp:10} {a:9.3f} {b_full:12.3f} {b_inc:+11.3f} {c:11.3f} {tc_a:8.3f} {tc_inc:+8.3f}")

    ab_same = [i for i in IMPUTERS
               if res["per_imputer"][i]["a_pass"] and res["per_imputer"][i]["b_pass"]]
    c_ok = [i for i in IMPUTERS if res["per_imputer"][i]["c_pass"]]
    G1 = len(ab_same) >= 2 and len(c_ok) >= 2
    res["criteria"] = {"ab_same_imputers": ab_same, "c_imputers": c_ok, "G1_pass": bool(G1)}
    print("\n" + "=" * 100)
    print(f"G1-a&b same-imputer passes: {ab_same}  |  G1-c passes: {c_ok}")
    verdict = ("PASS — truth-level information beyond φ exists; channel justified as EVAL-ONLY "
               "instrumentation" if G1 else
               "FAIL — channel not justified (see named failure outcomes in pre-registration §3)")
    print(f"G1 VERDICT: {verdict}")
    res["criteria"]["verdict"] = verdict
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({**res, "rows": rows}, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
