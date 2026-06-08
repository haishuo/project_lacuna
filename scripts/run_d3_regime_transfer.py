"""
scripts/run_d3_regime_transfer.py

D3 — regime-matched continuous transfer test (PROPOSAL-D3-regime-matched-transfer-spec.md; PI-approved
2026-06-07). Tests the footprint-regime COVERAGE hypothesis H1: held-out transfer improves when the
training pool is closer to the held-out target in footprint-regime space.

NO architecture change, NO hyperparameter tuning, NO new data, NO metadata, NO Level-2 deviation, NO
extra arms beyond the spec. SAME cfg as run_curve_tightened (frozen). Reports AUC + RPS + ECE + entropy.

Design (spec §2): held-out continuous D ∈ {NHANES-cont, HMDA}; SCF wealth = negative control. Pools:
  M  matched (moderate-continuous, regime-near: labor; +nhanes for HMDA)
  Xh mismatch-heavytail (wealth — ALSO continuous ⇒ isolates regime from continuity)
  Xo mismatch-ordinal (bfi+yrbss — full low-card datasets)
  M+Xh diverse-but-mismatched (matched + wealth)
Plus a SIZE-CONTROLLED M-vs-Xh (equal #continuous source columns) — the clean isolation contrast.
Pre-registered coverage from runs/d3_preregistration.json (locked, computed before training).

Continuous-only bases: every held-out test domain and every continuous pool is restricted to columns
with card>=30, so every uniformly-sampled top-coding target is a valid continuous target (removes the
categorical-dilution confound, §8 findings; spec §6). The ordinal pool stays full (low-card is its nature).

Run: /home/haishuo/miniconda3/envs/lacuna/bin/python -u scripts/run_d3_regime_transfer.py
"""

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from lacuna.core.rng import RNGState
from lacuna.data.catalog import create_default_catalog
from lacuna.data.ingestion import RawDataset
from lacuna.survey import metrics as M
from lacuna.survey.column_batching import collate_columns
from lacuna.survey.example_source import LODSurveyExampleSource
from lacuna.survey.level1_train import Level1Config, train_level1
from lacuna.survey.loss import rps_loss
from lacuna.survey.train import _make_examples

RB = Path("/mnt/data/lacuna/role_b")
PREREG = json.loads(Path("runs/d3_preregistration.json").read_text())["preregistered"]
OUT = Path("runs/d3_regime_transfer.json")
SEEDS = [2026, 7, 99, 13, 41, 57, 88, 101]            # 8 seeds (spec §4)
MIN_CARD = 30
cat = create_default_catalog()


def _load(name):
    if name.startswith("rb_"):
        df = pd.read_csv(RB / f"{name}.csv")
        return df.to_numpy(np.float32), list(df.columns)
    raw = cat.load(name)
    return np.asarray(raw.data, np.float32), list(raw.feature_names)


def _continuous_idx(mat, feats):
    out = []
    for j in range(mat.shape[1]):
        v = mat[:, j][np.isfinite(mat[:, j])]
        if np.unique(v).size >= MIN_CARD:
            out.append(j)
    return out


def cont_base(name, cap=None):
    """RawDataset restricted to continuous (card>=30) columns; optionally capped to the first `cap`."""
    mat, feats = _load(name)
    idx = _continuous_idx(mat, feats)
    if cap is not None:
        idx = idx[:cap]
    if len(idx) < 2:
        raise ValueError(f"{name}: only {len(idx)} continuous columns (<2); cannot form LOD examples")
    return RawDataset(data=mat[:, idx], feature_names=tuple(feats[j] for j in idx),
                      source="role_b_projected" if name.startswith("rb_") else "native", name=name)


def full_base(name):
    mat, feats = _load(name)
    return RawDataset(data=mat, feature_names=tuple(feats), source="native", name=name)


def n_cont(name):
    mat, feats = _load(name)
    return len(_continuous_idx(mat, feats))


def _cfg():
    # IDENTICAL to run_curve_tightened._cfg — frozen. No tuning.
    return Level1Config(delta_grid=[0.0, 2.5], beta1_range=(1.0, 1.0), target_rate=0.3, max_rows=384,
                        batch_size=16, train_size=600, max_epochs=40, patience=6, val_size=140,
                        test_size=140, m=16, e_col=32, coarse_scheme="binary")


@torch.no_grad()
def _evaluate(model, te, cfg):
    """Calibration-first metrics on held-out examples: AUC, RPS, ECE, entropy(bits), p(δ=0)."""
    model.eval()
    probs, labels, deltas = [], [], []
    for s in range(0, len(te), cfg.batch_size):
        cb = collate_columns(te[s:s + cfg.batch_size], max_rows=cfg.max_rows)
        probs.append(model.predict_proba(cb))           # temperature-calibrated
        labels.append((cb.delta > 0).long())
        deltas.append(cb.delta)
    P = torch.cat(probs); Y = torch.cat(labels); D = torch.cat(deltas)
    logp = torch.log(P.clamp_min(1e-9))                  # rps_loss applies softmax ⇒ recovers P
    return {"auc": float(roc_auc_score(Y.numpy(), P[:, 1].numpy())),
            "rps": float(rps_loss(logp, Y).item()),
            "ece": float(M.ece(P, Y)["ece"]),
            "entropy_bits": float(M.mean_predictive_entropy(P)),
            "p_delta0": float(M.p_delta_zero(P))}


def run_arm(train_pool, test_pool, git):
    cfg = _cfg(); src = lambda pool: LODSurveyExampleSource(pool, tau_quantile=0.70)
    va = [cont_base("survey_chile")]                     # fixed VAL — never train/test
    rows = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        out = train_level1(src(train_pool), src(va), src(test_pool), cfg, RNGState(seed=seed),
                           kind="ablation", run_id=f"d3-{seed}", git_commit=git,
                           timestamp="2026-06-07T12:00:00Z")
        te = _make_examples(src(test_pool), cfg, 200, RNGState(seed=777), stratify=True)
        rows.append(_evaluate(out["model"], te, cfg))
    agg = {}
    for k in rows[0]:
        a = np.array([r[k] for r in rows])
        agg[k] = {"mean": float(a.mean()), "se": float(a.std(ddof=1) / np.sqrt(len(a))),
                  "per_seed": [round(x, 3) for x in a.tolist()]}
    return agg


def main():
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    labor_names = ["survey_cps1985", "survey_cps1988", "survey_psid1976", "survey_psid7682", "survey_workinghours"]
    labor = [cont_base(n) for n in labor_names]
    wealth = [cont_base("rb_scf2022_wealth_cont")]
    nhanes = [cont_base("rb_nhanes_weight"), cont_base("rb_nhanes_poverty"), cont_base("rb_nhanes_income")]
    hmda = [cont_base("survey_hmda")]
    ordinal = [full_base("survey_bfi"), full_base("survey_yrbss")]   # low-card by nature (mismatch-ordinal)

    # ----- arms: (held-out D, pool label, pool, pre-registered coverage key) -----
    arms = [
        ("nhanes", "M_matched_labor",        labor,            ("nhanes", "matched_labor")),
        ("nhanes", "Xh_mismatch_wealth",     wealth,           ("nhanes", "mismatch_heavytail_wealth")),
        ("nhanes", "Xo_mismatch_ordinal",    ordinal,          ("nhanes", "mismatch_ordinal_bfi_yrbss")),
        ("nhanes", "MXh_diverse_M_plus_wealth", labor + wealth, ("nhanes", "diverse_mismatched_M_plus_wealth")),
        ("hmda",   "M_matched_labor_nhanes", labor + nhanes,   ("hmda", "matched_labor_nhanes")),
        ("hmda",   "Xh_mismatch_wealth",     wealth,           ("hmda", "mismatch_heavytail_wealth")),
        ("hmda",   "Xo_mismatch_ordinal",    ordinal,          ("hmda", "mismatch_ordinal_bfi_yrbss")),
        ("wealth", "control_all_moderate",   labor + nhanes + hmda, ("wealth", "control_all_moderate")),
    ]
    test_pool = {"nhanes": nhanes, "hmda": hmda, "wealth": wealth}

    print("=" * 104)
    print("D3 — REGIME-MATCHED CONTINUOUS TRANSFER (8 seeds; AUC/RPS/ECE/entropy; coverage from pre-registration)")
    print("=" * 104)
    results, pairs = [], []
    for D, label, pool, (ck_d, ck_p) in arms:
        cov = PREREG[ck_d]["pools"][ck_p]
        agg = run_arm(pool, test_pool[D], git)
        a, r, e, h = agg["auc"], agg["rps"], agg["ece"], agg["entropy_bits"]
        print(f"\n[D={D:6}] {label:26} cov={cov:.3f}")
        print(f"          AUC {a['mean']:.3f}±{a['se']:.3f}  RPS {r['mean']:.3f}±{r['se']:.3f}  "
              f"ECE {e['mean']:.3f}  ent {h['mean']:.3f} bits   AUC/seed {a['per_seed']}")
        results.append({"held_out": D, "pool": label, "coverage": cov, "metrics": agg})
        pairs.append((cov, a["mean"]))

    # ----- size-controlled M vs Xh (equal #continuous source columns) — clean isolation contrast -----
    K = min(n_cont("rb_scf2022_wealth_cont"), max(n_cont(n) for n in labor_names))
    richest = max(labor_names, key=n_cont)
    M_cap = [cont_base(richest, cap=K)]
    Xh_cap = [cont_base("rb_scf2022_wealth_cont", cap=K)]
    print("\n" + "-" * 104)
    print(f"SIZE-CONTROLLED isolation (held-out NHANES): K={K} continuous cols each; "
          f"M={richest} vs Xh=wealth (same #datasets, cols, cfg)")
    for label, pool in [("M_cap_" + richest, M_cap), ("Xh_cap_wealth", Xh_cap)]:
        agg = run_arm(pool, nhanes, git)
        a = agg["auc"]
        print(f"   {label:22} AUC {a['mean']:.3f}±{a['se']:.3f}   RPS {agg['rps']['mean']:.3f}   "
              f"ECE {agg['ece']['mean']:.3f}   ent {agg['entropy_bits']['mean']:.3f}")
        results.append({"held_out": "nhanes_sizematched", "pool": label, "coverage": None, "metrics": agg})

    # ----- strong test: coverage vs transfer across all main (D,pool) pairs -----
    cov_arr = np.array([c for c, _ in pairs]); auc_arr = np.array([x for _, x in pairs])
    def spearman(x, y):
        rx, ry = x.argsort().argsort().astype(float), y.argsort().argsort().astype(float)
        return float(np.corrcoef(rx, ry)[0, 1])
    slope = float(np.polyfit(cov_arr, auc_arr, 1)[0])
    print("\n" + "=" * 104)
    print(f"STRONG TEST — coverage vs transfer across {len(pairs)} (D,pool) pairs:")
    print(f"   Pearson(cov, AUC) = {float(np.corrcoef(cov_arr, auc_arr)[0,1]):+.3f}   "
          f"Spearman = {spearman(cov_arr, auc_arr):+.3f}   slope = {slope:+.4f} AUC/cov-unit")
    print("   H1 predicts NEGATIVE (more coverage-distance ⇒ less transfer).")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"results": results, "seeds": SEEDS,
                               "coverage_vs_transfer": {"pearson": float(np.corrcoef(cov_arr, auc_arr)[0, 1]),
                                                        "spearman": spearman(cov_arr, auc_arr), "slope": slope},
                               "git": git}, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
