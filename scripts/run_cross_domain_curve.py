"""
scripts/run_cross_domain_curve.py

Cross-DOMAIN learning curve with the LARGER codebook-curated NHANES role-B bases (cross-domain plan §3;
MASTER §8). Tests the PI's question: was the earlier FLAT curve caused by small/thin bases? Top-coding
OOF AUC vs #domains, fixed labor test, 3 seeds, max_rows=384. Cumulative:

  P1 labor               (cps1988, psid1976, psid7682)                     1 domain
  P2 +psychology +health (+ bfi, yrbss)                                    3 domains
  P3 +NHANES block       (+ rb_nhanes_weight ~3833, rb_nhanes_poverty ~4774)  5 domains (ONE block)

rb_nhanes_weight + rb_nhanes_poverty share NHANES-2017-18 respondents ⇒ ONE block; both are in TRAIN
(test = labor) so no train/test leakage. Read in DOMAINS, not files.

Run: python -u scripts/run_cross_domain_curve.py
"""

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from lacuna.core.rng import RNGState
from lacuna.data.catalog import create_default_catalog
from lacuna.data.ingestion import RawDataset
from lacuna.survey.column_batching import collate_columns
from lacuna.survey.example_source import LODSurveyExampleSource
from lacuna.survey.level1_train import Level1Config, train_level1
from lacuna.survey.train import _make_examples

RB = Path("/mnt/data/lacuna/role_b")
VAL = ["survey_chile", "survey_hmda"]
TEST = ["survey_cps1985", "survey_workinghours"]
SEEDS = [2026, 7, 99]


def _rb(name):
    df = pd.read_csv(RB / f"{name}.csv")
    return RawDataset(data=df.to_numpy(np.float32), feature_names=tuple(df.columns),
                      source="role_b_projected", name=name)


def _cfg():
    return Level1Config(delta_grid=[0.0, 2.5], beta1_range=(1.0, 1.0), target_rate=0.3, max_rows=384,
                        batch_size=16, train_size=600, max_epochs=40, patience=6, val_size=140,
                        test_size=140, m=16, e_col=32, coarse_scheme="binary")


@torch.no_grad()
def _auc(model, ex, cfg):
    model.eval(); p, y = [], []
    for s in range(0, len(ex), cfg.batch_size):
        cb = collate_columns(ex[s:s + cfg.batch_size], max_rows=cfg.max_rows)
        p.append(model.predict_proba(cb)[:, 1]); y.append((cb.delta > 0).long())
    return float(roc_auc_score(torch.cat(y).numpy(), torch.cat(p).numpy()))


def _point(cat, train_pool, n_domains, label, git):
    cfg = _cfg()
    src = lambda pool: LODSurveyExampleSource(pool, tau_quantile=0.70)
    va = [cat.load(n) for n in VAL]; te = [cat.load(n) for n in TEST]
    aucs = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        out = train_level1(src(train_pool), src(va), src(te), cfg, RNGState(seed=seed),
                           kind="ablation", run_id=f"xdom-{label}-{seed}", git_commit=git,
                           timestamp="2026-06-06T12:00:00Z")
        test_ex = _make_examples(src(te), cfg, 200, RNGState(seed=777), stratify=True)
        aucs.append(_auc(out["model"], test_ex, cfg))
    m, sd = float(np.mean(aucs)), float(np.std(aucs))
    print(f"  P[{n_domains}d] {label:34} AUC {m:.3f} ± {sd:.3f}  {[round(a,3) for a in aucs]}")
    return n_domains, m, sd


def main():
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    cat = create_default_catalog()
    L = lambda *names: [cat.load(n) for n in names]
    labor = L("survey_cps1988", "survey_psid1976", "survey_psid7682")
    p2 = labor + L("survey_bfi", "survey_yrbss")
    p3 = p2 + [_rb("rb_nhanes_weight"), _rb("rb_nhanes_poverty")]
    print("=" * 96)
    print("CROSS-DOMAIN CURVE w/ LARGER NHANES bases — top-coding OOF AUC vs #domains (labor test, m384)")
    print("=" * 96)
    pts = [_point(cat, labor, 1, "labor", git),
           _point(cat, p2, 3, "+psychology +health", git),
           _point(cat, p3, 5, "+NHANES weight+poverty (big)", git)]
    xs = np.array([p[0] for p in pts]); ys = np.array([p[1] for p in pts])
    slope = float(np.polyfit(xs, ys, 1)[0])
    print("\n" + "=" * 96)
    print(f"  curve: " + " -> ".join(f"{p[0]}d:{p[1]:.3f}" for p in pts))
    print(f"  SLOPE = {slope:+.4f}/domain | P2->P3 (add big NHANES) = {ys[2]-ys[1]:+.3f} | "
          f"edge P3-P1 = {ys[-1]-ys[0]:+.3f}")
    if ys[2] - ys[1] >= 0.02:
        v = ("BIGGER NHANES bases MOVE the curve (+%.3f) -> domain diversity helps when bases are large "
             "enough; the prior flat curve was a small-base artifact. Scaling hypothesis SUPPORTED." % (ys[2]-ys[1]))
    elif abs(ys[2] - ys[1]) < 0.02:
        v = ("Still ~FLAT even with larger NHANES bases (+%.3f) -> the flat curve is NOT just a small-base "
             "artifact; either labor-test transfer is the wrong probe (try leave-one-domain-out) or the "
             "signal is genuinely weak at this corpus scale -> external acquisition becomes the priority." % (ys[2]-ys[1]))
    else:
        v = "Negative: bigger NHANES bases hurt labor transfer (inspect)."
    print(f"  VERDICT: {v}")
    print("=" * 96)


if __name__ == "__main__":
    main()
