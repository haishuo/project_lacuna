"""
scripts/run_level1_stage1_m1.py

Stage-1 M1 — does the Level-1 φ-spine REPRODUCE the Stage-0 column-primary signal INSIDE the governed
pipeline (Stage-1 spec §0)? Binary δ0-vs-δ2.5, held-out leave-datasets-out on the 9 genuine surveys,
top-coding idiom vs own-value control, 3 seeds. Success: top-coding OOF AUC ≈ Stage-0 (~0.73) and ≫
the BERT-backbone (~0.55), while own-value stays lower. Leakage-gated; named_prior recorded.

Run: python -u scripts/run_level1_stage1_m1.py
"""

import subprocess
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from lacuna.core.rng import RNGState
from lacuna.data.catalog import create_default_catalog
from lacuna.survey.column_batching import collate_columns
from lacuna.survey.example_source import LODSurveyExampleSource, SurveyExampleSource
from lacuna.survey.level1_train import Level1Config, train_level1
from lacuna.survey.train import _make_examples

TRAIN = ["survey_bfi", "survey_cps1988", "survey_psid1976", "survey_psid7682", "survey_yrbss"]
VAL = ["survey_chile", "survey_hmda"]
TEST = ["survey_cps1985", "survey_workinghours"]   # continuity test split (PI sub-decision)
SEEDS = [2026, 7, 99]


def _src(family, pool):
    return LODSurveyExampleSource(pool, tau_quantile=0.70) if family == "lod" else SurveyExampleSource(pool)


def _cfg():
    return Level1Config(delta_grid=[0.0, 2.5], beta1_range=(1.0, 1.0), target_rate=0.3,
                        max_rows=384, batch_size=16, train_size=600, max_epochs=40,
                        patience=6, val_size=140, test_size=140, m=16, e_col=32, coarse_scheme="binary")


@torch.no_grad()
def _auc(model, examples, cfg):
    model.eval()
    p, y = [], []
    for s in range(0, len(examples), cfg.batch_size):
        cb = collate_columns(examples[s:s + cfg.batch_size], max_rows=cfg.max_rows)
        p.append(model.predict_proba(cb)[:, 1])
        y.append((cb.delta > 0).long())
    return float(roc_auc_score(torch.cat(y).numpy(), torch.cat(p).numpy()))


def _arm(cat, family, seed, git):
    cfg = _cfg()
    tr, va, te = (_src(family, [cat.load(n) for n in P]) for P in (TRAIN, VAL, TEST))
    torch.manual_seed(seed)
    out = train_level1(tr, va, te, cfg, RNGState(seed=seed), kind="ablation",
                       run_id=f"level1-m1-{family}-s{seed}", git_commit=git,
                       timestamp="2026-06-06T12:00:00Z", device="cpu")
    test_ex = _make_examples(_src(family, [cat.load(n) for n in TEST]), cfg, 200,
                             RNGState(seed=777), stratify=True)
    auc = _auc(out["model"], test_ex, cfg)
    return auc, out["leakage_pass"], out["manifest"]["named_prior"]["idiom_vocabulary"]


def main():
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    Path("runs").mkdir(exist_ok=True)
    cat = create_default_catalog()
    print("=" * 92)
    print("STAGE-1 M1 — φ-spine reproduces Stage-0 in-pipeline?  binary δ0-vs-δ2.5  test=%s" % TEST)
    print("reference: Stage-0 φ 0.735 ; BERT backbone 0.548")
    print("=" * 92)
    res = {}
    for fam in ("lod", "ownvalue"):
        aucs, leaks, idiom = [], [], None
        for seed in SEEDS:
            a, lk, idiom = _arm(cat, fam, seed, git)
            aucs.append(a); leaks.append(lk)
        m, sd = float(np.mean(aucs)), float(np.std(aucs))
        res[fam] = (m, sd)
        name = "top_coding" if fam == "lod" else "own_value"
        print(f"  {name:11} AUC {m:.3f} ± {sd:.3f}  seeds={[round(a,3) for a in aucs]}  "
              f"leak_pass={all(leaks)}  idiom={idiom}")
    lod_m = res["lod"][0]; ov_m = res["ownvalue"][0]
    print("\n" + "=" * 92)
    reproduced = lod_m >= 0.68 and lod_m >= ov_m + 0.05
    print(f"M1 VERDICT: top_coding {lod_m:.3f} (≥0.68 and ≫ own-value {ov_m:.3f})  -> "
          f"{'REPRODUCED — φ-spine carries the Stage-0 signal in-pipeline (≫ backbone 0.548)' if reproduced else 'NOT reproduced — investigate'}")
    print("=" * 92)


if __name__ == "__main__":
    main()
