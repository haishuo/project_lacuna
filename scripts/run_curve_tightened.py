"""
scripts/run_curve_tightened.py

Tightened cross-domain evidence (PI 2026-06-06) — NO architecture/hyperparameter/variant changes.
Two probes, both top-coding OOF AUC, max_rows=384, fixed VAL = chile+hmda (never train/test):

  A) MULTI-SEED cumulative curve (8 seeds): labor(1d) -> +psych+health(3d) -> +NHANES big(5d), test on
     labor (cps1985/workinghours). Does the +0.009/domain slope survive 8 seeds? (report mean±SE + 2-SE).
  B) BLOCK-AWARE LEAVE-ONE-DOMAIN-OUT (5 seeds): for held-out domain D, compare transfer to D from
     NARROW train (labor-core) vs DIVERSE train (labor-core + all other domains except D). Δ = does
     adding domains improve transfer to an UNSEEN domain? Reported PER DOMAIN (not averaged).
     NHANES weight+poverty = ONE block (same SEQN) -> always together, never split across train/test.

Run: python -u scripts/run_curve_tightened.py
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
CUM_SEEDS = [2026, 7, 99, 13, 41, 57, 88, 101]
LOO_SEEDS = [2026, 7, 99, 13, 41]
cat = create_default_catalog()


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


def _train_eval(train_pool, test_pool, seeds, tag, git):
    cfg = _cfg(); src = lambda pool: LODSurveyExampleSource(pool, tau_quantile=0.70)
    va = [cat.load(n) for n in ("survey_chile", "survey_hmda")]
    aucs = []
    for seed in seeds:
        torch.manual_seed(seed)
        out = train_level1(src(train_pool), src(va), src(test_pool), cfg, RNGState(seed=seed),
                           kind="ablation", run_id=f"tight-{tag}-{seed}", git_commit=git,
                           timestamp="2026-06-06T12:00:00Z")
        te = _make_examples(src(test_pool), cfg, 200, RNGState(seed=777), stratify=True)
        aucs.append(_auc(out["model"], te, cfg))
    return np.array(aucs)


def main():
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    L = lambda *n: [cat.load(x) for x in n]
    labor_core = L("survey_cps1988", "survey_psid1976", "survey_psid7682")
    bfi, yrbss = L("survey_bfi"), L("survey_yrbss")
    nhanes = [_rb("rb_nhanes_weight"), _rb("rb_nhanes_poverty")]
    scf_wealth = [_rb("rb_scf2022_wealth")]   # NEW domain `wealth` (SCF 2022, role-B, preferred ~4595)
    labor_test = L("survey_cps1985", "survey_workinghours")

    print("=" * 96); print("A) MULTI-SEED cumulative curve (8 seeds; labor test)"); print("=" * 96)
    pools = [("labor", 1, labor_core), ("+psych+health", 3, labor_core + bfi + yrbss),
             ("+NHANES big", 5, labor_core + bfi + yrbss + nhanes)]
    cum = []
    for label, nd, pool in pools:
        a = _train_eval(pool, labor_test, CUM_SEEDS, f"cum{nd}", git)
        se = a.std(ddof=1) / np.sqrt(len(a))
        cum.append((nd, a.mean(), se))
        print(f"  P[{nd}d] {label:20} AUC {a.mean():.3f} ± SE {se:.3f}  (n={len(a)})")
    slope = float(np.polyfit([c[0] for c in cum], [c[1] for c in cum], 1)[0])
    d = cum[2][1] - cum[0][1]; dse = np.sqrt(cum[2][2] ** 2 + cum[0][2] ** 2)
    print(f"  SLOPE={slope:+.4f}/domain | edge P3-P1={d:+.3f} (SE {dse:.3f}; "
          f"{'>2SE SIGNIFICANT' if d > 2 * dse else 'within 2SE — not significant'})")

    print("\n" + "=" * 96); print("B) LEAVE-ONE-DOMAIN-OUT (5 seeds): NARROW(labor) vs DIVERSE(+others), test on held-out D")
    print("=" * 96)
    loo = [("NHANES(weight+poverty)", nhanes, labor_core, labor_core + bfi + yrbss),
           ("health(yrbss)", yrbss, labor_core, labor_core + bfi + nhanes),
           ("psychology(bfi)", bfi, labor_core, labor_core + yrbss + nhanes),
           # NEW: the decisive held-out WEALTH test (large, continuous targets; plan §9). DIVERSE
           # excludes the held-out wealth domain; SCF is one block ⇒ never in train when held out.
           ("wealth(scf)", scf_wealth, labor_core, labor_core + bfi + yrbss + nhanes)]
    for dname, test_pool, narrow, diverse in loo:
        an = _train_eval(narrow, test_pool, LOO_SEEDS, f"loo-n-{dname[:4]}", git)
        ad = _train_eval(diverse, test_pool, LOO_SEEDS, f"loo-d-{dname[:4]}", git)
        delta = ad.mean() - an.mean(); dse = np.sqrt(an.var(ddof=1)/len(an) + ad.var(ddof=1)/len(ad))
        sig = "HELPS(>2SE)" if delta > 2 * dse else ("hurts" if delta < -2 * dse else "~flat")
        print(f"  test={dname:24} narrow {an.mean():.3f}±{an.std():.3f} -> diverse {ad.mean():.3f}±{ad.std():.3f}"
              f"  Δ={delta:+.3f} ({sig})")
    print("=" * 96)


if __name__ == "__main__":
    main()
