"""
scripts/run_p2p2c_consequence_ab.py

P2.2c — LOD model ladder + the critical A/B (PROPOSAL-P2.2c §6). The held-out MODEL test of the
detectability spectrum: train a target-conditioned δ-prior on semi-synthetic holes on real survey X,
validate on HELD-OUT real survey datasets — once for OWN-VALUE self-censoring (known flat boundary)
and once for LOD/top-coding (hypothesized detectable) — same δ-bins / RPS / calibration / leakage /
manifest / split. The arm-to-arm contrast is the test of whether the spectrum transfers to the
learned channel. Run: python -u scripts/run_p2p2c_consequence_ab.py
"""

import json
import subprocess
import time
from pathlib import Path

import torch

from lacuna.core.rng import RNGState
from lacuna.data.catalog import create_default_catalog
from lacuna.survey.example_source import LODSurveyExampleSource, SurveyExampleSource
from lacuna.survey.loss import rps_loss
from lacuna.survey.run_manifest import write_manifest
from lacuna.survey.train import TrainConfig, _forward_examples, _make_examples, train_delta_prior

POOL_TRAIN = ["survey_cps1988", "survey_yrbss", "survey_computers", "survey_psid7682"]
POOL_VAL = ["survey_chile", "survey_hmda"]
POOL_TEST = ["survey_cps1985", "survey_workinghours"]
UNIFORM_RPS = 0.190476


def _base_rate_rps(sheets, num_bins=7):
    bins = torch.tensor([s.delta_bin for s in sheets], dtype=torch.long)
    marg = torch.bincount(bins, minlength=num_bins).float()
    marg = marg / marg.sum()
    logits = torch.log(marg.clamp(min=1e-9)).unsqueeze(0).repeat(len(bins), 1)
    return float(rps_loss(logits, bins).item())


def _run_arm(name, make_source, tr, va, te, cfg, git, seed):
    """make_source(pool) -> an ExampleSource for that pool (own-value or LOD)."""
    t0 = time.time()
    out = train_delta_prior(make_source(tr), make_source(va), make_source(te), cfg,
                            RNGState(seed=seed), kind="main", run_id=f"p2p2c-cons-{name}",
                            git_commit=git, timestamp="2026-06-04T12:00:00Z")
    out["manifest"]["wall_clock_seconds"] = round(time.time() - t0, 1)
    # per-example RPS SE on a fresh held-out test draw + base-rate ref
    test_ex = _make_examples(make_source(te), cfg, cfg.test_size, RNGState(seed=900 + seed), stratify=True)
    logits, labels, _, sheets = _forward_examples(out["model"], test_ex, cfg)
    T = out["manifest"]["temperature"]
    per = rps_loss(logits / T, labels, reduction="none")
    rps_m = float(per.mean()); se = float(per.std(unbiased=True) / (len(per) ** 0.5))
    base = _base_rate_rps(sheets)
    b = out["results"]["test_after_temperature"]
    row = {
        "arm": name, "family": out["manifest"]["generator_family"],
        "rps": rps_m, "rps_se": se, "base_rate_rps": base,
        "uni_minus_rps_over_se": (UNIFORM_RPS - rps_m) / se if se > 0 else 0.0,
        "base_minus_rps_over_se": (base - rps_m) / se if se > 0 else 0.0,
        "bin_acc": b["bin_accuracy"], "adj_acc": b["adjacent_accuracy"],
        "entropy_bits": b["entropy_bits"], "p_delta0": b["p_delta0"], "ece": b["ece"],
        "leakage_pass": out["leakage_pass"], "wall": out["manifest"]["wall_clock_seconds"],
    }
    Path("runs").mkdir(exist_ok=True)
    write_manifest(Path(f"runs/p2p2c-cons-{name}.json"), out["manifest"])
    return row


def main():
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    cat = create_default_catalog()
    tr = [cat.load(n) for n in POOL_TRAIN]
    va = [cat.load(n) for n in POOL_VAL]
    te = [cat.load(n) for n in POOL_TEST]
    cfg = TrainConfig(
        delta_grid=[0.0, 0.15, 0.4, 0.75, 1.25, 1.75, 2.5], beta1_range=(0.0, 2.0),
        target_rate=0.3, max_rows=512, max_cols=12, batch_size=16,
        train_batches_per_epoch=40, max_epochs=16, patience=5, val_size=140, test_size=140,
        hidden_dim=96, evidence_dim=48, n_layers=2, n_heads=4, target_conditioned=True,
        consequence_features=True,
    )

    print("=" * 96)
    print("P2.2c CONSEQUENCE-FEATURE A/B (own-value vs LOD) — same harness + fixed consequence features")
    print(f"  train={POOL_TRAIN} val={POOL_VAL} test={POOL_TEST}  max_rows={cfg.max_rows}")
    print("=" * 96)

    own_source = lambda pool: SurveyExampleSource(pool)
    lod_source = lambda pool: LODSurveyExampleSource(pool, tau_quantile=0.70)
    rows = []
    rows.append(_run_arm("ownvalue", own_source, tr, va, te, cfg, git, 2026))
    rows.append(_run_arm("lod", lod_source, tr, va, te, cfg, git, 2026))

    hdr = (f"{'arm':>9} | {'family':>22} | {'rps':>7} | {'(uni-rps)/SE':>12} | {'(base-rps)/SE':>13} | "
           f"{'bin':>5} | {'adj':>5} | {'H bits':>6} | {'P(d0)':>5} | {'leak':>5}")
    print("\n" + hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['arm']:>9} | {r['family']:>22} | {r['rps']:>7.4f} | {r['uni_minus_rps_over_se']:>+12.2f} | "
              f"{r['base_minus_rps_over_se']:>+13.2f} | {r['bin_acc']:>5.3f} | {r['adj_acc']:>5.3f} | "
              f"{r['entropy_bits']:>6.3f} | {r['p_delta0']:>5.3f} | {str(r['leakage_pass']):>5}")

    ov = next(r for r in rows if r["arm"] == "ownvalue")
    lod = next(r for r in rows if r["arm"] == "lod")
    lod_learns = lod["uni_minus_rps_over_se"] > 2 and lod["base_minus_rps_over_se"] > 2 and lod["adj_acc"] > 0.45
    ov_floors = ov["uni_minus_rps_over_se"] < 2
    print("\n" + "=" * 96)
    print("A/B VERDICT (the detectability spectrum, in the learned channel)")
    print("=" * 96)
    print(f"  LOD learns out-of-family (beats uniform & base by >2 SE, adj>0.45)? {lod_learns}")
    print(f"  own-value still floors? {ov_floors}")
    if lod_learns and ov_floors:
        verdict = ("SPECTRUM CONFIRMED IN THE LEARNED CHANNEL: LOD learns on held-out real survey X "
                   "while own-value floors — different idioms occupy different detectability regions.")
    elif lod_learns:
        verdict = "LOD learns; own-value also shows signal — re-check the own-value arm/contrast."
    else:
        verdict = ("LOD does NOT beat uniform out-of-family despite the oracle signal -> transfer/"
                   "representation gap on real survey X (NOT absence of signal — the oracle proved it).")
    print(f"  VERDICT: {verdict}")
    Path("runs/p2p2c-cons-summary.json").write_text(json.dumps({"git": git, "rows": rows, "verdict": verdict}, indent=2))
    print("\nsaved: runs/p2p2c-cons-{ownvalue,lod}.json + summary")


if __name__ == "__main__":
    main()
