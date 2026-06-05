"""
scripts/run_p2p2b_cardinality_probe.py

P2.2b — cardinality / strong-δ probe. Separates "δ signal present but not represented" from
"intrinsically near-absent on real survey columns" (the last cheap diagnostic before any
representation change). See docs/feasibility-p2p2b-cardinality-probe-note.md.

Safeguard first (cardinality distribution + strata). Part A: cardinality-stratified 7-bin δ-prior
(does high-cardinality/continuous recover δ while low-card floors?). Part B: strong-δ-only contrast
δ∈{0,2.5}, AUC per stratum (labeled DIAGNOSTIC, not the P2 objective). No architecture/feature
changes. Run: python -u scripts/run_p2p2b_cardinality_probe.py
"""

import json
import subprocess
import time
from pathlib import Path

import torch

from lacuna.core.rng import RNGState
from lacuna.data.catalog import create_default_catalog
from lacuna.survey import metrics as M
from lacuna.survey.column_stats import cardinality_distribution, target_cardinality_table
from lacuna.survey.example_source import StratifiedRealXSource
from lacuna.survey.loss import rps_loss
from lacuna.survey.train import TrainConfig, _forward_examples, _make_examples, train_delta_prior

POOL = ["survey_cps1988", "survey_yrbss", "survey_computers",
        "survey_psid7682", "survey_chile", "survey_hmda", "survey_cps1985"]
# fixed, interpretable cardinality thresholds
STRATA_DEF = [("low(<=8)", 0, 8), ("medium(9-60)", 9, 60), ("high(>60)", 61, 10 ** 9)]
UNIFORM_RPS = 0.190476


def _strata_from_cardinality(table):
    strata, summary = [], []
    for label, lo, hi in STRATA_DEF:
        members = [r for r in table if lo <= r["n_unique"] <= hi]
        strata.append([(r["dataset"], r["target_idx"]) for r in members])
        us = [r["n_unique"] for r in members]
        summary.append({"label": label, "lo": lo, "hi": hi, "n_pairs": len(members),
                        "nunique_min": min(us) if us else None, "nunique_max": max(us) if us else None,
                        "examples": [f"{r['dataset']}.{r['target_name']}({r['n_unique']})" for r in members[:4]]})
    return strata, summary


def _auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Tie-corrected rank-based AUC = P(score|y=1 > score|y=0), ties=0.5. No sklearn."""
    n_pos = float((labels == 1).sum()); n_neg = float((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = torch.argsort(scores)
    sorted_s = scores[order]
    n = scores.numel()
    rank_of_sorted = torch.empty(n, dtype=torch.float64)
    i = 0
    while i < n:  # average ranks within tie groups
        j = i
        while j + 1 < n and float(sorted_s[j + 1]) == float(sorted_s[i]):
            j += 1
        rank_of_sorted[i:j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1
    ranks = torch.empty(n, dtype=torch.float64)
    ranks[order] = rank_of_sorted
    r_pos = float(ranks[labels == 1].sum())
    return (r_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def _cfg(delta_grid):
    return TrainConfig(
        delta_grid=delta_grid, beta1_range=(0.0, 2.0), target_rate=0.3,
        max_rows=1024, max_cols=8, batch_size=16,
        train_batches_per_epoch=40, max_epochs=16, patience=5,
        val_size=105, test_size=105,
        hidden_dim=96, evidence_dim=48, n_layers=2, n_heads=4, target_conditioned=True,
    )


def main() -> None:
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    cat = create_default_catalog()
    datasets = {n: cat.load(n) for n in POOL}
    table = target_cardinality_table(list(datasets.values()))
    dist = cardinality_distribution(table)
    strata, summary = _strata_from_cardinality(table)

    print("=" * 80)
    print("P2.2b CARDINALITY PROBE — target n_unique distribution (safeguard)")
    print("=" * 80)
    print(f"  n candidate targets={dist['n_targets']}  min={dist['min']} q25={dist['q25']:.0f} "
          f"median={dist['median']:.0f} q75={dist['q75']:.0f} max={dist['max']}")
    for s in summary:
        print(f"  {s['label']:>14}: n_pairs={s['n_pairs']:2d}  nunique[{s['nunique_min']},{s['nunique_max']}]  "
              f"e.g. {', '.join(s['examples'])}")
    Path("runs").mkdir(exist_ok=True)
    Path("runs/p2p2b-cardinality-dist.json").write_text(json.dumps(
        {"git": git, "distribution": dist, "strata": summary, "table": table}, indent=2))

    usable = [(st, sm) for st, sm in zip(strata, summary) if len(st) > 0]

    def per_stratum_eval(model, cfg, tag, binary=False):
        print(f"\n{tag}")
        hdr = (f"{'stratum':>14} | {'rps':>7} | {'(uni-rps)/SE':>12} | {'bin':>5} | {'adj':>5} | "
               f"{'entropy':>7} | {'P(d=0)':>6}" + (" | {:>5}".format("AUC") if binary else ""))
        print(hdr); print("-" * len(hdr))
        results = []
        for st, sm in usable:
            ssrc = StratifiedRealXSource(datasets, [st])
            ex = _make_examples(ssrc, cfg, 120, RNGState(seed=700 + sm["lo"]), stratify=True)
            logits, labels, _, _ = _forward_examples(model, ex, cfg)
            T = 1.0
            per = rps_loss(logits, labels, reduction="none")
            rps_m = float(per.mean()); se = float(per.std(unbiased=True) / (len(per) ** 0.5))
            probs = torch.softmax(logits, dim=-1)
            row = {"stratum": sm["label"], "rps": rps_m,
                   "uni_minus_rps_over_se": (UNIFORM_RPS - rps_m) / se if se > 0 else 0.0,
                   "bin_acc": M.bin_accuracy(probs, labels), "adj_acc": M.adjacent_accuracy(probs, labels),
                   "entropy_bits": M.mean_predictive_entropy(probs), "p_delta0": M.p_delta_zero(probs)}
            line = (f"{sm['label']:>14} | {rps_m:>7.4f} | {row['uni_minus_rps_over_se']:>+12.2f} | "
                    f"{row['bin_acc']:>5.3f} | {row['adj_acc']:>5.3f} | {row['entropy_bits']:>7.3f} | "
                    f"{row['p_delta0']:>6.3f}")
            if binary:
                score = 1.0 - probs[:, 0]  # P(δ>0)
                ylab = (labels > 0).long()
                row["auc"] = _auc(score, ylab)
                line += f" | {row['auc']:>5.3f}"
            print(line)
            results.append(row)
        return results

    # ---- Part A: cardinality-stratified 7-bin ----
    cfgA = _cfg([0.0, 0.15, 0.4, 0.75, 1.25, 1.75, 2.5])
    src = lambda: StratifiedRealXSource(datasets, [s for s in strata if len(s) > 0])
    t0 = time.time()
    outA = train_delta_prior(src(), src(), src(), cfgA, RNGState(seed=2026),
                             kind="main", run_id="p2p2b-cardA", git_commit=git,
                             timestamp="2026-06-04T12:00:00Z")
    print(f"\n[Part A trained] wall={time.time()-t0:.0f}s leakage_pass={outA['leakage_pass']} "
          f"overall_test_rps={outA['results']['test_after_temperature']['rps']:.4f}")
    resA = per_stratum_eval(outA["model"], cfgA,
                            "PART A — cardinality-stratified 7-bin (uniform_rps=0.1905, max H=2.807)")

    # ---- Part B: strong-δ-only diagnostic ----
    cfgB = _cfg([0.0, 2.5])
    t0 = time.time()
    outB = train_delta_prior(src(), src(), src(), cfgB, RNGState(seed=4242),
                             kind="ablation", run_id="p2p2b-cardB-strongdelta", git_commit=git,
                             timestamp="2026-06-04T12:00:00Z")
    print(f"\n[Part B trained] wall={time.time()-t0:.0f}s leakage_pass={outB['leakage_pass']}  "
          f"(DIAGNOSTIC: δ∈{{0,2.5}}, AUC of P(δ>0))")
    resB = per_stratum_eval(outB["model"], cfgB,
                            "PART B — strong-δ contrast δ∈{0,2.5} (AUC=0.5 chance)", binary=True)

    # ---- decision summary ----
    def beats(r):
        return r["uni_minus_rps_over_se"] > 2 and r["adj_acc"] > 0.45
    high_A = next((r for r in resA if r["stratum"].startswith("high")), None)
    low_A = next((r for r in resA if r["stratum"].startswith("low")), None)
    high_B = next((r for r in resB if r["stratum"].startswith("high")), None)
    print("\n" + "=" * 80)
    print("DECISION SUMMARY")
    print("=" * 80)
    print(f"  Part A: high-card learns? {beats(high_A) if high_A else 'n/a'}  "
          f"low-card learns? {beats(low_A) if low_A else 'n/a'}")
    print(f"  Part B: strong-δ AUC high-card = {high_B['auc']:.3f}  "
          f"(>0.7 => strong effect readable; ~0.5 => invisible)" if high_B else "  Part B: n/a")
    if high_A and beats(high_A) and low_A and not beats(low_A):
        verdict = "CARDINALITY-GATED: signal present on high-card; P2 needs cardinality-aware abstention/features"
    elif high_B and high_B["auc"] > 0.7 and not (high_A and beats(high_A)):
        verdict = "RESOLUTION/REPRESENTATION: strong-δ readable but 7-bin not -> coarse/curriculum or residual features"
    elif high_B and high_B["auc"] < 0.6:
        verdict = "INTRINSICALLY NEAR-ABSENT: even strong-δ on high-card unreadable -> revise P2 to wide-prior + sensitivity reporting"
    else:
        verdict = "MIXED — inspect table"
    print(f"  VERDICT: {verdict}")

    from lacuna.survey.run_manifest import write_manifest
    outA["manifest"]["cardinality_probe"] = {"distribution": dist, "strata": summary,
                                             "partA": resA, "partB": resB, "verdict": verdict}
    write_manifest(Path("runs/p2p2b-cardinality.json"), outA["manifest"])
    print("\nsaved: runs/p2p2b-cardinality.json")


if __name__ == "__main__":
    main()
