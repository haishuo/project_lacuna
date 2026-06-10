"""
scripts/preregister_detectability_gate.py

E-JUSTIFY/E-FALSIFY step 1 (PROPOSAL-E-JUSTIFY-E-FALSIFY-detectability-study §2/§9.1): compute and
COMMIT the coverage-gate thresholds θ_gate BEFORE any training. Descriptive statistics only — no model,
no training, no probing. Deterministic.

Locked rule (spec §2): θ_gate(pool) = the 95th percentile of within-pool LEAVE-ONE-COLUMN-OUT
self-coverage — each training column's regime distance to the nearest OTHER pool column, in the
standardized 4-D footprint space (l_skew, l_kurt, log10_card, has_neg; D2 metric). A column is gated
OUT (UNKNOWN) iff its coverage-distance to the pool exceeds θ_gate.

Pools (the D3 matched arms, continuous-only as trained):
  pool_nhanes (held-out NHANES subject) = labor 5 surveys, continuous cols
  pool_hmda   (held-out HMDA subject)   = labor + NHANES bases, continuous cols
Standardization: z-score each coordinate across ALL study columns (pools ∪ held-out ∪ wealth),
as in the D3 pre-registration (one shared regime space).

Run: /home/haishuo/miniconda3/envs/lacuna/bin/python -u scripts/preregister_detectability_gate.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from lacuna.data.catalog import create_default_catalog
from lacuna.survey.regime_descriptors import PRIMARY_KEYS, shape_descriptors

RB = Path("/mnt/data/lacuna/role_b")
OUT = Path("runs/detectability_gate.json")
MIN_CARD = 30

LABOR = ["survey_cps1985", "survey_cps1988", "survey_psid1976", "survey_psid7682", "survey_workinghours"]
NHANES = ["rb_nhanes_weight", "rb_nhanes_poverty", "rb_nhanes_income"]
POOLS = {"pool_nhanes": LABOR, "pool_hmda": LABOR + NHANES}
EVAL_SETS = {"nhanes": NHANES, "hmda": ["survey_hmda"], "wealth": ["rb_scf2022_wealth_cont"]}


def _cont_cols(name, cat):
    if name.startswith("rb_"):
        df = pd.read_csv(RB / f"{name}.csv")
        mat, feats = df.to_numpy(np.float64), list(df.columns)
    else:
        raw = cat.load(name)
        mat, feats = np.asarray(raw.data, np.float64), list(raw.feature_names)
    out = []
    for j, f in enumerate(feats):
        v = mat[:, j][np.isfinite(mat[:, j])]
        if v.size >= 8 and np.unique(v).size >= MIN_CARD:
            out.append((f"{name}.{f}", v))
    return out


def main():
    cat = create_default_catalog()
    all_names = sorted(set(LABOR + NHANES + ["survey_hmda", "rb_scf2022_wealth_cont"]))
    cols = {}
    for n in all_names:
        for cid, v in _cont_cols(n, cat):
            cols[cid] = shape_descriptors(v)
    ids = sorted(cols)
    X = np.array([[cols[c][k] for k in PRIMARY_KEYS] for c in ids])
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    Z = {c: (X[i] - mu) / sd for i, c in enumerate(ids)}

    def of(dataset_names):
        return [c for c in ids if c.split(".")[0] in dataset_names]

    out = {"rule": "theta = 95th pct of within-pool leave-one-column-out self-coverage",
           "primary_keys": list(PRIMARY_KEYS), "standardize_mu": mu.tolist(), "standardize_sd": sd.tolist(),
           "pools": {}, "coverage": {}}

    print("=" * 96)
    print("DETECTABILITY GATE PRE-REGISTRATION (descriptive; committed BEFORE any training)")
    print("=" * 96)
    for pname, pdatasets in POOLS.items():
        pcols = of(pdatasets)
        self_cov = [min(np.linalg.norm(Z[c] - Z[t]) for t in pcols if t != c) for c in pcols]
        theta = float(np.percentile(self_cov, 95))
        out["pools"][pname] = {"datasets": pdatasets, "n_columns": len(pcols),
                               "theta_gate": round(theta, 4),
                               "self_coverage": {c: round(s, 4) for c, s in zip(pcols, self_cov)}}
        print(f"\n{pname}: {len(pcols)} continuous cols  θ_gate = {theta:.4f}")
        # coverage + gate state for every eval column vs this pool
        out["coverage"][pname] = {}
        for ev, enames in EVAL_SETS.items():
            states = {}
            for c in of(enames):
                cov = float(min(np.linalg.norm(Z[c] - Z[t]) for t in pcols))
                states[c] = {"cov": round(cov, 4), "gated_out": bool(cov > theta)}
            n_out = sum(s["gated_out"] for s in states.values())
            out["coverage"][pname][ev] = states
            print(f"   {ev:7} vs {pname:11}: {n_out}/{len(states)} gated OUT  "
                  f"{[(c.split('.',1)[1], s['cov'], 'OUT' if s['gated_out'] else 'in') for c, s in states.items()]}")
        # anchors (in-pool) use leave-own-column-out coverage at cell time; recorded via self_coverage above

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
