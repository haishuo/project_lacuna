"""
scripts/preregister_d3_coverage.py

D3 PRE-REGISTRATION (SCF consolidation memo §10 → D3): compute, BEFORE any training, the predicted
footprint-regime coverage of each held-out continuous domain by each candidate training pool. These
numbers are the pre-registered predictions the D3 transfer test will be scored against.

NO model, NO training, NO acquisition — deterministic descriptive statistics using the D2 metric
(`lacuna.survey.regime_descriptors`). Coverage on the held-out domain's CONTINUOUS columns (card≥30),
nearest-neighbour in the standardized 4-D regime space (l_skew, l_kurt, log10_card, has_neg).

Run: /home/haishuo/miniconda3/envs/lacuna/bin/python -u scripts/preregister_d3_coverage.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from lacuna.data.catalog import create_default_catalog
from lacuna.survey.regime_descriptors import PRIMARY_KEYS, shape_descriptors

RB = Path("/mnt/data/lacuna/role_b")
OUT = Path("runs/d3_preregistration.json")

SETS = {
    "labor": ["survey_cps1985", "survey_cps1988", "survey_psid1976", "survey_psid7682", "survey_workinghours"],
    "nhanes": ["rb_nhanes_weight", "rb_nhanes_poverty", "rb_nhanes_income", "rb_nhanes_demographics"],
    "hmda": ["survey_hmda"], "chile": ["survey_chile"],
    "bfi": ["survey_bfi"], "yrbss": ["survey_yrbss"], "wealth": ["rb_scf2022_wealth_cont"],
}


def _cols_of(names, cat):
    out = []
    for n in names:
        if n.startswith("rb_"):
            df = pd.read_csv(RB / f"{n}.csv"); mat, feats = df.to_numpy(np.float64), list(df.columns)
        else:
            raw = cat.load(n); mat, feats = np.asarray(raw.data, np.float64), list(raw.feature_names)
        for j, f in enumerate(feats):
            v = mat[:, j]; v = v[np.isfinite(v)]
            if v.size >= 8 and np.unique(v).size >= 2:
                out.append((f"{n}.{f}", v))
    return out


def main():
    cat = create_default_catalog()
    desc = {}
    for s, names in SETS.items():
        for cid, v in _cols_of(names, cat):
            desc[cid] = (s, shape_descriptors(v))
    ids = list(desc)
    X = np.array([[desc[c][1][k] for k in PRIMARY_KEYS] for c in ids], dtype=np.float64)
    mu, sd = X.mean(0), X.std(0); sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    zc = {c: Z[i] for i, c in enumerate(ids)}
    cols_in = lambda sets: [c for c in ids if desc[c][0] in sets]
    cont_in = lambda sets: [c for c in ids if desc[c][0] in sets and desc[c][1]["card"] >= 30]

    def coverage(Dcols, pool_sets):
        pool = [c for c in cols_in(pool_sets) if c not in Dcols]
        return float(np.mean([min(np.linalg.norm(zc[c] - zc[t]) for t in pool) for c in Dcols]))

    plan = {
        "nhanes": [("matched_labor", ["labor"]), ("mismatch_heavytail_wealth", ["wealth"]),
                   ("mismatch_ordinal_bfi_yrbss", ["bfi", "yrbss"]),
                   ("diverse_mismatched", ["labor", "wealth", "bfi", "yrbss"])],
        "hmda": [("matched_labor_nhanes", ["labor", "nhanes"]), ("mismatch_heavytail_wealth", ["wealth"]),
                 ("mismatch_ordinal_bfi_yrbss", ["bfi", "yrbss"])],
        "wealth": [("control_all_moderate", ["labor", "nhanes", "hmda"])],
    }
    out = {"metric": list(PRIMARY_KEYS), "standardize_mu": mu.tolist(), "standardize_sd": sd.tolist(),
           "preregistered": {}}
    print("PRE-REGISTERED coverage_distance(D_continuous | pool)  [lower ⇒ predict MORE transfer]\n")
    for D, pools in plan.items():
        Dcols = cont_in([D])
        out["preregistered"][D] = {"n_continuous_cols": len(Dcols), "pools": {}}
        print(f"== held-out D = {D}  ({len(Dcols)} continuous cols) ==")
        for label, ps in pools:
            cov = coverage(Dcols, ps)
            out["preregistered"][D]["pools"][label] = round(cov, 4)
            print(f"   {label:30} cov={cov:.3f}")
        print()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
