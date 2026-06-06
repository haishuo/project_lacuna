"""
scripts/build_nhanes_role_b.py

Codebook-aware role-B projection of the FULL NHANES 2017-18 weight + poverty modules
(PROPOSAL-NHANES-rolebase-projection-spec; PI-approved 2026-06-06). Weight + poverty ONLY (no income
brackets this pass). Role-A (natural missingness) archived separately; role-B = complete-case over the
SELECTED target+predictor columns. Per-VARIABLE sentinel map (conservative: default NO recode; age
never recoded). Quality checks (§9): sentinel-spike-frac <= 0.002 + plausible-range; failing targets
dropped. Tiered usability (§b): preferred>=1500 / acceptable>=750 / fragile<750.

Run: python -u scripts/build_nhanes_role_b.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat

INC = Path("/mnt/data/lacuna/incoming")
RB = Path("/mnt/data/lacuna/role_b")
RA = Path("/mnt/data/lacuna/role_a")

# Per-variable sentinel map (NHANES codebook). Default for any var NOT listed = [] (no recode).
SENTINELS = {
    "WHD010": [7777, 9999], "WHD020": [7777, 9999], "WHD050": [7777, 9999],
    "WHD110": [7777, 9999], "WHD120": [7777, 9999], "WHD140": [7777, 9999],
    "DMDEDUC2": [7, 9],                 # education 1-5 valid; 7=refuse 9=DK
    # poverty/age/gender/family-size: NO recode (valid values; conservative — age 7/9/77 are real)
    "INDFMPIR": [], "RIDAGEYR": [], "RIAGENDR": [], "DMDFMSIZ": [],
}
PLAUSIBLE = {"WHD010": (40, 90), "WHD020": (50, 700), "WHD050": (50, 700), "WHD110": (50, 700),
             "WHD120": (50, 700), "WHD140": (50, 700), "INDFMPIR": (0, 5), "RIDAGEYR": (0, 100)}
REFUSE_CODES = (7, 9, 77, 99, 777, 999, 7777, 9999, 77777, 99999)
SPIKE_TOL = 0.002

BASES = [
    {"name": "rb_nhanes_weight", "domain": "health", "modules": ["WHQ_J", "DEMO_J"],
     "targets": ["WHD020", "WHD050", "WHD110", "WHD120", "WHD140"],
     "predictors": ["RIDAGEYR", "RIAGENDR", "WHD010", "DMDEDUC2"]},
    {"name": "rb_nhanes_poverty", "domain": "demographics", "modules": ["DEMO_J"],
     "targets": ["INDFMPIR"],
     "predictors": ["RIDAGEYR", "RIAGENDR", "DMDEDUC2", "DMDFMSIZ"]},
]


def _load(mod):
    df, _ = pyreadstat.read_xport(str(INC / f"{mod}.xpt"))
    return df


def _recode(df):
    out = df.copy()
    for c in out.columns:
        codes = SENTINELS.get(c, [])
        if codes:
            out.loc[out[c].isin(codes), c] = np.nan
    return out


def _spike_frac(v):
    v = v[~np.isnan(v)]
    return float(np.isin(v, np.asarray(REFUSE_CODES, float)).mean()) if v.size else 1.0


def build(spec):
    # merge modules on SEQN
    df = _load(spec["modules"][0])
    for m in spec["modules"][1:]:
        df = df.merge(_load(m), on="SEQN", how="left", suffixes=("", "_dup"))
    cols = [c for c in spec["targets"] + spec["predictors"] if c in df.columns]
    sub = _recode(df[["SEQN"] + cols])
    # role-A archive: selected cols WITH natural+sentinel NaN (never supervised)
    RA.mkdir(parents=True, exist_ok=True)
    sub.to_csv(RA / f"{spec['name']}_roleA.csv", index=False)
    # role-B: complete-case over selected target+predictor cols (drop SEQN)
    proj = sub[cols].dropna()
    # quality checks per target
    kept_targets, checks = [], {}
    for t in spec["targets"]:
        if t not in proj.columns:
            continue
        v = proj[t].to_numpy(float)
        sp = _spike_frac(v)
        lo, hi = PLAUSIBLE.get(t, (-np.inf, np.inf))
        in_range = bool(np.all((v >= lo) & (v <= hi)))
        ok = sp <= SPIKE_TOL and in_range and (np.unique(v).size >= 30)
        checks[t] = {"spike_frac": round(sp, 5), "in_range": in_range,
                     "card": int(np.unique(v).size), "min": float(v.min()), "max": float(v.max()),
                     "median": float(np.median(v)), "pass": ok}
        if ok:
            kept_targets.append(t)
    n = len(proj)
    tier = "preferred" if n >= 1500 else ("acceptable" if n >= 750 else "fragile")
    prov = {"name": spec["name"], "domain": spec["domain"], "source_block": "NHANES-2017-18",
            "flag": "projected_from_naturally_missing", "modules": spec["modules"], "merge_key": "SEQN",
            "projection": "complete_case_from_naturally_missing", "sentinel_map": SENTINELS,
            "targets_requested": spec["targets"], "targets_passed": kept_targets,
            "predictors": spec["predictors"], "rows_retained": n, "cols_kept": list(proj.columns),
            "usability_tier": tier, "quality_checks": checks,
            "role_a_archive": str(RA / f"{spec['name']}_roleA.csv")}
    print(f"\n### {spec['name']}  ({spec['domain']}, NHANES-2017-18)  role-B {n}x{proj.shape[1]}  "
          f"[{tier}]  | {len(kept_targets)}/{len(spec['targets'])} targets pass")
    for t, c in checks.items():
        print(f"      {t:10} card={c['card']:5d} range=[{c['min']:.1f},{c['max']:.1f}] "
              f"median={c['median']:.1f} spike={c['spike_frac']:.4f} in_range={c['in_range']} "
              f"-> {'PASS' if c['pass'] else 'DROP'}")
    if not kept_targets or n < 750:
        print(f"   ⇒ {spec['name']}: no passing target or fragile (<750) — NOT written for training")
        return {**prov, "written": False}
    # write role-B (only passing targets + predictors)
    RB.mkdir(parents=True, exist_ok=True)
    keep_cols = kept_targets + [p for p in spec["predictors"] if p in proj.columns]
    proj[keep_cols].to_csv(RB / f"{spec['name']}.csv", index=False)
    (RB / f"{spec['name']}.provenance.json").write_text(json.dumps(prov, indent=2))
    print(f"   wrote {RB}/{spec['name']}.csv ({len(keep_cols)} cols) + role-A archive")
    return {**prov, "written": True}


def main():
    print("=" * 96)
    print("NHANES FULL-MODULE ROLE-B (weight + poverty; per-variable sentinels; quality-checked)")
    print("=" * 96)
    out = [build(s) for s in BASES]
    RB.mkdir(parents=True, exist_ok=True)
    (RB / "INDEX_nhanes.json").write_text(json.dumps(out, indent=2))
    usable = [o for o in out if o["written"]]
    print(f"\n{len(usable)}/{len(out)} bases written for training "
          f"(tiers: {[ (o['name'], o['usability_tier']) for o in out ]})")


if __name__ == "__main__":
    main()
