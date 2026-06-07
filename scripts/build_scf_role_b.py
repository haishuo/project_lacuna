"""
scripts/build_scf_role_b.py

Role-B projection of the Survey of Consumer Finances 2022 Summary Extract Public Data into the
new `wealth` domain (PROPOSAL-SCF-wealth-acquisition-plan; PI-approved 2026-06-06).

Binding PI decisions baked in:
  • SCF 2022 ONLY; the *Summary Extract* file `rscfp2022.dta` (readable names, not X-codes).
  • IMPLICATE 1 ONLY (`y1 % 10 == 1`) — the 5 implicates are imputed copies of the SAME 4,595
    families, NOT 5× data (counting-discipline / slice-inflation trap, framework §1). One row/family.
  • ROLE-B ONLY: the public file is multiply imputed ⇒ item-nonresponse is already filled ⇒ there is
    NO natural-missingness role-A archive to preserve (documented exception to the dual-output rule;
    `role_a_archive = none`). We never train on natural missingness; here there is none to keep.
  • Targets = net worth + income + major asset aggregates: networth, income, asset, fin, nfin.
    (houses/debt EXCLUDED — zero-inflated single components, not aggregates.)
  • NO sentinel recode: SCF summary-extract dollar values are imputed and carry no refuse/DK codes
    (default-no-recode, framework §3). The refuse-code spike check is still computed as an auditable
    guard (expected ≈0). networth carries legitimate NEGATIVE values (debt > assets) — the plausible
    range admits negatives (the age-77 warning in reverse: do not reject valid negative net worth).
  • Exclude weight/id/design columns (wgt, y1, yy1) by NOT selecting them.

Quality gates (framework §3, identical to the NHANES build): per target — refuse-spike-frac ≤ 0.002
AND all values within plausible range AND cardinality ≥ 30; failing targets dropped + logged. Tiered
usability: preferred ≥1500 / acceptable ≥750 / fragile <750.

Run: /home/haishuo/miniconda3/envs/lacuna/bin/python -u scripts/build_scf_role_b.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path("/mnt/data/lacuna/incoming/scf2022_summary/rscfp2022.dta")
RB = Path("/mnt/data/lacuna/role_b")

NAME = "rb_scf2022_wealth"
DOMAIN = "wealth"
SOURCE_BLOCK = "scf"                       # SCF waves share this block (counting discipline §1)
TARGETS = ["networth", "income", "asset", "fin", "nfin"]
PREDICTORS = ["age", "edcl", "married", "kids", "famstruct", "racecl4", "occat1", "lf"]

# SCF summary-extract values are imputed dollars / coded demographics: NO refuse/DK sentinels.
SENTINELS: dict = {}                       # default no-recode for every selected variable
REFUSE_CODES = (7, 9, 77, 99, 777, 999, 7777, 9999, 77777, 99999)   # auditable guard only (expect ≈0)
SPIKE_TOL = 0.002
MIN_CARD = 30
# Plausible ranges (admit NEGATIVE net worth; generous upper bounds only reject absurd/garbage values).
PLAUSIBLE = {
    "networth": (-1e9, 1e10), "income": (-1e7, 1e10), "asset": (0.0, 1e10),
    "fin": (0.0, 1e10), "nfin": (0.0, 1e10),
}


def _spike_frac(v: np.ndarray) -> float:
    v = v[~np.isnan(v)]
    return float(np.isin(v, np.asarray(REFUSE_CODES, float)).mean()) if v.size else 1.0


def build() -> dict:
    if not SRC.exists():
        raise FileNotFoundError(f"SCF summary extract not found at {SRC}; run the acquisition step first")
    df = pd.read_stata(str(SRC), convert_categoricals=False)
    for c in ("y1", "yy1", "wgt"):
        if c not in df.columns:
            raise KeyError(f"expected SCF id/weight column {c!r} missing — wrong file? cols={len(df.columns)}")

    # IMPLICATE 1 ONLY — robust selection (y1 = yy1*10 + implicate ⇒ implicate = y1 % 10).
    impl = df["y1"].to_numpy(np.int64) % 10
    if set(np.unique(impl)) != {1, 2, 3, 4, 5}:
        raise ValueError(f"unexpected implicate indices {sorted(set(np.unique(impl)))}; expected 1..5")
    df1 = df[impl == 1].copy()
    n_households = df1["yy1"].nunique()
    if len(df1) != n_households:
        raise ValueError(f"implicate-1 not one-row-per-household: {len(df1)} rows, {n_households} households")

    cols = [c for c in TARGETS + PREDICTORS if c in df1.columns]
    missing = [c for c in TARGETS + PREDICTORS if c not in df1.columns]
    if missing:
        raise KeyError(f"selected columns absent from SCF summary extract: {missing}")
    sub = df1[cols]                         # NO sentinel recode (SENTINELS empty)

    # role-B: complete-case over the SELECTED target+predictor columns (SCF is fully imputed ⇒ no drop).
    proj = sub.dropna()
    n = len(proj)

    kept_targets, checks = [], {}
    for t in TARGETS:
        v = proj[t].to_numpy(float)
        sp = _spike_frac(v)
        lo, hi = PLAUSIBLE.get(t, (-np.inf, np.inf))
        in_range = bool(np.all((v >= lo) & (v <= hi)))
        card = int(np.unique(v).size)
        ok = sp <= SPIKE_TOL and in_range and card >= MIN_CARD
        checks[t] = {"card": card, "spike_frac": round(sp, 6), "in_range": in_range,
                     "min": float(v.min()), "max": float(v.max()), "median": float(np.median(v)),
                     "neg_frac": round(float((v < 0).mean()), 4), "zero_frac": round(float((v == 0).mean()), 4),
                     "pass": ok}
        if ok:
            kept_targets.append(t)

    tier = "preferred" if n >= 1500 else ("acceptable" if n >= 750 else "fragile")
    prov = {
        "name": NAME, "domain": DOMAIN, "source": "SCF-2022-SummaryExtract", "source_block": SOURCE_BLOCK,
        "wave": "2022", "country": "US", "file": str(SRC), "implicate_rule": "implicate 1 only (y1 % 10 == 1)",
        "n_implicates_in_file": 5, "n_households": int(n_households),
        "flag": "projected_from_naturally_missing", "projection": "complete_case_single_implicate",
        "role_a_archive": "none (multiply-imputed public file — no natural missingness; PI-approved role-B-only)",
        "natural_missingness_available": False,
        "sentinel_map": SENTINELS, "sentinel_policy": "no recode (imputed summary extract, no refuse/DK codes)",
        "excluded_design_cols": ["wgt", "y1", "yy1"],
        "targets_requested": TARGETS, "targets_passed": kept_targets, "predictors": PREDICTORS,
        "rows_retained": n, "cols_kept": list(proj.columns), "usability_tier": tier,
        "quality_checks": checks,
        "license": "public domain (U.S. Federal Reserve Board)",
    }

    print("=" * 96)
    print("SCF 2022 SUMMARY EXTRACT → role-B `wealth` (implicate 1; role-B-only; quality-checked)")
    print("=" * 96)
    print(f"\n### {NAME}  ({DOMAIN}, SCF-2022)  role-B {n}x{proj.shape[1]}  [{tier}]  "
          f"| {len(kept_targets)}/{len(TARGETS)} targets pass  | households={n_households}")
    for t, c in checks.items():
        print(f"      {t:9} card={c['card']:5d} range=[{c['min']:.0f},{c['max']:.0f}] median={c['median']:.0f} "
              f"neg={c['neg_frac']:.3f} zero={c['zero_frac']:.3f} spike={c['spike_frac']:.5f} "
              f"in_range={c['in_range']} -> {'PASS' if c['pass'] else 'DROP'}")

    if not kept_targets or n < 750:
        print(f"   ⇒ {NAME}: no passing target or fragile (<750) — NOT written for training")
        return {**prov, "written": False}

    RB.mkdir(parents=True, exist_ok=True)
    keep_cols = kept_targets + [p for p in PREDICTORS if p in proj.columns]
    proj[keep_cols].to_csv(RB / f"{NAME}.csv", index=False)
    (RB / f"{NAME}.provenance.json").write_text(json.dumps(prov, indent=2))
    (RB / "INDEX_scf.json").write_text(json.dumps({**prov, "written": True}, indent=2))
    print(f"   wrote {RB}/{NAME}.csv ({len(keep_cols)} cols) + provenance (role-A: none, imputed)")
    return {**prov, "written": True}


if __name__ == "__main__":
    build()
