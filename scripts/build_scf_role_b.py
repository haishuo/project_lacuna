"""
scripts/build_scf_role_b.py

Role-B projection of the Survey of Consumer Finances 2022 Summary Extract Public Data into the
new `wealth` domain (PROPOSAL-SCF-wealth-acquisition-plan; PI-approved 2026-06-06).

Emits TWO role-B bases from the same source (one row per family, implicate 1):

  • rb_scf2022_wealth_cont  — CANONICAL for the continuous top-coding idiom. Columns are CONTINUOUS
    ONLY (5 wealth aggregates + continuous `age`). Use this for the held-out wealth test.
  • rb_scf2022_wealth       — as-built (5 continuous targets + 8 categorical/ordinal predictors),
    faithful to plan §5. Retained to DOCUMENT a confound, not for the headline (see below).

WHY TWO (an ingestion finding): the frozen example pipeline (`select_target_predictor`) samples the
censored target UNIFORMLY over all non-constant columns — there is NO cardinality filter. So any
categorical/ordinal predictor in a base can be drawn as a "top-coding target," which is degenerate for
a continuous idiom. With 8/13 categorical columns the as-built SCF base's held-out top-coding AUC sits
BELOW chance (~0.44) — a dilution artifact, not anti-transfer. Restricting the base to CONTINUOUS
columns makes every uniformly-sampled target a valid continuous idiom target and removes the confound
(narrow held-out AUC ~0.66). The framework's continuous-target requirement binds at target selection;
the continuous-only base honors it. (NHANES bases carry ~22% categorical columns — a milder version of
the same effect — but are left unchanged for reproducibility; SCF's 62% made it acute.)

Binding PI decisions baked in:
  • SCF 2022 ONLY; the *Summary Extract* `rscfp2022.dta` (readable names, not X-codes).
  • IMPLICATE 1 ONLY (`y1 % 10 == 1`) — the 5 implicates are imputed copies of the SAME 4,595
    families, NOT 5× data (counting-discipline / slice-inflation trap, framework §1).
  • ROLE-B ONLY: multiply-imputed public file ⇒ no natural-missingness role-A archive (documented
    exception; `role_a_archive = none`). Never train on natural missingness; here there is none.
  • Targets = net worth + income + major asset aggregates: networth, income, asset, fin, nfin.
  • NO sentinel recode (imputed dollars, no refuse/DK codes); plausible range admits NEGATIVE net worth.
  • Exclude weight/id/design columns (wgt, y1, yy1) by NOT selecting them.

Quality gates (framework §3): per target — refuse-spike-frac ≤ 0.002 AND within plausible range AND
cardinality ≥ 30; failing targets dropped + logged. Tier: preferred ≥1500 / acceptable ≥750 / fragile <750.

Run: /home/haishuo/miniconda3/envs/lacuna/bin/python -u scripts/build_scf_role_b.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path("/mnt/data/lacuna/incoming/scf2022_summary/rscfp2022.dta")
RB = Path("/mnt/data/lacuna/role_b")

DOMAIN = "wealth"
SOURCE_BLOCK = "scf"                        # SCF waves share this block (counting discipline §1)
TARGETS = ["networth", "income", "asset", "fin", "nfin"]
CAT_PREDICTORS = ["edcl", "married", "kids", "famstruct", "racecl4", "occat1", "lf"]
CONT_PREDICTOR = ["age"]                    # continuous (card 78); the lone predictor in the canonical base

# Two bases: canonical continuous-only, and the as-built (confound-documenting) variant.
BASES = [
    {"name": "rb_scf2022_wealth_cont", "predictors": CONT_PREDICTOR, "role": "canonical (continuous-only)"},
    {"name": "rb_scf2022_wealth", "predictors": CONT_PREDICTOR + CAT_PREDICTORS,
     "role": "as-built (5 cont targets + categorical predictors; documents the uniform-sampling confound)"},
]

SENTINELS: dict = {}                        # SCF summary extract is imputed: no refuse/DK sentinels
REFUSE_CODES = (7, 9, 77, 99, 777, 999, 7777, 9999, 77777, 99999)   # auditable guard only (expect ≈0)
SPIKE_TOL = 0.002
MIN_CARD = 30
PLAUSIBLE = {                               # admit NEGATIVE net worth; bounds only reject absurd values
    "networth": (-1e9, 1e10), "income": (-1e7, 1e10), "asset": (0.0, 1e10),
    "fin": (0.0, 1e10), "nfin": (0.0, 1e10),
}


def _spike_frac(v: np.ndarray) -> float:
    v = v[~np.isnan(v)]
    return float(np.isin(v, np.asarray(REFUSE_CODES, float)).mean()) if v.size else 1.0


def _load_implicate1() -> tuple:
    if not SRC.exists():
        raise FileNotFoundError(f"SCF summary extract not found at {SRC}; run the acquisition step first")
    df = pd.read_stata(str(SRC), convert_categoricals=False)
    for c in ("y1", "yy1", "wgt"):
        if c not in df.columns:
            raise KeyError(f"expected SCF id/weight column {c!r} missing — wrong file?")
    impl = df["y1"].to_numpy(np.int64) % 10
    if set(np.unique(impl)) != {1, 2, 3, 4, 5}:
        raise ValueError(f"unexpected implicate indices {sorted(set(np.unique(impl)))}; expected 1..5")
    df1 = df[impl == 1].copy()
    n_h = df1["yy1"].nunique()
    if len(df1) != n_h:
        raise ValueError(f"implicate-1 not one-row-per-household: {len(df1)} rows, {n_h} households")
    return df1, n_h


def build(df1, n_households, spec) -> dict:
    cols = TARGETS + spec["predictors"]
    missing = [c for c in cols if c not in df1.columns]
    if missing:
        raise KeyError(f"selected columns absent from SCF summary extract: {missing}")
    proj = df1[cols].dropna()              # SCF fully imputed ⇒ no drop; NO sentinel recode
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
    n_cont_cols = len(kept_targets) + len(CONT_PREDICTOR)
    prov = {
        "name": spec["name"], "role": spec["role"], "domain": DOMAIN, "source": "SCF-2022-SummaryExtract",
        "source_block": SOURCE_BLOCK, "wave": "2022", "country": "US", "file": str(SRC),
        "implicate_rule": "implicate 1 only (y1 % 10 == 1)", "n_implicates_in_file": 5,
        "n_households": int(n_households), "flag": "projected_from_naturally_missing",
        "projection": "complete_case_single_implicate",
        "role_a_archive": "none (multiply-imputed public file — no natural missingness; PI-approved role-B-only)",
        "natural_missingness_available": False, "sentinel_map": SENTINELS,
        "sentinel_policy": "no recode (imputed summary extract, no refuse/DK codes)",
        "excluded_design_cols": ["wgt", "y1", "yy1"],
        "uniform_target_sampling_note": (
            "pipeline samples the censored target uniformly over non-constant columns; categorical "
            "predictors become degenerate top-coding targets. continuous_col_fraction reported below."),
        "continuous_col_fraction": round(n_cont_cols / proj.shape[1], 3),
        "targets_requested": TARGETS, "targets_passed": kept_targets, "predictors": spec["predictors"],
        "rows_retained": n, "cols_kept": list(proj.columns), "usability_tier": tier,
        "quality_checks": checks, "license": "public domain (U.S. Federal Reserve Board)",
    }

    print(f"\n### {spec['name']}  [{spec['role']}]  ({DOMAIN}, SCF-2022)  role-B {n}x{proj.shape[1]}  "
          f"[{tier}]  | {len(kept_targets)}/{len(TARGETS)} targets pass | cont_frac={prov['continuous_col_fraction']}")
    for t, c in checks.items():
        print(f"      {t:9} card={c['card']:5d} range=[{c['min']:.0f},{c['max']:.0f}] median={c['median']:.0f} "
              f"neg={c['neg_frac']:.3f} zero={c['zero_frac']:.3f} spike={c['spike_frac']:.5f} "
              f"-> {'PASS' if c['pass'] else 'DROP'}")
    if not kept_targets or n < 750:
        print(f"   ⇒ {spec['name']}: no passing target or fragile (<750) — NOT written")
        return {**prov, "written": False}

    RB.mkdir(parents=True, exist_ok=True)
    keep_cols = kept_targets + [p for p in spec["predictors"] if p in proj.columns]
    proj[keep_cols].to_csv(RB / f"{spec['name']}.csv", index=False)
    (RB / f"{spec['name']}.provenance.json").write_text(json.dumps(prov, indent=2))
    print(f"   wrote {RB}/{spec['name']}.csv ({len(keep_cols)} cols) + provenance (role-A: none, imputed)")
    return {**prov, "written": True}


def main():
    print("=" * 96)
    print("SCF 2022 SUMMARY EXTRACT → role-B `wealth` (implicate 1; role-B-only; canonical=continuous-only)")
    print("=" * 96)
    df1, n_h = _load_implicate1()
    print(f"implicate-1 rows: {len(df1)}  households: {n_h}")
    out = [build(df1, n_h, s) for s in BASES]
    (RB / "INDEX_scf.json").write_text(json.dumps(out, indent=2))
    print(f"\n{sum(o['written'] for o in out)}/{len(out)} bases written "
          f"(canonical for the held-out wealth test = rb_scf2022_wealth_cont)")


if __name__ == "__main__":
    main()
