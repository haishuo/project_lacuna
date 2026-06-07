"""
scripts/build_regime_map.py

D2 (SCF consolidation memo): a footprint-geometry / distribution-regime MAP over the EXISTING corpus,
and a test of whether regime-distance POSTDICTS the observed leave-one-domain-out transfer (Δ).

NO model, NO training, NO acquisition — descriptive statistics on the role-B/native columns the curve
already used. Deterministic.

Method:
  1. For every non-constant column of each curve domain, compute the scale-invariant regime vector
     (l_skew, l_kurt, log10_card, has_neg) — `lacuna.survey.regime_descriptors`.
  2. Standardize each coordinate across ALL columns (z-score) ⇒ a common regime space.
  3. coverage_distance(D | pool) = mean over D's columns c of  min over pool columns t (t∉D) of
     ||z(c) - z(t)||  — "how well does the training POOL cover D's footprint geometry?" (lower = better).
  4. coverage_gain(D) = coverage_distance(D | narrow) − coverage_distance(D | diverse)  (≥0 ⇒ adding the
     diverse domains brings training geometrically closer to D).
  5. POSTDICTION: H1 predicts observed Δ rises with coverage_gain. Report both, plus the isolation of each
     domain (nearest-neighbour distance to the rest of the corpus). n=4 held-out domains ⇒ ordering, not
     proof.

Run: /home/haishuo/miniconda3/envs/lacuna/bin/python -u scripts/build_regime_map.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from lacuna.data.catalog import create_default_catalog
from lacuna.survey.regime_descriptors import PRIMARY_KEYS, shape_descriptors

RB = Path("/mnt/data/lacuna/role_b")
OUT = Path("runs/regime_map.json")

# Domains exactly as the curve's pools define them (run_curve_tightened.py).
NATIVE = {"labor": ["survey_cps1988", "survey_psid1976", "survey_psid7682"],
          "psychology": ["survey_bfi"], "health": ["survey_yrbss"]}
ROLE_B = {"nhanes": ["rb_nhanes_weight", "rb_nhanes_poverty"], "wealth": ["rb_scf2022_wealth_cont"]}

# Observed leave-one-domain-out Δ (continuous-only wealth; findings §7–§8). narrow=labor.
OBSERVED = {
    "psychology": {"delta": +0.064, "diverse": ["labor", "health", "nhanes"], "sig": "HELPS"},
    "health":     {"delta": +0.037, "diverse": ["labor", "psychology", "nhanes"], "sig": "HELPS"},
    "nhanes":     {"delta": +0.045, "diverse": ["labor", "psychology", "health"], "sig": "noisy"},
    "wealth":     {"delta": -0.053, "diverse": ["labor", "psychology", "health", "nhanes"], "sig": "flat"},
}


def _columns_of(domain, names, cat):
    """List of (col_id, observed_values) for every non-constant column across a domain's datasets."""
    cols = []
    for name in names:
        if name in ROLE_B.get(domain, []) or name.startswith("rb_"):
            df = pd.read_csv(RB / f"{name}.csv")
            mat, feats = df.to_numpy(np.float64), list(df.columns)
        else:
            raw = cat.load(name)
            mat, feats = np.asarray(raw.data, np.float64), list(raw.feature_names)
        for j, f in enumerate(feats):
            v = mat[:, j]
            v = v[np.isfinite(v)]
            if v.size >= 8 and np.unique(v).size >= 2:
                cols.append((f"{name}.{f}", v))
    return cols


def main():
    cat = create_default_catalog()
    domain_names = {**NATIVE, **ROLE_B}
    # 1. descriptors per column, tagged by domain
    per_domain_cols, rows = {}, []
    for dom, names in domain_names.items():
        cols = _columns_of(dom, names, cat)
        descs = [(cid, shape_descriptors(v)) for cid, v in cols]
        per_domain_cols[dom] = descs
        for cid, d in descs:
            rows.append({"domain": dom, "col": cid, **{k: d[k] for k in PRIMARY_KEYS},
                         "card": d["card"], "skew": round(d["skew"], 2),
                         "excess_kurt": round(d["excess_kurt"], 1)})
    # 2. standardize the primary coordinates across ALL columns
    X = np.array([[r[k] for k in PRIMARY_KEYS] for r in rows], dtype=np.float64)
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    zmap = {r["col"]: Z[i] for i, r in enumerate(rows)}

    def cov_dist(D, pool_domains):
        pool = [c for pd_ in pool_domains for (c, _) in per_domain_cols[pd_]]
        dists = []
        for (c, _) in per_domain_cols[D]:
            zc = zmap[c]
            dists.append(min(np.linalg.norm(zc - zmap[t]) for t in pool))
        return float(np.mean(dists))

    # 3. domain × regime MAP (median scale-invariant coordinates)
    print("=" * 100)
    print("DOMAIN × REGIME MAP (median scale-invariant footprint coordinates per domain)")
    print("=" * 100)
    print(f"{'domain':12} {'n_col':>5} {'L-skew':>8} {'L-kurt':>8} {'log10card':>10} "
          f"{'has_neg':>8} {'med_card':>9} {'med_skew':>9} {'med_exkurt':>11}")
    regime_table = {}
    for dom in domain_names:
        ds = [d for _, d in per_domain_cols[dom]]
        med = {k: float(np.median([d[k] for d in ds])) for k in
               ("l_skew", "l_kurt", "log10_card", "has_neg", "card", "skew", "excess_kurt")}
        regime_table[dom] = {**med, "n_col": len(ds)}
        print(f"{dom:12} {len(ds):5d} {med['l_skew']:8.3f} {med['l_kurt']:8.3f} {med['log10_card']:10.2f} "
              f"{med['has_neg']:8.2f} {med['card']:9.0f} {med['skew']:9.2f} {med['excess_kurt']:11.1f}")

    # 4. coverage gain + isolation, vs observed Δ (POSTDICTION)
    print("\n" + "=" * 100)
    print("POSTDICTION — does regime coverage_gain track observed leave-one-domain-out Δ?")
    print("=" * 100)
    all_doms = list(domain_names)
    post = []
    print(f"{'held-out D':12} {'obs Δ':>8} {'sig':>6} {'cov_narrow':>11} {'cov_diverse':>12} "
          f"{'cov_GAIN':>9} {'isolation':>10}")
    for D, info in OBSERVED.items():
        cn = cov_dist(D, ["labor"])
        cd = cov_dist(D, info["diverse"])
        gain = cn - cd
        isolation = cov_dist(D, [x for x in all_doms if x != D])  # nearest other-domain column distance
        post.append({"domain": D, "delta": info["delta"], "cov_narrow": cn, "cov_diverse": cd,
                     "cov_gain": gain, "isolation": isolation, "sig": info["sig"]})
        print(f"{D:12} {info['delta']:+8.3f} {info['sig']:>6} {cn:11.3f} {cd:12.3f} {gain:9.3f} {isolation:10.3f}")

    # rank agreement between coverage_gain and observed Δ (Spearman over n=4 — indicative only)
    g = np.array([p["cov_gain"] for p in post]); dl = np.array([p["delta"] for p in post])
    iso = np.array([p["isolation"] for p in post])
    def spearman(a, b):
        ra, rb = a.argsort().argsort().astype(float), b.argsort().argsort().astype(float)
        return float(np.corrcoef(ra, rb)[0, 1])
    print("\n  rank corr (coverage_gain , observed Δ)  =", round(spearman(g, dl), 3), " [n=4, indicative]")
    print("  rank corr (isolation     , observed Δ)  =", round(spearman(iso, dl), 3),
          " [negative expected: more isolated ⇒ less transfer]")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"regime_table": regime_table, "postdiction": post,
                               "standardize_mu": mu.tolist(), "standardize_sd": sd.tolist(),
                               "primary_keys": list(PRIMARY_KEYS)}, indent=2))
    print(f"\n  wrote {OUT}")


if __name__ == "__main__":
    main()
