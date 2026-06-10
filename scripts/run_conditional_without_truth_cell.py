"""
scripts/run_conditional_without_truth_cell.py

The EMPTY CELL — conditional-without-truth, executed exactly per the LOCKED pre-registration
(`PREREGISTRATION-conditional-without-truth-cell.md`, commit e1374f9; PI-approved). Isolates the source
of the G1 gain: truth-bottleneck vs conditional-structure. No tuning; bands/stability rule locked.

Protocol identity with G1 (binding): reuses run_g1_imputation_channel's cont_base/mech_mask/
pooled_oof_auc VERBATIM, the same cell enumeration and per-cell RNG seeds (31000+i) so the realized
examples and mechanism masks are BIT-IDENTICAL to G1's — verified by recomputing the frozen base17
features and asserting equality against the stored G1 rows BEFORE any new feature is trusted.
ONLY the features change: the 16 no-truth features (mech view only; leak guards L1–L8; the paired-MCAR
view is FORBIDDEN — truth access in disguise).

Decision (locked §5): per-imputer recovery fraction rho = (AUC_cell − 0.574) / (AUC_G1 − 0.574),
anchors G1 = {linear .900, mice .900, rf .989, gbm .993, nn .994}; bands truth-bottleneck rho<=0.25 or
AUC<=0.62 / conditional-structure rho>=0.60 or AUC>=0.80 / mixed between; verdict = band shared by >=3
of the 4 DISTINCT classes (mice≡linear), else "imputer-dependent (mixed)".

Run: /home/haishuo/miniconda3/envs/lacuna/bin/python -u scripts/run_conditional_without_truth_cell.py
"""

import importlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from lacuna.core.rng import RNGState
from lacuna.survey.consequence_features import compute_consequence_features
from lacuna.survey.imputation_channel import IMPUTERS, NO_TRUTH_FEATURE_NAMES, no_truth_features

G1 = importlib.import_module("scripts.run_g1_imputation_channel")  # protocol source of truth
G1_ROWS = json.loads(Path("runs/g1_imputation_channel.json").read_text())["rows"]
OUT = Path("runs/conditional_without_truth_cell.json")

PHI_ANCHOR = {"own_value": 0.574, "top_coding": 0.633}
G1_ANCHOR = {"own_value": {"linear": 0.900, "mice_lite": 0.900, "rf": 0.989, "gbm": 0.993, "nn": 0.994},
             "top_coding": {"linear": 0.885, "mice_lite": 0.885, "rf": 0.947, "gbm": 0.943, "nn": 0.883}}
DISTINCT = ("linear", "rf", "gbm", "nn")  # mice_lite ≡ linear (complete predictors)


def build_rows():
    """G1's enumeration verbatim; base17 identity-checked against the stored G1 rows."""
    rows, cell_i, ex_i, g1_ptr, checked = [], 0, 0, 0, 0
    for dom in sorted(G1.DOMAINS):
        for name in G1.DOMAINS[dom]:
            base = G1.cont_base(name)
            for t_idx, t_name in enumerate(base.feature_names):
                for idiom in G1.IDIOMS:
                    for delta in G1.DELTAS:
                        rng = RNGState(seed=31000 + cell_i)
                        cell_i += 1
                        for e in range(G1.N_EX):
                            sub = G1.subsample_raw(base, max_rows=G1.MAX_ROWS, rng=rng.spawn())
                            X_t = torch.from_numpy(np.asarray(sub.data, np.float32))
                            try:
                                m_mech = G1.mech_mask(X_t, t_idx, idiom, delta, rng)
                                _ = G1.mcar_pair_mask(m_mech, rng.spawn())  # consume G1's spawn; UNUSED (L1)
                                # bit-identity check: recomputed base17 must equal the stored G1 row
                                R = torch.ones_like(X_t, dtype=torch.bool)
                                R[:, t_idx] = torch.from_numpy(m_mech)
                                b17 = compute_consequence_features(X_t * R.float(), R, t_idx).numpy()
                                g1r = G1_ROWS[g1_ptr]
                                if (g1r["dataset"], g1r["target"], g1r["idiom"], g1r["delta"]) != \
                                        (name, t_name, idiom, delta):
                                    raise RuntimeError(f"enumeration drift at G1 row {g1_ptr}")
                                if not np.allclose(b17, np.array(g1r["base17"]), atol=1e-5):
                                    raise RuntimeError(f"base17 mismatch at G1 row {g1_ptr} — examples NOT identical")
                                g1_ptr += 1; checked += 1
                                # the no-truth view: punched target cells = NaN; truth never passed
                                view = X_t.numpy().astype(np.float64)
                                view[~m_mech, t_idx] = np.nan
                                feats = {}
                                hr = RNGState(seed=90000 + ex_i)  # holdout seed from example index only (L8)
                                for imp in IMPUTERS:
                                    feats[imp] = no_truth_features(view, t_idx, imputer=imp,
                                                                   seed=1000 + e, holdout_rng=hr.spawn())
                            except ValueError as err:
                                print(f"  skip {name}.{t_name} {idiom} d={delta} ex{e}: {err}")
                                ex_i += 1
                                continue
                            rows.append({"domain": dom, "dataset": name, "target": t_name,
                                         "idiom": idiom, "delta": delta, "base17": g1r["base17"],
                                         **{f"{imp}.{k}": v for imp in IMPUTERS
                                            for k, v in feats[imp].items()}})
                            ex_i += 1
        print(f"[{time.strftime('%H:%M:%S')}] domain {dom} done ({len(rows)} rows; {checked} identity-checked)")
    if g1_ptr != len(G1_ROWS):
        raise RuntimeError(f"row count mismatch: consumed {g1_ptr} of {len(G1_ROWS)} G1 rows")
    return rows


def main():
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    print("=" * 100)
    print("EMPTY CELL — conditional WITHOUT truth (locked prereg e1374f9; G1 protocol identity verified)")
    print("=" * 100)
    rows = build_rows()
    print(f"total rows: {len(rows)} (all base17-identity-checked against G1)")

    cell = lambda imp: (lambda r: [r[f"{imp}.{k}"] for k in NO_TRUTH_FEATURE_NAMES])
    base = lambda r: r["base17"]
    both = lambda imp: (lambda r: r["base17"] + [r[f"{imp}.{k}"] for k in NO_TRUTH_FEATURE_NAMES])
    y_delta = lambda r: int(r["delta"] > 0)

    res = {"git": git, "n_rows": len(rows), "per_idiom": {}}
    for idiom in ("own_value", "top_coding"):
        sel = lambda r, idm=idiom: r["idiom"] == idm
        res["per_idiom"][idiom] = {}
        print(f"\n== {idiom} (phi anchor {PHI_ANCHOR[idiom]}) ==")
        print(f"{'imputer':10} {'AUC_cell':>8} {'rho':>7} {'band':>22} {'base+cell':>9} {'incr':>7}")
        for imp in IMPUTERS:
            auc = G1.pooled_oof_auc(rows, sel, y_delta, cell(imp))
            full = G1.pooled_oof_auc(rows, sel, y_delta, both(imp))
            rho = (auc - PHI_ANCHOR[idiom]) / (G1_ANCHOR[idiom][imp] - PHI_ANCHOR[idiom])
            band = ("truth_bottleneck" if (rho <= 0.25 or auc <= 0.62) else
                    "conditional_structure" if (rho >= 0.60 or auc >= 0.80) else "mixed")
            res["per_idiom"][idiom][imp] = {"auc_cell": auc, "rho": rho, "band": band,
                                            "auc_base_plus_cell": full,
                                            "increment": full - PHI_ANCHOR[idiom]}
            print(f"{imp:10} {auc:8.3f} {rho:7.3f} {band:>22} {full:9.3f} {full-PHI_ANCHOR[idiom]:+7.3f}")
        # idiom-separation, reported not gated
    isep = {imp: G1.pooled_oof_auc(rows, lambda r: r["delta"] > 0,
                                   lambda r: int(r["idiom"] == "top_coding"), cell(imp))
            for imp in IMPUTERS}
    print(f"\nidiom separation at delta=2.5 (reported): "
          f"{ {k: round(v,3) for k, v in isep.items()} }")
    res["idiom_separation"] = isep

    # locked stability rule: band shared by >=3 of the 4 DISTINCT classes (primary = own_value)
    bands = [res["per_idiom"]["own_value"][i]["band"] for i in DISTINCT]
    counts = {b: bands.count(b) for b in set(bands)}
    verdict_band = next((b for b, c in counts.items() if c >= 3), "imputer_dependent_mixed")
    res["verdict"] = {"primary_bands_distinct_classes": dict(zip(DISTINCT, bands)),
                      "verdict_band": verdict_band}
    print("\n" + "=" * 100)
    print(f"PRIMARY (own_value) bands by distinct class: {dict(zip(DISTINCT, bands))}")
    print(f"VERDICT (locked stability rule, >=3/4): {verdict_band}")
    OUT.write_text(json.dumps({**res, "rows": rows}, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
