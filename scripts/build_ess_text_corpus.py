"""
scripts/build_ess_text_corpus.py

Build the ESS item-TEXT -> missingness-BEHAVIOR corpus: join ESS11 variable label text (from the
.dta, ESS ERIC 2026, doi:10.21338/ess11e04_1 — see docs/DATA-CITATIONS.md) to the per-item
documented nonresponse counts (refusal/DK/no-answer/not-applicable sentinel families) computed
from the coded CSV (same edition 4.1). Columns admitted by the same conservative out-of-range
sentinel filter as the Stage-1/2 showdown (age-77 trap guard).

Output: /mnt/data/lacuna/role_b/ess_text_corpus.csv — one row per item:
(survey, var, text, n_valid, n_refused, n_dontknow, n_noanswer, n_skip, refusal_rate, dk_rate).

Run: /home/haishuo/miniconda3/envs/lacuna/bin/python -u scripts/build_ess_text_corpus.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

from lacuna.survey.ess_codes import resolve_ess_column

CSV = Path("/mnt/data/lacuna/rejected/ESS11e04_1.csv")
DTA = Path("/mnt/data/lacuna/incoming/ESS11e04_1.dta")
OUT = Path("/mnt/data/lacuna/role_b/ess_text_corpus.csv")


def main():
    labels = pd.read_stata(DTA, iterator=True).variable_labels()
    ess = pd.read_csv(CSV, low_memory=False)
    num = ess.select_dtypes(include=[np.number])
    rows = []
    for c in num.columns:
        codes = resolve_ess_column(num[c].values)
        if codes is None:
            continue
        text = labels.get(c, "").strip()
        if not text or text.lower() == c.lower():
            continue
        v = num[c].dropna().values
        n_ref = int(np.isin(v, list(codes.refusal)).sum())
        n_dk = int(np.isin(v, list(codes.dont_know)).sum())
        n_na = int(np.isin(v, list(codes.no_answer)).sum())
        n_skip = int(np.isin(v, list(codes.not_applicable)).sum())
        n_valid = int(len(v) - n_ref - n_dk - n_na - n_skip)
        denom = max(n_valid + n_ref + n_dk + n_na, 1)   # eligible respondents (skip excluded)
        rows.append({"survey": "ESS11", "var": c, "text": text, "n_valid": n_valid,
                     "n_refused": n_ref, "n_dontknow": n_dk, "n_noanswer": n_na, "n_skip": n_skip,
                     "refusal_rate": n_ref / denom, "dk_rate": n_dk / denom})
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("no ESS items extracted — label join or filter failed")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT}: {len(df)} items with TEXT + behavior labels")
    print(f"  total refusals {int(df.n_refused.sum())} | items refusal_rate>=0.01: {(df.refusal_rate>=0.01).sum()}")
    top = df.sort_values("refusal_rate", ascending=False).head(8)
    print("  highest refusal-rate items:")
    for _, r in top.iterrows():
        print(f"    [{r['refusal_rate']:.3f}] {r['var']}: {r['text'][:80]}")
    low = df.sort_values("refusal_rate").head(3)
    print("  lowest:")
    for _, r in low.iterrows():
        print(f"    [{r['refusal_rate']:.3f}] {r['var']}: {r['text'][:80]}")


if __name__ == "__main__":
    main()
