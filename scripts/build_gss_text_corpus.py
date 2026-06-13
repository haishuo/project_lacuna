"""
scripts/build_gss_text_corpus.py

Build the GSS item-TEXT -> missingness-BEHAVIOR corpus from the GSS 1972-2024 cumulative file
(NORC; see docs/data/DATA-CITATIONS.md). GSS is the CLEANEST label source: refusal/DK/no-answer/skip
are EXPLICITLY TYPED Stata extended-missing codes (.r/.d/.n/.i/.s) — no sentinel inference, no
width heuristics, the age-77 trap is impossible by construction.

Per variable: label text + counts of r/d/n/i/s + valid; rates over eligible respondents
(valid + r + d + n; 'i'=IAP routing and 's'=skipped excluded from the denominator). Variables
must have label text and >=200 eligible respondents.

Output: /mnt/data/lacuna/role_b/gss_text_corpus.csv
Run: /home/haishuo/miniconda3/envs/lacuna/bin/python -u scripts/build_gss_text_corpus.py
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat

DTA = Path("/mnt/data/lacuna/incoming/gss7224_r3.dta")
OUT = Path("/mnt/data/lacuna/role_b/gss_text_corpus.csv")
MIN_ELIGIBLE = 200
CHUNK = 400  # columns per read pass


def main():
    _, meta = pyreadstat.read_dta(str(DTA), metadataonly=True, encoding="latin1")
    cols = meta.column_names
    labels = meta.column_names_to_labels
    print(f"{len(cols)} variables in cumulative file")
    rows = []
    t0 = time.time()
    for s in range(0, len(cols), CHUNK):
        sub = cols[s:s + CHUNK]
        df, _ = pyreadstat.read_dta(str(DTA), usecols=sub, user_missing=True, encoding="latin1")
        for c in sub:
            text = (labels.get(c) or "").strip()
            if not text or text.lower() == c.lower():
                continue
            v = df[c]
            sv = v[v.notna()].astype(str)
            n_r = int((sv == "r").sum()); n_d = int((sv == "d").sum())
            n_n = int((sv == "n").sum()); n_i = int((sv == "i").sum()); n_s = int((sv == "s").sum())
            n_other_codes = int(sv.str.fullmatch(r"[a-z]").sum()) - (n_r + n_d + n_n + n_i + n_s)
            n_valid = int(len(sv) - n_r - n_d - n_n - n_i - n_s - max(n_other_codes, 0))
            eligible = n_valid + n_r + n_d + n_n
            if eligible < MIN_ELIGIBLE or n_valid <= 0:
                continue
            rows.append({"survey": "GSS", "var": c, "text": text, "n_valid": n_valid,
                         "n_refused": n_r, "n_dontknow": n_d, "n_noanswer": n_n,
                         "n_skip": n_i + n_s,
                         "refusal_rate": n_r / eligible, "dk_rate": n_d / eligible})
        print(f"  [{time.time()-t0:6.1f}s] {min(s+CHUNK,len(cols))}/{len(cols)} cols, {len(rows)} items kept")
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("no GSS items extracted")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"\nwrote {OUT}: {len(df)} items")
    print(f"  total refusals {int(df.n_refused.sum())} | items refusal_rate>=0.01: {(df.refusal_rate>=0.01).sum()}")
    for _, r in df.sort_values("refusal_rate", ascending=False).head(8).iterrows():
        print(f"  [{r['refusal_rate']:.3f}] {r['var']}: {r['text'][:75]}")


if __name__ == "__main__":
    main()
