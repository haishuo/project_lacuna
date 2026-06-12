"""
scripts/build_nhanes_text_corpus.py

Build the item-TEXT -> missingness-BEHAVIOR corpus from real NHANES codebooks + data. For each
variable: parse the codebook for its English question text AND the codes its value-table labels
"Refused"/"Don't know" (codebook-grounded — avoids the age-77 trap), then count those codes in the
real .xpt. Emits one row per item: (survey, module, var, label, text, n, n_valid, n_refused,
n_dontknow, refusal_rate, dk_rate). This is the seed corpus for the semantic channel (item text ->
refusal behavior). NHANES = ONE survey block (same respondents); diverse TEXT across modules.

Run: /home/haishuo/miniconda3/envs/lacuna/bin/python -u scripts/build_nhanes_text_corpus.py
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

CB = Path("/mnt/data/lacuna/incoming/codebooks")
XPT = Path("/mnt/data/lacuna/incoming")
OUT = Path("/mnt/data/lacuna/role_b/nhanes_text_corpus.csv")
MODULES = ["DPQ", "INQ", "WHQ", "DEMO", "DUQ"]


def parse_codebook(htm):
    """Return {var: {'label':.., 'text':.., 'refused':set(codes), 'dontknow':set(codes)}}."""
    h = htm.read_text(encoding="utf-8", errors="ignore")
    out = {}
    # split into per-variable detail blocks at the vartitle headers
    blocks = re.split(r'<h3 class="vartitle" id="([A-Z0-9_]+)">', h)
    # blocks[0] = preamble; then alternating (varname, body)
    for k in range(1, len(blocks), 2):
        var = blocks[k]
        body = blocks[k + 1]
        label_m = re.search(r'%s\s*-\s*([^<]+)</h3>' % re.escape(var), "<h3 class=\"vartitle\" id=\"%s\">%s" % (var, body[:200]))
        label = label_m.group(1).strip() if label_m else ""
        text_m = re.search(r'English Text:\s*</dt>\s*<dd[^>]*>(.*?)</dd>', body, re.S)
        text = re.sub(r'\s+', " ", re.sub(r'<[^>]+>', " ", text_m.group(1))).strip() if text_m else ""
        refused, dk = set(), set()
        # value table rows: <td>CODE</td> <td>DESCRIPTION</td>
        vtab = body.split("</table>")[0]
        for code, desc in re.findall(r'<td[^>]*>\s*([0-9]+)\s*</td>\s*<td[^>]*>(.*?)</td>', vtab, re.S):
            d = re.sub(r'<[^>]+>', " ", desc).lower()
            if "refus" in d:
                refused.add(int(code))
            elif "don't know" in d or "dont know" in d or "don&#39;t know" in d:
                dk.add(int(code))
        out[var] = {"label": label, "text": text, "refused": refused, "dontknow": dk}
    return out


def main():
    rows = []
    for mod in MODULES:
        cb = parse_codebook(CB / f"{mod}_J.htm")
        data = pd.read_sas(XPT / f"{mod}_J.xpt", format="xport")
        for var, info in cb.items():
            if var not in data.columns or not info["text"]:
                continue
            s = data[var]
            ref = info["refused"] or {7, 77, 7777}      # fall back to NHANES convention if table silent
            dkc = info["dontknow"] or {9, 99, 9999}
            n_ref = int(s.isin(list(ref)).sum())
            n_dk = int(s.isin(list(dkc)).sum())
            n_valid = int(s.notna().sum() - n_ref - n_dk)
            n = int(len(s))
            if n_valid <= 0:
                continue
            denom = max(n_valid + n_ref + n_dk, 1)
            rows.append({"survey": "NHANES", "module": mod, "var": var, "label": info["label"],
                         "text": info["text"], "n": n, "n_valid": n_valid, "n_refused": n_ref,
                         "n_dontknow": n_dk, "refusal_rate": n_ref / denom, "dk_rate": n_dk / denom})
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("no items extracted — codebook parse failed")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT}: {len(df)} items with TEXT + behavior labels")
    print(f"  modules: {df['module'].value_counts().to_dict()}")
    print(f"  items with >=1 refusal: {(df['n_refused'] > 0).sum()} | >=20 refusals: {(df['n_refused'] >= 20).sum()}")
    print(f"  refusal_rate: median {df['refusal_rate'].median():.4f} max {df['refusal_rate'].max():.4f}")
    top = df.sort_values("n_refused", ascending=False).head(6)
    print("  most-refused items (text -> refusal_rate):")
    for _, r in top.iterrows():
        print(f"    [{r['refusal_rate']:.3f}] {r['var']}: {r['text'][:90]}")


if __name__ == "__main__":
    main()
