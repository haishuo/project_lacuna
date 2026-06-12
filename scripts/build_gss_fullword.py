"""
scripts/build_gss_fullword.py

Upgrade the GSS text corpus with FULL question wording from the GSS 2024 Codebook PDF (NORC;
docs/DATA-CITATIONS.md). The codebook's INDEX OF VARIABLES lists entries as
"VARNAME <section-number> <question text...>"; we split the concatenated PDF text on that
pattern and map varname -> wording. Conservative: items keep their short label when no entry is
found (full_text empty), never guessed. Kills the REGISTER confound (GSS telegraphic labels vs
ESS interview paragraphs) flagged in the three-instrument probe.

Output: rewrites /mnt/data/lacuna/role_b/gss_text_corpus.csv with (full_text) added.
Run: /home/haishuo/miniconda3/envs/lacuna/bin/python -u scripts/build_gss_fullword.py
"""

import re
from pathlib import Path

import pandas as pd
from pypdf import PdfReader

PDF = Path("/mnt/data/lacuna/incoming/GSS 2024 Codebook R3.pdf")
CORPUS = Path("/mnt/data/lacuna/role_b/gss_text_corpus.csv")
HEADER_PAT = re.compile(
    r"General Social Survey Codebook Produced on: [^|]*?\(Release \d\)\s*"
    r"|Codebook and Unweighted Frequencies for the 2024 General Social Survey \(GSS\)\s*"
    r"|SECTION INDEX OF VARIABLES Page \d+\s*"
    r"|Variable Name Section SAS_LABEL\s*")
# entry delimiter: VARNAME (upper alnum, >=3 chars) + section number like 1.000
ENTRY_PAT = re.compile(r"\b([A-Z][A-Z0-9_]{2,15})\s+\d{1,2}\.\d{3}\s+")


def main():
    reader = PdfReader(PDF)
    text = " ".join((p.extract_text() or "") for p in reader.pages)
    text = re.sub(r"\s+", " ", text)
    text = HEADER_PAT.sub(" ", text)
    parts = ENTRY_PAT.split(text)
    # parts: [pre, VAR1, body1, VAR2, body2, ...]
    wording = {}
    for k in range(1, len(parts) - 1, 2):
        var = parts[k].lower()
        body = parts[k + 1].strip()
        # body may include trailing interviewer junk; cap length, require sentence-like content
        if 15 <= len(body) <= 1200 and re.search(r"[a-z]", body):
            if var not in wording or len(body) > len(wording[var]):
                wording[var] = body[:600]
    print(f"extracted wording for {len(wording)} variables from codebook index")
    df = pd.read_csv(CORPUS)
    df["full_text"] = df["var"].map(wording).fillna("")
    direct = (df["full_text"] != "").sum()
    # FAMILY PROPAGATION: wave-variants (income06/income98/...) are the same question re-fielded;
    # inherit wording from any family member (var with trailing digits stripped). Flagged in
    # full_text_source so inherited wording is auditable.
    fam = df["var"].str.replace(r"\d+$", "", regex=True)
    fam_word = {}
    for f, w in zip(fam, df["full_text"]):
        if w and (f not in fam_word or len(w) > len(fam_word[f])):
            fam_word[f] = w
    for b, w in wording.items():
        fb = re.sub(r"\d+$", "", b)
        if fb not in fam_word:
            fam_word[fb] = w
    inherit = (df["full_text"] == "") & fam.map(lambda f: f in fam_word)
    df.loc[inherit, "full_text"] = fam.map(fam_word)[inherit]
    df["full_text_source"] = ""
    df.loc[df["full_text"] != "", "full_text_source"] = "codebook_index"
    df.loc[inherit, "full_text_source"] = "family_inherited"
    matched = (df["full_text"] != "").sum()
    print(f"  direct {direct} + family-inherited {int(inherit.sum())}")
    df.to_csv(CORPUS, index=False)
    print(f"rewrote {CORPUS}: {matched}/{len(df)} items now carry full wording")
    chk = df[df["var"].isin(["rincome", "income", "attend", "pray", "partyid"])]
    for _, r in chk.iterrows():
        print(f"  {r['var']}: {r['full_text'][:100] or '(unmatched)'}")


if __name__ == "__main__":
    main()
