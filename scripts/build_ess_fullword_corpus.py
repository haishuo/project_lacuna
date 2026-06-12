"""
scripts/build_ess_fullword_corpus.py

Upgrade the ESS text corpus with FULL question wording from the ESS Round 11 Source Questionnaire
PDF (ESS ERIC; see docs/DATA-CITATIONS.md). The questionnaire indexes items by question number
(e.g. C11), not variable name, so questions are matched to corpus items by token-overlap between
the .dta variable label and the question wording. Conservative: a match below the overlap floor
keeps the short label only (full_text empty) — never guess.

Output: rewrites /mnt/data/lacuna/role_b/ess_text_corpus.csv with added columns
(qnum, full_text, match_score). Items keep working with the short label when unmatched.

Run: /home/haishuo/miniconda3/envs/lacuna/bin/python -u scripts/build_ess_fullword_corpus.py
"""

import re
from pathlib import Path

import pandas as pd
from pypdf import PdfReader

PDF = Path("/mnt/data/lacuna/incoming/ESS Round 11 Source Questionnaire_FINAL_Alert 04.pdf")
CORPUS = Path("/mnt/data/lacuna/role_b/ess_text_corpus.csv")
MATCH_FLOOR = 0.5
STOP = {"the", "a", "an", "of", "to", "in", "on", "at", "and", "or", "is", "are", "you", "your",
        "how", "do", "did", "does", "any", "for", "with", "from", "as", "by", "be", "been",
        "what", "which", "who", "all", "this", "that", "card", "still", "please", "tell", "me",
        "using", "would", "say", "have", "has", "ever", "apart"}


def tokens(s):
    return {w for w in re.findall(r"[a-z]+", s.lower()) if w not in STOP and len(w) > 2}


def extract_questions(pdf_path):
    """Split the questionnaire into (qnum, wording) blocks at question-number markers."""
    reader = PdfReader(pdf_path)
    full = "\n".join((p.extract_text() or "") for p in reader.pages)
    full = re.sub(r"\s+", " ", full)
    # question markers: letter+number at a word boundary (A1..F41, plus module letters)
    parts = re.split(r"\b([A-Z]\d{1,2}[a-z]?)\s", full)
    out = {}
    for k in range(1, len(parts) - 1, 2):
        qnum, body = parts[k], parts[k + 1]
        # wording = text up to the first answer-scale signature or next instruction
        cut = re.split(r"\(Refusal\)|\(Don.t know\)|ASK IF|GO TO|INTERVIEWER", body)[0]
        wording = cut.strip()
        if 15 <= len(wording) <= 600:
            # keep the longest body seen per qnum (rotating-module duplicates)
            if qnum not in out or len(wording) > len(out[qnum]):
                out[qnum] = wording
    return out


def main():
    df = pd.read_csv(CORPUS)
    questions = extract_questions(PDF)
    print(f"extracted {len(questions)} question blocks from the source questionnaire")
    qtok = {q: tokens(w) for q, w in questions.items()}
    qnums, fulls, scores = [], [], []
    for _, row in df.iterrows():
        lt = tokens(str(row["text"]))
        best_q, best_s = "", 0.0
        if lt:
            for q, wt in qtok.items():
                if not wt:
                    continue
                s = len(lt & wt) / len(lt)
                if s > best_s:
                    best_q, best_s = q, s
        if best_s >= MATCH_FLOOR:
            qnums.append(best_q); fulls.append(questions[best_q]); scores.append(round(best_s, 3))
        else:
            qnums.append(""); fulls.append(""); scores.append(round(best_s, 3))
    df["qnum"], df["full_text"], df["match_score"] = qnums, fulls, scores
    df.to_csv(CORPUS, index=False)
    matched = (df["full_text"] != "").sum()
    print(f"rewrote {CORPUS}: {matched}/{len(df)} items matched to full wording (floor {MATCH_FLOOR})")
    for _, r in df[df["full_text"] != ""].sort_values("n_refused", ascending=False).head(3).iterrows():
        print(f"  [{r['var']} <- {r['qnum']} @ {r['match_score']}] {r['full_text'][:110]}")
    for _, r in df[df["full_text"] == ""].sort_values("n_refused", ascending=False).head(3).iterrows():
        print(f"  UNMATCHED [{r['var']}] label: {r['text'][:80]}")


if __name__ == "__main__":
    main()
