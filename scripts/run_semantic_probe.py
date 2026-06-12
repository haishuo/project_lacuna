"""
scripts/run_semantic_probe.py

SEMANTIC PROBE (Arm 1 vs Arm 2 of the semantic-channel design): does item TEXT predict real
refusal behavior — and does a frozen off-the-shelf embedding beat a keyword/TF-IDF null?
(Arm 3, the behavior-TRAINED encoder, is a separate later experiment; this probe establishes
whether the semantic signal exists at all and what the commodity ceiling is.)

Items: ESS11 corpus (349 items; full question wording where matched, else label text) + NHANES
corpus (151 items, codebook question text). Targets: per-item refusal_rate (primary; Spearman +
top-vs-bottom-tertile AUC) and dk_rate (secondary).

Leakage guards:
  - FAMILY BLOCKS within ESS: country-variants of one question (prtvt* etc.) share a variable
    prefix; grouped 5-fold CV never splits a family across train/test (paraphrase-leak guard).
  - LEAVE-INSTRUMENT-OUT: train ESS -> test NHANES (and reverse) with metrics computed WITHIN the
    held-out instrument (rates are not comparable across instruments).

Arms (both shallow heads; the question is the TEXT REPRESENTATION):
  A1 keyword null: TF-IDF unigram+bigram -> ridge / LR.
  A2 frozen embedding: all-MiniLM-L6-v2 (384d, frozen) -> ridge / LR.

Deterministic (fixed seeds); fail-loud. Run:
  /home/haishuo/miniconda3/envs/lacuna/bin/python -u scripts/run_semantic_probe.py
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score

ESS = Path("/mnt/data/lacuna/role_b/ess_text_corpus.csv")
NH = Path("/mnt/data/lacuna/role_b/nhanes_text_corpus.csv")
OUT = Path("runs/semantic_probe.json")
SEED = 2026


def family(var):
    """ESS family block: strip trailing country-code-ish suffix; conservative prefix grouping."""
    m = re.match(r"([a-z]+?)(?:[a-z]{0,2}\d*)$", var)
    base = var[:5] if len(var) > 5 else var
    return base


def load_items():
    ess = pd.read_csv(ESS)
    ess["item_text"] = np.where(ess["full_text"].fillna("") != "", ess["full_text"], ess["text"])
    ess["instrument"] = "ESS11"
    ess["block"] = ess["var"].map(family)
    nh = pd.read_csv(NH)
    nh["item_text"] = nh["text"]
    nh["instrument"] = "NHANES"
    nh["block"] = "nh_" + nh["module"]
    cols = ["instrument", "var", "item_text", "refusal_rate", "dk_rate", "block"]
    df = pd.concat([ess[cols], nh[cols]], ignore_index=True)
    df = df[df["item_text"].str.len() > 5].reset_index(drop=True)
    return df


def embed_texts(texts):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return np.asarray(model.encode(list(texts), batch_size=64, show_progress_bar=False))


def tertile_labels(y):
    lo, hi = np.quantile(y, [1 / 3, 2 / 3])
    lab = np.full(len(y), -1)
    lab[y <= lo] = 0
    lab[y >= hi] = 1
    return lab


def eval_split(Xtr_txt, Xte_txt, Etr, Ete, ytr, yte, seed):
    """Return {arm: {spearman, tertile_auc}} for one train/test split."""
    res = {}
    # A1 TF-IDF
    tf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=5000)
    Ttr = tf.fit_transform(Xtr_txt)
    Tte = tf.transform(Xte_txt)
    for arm, (ftr, fte) in [("tfidf", (Ttr, Tte)), ("embed", (Etr, Ete))]:
        ridge = Ridge(alpha=1.0).fit(ftr, ytr)
        pred = ridge.predict(fte)
        sp = float(spearmanr(yte, pred).statistic) if len(set(yte)) > 1 else float("nan")
        # tertile AUC: classify high vs low refusal within the test set
        lab_te = tertile_labels(yte)
        m = lab_te >= 0
        auc = float("nan")
        if m.sum() >= 10 and len(set(lab_te[m])) == 2:
            lab_tr = tertile_labels(ytr)
            mt = lab_tr >= 0
            lr = LogisticRegression(max_iter=2000, random_state=seed).fit(
                ftr[mt] if not hasattr(ftr, "toarray") else ftr[mt], lab_tr[mt])
            sc = lr.predict_proba(fte[m] if not hasattr(fte, "toarray") else fte[m])[:, 1]
            auc = float(roc_auc_score(lab_te[m], sc))
        res[arm] = {"spearman": sp, "tertile_auc": auc}
    return res


def main():
    rng = np.random.default_rng(SEED)
    df = load_items()
    print(f"items: {len(df)} ({df.instrument.value_counts().to_dict()}); "
          f"ESS families: {df[df.instrument=='ESS11'].block.nunique()}")
    E = embed_texts(df["item_text"])
    print(f"embeddings: {E.shape}")
    results = {}

    for target in ("refusal_rate", "dk_rate"):
        y = df[target].to_numpy()
        # ---- grouped 5-fold CV within ESS (family blocks) ----
        ess_idx = np.where(df.instrument == "ESS11")[0]
        blocks = df.loc[ess_idx, "block"].to_numpy()
        ub = rng.permutation(np.unique(blocks))
        fold_of_block = {b: i % 5 for i, b in enumerate(ub)}
        folds = np.array([fold_of_block[b] for b in blocks])
        agg = {"tfidf": {"spearman": [], "tertile_auc": []},
               "embed": {"spearman": [], "tertile_auc": []}}
        for f in range(5):
            tr = ess_idx[folds != f]
            te = ess_idx[folds == f]
            r = eval_split(df.loc[tr, "item_text"], df.loc[te, "item_text"],
                           E[tr], E[te], y[tr], y[te], SEED + f)
            for arm in agg:
                for k in agg[arm]:
                    agg[arm][k].append(r[arm][k])
        ess_cv = {arm: {k: float(np.nanmean(v)) for k, v in d.items()} for arm, d in agg.items()}

        # ---- leave-instrument-out: train ESS -> test NHANES ----
        tr = np.where(df.instrument == "ESS11")[0]
        te = np.where(df.instrument == "NHANES")[0]
        ess_to_nh = eval_split(df.loc[tr, "item_text"], df.loc[te, "item_text"],
                               E[tr], E[te], y[tr], y[te], SEED)
        results[target] = {"ess_grouped_cv": ess_cv, "ess_to_nhanes": ess_to_nh}

        print(f"\n=== target: {target} ===")
        print(f"  ESS grouped-family 5-fold CV (paraphrase-leak guarded):")
        for arm in ("tfidf", "embed"):
            c = ess_cv[arm]
            print(f"    {arm:6s}: Spearman {c['spearman']:+.3f} | tertile AUC {c['tertile_auc']:.3f}")
        print(f"  ESS -> NHANES (leave-instrument-out, within-NHANES metrics):")
        for arm in ("tfidf", "embed"):
            c = ess_to_nh[arm]
            print(f"    {arm:6s}: Spearman {c['spearman']:+.3f} | tertile AUC {c['tertile_auc']:.3f}")

    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
