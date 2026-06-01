#!/usr/bin/env python3
"""
Stage P1 (ADR-0008): build + validate the FROZEN metadata->prior table.

The deployed prior channel is a frozen, auditable mapping `semantic class -> Dirichlet prior over
(MCAR, MAR, MNAR)` (lacuna.priors.metadata_prior). At inference a local model labels each column's
semantic class from its metadata (scripts/metadata_prior_bakeoff.py); this table turns that label into a
prior. This script:

  1. Emits the frozen table (the auditable artifact, ADR-0008 commitment 3) -> prior_table.json.
  2. Validates it on the grounded benchmark TWO ways (ADR-0008 commitment 2 — the prior is curated, the
     LLM only supplies the semantic label):
       - CURATED (zero-LLM): gold semantic class -> prior -> argmax vs gold mechanism. A consistency check
         that the table encodes the grounded semantic->mechanism map, by grounding tier.
       - LLM-DRIVEN: the local model's PREDICTED semantic class (from the bakeoff results) -> prior ->
         argmax vs gold mechanism. The realistic deployed-prior accuracy.
  3. Reports the prior's deliberate under-confidence (favoured-class probability ~0.52-0.70, capped so the
     data likelihood can override it — ADR-0008 commitment 1) and its abstention behaviour on the
     genuinely-indeterminate columns.

Deterministic; no RNG, no model calls (it reuses the already-computed bakeoff predictions). Usage:
    python scripts/stageP1_build_prior_table.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lacuna.priors.metadata_prior import (
    SEMANTIC_PRIOR_SPEC, CLASS_NAMES, semantic_prior_alpha, prior_mean,
)

BENCH = PROJECT_ROOT / "scripts" / "metadata_prior" / "benchmark.json"
BAKEOFF = PROJECT_ROOT / "scripts" / "metadata_prior" / "bakeoff_results.json"
OUT = PROJECT_ROOT / "scripts" / "metadata_prior" / "prior_table.json"
_LLM_MODEL = "Qwen2.5-14B"   # the bakeoff's strongest local model; the deployed prior-author
_LLM_COND = "name+desc"


def build_table():
    """The frozen semantic -> prior mapping (the auditable artifact)."""
    table = {}
    for cls, (favored, r) in SEMANTIC_PRIOR_SPEC.items():
        a = semantic_prior_alpha(cls)
        table[cls] = {
            "alpha": [round(float(x), 4) for x in a],
            "favored": CLASS_NAMES[favored] if favored is not None else None,
            "target_reliability": round(float(r), 4),
            "favored_prob": round(float(prior_mean(a).max()), 4),
        }
    return table


def _argmax_mech(semantic_class):
    """Mechanism the prior favours for a semantic class, or None if it abstains (flat / unknown)."""
    if semantic_class not in SEMANTIC_PRIOR_SPEC or semantic_class == "indeterminate":
        return None
    favored, _ = SEMANTIC_PRIOR_SPEC[semantic_class]
    return CLASS_NAMES[favored] if favored is not None else None


def _score(pairs):
    """pairs: list of (gold_mechanism, semantic_class, grounding). Returns prior-only metrics."""
    clear = [(g, s, gr) for (g, s, gr) in pairs if g != "INDETERMINATE"]
    absta = [(g, s, gr) for (g, s, gr) in pairs if g == "INDETERMINATE"]

    def acc(subset):
        if not subset:
            return None
        committed = [(g, _argmax_mech(s)) for (g, s, _gr) in subset]
        # accuracy counts an abstention (None) on a clear column as a miss
        return round(sum(1 for g, p in committed if p == g) / len(subset), 3)

    strong = [(g, s, gr) for (g, s, gr) in clear if gr == "strong"]
    return {
        "acc_clear": acc(clear),
        "acc_strong": acc(strong),
        "acc_consensus": acc([t for t in clear if t[2] == "consensus"]),
        "acc_weak": acc([t for t in clear if t[2] == "weak"]),
        "abstain_recall": round(sum(1 for (_g, s, _gr) in absta if _argmax_mech(s) is None) / len(absta), 3)
        if absta else None,
        "over_abstain": round(sum(1 for (_g, s, _gr) in clear if _argmax_mech(s) is None) / len(clear), 3)
        if clear else None,
        "n_clear": len(clear), "n_strong": len(strong), "n_abstain": len(absta),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", type=Path, default=OUT)
    args = ap.parse_args()

    bench = json.loads(BENCH.read_text())
    cols = bench["columns"]
    table = build_table()

    # CURATED path — gold semantic labels (zero LLM).
    curated_pairs = [(c["gold_mechanism"], c["gold_semantic"], c["grounding"]) for c in cols]
    curated = _score(curated_pairs)

    # LLM-DRIVEN path — the local model's predicted semantic labels from the bakeoff.
    llm = None
    if BAKEOFF.exists():
        rep = json.loads(BAKEOFF.read_text())
        rows = rep.get("models", {}).get(_LLM_MODEL, {}).get("rows", [])
        pred = {r["id"]: r["pred_semantic"] for r in rows if r["condition"] == _LLM_COND}
        if pred:
            gold_g = {c["id"]: c["gold_mechanism"] for c in cols}
            gold_gr = {c["id"]: c["grounding"] for c in cols}
            llm_pairs = [(gold_g[i], (pred.get(i) or "indeterminate"), gold_gr[i])
                         for i in gold_g if i in pred]
            llm = _score(llm_pairs)
            llm["n_scored"] = len(llm_pairs)
            llm["model"] = f"{_LLM_MODEL} ({_LLM_COND})"

    report = {"table": table, "validation": {"curated": curated, "llm_driven": llm},
              "note": "Prior strengths are capped (favoured prob <= 0.70) so the data likelihood can "
                      "override the prior (ADR-0008 commitment 1); the prior is a nudge, re-fit in P2."}
    args.output.write_text(json.dumps(report, indent=2))

    # --- print ---
    print("=" * 88)
    print("STAGE P1 — frozen metadata->prior table (ADR-0008)")
    print("=" * 88)
    print(f"  {'semantic class':22s} {'favored':>8s} {'alpha (MCAR/MAR/MNAR)':>26s} {'P(favored)':>11s}")
    for cls, t in table.items():
        print(f"  {cls:22s} {str(t['favored']):>8s} {str(t['alpha']):>26s} {t['favored_prob']:>11.3f}")
    print("\n  --- prior-only validation on the grounded benchmark ---")
    print(f"  CURATED (gold semantic labels): clear {curated['acc_clear']} | strong {curated['acc_strong']}"
          f" | consensus {curated['acc_consensus']} | weak {curated['acc_weak']} | "
          f"abstain_recall {curated['abstain_recall']}")
    if llm:
        print(f"  LLM-DRIVEN ({llm['model']}): clear {llm['acc_clear']} | strong {llm['acc_strong']}"
              f" | consensus {llm['acc_consensus']} | weak {llm['acc_weak']} | "
              f"abstain_recall {llm['abstain_recall']} | over_abstain {llm['over_abstain']}")
    else:
        print("  LLM-DRIVEN: (bakeoff results not found; run scripts/metadata_prior_bakeoff.py)")
    print("\n  Note: favoured-class prior prob is capped at 0.70 so the data can override it (commitment 1);")
    print("  strengths are a documented P1 starting point, re-fit against real likelihoods in P2.")
    print("=" * 88)
    print(f"\nWrote frozen prior table -> {args.output}")


if __name__ == "__main__":
    main()
