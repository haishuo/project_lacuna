#!/usr/bin/env python3
"""
Stage P4 (ADR-0008): gate the fact-tier prior strength on classifier CONFIDENCE.

P3 showed a strong fact-tier prior amplifies a misclassification INTO the fact tier (survey_bfi wrongly
labelled skip_gated -> a strong MAR prior the data struggles to override). The fix: de-rate the fact-tier
strength by the classifier's confidence in the label, where confidence is SELF-CONSISTENCY — classify each
anchor's metadata k times under sampling and take the modal-label frequency. A genuine design fact (a
rotated booklet -> planned_random every time) classifies stably -> keeps full strength; an ambiguous column
classifies unstably -> the fact-tier prior drops toward an overridable hunch. Self-consistency is
metadata-only, so it does NOT couple the prior to the data channel (ADR-0008 commitments 1-3).

Reports per anchor: modal semantic label + self-consistency confidence, and the combined read under the
UNGATED vs the GATED prior — to confirm the gate (a) preserves the PISA MCAR-by-design fix (high confidence)
and (b) de-rates the bfi-style misclassification (low confidence). Honest limit: self-consistency cannot
catch a CONFIDENTLY-wrong classification (stable but incorrect); reported if it occurs.

Deterministic: each sample is seeded. Usage:
    python scripts/stageP4_confidence_gate.py
"""

import argparse
import gc
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lacuna.core.rng import RNGState
from lacuna.config import load_config
from lacuna.data.composition_batch import N_FOOTPRINT_FEATURES
from lacuna.models.composition_head import CompositionHead
from lacuna.priors.metadata_prior import (
    CLASS_NAMES, semantic_tier, semantic_prior_alpha, gated_semantic_prior_alpha,
    combine_prior_likelihood, prior_mean,
)
from lacuna_survey.anchors import ANCHORS
from scripts.stageC_composition_head import init_encoder
from scripts.metadata_prior_bakeoff import load_model, SYSTEM, make_user, parse_answer
from scripts.stageP3_anchors_combined import anchor_raw_likelihood, ANCHOR_META, _FACT_TIER

BASELINE = "/mnt/artifacts/project_lacuna/runs/stage0_general_baseline"
_QWEN = "Qwen/Qwen2.5-14B-Instruct"


@torch.no_grad()
def sampled_generate(tok, model, messages, device, max_new_tokens, temperature, seed):
    torch.manual_seed(seed)
    inputs = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(device)
    out = model.generate(inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature,
                         top_p=0.95, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, inputs.shape[1]:], skip_special_tokens=True)


def self_consistency_classify(device, k, temperature, seed):
    """For each anchor: k sampled classifications -> (modal semantic label, confidence=modal freq, labels)."""
    tok, model = load_model(_QWEN, "4bit", device)
    out = {}
    for a in ANCHORS:
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": make_user(ANCHOR_META[a.slug], "name+desc")}]
        labels = []
        for j in range(k):
            _, sem, _ = parse_answer(sampled_generate(tok, model, msgs, device, 200, temperature,
                                                      seed + 1000 * j))
            labels.append(sem or "indeterminate")
        modal, cnt = Counter(labels).most_common(1)[0]
        out[a.slug] = {"modal": modal, "confidence": cnt / k, "labels": labels}
    del model, tok
    gc.collect()
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline-checkpoint", default=f"{BASELINE}/checkpoints/best_model.pt")
    ap.add_argument("--baseline-config", default=f"{BASELINE}/config.yaml")
    ap.add_argument("--heads-checkpoint",
                    default=f"{BASELINE}/checkpoints/stageC_composition_frozen-probe+footprint.pt")
    ap.add_argument("--head-hidden", type=int, default=64)
    ap.add_argument("--k", type=int, default=7, help="self-consistency samples per anchor")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--n-draws", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260601)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output", type=Path, default=Path(f"{BASELINE}/stageP4_confidence_gate.json"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    print(f"Stage P4 — phase 1: self-consistency classify (k={args.k}, T={args.temperature}) ...", flush=True)
    sc = self_consistency_classify(args.device, args.k, args.temperature, args.seed)

    cfg = load_config(args.baseline_config)
    dims = dict(hidden_dim=cfg.model.hidden_dim, evidence_dim=cfg.model.evidence_dim,
                n_layers=cfg.model.n_layers, n_heads=cfg.model.n_heads,
                max_cols=cfg.data.max_cols, dropout=cfg.model.dropout)
    encoder = init_encoder(args.baseline_checkpoint, dims, args.device)
    state = torch.load(args.heads_checkpoint, map_location="cpu", weights_only=False)["head_states"]
    heads = []
    for hs in state:
        h = CompositionHead(cfg.model.evidence_dim, hidden_dim=args.head_hidden,
                            dropout=cfg.model.dropout, n_extra_features=N_FOOTPRINT_FEATURES)
        h.load_state_dict(hs)
        heads.append(h.to(args.device).eval())
    rng = RNGState(seed=args.seed)
    print("phase 2: likelihood + ungated-vs-gated combine ...\n", flush=True)

    rows = []
    for a in ANCHORS:
        modal, conf = sc[a.slug]["modal"], sc[a.slug]["confidence"]
        like = anchor_raw_likelihood(encoder, heads, a.slug, rng=rng.spawn(),
                                     max_rows=cfg.data.max_rows, max_cols=cfg.data.max_cols,
                                     device=args.device, n_draws=args.n_draws)
        ungated = prior_mean(combine_prior_likelihood(semantic_prior_alpha(modal), like))
        gated = prior_mean(combine_prior_likelihood(gated_semantic_prior_alpha(modal, conf), like))
        rows.append({
            "slug": a.slug, "consensus": a.label_name, "anchor_tier": "fact" if a.slug in _FACT_TIER else "gut",
            "modal_semantic": modal, "semantic_tier": semantic_tier(modal), "confidence": round(conf, 3),
            "labels": sc[a.slug]["labels"],
            "ungated": [round(float(x), 3) for x in ungated], "gated": [round(float(x), 3) for x in gated],
            "ungated_argmax": CLASS_NAMES[int(np.argmax(ungated))],
            "gated_argmax": CLASS_NAMES[int(np.argmax(gated))],
        })

    print("=" * 112)
    print("STAGE P4 — confidence-gated fact-tier prior (self-consistency); ungated vs gated combined read")
    print("=" * 112)
    print(f"  {'anchor':28s} {'cons':4s} {'modal_semantic':20s} {'tier':5s} {'conf':5s} "
          f"{'ungated(M/A/N)':17s} {'gated(M/A/N)':17s} {'unG':4s} {'gat':4s}")
    for r in sorted(rows, key=lambda x: (x["semantic_tier"], x["slug"])):
        print(f"  {r['slug']:28s} {r['consensus']:4s} {r['modal_semantic']:20s} {r['semantic_tier']:5s} "
              f"{r['confidence']:5.2f} {str(r['ungated']):17s} {str(r['gated']):17s} "
              f"{r['ungated_argmax'][:4]:4s} {r['gated_argmax'][:4]:4s}")

    fact_pred = [r for r in rows if r["semantic_tier"] == "fact"]
    pisa = [r for r in rows if r["slug"].startswith("pisa")]
    changed = [r for r in rows if r["ungated_argmax"] != r["gated_argmax"]]
    summary = {
        "k": args.k, "temperature": args.temperature,
        "fact_label_confidence": {r["slug"]: r["confidence"] for r in fact_pred},
        "pisa_gated_f_mcar": {r["slug"]: r["gated"][0] for r in pisa},
        "pisa_gated_argmax": {r["slug"]: r["gated_argmax"] for r in pisa},
        "argmax_changed_by_gate": [{"slug": r["slug"], "modal": r["modal_semantic"], "conf": r["confidence"],
                                    "ungated": r["ungated_argmax"], "gated": r["gated_argmax"]}
                                   for r in changed],
    }
    print("\n  fact-tier label self-consistency (high = trust the strong prior):")
    for r in sorted(fact_pred, key=lambda x: x["confidence"], reverse=True):
        print(f"    {r['slug']:28s} modal={r['modal_semantic']:16s} conf={r['confidence']:.2f}  "
              f"labels={r['labels']}")
    print(f"\n  PISA gated argmax (fix preserved if high-confidence planned_random): {summary['pisa_gated_argmax']}")
    print(f"  argmax changed by the gate (de-rated low-confidence fact labels): "
          f"{[c['slug'] + '(' + c['ungated'] + '->' + c['gated'] + ',conf' + str(c['conf']) + ')' for c in summary['argmax_changed_by_gate']]}")
    print("=" * 112)
    args.output.write_text(json.dumps({"summary": summary, "anchors": rows}, indent=2))
    print(f"\nWrote -> {args.output}")


if __name__ == "__main__":
    main()
