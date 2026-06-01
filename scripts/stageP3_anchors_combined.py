#!/usr/bin/env python3
"""
Stage P3 (ADR-0008): the metadata prior x data likelihood, end-to-end on the real survey anchors.

Runs the DEPLOYED pipeline on real data: a local model (Qwen) reads each anchor's analyst-metadata
(domain + what the dominant missing variable is + design — NEVER the mechanism label or a mechanism-naming
citation, to avoid circularity) and labels its semantic class; the frozen P1 table turns that into a prior;
the composition head supplies the data likelihood; they are pooled in RAW evidence space (Stage P2). Per
anchor we report the data-only read, the prior, the combined read, their DISAGREEMENT, and the argmax vs
the textbook consensus.

Honest framing (the whole point — there is NO mechanism ground truth on real data):
  - The anchor consensus labels are themselves elicited judgement (gut feelings); validating the prior
    against them is SEMI-CIRCULAR (it mostly confirms the model encoded the textbook). So the headline
    validation is the FACT tier, where the metadata states a design fact the data channel cannot see:
      * PISA rotated booklets -> MCAR-by-design (does the prior FIX the footprint's structured misread?).
      * UCLA price-quote skip-logic -> MAR-by-design.
  - The sensitive-item NHANES anchors (income/weight/drug/depression -> MNAR) are the GUT-FEELING tier:
    reported as weaker face validity, and `survey_chile` is a built-in disagreement (income is sensitive
    -> the prior leans MNAR, but the textbook reads it MAR-given-demographics — a reasonable-but-wrong prior).
  - A wrong prior amplifies error on the non-identifiable axis and the data cannot rescue it (Stage P2);
    the deliverable is therefore the DISAGREEMENT report, not a hidden correction.

Deterministic via explicit seeds. Two phases (Qwen freed before the composition instrument loads, to fit
16 GB). Usage:
    python scripts/stageP3_anchors_combined.py
"""

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lacuna.core.rng import RNGState
from lacuna.core.types import ObservedDataset
from lacuna.config import load_config
from lacuna.data.missingness_footprint import missingness_footprint, FOOTPRINT_FEATURES
from lacuna.data.tokenization import tokenize_and_batch
from lacuna.data.composition_batch import N_FOOTPRINT_FEATURES
from lacuna.models.composition_head import CompositionHead, ensemble_alpha, composition_mean
from lacuna.priors.metadata_prior import (
    CLASS_NAMES, semantic_prior_alpha, combine_prior_likelihood, prior_mean, channel_disagreement,
)
from lacuna_survey.anchors import ANCHORS
from scripts.stageC_composition_head import init_encoder, forward_alpha
from scripts.metadata_prior_bakeoff import load_model, generate, SYSTEM, make_user, parse_answer

BASELINE = "/mnt/artifacts/project_lacuna/runs/stage0_general_baseline"
ANCHOR_DIR = PROJECT_ROOT / "lacuna_survey" / "evaluation_data"
_QWEN = "Qwen/Qwen2.5-14B-Instruct"

# Analyst metadata for each anchor's DOMINANT missing variable — domain + what it is + design ONLY.
# NO mechanism words (MCAR/MAR/MNAR) and no mechanism-naming citation: the model must INFER the prior.
# FACT-tier entries state a design fact the footprint cannot observe (rotation / skip-logic).
ANCHOR_META = {
    "survey_bfi": dict(column_name="item_E1", domain="Big Five personality questionnaire",
                       description="Response to a Likert personality item (e.g. 'Am full of ideas')."),
    "survey_chile": dict(column_name="income", domain="National plebiscite vote-intention survey",
                         description="Respondent's monthly household income."),
    "survey_cars93": dict(column_name="Luggage.room", domain="Automobile attributes dataset",
                          description="A vehicle physical/price attribute (e.g. luggage capacity, price)."),
    "survey_survey": dict(column_name="Height", domain="University student survey",
                          description="Student's reported height and similar physical attributes."),
    "survey_yrbss": dict(column_name="weight", domain="Youth risk-behavior surveillance phone survey",
                         description="Self-reported body weight and adolescent risk-behavior items."),
    "survey_gssvocab": dict(column_name="ageGroup", domain="General Social Survey vocabulary module",
                            description="Respondent year of birth / age and vocabulary-test score."),
    "survey_ucla_textbooks": dict(column_name="uclaNew", domain="University textbook-purchase survey",
                                  description="The price a student would pay for a textbook, recorded only "
                                              "for textbooks that the student actually purchased."),
    "survey_nhanes_demographics": dict(column_name="INDFMPIR", domain="NHANES demographics",
                                       description="Ratio of family income to the federal poverty guideline."),
    "pisa2018_gbr_rotation": dict(column_name="ST196 reading-attitude item",
                                  domain="PISA international student assessment",
                                  description="A reading-attitude questionnaire item administered to only a "
                                              "random subset of the rotated test booklets (a planned, "
                                              "randomly-assigned booklet design)."),
    "pisa2022_deu_rotation": dict(column_name="ST315 questionnaire item",
                                  domain="PISA international student assessment",
                                  description="A questionnaire item present on only a randomly-assigned "
                                              "subset of rotated test booklets (planned booklet rotation)."),
    "nhanes_inq_income": dict(column_name="INDFMMPI", domain="NHANES income module",
                              description="Monthly family income-to-poverty index."),
    "nhanes_whq_weight": dict(column_name="WHD020", domain="NHANES weight-history module",
                              description="Self-reported current and historical body weight."),
    "nhanes_duq_drug": dict(column_name="DUQ200", domain="NHANES drug-use questionnaire",
                            description="Whether the respondent has ever used marijuana, cocaine, or heroin "
                                        "(self-reported illicit drug use)."),
    "nhanes_dpq_phq9": dict(column_name="DPQ020", domain="NHANES mental-health depression screener",
                            description="A PHQ-9 depression-screener item (e.g. 'feeling down, depressed, or "
                                        "hopeless over the last two weeks')."),
}
# The FACT tier: the metadata states a design fact (rotation / skip-logic) the data channel cannot see.
_FACT_TIER = {"pisa2018_gbr_rotation", "pisa2022_deu_rotation", "survey_ucla_textbooks"}


def classify_semantics(device, max_new_tokens=200):
    """Phase 1: Qwen labels each anchor's dominant-missing-variable metadata -> semantic class."""
    tok, model = load_model(_QWEN, "4bit", device)
    out = {}
    for a in ANCHORS:
        meta = ANCHOR_META[a.slug]
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": make_user(meta, "name+desc")}]
        _, sem, _ = parse_answer(generate(tok, model, msgs, device, max_new_tokens))
        out[a.slug] = sem or "indeterminate"
    del model, tok
    gc.collect()
    torch.cuda.empty_cache()
    return out


def _load_anchor(slug):
    import pandas as pd
    df = pd.read_csv(ANCHOR_DIR / f"{slug}_real.csv")
    num = df.select_dtypes(include=[np.number])
    drop = [c for c in num.columns if c.upper() in ("SEQN", "ID", "RESPONDENT_ID")]
    num = num.drop(columns=drop)
    return num.values.astype(np.float32), tuple(num.columns.tolist())


def anchor_raw_likelihood(encoder, heads, slug, *, rng, max_rows, max_cols, device, n_draws=8):
    """Phase 2: RAW (uncalibrated) ensemble Dirichlet alpha [3] for one anchor (P2: combine raw)."""
    values, fnames = _load_anchor(slug)
    n = values.shape[0]
    draws = n_draws if n > max_rows else 1
    alphas = []
    for _ in range(draws):
        v = values[np.sort(rng.choice(n, size=max_rows, replace=False))] if n > max_rows else values
        mask = ~np.isnan(v)
        x = np.nan_to_num(v, nan=0.0)
        obs = ObservedDataset(x=torch.from_numpy(x), r=torch.from_numpy(mask), n=x.shape[0], d=x.shape[1],
                              feature_names=fnames, dataset_id=slug)
        fp = missingness_footprint(obs.x, obs.r)
        extra = torch.tensor([[fp[k] for k in FOOTPRINT_FEATURES]], dtype=torch.float32).to(device)
        b = tokenize_and_batch([obs], max_rows=max_rows, max_cols=max_cols).to(device)
        with torch.no_grad():
            per_model = torch.stack([forward_alpha(encoder, h, b, extra).cpu() for h in heads], 0)
        alphas.append(ensemble_alpha(per_model).numpy()[0])
    return np.mean(alphas, axis=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline-checkpoint", default=f"{BASELINE}/checkpoints/best_model.pt")
    ap.add_argument("--baseline-config", default=f"{BASELINE}/config.yaml")
    ap.add_argument("--heads-checkpoint",
                    default=f"{BASELINE}/checkpoints/stageC_composition_frozen-probe+footprint.pt")
    ap.add_argument("--head-hidden", type=int, default=64)
    ap.add_argument("--n-draws", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260601)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output", type=Path, default=Path(f"{BASELINE}/stageP3_anchors_combined.json"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    print("Stage P3 — phase 1: Qwen labels anchor metadata (mechanism-blind) ...", flush=True)
    semantics = classify_semantics(args.device)

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
    print("phase 2: composition likelihood per anchor + combine ...\n", flush=True)

    rows = []
    for a in ANCHORS:
        sem = semantics[a.slug]
        prior_a = semantic_prior_alpha(sem)
        like_a = anchor_raw_likelihood(encoder, heads, a.slug, rng=rng.spawn(),
                                       max_rows=cfg.data.max_rows, max_cols=cfg.data.max_cols,
                                       device=args.device, n_draws=args.n_draws)
        comb_a = combine_prior_likelihood(prior_a, like_a)
        data_only, prior_m, combined = prior_mean(like_a), prior_mean(prior_a), prior_mean(comb_a)
        rows.append({
            "slug": a.slug, "consensus": a.label_name, "tier": "fact" if a.slug in _FACT_TIER else "gut",
            "qwen_semantic": sem, "prior_argmax": CLASS_NAMES[int(np.argmax(prior_m))],
            "data_only": [round(float(x), 3) for x in data_only],
            "prior": [round(float(x), 3) for x in prior_m],
            "combined": [round(float(x), 3) for x in combined],
            "data_argmax": CLASS_NAMES[int(np.argmax(data_only))],
            "combined_argmax": CLASS_NAMES[int(np.argmax(combined))],
            "disagreement": round(channel_disagreement(prior_a, like_a), 3),
            "f_mcar_shift": round(float(combined[0] - data_only[0]), 3),
        })

    def hit(r, key):
        return r[f"{key}_argmax"] == r["consensus"]

    print("=" * 110)
    print("STAGE P3 — metadata prior x data likelihood on real anchors (data-only -> combined; raw evidence)")
    print("=" * 110)
    print(f"  {'anchor':28s} {'cons':4s} {'tier':4s} {'qwen_sem':20s} {'data(M/A/N)':17s} "
          f"{'combined(M/A/N)':17s} {'dataA':5s} {'combA':5s} {'disag':5s}")
    for r in sorted(rows, key=lambda x: (x["tier"], x["consensus"], x["slug"])):
        print(f"  {r['slug']:28s} {r['consensus']:4s} {r['tier']:4s} {r['qwen_semantic']:20s} "
              f"{str(r['data_only']):17s} {str(r['combined']):17s} {r['data_argmax'][:5]:5s} "
              f"{r['combined_argmax'][:5]:5s} {r['disagreement']:5.2f}")

    fact = [r for r in rows if r["tier"] == "fact"]
    gut = [r for r in rows if r["tier"] == "gut"]
    pisa = [r for r in rows if r["slug"].startswith("pisa")]
    summary = {
        "n_anchors": len(rows),
        "fact_tier": {
            "consensus_hit_data_only": sum(hit(r, "data") for r in fact), "n": len(fact),
            "consensus_hit_combined": sum(hit(r, "combined") for r in fact),
            "pisa_mean_f_mcar_data_only": round(float(np.mean([r["data_only"][0] for r in pisa])), 3),
            "pisa_mean_f_mcar_combined": round(float(np.mean([r["combined"][0] for r in pisa])), 3)},
        "gut_tier": {
            "consensus_hit_data_only": sum(hit(r, "data") for r in gut), "n": len(gut),
            "consensus_hit_combined": sum(hit(r, "combined") for r in gut)},
        "disagreement_cases": [r["slug"] for r in rows if r["prior_argmax"] != r["data_argmax"]],
    }
    f, g = summary["fact_tier"], summary["gut_tier"]
    print(f"\n  FACT tier (design stated in metadata; the real check): consensus argmax "
          f"{f['consensus_hit_data_only']}/{f['n']} data-only -> {f['consensus_hit_combined']}/{f['n']} combined")
    print(f"    PISA MCAR-by-design fix: mean f_MCAR {f['pisa_mean_f_mcar_data_only']} (data, reads structured)"
          f" -> {f['pisa_mean_f_mcar_combined']} (combined, prior supplies value-independence)")
    print(f"  GUT tier (sensitive-item consensus; semi-circular face validity): consensus argmax "
          f"{g['consensus_hit_data_only']}/{g['n']} data-only -> {g['consensus_hit_combined']}/{g['n']} combined")
    print(f"  prior-vs-data disagreement (reported, not hidden): {summary['disagreement_cases']}")
    print("=" * 110)

    args.output.write_text(json.dumps({"summary": summary, "anchors": rows}, indent=2))
    print(f"\nWrote -> {args.output}")


if __name__ == "__main__":
    main()
