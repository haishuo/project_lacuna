#!/usr/bin/env python3
"""
Metadata-prior bakeoff: can a LOCAL model author the missingness-mechanism prior a human analyst holds?

Premise (from the Lacuna design discussion): MAR-vs-MNAR is non-identifiable from the data alone
(Molenberghs), but an analyst is never working from the data alone — they know what the columns MEAN.
"Income" raises the prior on self-censoring MNAR; a lab assay with a detection limit raises the prior on
detection MNAR; a skip-gated follow-up is MAR-by-design; a randomized booklet rotation is MCAR-by-design.
That is a PRIOR authored from metadata (column name + codebook description + domain) — exactly the
extra-data channel Molenberghs says you must supply. This script asks: does a small, local, HIPAA-safe,
reproducible model produce that prior as well as a human/frontier model would? If yes, the deployed prior
needs no frontier API in the loop.

The model sees ONLY the metadata an analyst already has (NO data values, NO gold). It outputs a structured
prior {mechanism in MCAR/MAR/MNAR/INDETERMINATE, semantic class, confidence}. We score against a benchmark
whose gold is GROUNDED in real structure where possible (published limit-of-detection flags -> detection
MNAR; explicit skip-logic -> MAR-by-design; randomized administration -> MCAR-by-design; the survey-anchor
consensus for sensitive items -> self-censoring MNAR), with weaker/abstain cases included to probe
over-confidence.

Two conditions isolate the user's "codebooks shrink the model you need" hypothesis:
  - name+desc : column name + description + domain (what an analyst actually has) -- the realistic task.
  - name-only : column name + domain only (the cryptic-code stress test).

Deterministic: greedy decoding (do_sample=False), fixed seed. Usage:
    python scripts/metadata_prior_bakeoff.py                       # full lineup, both conditions
    python scripts/metadata_prior_bakeoff.py --models Qwen2.5-3B   # one model
    python scripts/metadata_prior_bakeoff.py --conditions name+desc
"""

import argparse
import gc
import json
import re
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

BENCH = PROJECT_ROOT / "scripts" / "metadata_prior" / "benchmark.json"
OUT = PROJECT_ROOT / "scripts" / "metadata_prior" / "bakeoff_results.json"

# (display name, HF repo, quantization). 4-bit for >=7B to fit 16 GB; bf16 otherwise.
LINEUP = [
    ("Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B-Instruct", "bf16"),
    ("Qwen2.5-1.5B", "Qwen/Qwen2.5-1.5B-Instruct", "bf16"),
    ("Qwen2.5-3B", "Qwen/Qwen2.5-3B-Instruct", "bf16"),
    ("Phi-3.5-mini", "microsoft/Phi-3.5-mini-instruct", "bf16"),
    ("Qwen2.5-7B", "Qwen/Qwen2.5-7B-Instruct", "4bit"),
    ("Qwen2.5-14B", "Qwen/Qwen2.5-14B-Instruct", "4bit"),
]

MECHS = ("MCAR", "MAR", "MNAR", "INDETERMINATE")
SEMANTICS = ("lab_lod", "sensitive_disclosure", "skip_gated", "planned_random",
             "administrative", "demographic_core", "routine_measure", "indeterminate")

SYSTEM = (
    "You are a survey-methodology and clinical-data expert assessing MISSING DATA. For a single column "
    "you are given only its metadata (name, description, dataset domain) — never the data values. Using "
    "world knowledge about how missingness arises (human disclosure behaviour, measurement physics, "
    "questionnaire design), state the most likely PRIOR over the missingness mechanism for that column.\n\n"
    "Mechanism options:\n"
    "  MCAR  = missing completely at random (value-independent AND not driven by other variables; e.g. "
    "randomized/planned-missing designs, administrative).\n"
    "  MAR   = missing depends on OTHER observed variables but not the value itself (e.g. skip-logic gated "
    "on an observed answer; missing-by-design conditional on an observed gate).\n"
    "  MNAR  = missing depends on the value ITSELF (e.g. sensitive items people hide at the extremes = "
    "self-censoring; lab assays where values below a detection limit are unobserved).\n"
    "  INDETERMINATE = the metadata does not determine a mechanism; abstain.\n\n"
    "Semantic classes: lab_lod, sensitive_disclosure, skip_gated, planned_random, administrative, "
    "demographic_core, routine_measure, indeterminate.\n\n"
    "Think in one or two short sentences, then end with EXACTLY one line:\n"
    "FINAL: mechanism=<MCAR|MAR|MNAR|INDETERMINATE>; semantic=<one class>; confidence=<0.0-1.0>"
)


def make_user(col, condition):
    if condition == "name-only":
        return (f"Dataset domain: {col['domain']}\nColumn name: {col['column_name']}\n"
                f"(No description available.)\n\nAssess the missingness mechanism prior.")
    return (f"Dataset domain: {col['domain']}\nColumn name: {col['column_name']}\n"
            f"Description: {col['description']}\n\nAssess the missingness mechanism prior.")


_MECH_RE = re.compile(r"mechanism\s*=\s*(MCAR|MAR|MNAR|INDETERMINATE)", re.I)
_SEM_RE = re.compile(r"semantic\s*=\s*([a-z_]+)", re.I)
_CONF_RE = re.compile(r"confidence\s*=\s*([01](?:\.\d+)?)", re.I)


def parse_answer(text):
    """Extract (mechanism, semantic, confidence) from the model output; lenient fallbacks."""
    tail = text[-400:]  # the FINAL line is at the end
    mech = _MECH_RE.search(tail) or _MECH_RE.search(text)
    sem = _SEM_RE.search(tail) or _SEM_RE.search(text)
    conf = _CONF_RE.search(tail) or _CONF_RE.search(text)
    mechanism = mech.group(1).upper() if mech else None
    if mechanism is None:  # last-ditch: any bare mechanism token near the end
        for m in MECHS:
            if re.search(rf"\b{m}\b", tail):
                mechanism = m
                break
    semantic = sem.group(1).lower() if sem and sem.group(1).lower() in SEMANTICS else None
    confidence = float(conf.group(1)) if conf else None
    return mechanism, semantic, confidence


def load_model(repo, quant, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
    kw = dict(trust_remote_code=True)
    if quant == "4bit":
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
        kw["device_map"] = device
    else:
        kw["torch_dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(repo, **kw)
    if quant != "4bit":
        model = model.to(device)
    model.eval()
    return tok, model


@torch.no_grad()
def generate(tok, model, messages, device, max_new_tokens):
    inputs = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(device)
    out = model.generate(inputs, max_new_tokens=max_new_tokens, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, inputs.shape[1]:], skip_special_tokens=True)


def run_model(name, repo, quant, columns, conditions, device, max_new_tokens):
    print(f"\n=== {name} ({repo}, {quant}) ===", flush=True)
    tok, model = load_model(repo, quant, device)
    results = []
    for cond in conditions:
        for col in columns:
            messages = [{"role": "system", "content": SYSTEM},
                        {"role": "user", "content": make_user(col, cond)}]
            text = generate(tok, model, messages, device, max_new_tokens)
            mech, sem, conf = parse_answer(text)
            results.append({"id": col["id"], "condition": cond, "gold_mechanism": col["gold_mechanism"],
                            "gold_semantic": col["gold_semantic"], "grounding": col["grounding"],
                            "pred_mechanism": mech, "pred_semantic": sem, "pred_confidence": conf,
                            "raw_tail": text[-200:]})
    del model, tok
    gc.collect()
    torch.cuda.empty_cache()
    return results


def score(results, columns):
    """Per (model implicit) metrics, split by condition. `results` is one model's rows."""
    gold = {c["id"]: c for c in columns}
    out = {}
    for cond in sorted(set(r["condition"] for r in results)):
        rows = [r for r in results if r["condition"] == cond]
        clear = [r for r in rows if r["gold_mechanism"] != "INDETERMINATE"]
        strong = [r for r in clear if gold[r["id"]]["grounding"] == "strong"]
        absta = [r for r in rows if r["gold_mechanism"] == "INDETERMINATE"]
        parse_fail = sum(1 for r in rows if r["pred_mechanism"] is None)

        def r3(x):
            return round(x, 3) if x is not None else None

        def acc(subset):
            return (r3(sum(1 for r in subset if r["pred_mechanism"] == r["gold_mechanism"]) / len(subset))
                    if subset else None)

        sem_acc = (r3(sum(1 for r in clear if r["pred_semantic"] == r["gold_semantic"]) / len(clear))
                   if clear else None)
        over_abstain = (r3(sum(1 for r in clear if r["pred_mechanism"] == "INDETERMINATE") / len(clear))
                        if clear else None)
        # confusion over clear columns (gold rows x pred cols)
        conf = {g: {p: 0 for p in MECHS} for g in ("MCAR", "MAR", "MNAR")}
        for r in clear:
            p = r["pred_mechanism"] or "PARSE_FAIL"
            conf.setdefault(r["gold_mechanism"], {}).setdefault(p, 0)
            conf[r["gold_mechanism"]][p] = conf[r["gold_mechanism"]].get(p, 0) + 1
        out[cond] = {
            "mechanism_acc_clear": acc(clear), "mechanism_acc_strong": acc(strong),
            "mechanism_acc_consensus": acc([r for r in clear if gold[r["id"]]["grounding"] == "consensus"]),
            "mechanism_acc_weak": acc([r for r in clear if gold[r["id"]]["grounding"] == "weak"]),
            "abstain_recall": acc(absta), "over_abstain_rate": over_abstain,
            "semantic_acc_clear": sem_acc, "parse_fail": parse_fail, "n_clear": len(clear),
            "n_strong": len(strong), "n_abstain": len(absta), "confusion_clear": conf,
        }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=[n for n, _, _ in LINEUP],
                    help="subset of model display names to run")
    ap.add_argument("--conditions", nargs="+", default=["name+desc", "name-only"],
                    choices=["name+desc", "name-only"])
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output", type=Path, default=OUT)
    ap.add_argument("--seed", type=int, default=20260601)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    bench = json.loads(BENCH.read_text())
    columns = bench["columns"]
    majority = max(("MCAR", "MAR", "MNAR"),
                   key=lambda m: sum(1 for c in columns if c["gold_mechanism"] == m))
    maj_clear = [c for c in columns if c["gold_mechanism"] != "INDETERMINATE"]
    maj_acc = sum(1 for c in maj_clear if c["gold_mechanism"] == majority) / len(maj_clear)
    print(f"Benchmark: {len(columns)} columns | majority class '{majority}' = {maj_acc:.3f} on clear cols "
          f"| conditions {args.conditions}")

    report = {"benchmark_n": len(columns), "majority_class": majority, "majority_acc_clear": round(maj_acc, 4),
              "conditions": args.conditions, "models": {}}
    todo = [(n, r, q) for (n, r, q) in LINEUP if n in args.models]
    for name, repo, quant in todo:
        try:
            res = run_model(name, repo, quant, columns, args.conditions, args.device, args.max_new_tokens)
        except Exception as e:  # noqa: BLE001
            print(f"  !! {name} failed: {e}", flush=True)
            report["models"][name] = {"error": str(e)}
            continue
        metrics = score(res, columns)
        report["models"][name] = {"quant": quant, "metrics": metrics, "rows": res}
        for cond in args.conditions:
            m = metrics[cond]
            print(f"  [{name} | {cond}] mech_acc clear={m['mechanism_acc_clear']} "
                  f"strong={m['mechanism_acc_strong']} | abstain_recall={m['abstain_recall']} "
                  f"over_abstain={m['over_abstain_rate']} | sem_acc={m['semantic_acc_clear']} "
                  f"| parse_fail={m['parse_fail']}", flush=True)
        args.output.write_text(json.dumps(report, indent=2))  # checkpoint after each model

    # --- summary table ---
    print("\n" + "=" * 92)
    print("METADATA-PRIOR BAKEOFF — mechanism accuracy (clear cols) by model x condition")
    print(f"(majority-class baseline = {maj_acc:.3f}; n_clear={len(maj_clear)}, n_strong="
          f"{sum(1 for c in columns if c['grounding']=='strong')}, n_abstain="
          f"{sum(1 for c in columns if c['gold_mechanism']=='INDETERMINATE')})")
    print("=" * 92)
    hdr = f"  {'model':14s}"
    for cond in args.conditions:
        hdr += f" | {cond:>10s} clear/strong/abst"
    print(hdr)
    for name, _, _ in todo:
        info = report["models"].get(name, {})
        if "error" in info:
            print(f"  {name:14s}  ERROR: {info['error'][:60]}")
            continue
        line = f"  {name:14s}"
        for cond in args.conditions:
            m = info["metrics"][cond]
            line += (f" | {str(m['mechanism_acc_clear']):>10s} "
                     f"{m['mechanism_acc_strong']}/{m['abstain_recall']}")
        print(line)
    print("=" * 92)
    args.output.write_text(json.dumps(report, indent=2))
    print(f"\nWrote -> {args.output}")


if __name__ == "__main__":
    main()
