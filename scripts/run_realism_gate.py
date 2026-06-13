"""
scripts/run_realism_gate.py

Run the L1 realism gate (GENERATOR-DESIGN-charter §6) across a registry of
generators against a real survey corpus (ESS / NHANES). For each generator it
injects missingness into the corpus's complete-case real-X substrate and scores
the resulting masks against the real masks with the C2ST + fidelity panel.

A generator failing the gate is a FINDING (its masks do not look like real
missingness), reported, never hidden. A generator whose ``apply_to`` errors on
real-scale X is recorded as an error row (this surfaces the saturation hazard).

Deterministic given --seed. Offline. Run:
  /home/haishuo/miniconda3/envs/lacuna/bin/python -u scripts/run_realism_gate.py \
      --corpus ess --config lacuna_tabular_110 --n-cols 12
"""

import argparse
import dataclasses
import json
import time
from pathlib import Path

from lacuna.generators import load_registry_from_config
from lacuna.analysis.realism_gate import (
    load_ess_corpus,
    load_nhanes_corpus,
    run_realism_gate,
    format_gate_table,
)

_LOADERS = {"ess": load_ess_corpus, "nhanes": load_nhanes_corpus}


def _result_to_dict(r) -> dict:
    d = {
        "generator_id": r.generator_id,
        "generator_name": r.generator_name,
        "corpus": r.corpus_name,
        "d": r.d,
        "verdict": r.verdict,
        "auc_pass": r.auc_pass,
        "c2st": dataclasses.asdict(r.c2st),
        "fidelity": dataclasses.asdict(r.fidelity),
    }
    # tuples-of-tuples -> JSON-friendly lists
    d["c2st"]["top_features"] = [list(t) for t in r.c2st.top_features]
    d["c2st"]["top_rate_gaps"] = [list(t) for t in r.c2st.top_rate_gaps]
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=sorted(_LOADERS), default="ess")
    ap.add_argument("--config", default="lacuna_tabular_110")
    ap.add_argument("--n-cols", type=int, default=12)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--auc-pass", type=float, default=0.60)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = Path(args.out) if args.out else Path(f"runs/realism_gate_{args.corpus}.json")
    out.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 96)
    print(f"REALISM GATE — corpus={args.corpus} config={args.config} "
          f"n_cols={args.n_cols} seed={args.seed}")
    print("=" * 96)

    t0 = time.time()
    corpus = _LOADERS[args.corpus](n_cols=args.n_cols, seed=args.seed)
    print(f"[{time.strftime('%H:%M:%S')}] corpus loaded: d={corpus.d} "
          f"n_real={corpus.n_real} n_complete={corpus.n_complete} "
          f"blocks={len(corpus.block_names)} block_aware={corpus.block_aware}")
    print(f"  columns: {corpus.columns}")
    print(f"  real overall rate: {float(corpus.M_real.mean()):.4f}")

    registry = load_registry_from_config(args.config)
    print(f"[{time.strftime('%H:%M:%S')}] registry K={registry.K}; running gate...")

    results, errors = [], []
    for g in registry:
        try:
            r = run_realism_gate(g, corpus, seed=args.seed, auc_pass=args.auc_pass)
            results.append(r)
        except Exception as e:  # noqa: BLE001 — record per-generator failure, keep sweeping
            errors.append({"generator_id": g.generator_id, "generator_name": g.name,
                           "error": f"{type(e).__name__}: {e}"})
            print(f"  [error] {g.name}: {type(e).__name__}: {e}")

    print("\n" + format_gate_table(results))
    if errors:
        print(f"\n{len(errors)} generator(s) errored (see JSON).")

    n_pass = sum(1 for r in results if r.verdict == "pass")
    payload = {
        "corpus": args.corpus,
        "config": args.config,
        "n_cols": args.n_cols,
        "seed": args.seed,
        "auc_pass": args.auc_pass,
        "block_aware": corpus.block_aware,
        "columns": list(corpus.columns),
        "real_overall_rate": float(corpus.M_real.mean()),
        "n_pass": n_pass,
        "n_total": len(results),
        "n_errors": len(errors),
        "results": [_result_to_dict(r) for r in results],
        "errors": errors,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nPASS {n_pass}/{len(results)}; wrote {out} ({payload['elapsed_sec']}s)")


if __name__ == "__main__":
    main()
