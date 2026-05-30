#!/usr/bin/env python3
"""
Stage A (ADR-0007) — build the REAL-side missingness-footprint corpus.

Expands the real side of the realism check beyond the 14 labelled survey anchors by sampling
analyst-realistic missingness blocks from the large raw NHANES tables (`real_mask_sampler`) and
computing each block's observable footprint (`missingness_footprint`). Footprint matching needs no
mechanism labels, so the corpus can be grown cheaply (ADR-0007 commitment 5). Writes the corpus
(tiny footprint vectors) for the synthetic-vs-real gap report to consume.

Deterministic via an explicit seed. Usage:
    python scripts/stagea_real_footprints.py [--blocks-per-source 200] [--seed 0]
"""

import argparse
import gc
import glob
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lacuna.data.missingness_footprint import missingness_footprint, FOOTPRINT_FEATURES
from lacuna.data.real_mask_sampler import sample_mask_blocks

ANCHOR_DIR = PROJECT_ROOT / "lacuna_survey" / "evaluation_data"
NHANES_TABLES = [
    "/mnt/data/lacuna/nhanes/demographics_clean.csv",
    "/mnt/data/lacuna/nhanes/questionnaire_clean.csv",
]
OUT = Path("/mnt/artifacts/project_lacuna/composition/stagea_real_footprints.json")


def _numeric_values(df: pd.DataFrame) -> np.ndarray:
    num = df.select_dtypes("number")
    drop = [c for c in num.columns if c.upper() in ("SEQN", "ID", "RESPONDENT_ID")]
    return num.drop(columns=drop).values.astype(float)


def _footprints_from_anchors() -> list:
    out = []
    for f in sorted(glob.glob(str(ANCHOR_DIR / "*_real.csv"))):
        x = _numeric_values(pd.read_csv(f))
        r = ~np.isnan(x)
        out.append({"source": "anchor", "name": os.path.basename(f).replace("_real.csv", ""),
                    "n": int(x.shape[0]), "d": int(x.shape[1]),
                    "miss": round(float(np.isnan(x).mean()), 4),
                    "footprint": missingness_footprint(x, r)})
    return out


def _footprints_from_table(path: str, rng: np.random.Generator, n_blocks: int) -> list:
    if not os.path.exists(path):
        print(f"  skip (missing): {path}"); return []
    print(f"  loading {os.path.basename(path)} ...", flush=True)
    values = _numeric_values(pd.read_csv(path))
    print(f"    table {values.shape}, overall miss {np.isnan(values).mean():.1%}; sampling blocks ...",
          flush=True)
    blocks = sample_mask_blocks(values, rng, n_blocks=n_blocks,
                                min_cols=4, max_cols=25, min_rows=200, max_rows=3000,
                                miss_lo=0.02, miss_hi=0.6, max_tries_per_block=60)
    out = []
    for x, r in blocks:
        out.append({"source": os.path.basename(path).replace(".csv", ""),
                    "n": int(x.shape[0]), "d": int(x.shape[1]),
                    "miss": round(float(np.isnan(x).mean()), 4),
                    "footprint": missingness_footprint(x, r)})
    del values; gc.collect()
    print(f"    accepted {len(out)}/{n_blocks} blocks", flush=True)
    return out


def _summarise(label: str, corpus: list) -> dict:
    if not corpus:
        print(f"\n[{label}] empty"); return {}
    arr = {k: np.array([c["footprint"][k] for c in corpus]) for k in FOOTPRINT_FEATURES}
    print(f"\n[{label}] n={len(corpus)}  (per-feature mean ± sd)")
    summ = {}
    for k in FOOTPRINT_FEATURES:
        m, s = float(arr[k].mean()), float(arr[k].std())
        summ[k] = [round(m, 4), round(s, 4)]
        print(f"  {k:22s} {m:7.3f} ± {s:6.3f}")
    return summ


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--blocks-per-source", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", type=Path, default=OUT)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    print("=== anchors ===")
    anchors = _footprints_from_anchors()
    print(f"  {len(anchors)} labelled anchors")

    print("=== sampled NHANES blocks ===")
    sampled = []
    for path in NHANES_TABLES:
        sampled += _footprints_from_table(path, rng, args.blocks_per_source)

    corpus = anchors + sampled
    _summarise("anchors only", anchors)
    _summarise("FULL real corpus (anchors + sampled)", corpus)

    # Realised missingness of the sampled blocks (confound/realism sanity).
    if sampled:
        ms = np.array([c["miss"] for c in sampled])
        print(f"\nsampled-block miss rate: mean {ms.mean():.3f}, range [{ms.min():.3f}, {ms.max():.3f}]")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(
        {"n_anchors": len(anchors), "n_sampled": len(sampled),
         "features": list(FOOTPRINT_FEATURES), "corpus": corpus}, indent=2))
    print(f"\nWrote {len(corpus)} real footprints -> {args.output}")


if __name__ == "__main__":
    main()
