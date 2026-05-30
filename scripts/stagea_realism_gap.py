#!/usr/bin/env python3
"""
Stage A (ADR-0007) — the synthetic-vs-real realism GAP report.

Generates a corpus of footprints from our CURRENT synthetic composer (full diversity), loads the
real corpus (`stagea_real_footprints.py`), and quantifies the gap two ways:

  1. A real-vs-synthetic discriminator (standardised logistic regression, 5-fold CV) on the 20-D
     footprint -> ROC-AUC. AUC ~0.5 = indistinguishable (realistic); ~1.0 = trivially separable
     (unrealistic). The discriminator's standardised coefficients name WHICH features separate them.
  2. A per-feature table: real vs synthetic mean ± sd and the standardised gap (which direction we
     are off, and by how much).

This is the discriminator-as-RULER (ADR-0007 commitment 5): we read it to find where the generator
is unrealistic, we do NOT train against it. The output is the Stage B spec.

Deterministic via explicit seeds. Usage:
    python scripts/stagea_realism_gap.py [--n-synth 400] [--seed 0]
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

from lacuna.core.rng import RNGState
from lacuna.data.catalog import create_default_catalog
from lacuna.data.ingestion import RawDataset
from lacuna.data.semisynthetic import subsample_raw
from lacuna.data.mixed_batch import sample_column_classes
from lacuna.data.mixed_missingness import compose_mixed_missingness
from lacuna.data.composition_target import sample_composition_target
from lacuna.data.composition_sampler import compose_composition_missingness
from lacuna.data.missingness_footprint import missingness_footprint, FOOTPRINT_FEATURES

REAL = Path("/mnt/artifacts/project_lacuna/composition/stagea_real_footprints.json")
OUT = Path("/mnt/artifacts/project_lacuna/composition/stagea_gap.json")
MAX_D = 25  # cap columns to the real corpus's regime


def _complete_datasets():
    cat = create_default_catalog()
    out = []
    for name in cat.list_datasets():
        try:
            raw = cat.load(name)
        except Exception:  # noqa: BLE001
            continue
        if raw.d >= 4 and raw.n >= 100 and np.isfinite(raw.data).all():
            out.append(raw)
    return out


def _synthetic_footprints(n_synth: int, seed: int, generator: str = "legacy",
                          block_rate_share: float = 0.7) -> list:
    """Footprints of n_synth synthetic datasets in the real (n, d, miss) regime.

    generator="legacy"      -> the CURRENT full-diversity per-column composer (Stage A baseline,
                               AUC 1.000). Bit-identical to the original gap report.
    generator="composition" -> the Stage-B composition-controlled, realism-tuned generator: a target
                               composition is drawn from the broad simplex prior and realised with
                               the joint/block mechanisms (ADR-0007). `block_rate_share` tunes how
                               much of each structured class's missingness is block (co-missing).
    """
    rng = RNGState(seed=seed)
    datasets = _complete_datasets()
    if not datasets:
        raise RuntimeError("no complete datasets available to punch holes in")
    out = []
    for _ in range(n_synth):
        raw = datasets[rng.randint(0, len(datasets), (1,)).item()]
        # column subset to the real regime
        if raw.d > MAX_D:
            cols = sorted(int(c) for c in rng.choice(raw.d, size=MAX_D, replace=False))
            raw = RawDataset(data=raw.data[:, cols],
                             feature_names=tuple(raw.feature_names[c] for c in cols),
                             name=raw.name)
        target_n = rng.randint(200, 3001, (1,)).item()
        raw_sub = subsample_raw(raw, max_rows=target_n, rng=rng.spawn())
        if generator == "composition":
            target = sample_composition_target(rng.spawn())
            res = compose_composition_missingness(
                raw_sub, target, rng.spawn(), block_rate_share=block_rate_share)
        elif generator == "legacy":
            classes = sample_column_classes(raw_sub.d, rng.spawn(), p_observed=0.25)
            res = compose_mixed_missingness(
                raw_sub, classes, rng.spawn(),
                target_miss_rate=0.3, mnar_diverse=True, mar_diverse=True, compensate_rate=True)
        else:
            raise ValueError(f"unknown generator {generator!r}; use 'legacy' or 'composition'")
        out.append(missingness_footprint(res.observed.x, res.observed.r))
    return out


def _matrix(corpus: list) -> np.ndarray:
    return np.array([[c[k] for k in FOOTPRINT_FEATURES] for c in corpus], dtype=float)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-synth", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-seeds", type=int, default=1,
                    help="repeat over this many seeds (seed, seed+1, ...); report AUC mean±sd. "
                         ">=5 is the reporting standard for a stochastic generator.")
    ap.add_argument("--generator", choices=["legacy", "composition"], default="legacy",
                    help="legacy=current per-column composer (Stage-A baseline); "
                         "composition=Stage-B composition-controlled generator")
    ap.add_argument("--block-rate-share", type=float, default=0.85,
                    help="composition generator: fraction of each structured class's missingness "
                         "placed in co-missing blocks (higher = more structure)")
    ap.add_argument("--real", type=Path, default=REAL)
    ap.add_argument("--output", type=Path, default=OUT)
    args = ap.parse_args()

    real_corpus = json.loads(args.real.read_text())["corpus"]
    real = np.array([[c["footprint"][k] for k in FOOTPRINT_FEATURES] for c in real_corpus], dtype=float)
    y_base = [1] * len(real)
    print(f"real footprints: {real.shape[0]}")
    print(f"generating {args.n_synth} x {args.n_seeds} seed(s) synthetic footprints "
          f"({args.generator} generator"
          + (f", block_rate_share={args.block_rate_share}" if args.generator == "composition" else "")
          + ") ...", flush=True)

    # --- discriminator (real=1, synth=0), once per seed -> AUC distribution ---
    per_seed_auc, synth_pool = [], []
    for s in range(args.seed, args.seed + args.n_seeds):
        synth = _matrix(_synthetic_footprints(args.n_synth, s, args.generator, args.block_rate_share))
        synth_pool.append(synth)
        X = np.vstack([real, synth])
        y = np.array(y_base + [0] * len(synth))
        Xs = StandardScaler().fit_transform(X)
        a = float(np.mean(cross_val_score(LogisticRegression(max_iter=2000, C=1.0), Xs, y,
                                          cv=5, scoring="roc_auc")))
        per_seed_auc.append(a)
        print(f"  seed {s}: AUC {a:.4f}", flush=True)
    auc_mean, auc_sd = float(np.mean(per_seed_auc)), float(np.std(per_seed_auc))

    # Pool all seeds' synth for a stable per-feature table + fitted coefficients.
    synth = np.vstack(synth_pool)
    X = np.vstack([real, synth])
    y = np.array(y_base + [0] * len(synth))
    Xs = StandardScaler().fit_transform(X)
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(Xs, y)
    coefs = sorted(zip(FOOTPRINT_FEATURES, clf.coef_[0]), key=lambda t: -abs(t[1]))
    auc = auc_mean

    # --- per-feature gap ---
    rm, rs = real.mean(0), real.std(0)
    sm, ss = synth.mean(0), synth.std(0)
    pooled = np.sqrt((rs ** 2 + ss ** 2) / 2.0) + 1e-9
    gap = (rm - sm) / pooled  # +ve = real higher than synthetic

    print("\n" + "=" * 78)
    print(f"REALISM GAP — discriminator ROC-AUC = {auc_mean:.3f} ± {auc_sd:.3f} "
          f"over {args.n_seeds} seed(s)   (0.5 = indistinguishable, 1.0 = trivially separable)")
    print("=" * 78)
    print("\ntop separating features (|standardised logreg coef|; sign: +=>real-leaning):")
    for name, c in coefs[:8]:
        print(f"  {name:22s} coef {c:+6.2f}   real {rm[FOOTPRINT_FEATURES.index(name)]:6.3f}"
              f" vs synth {sm[FOOTPRINT_FEATURES.index(name)]:6.3f}   gap {gap[FOOTPRINT_FEATURES.index(name)]:+5.2f}σ")

    print("\nper-feature (real mean±sd | synth mean±sd | standardised gap, real−synth):")
    order = np.argsort(-np.abs(gap))
    for i in order:
        k = FOOTPRINT_FEATURES[i]
        print(f"  {k:22s} {rm[i]:6.3f}±{rs[i]:5.3f} | {sm[i]:6.3f}±{ss[i]:5.3f} | {gap[i]:+5.2f}σ")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "generator": args.generator, "block_rate_share": args.block_rate_share, "seed": args.seed,
        "n_seeds": args.n_seeds, "auc_mean": round(auc_mean, 4), "auc_sd": round(auc_sd, 4),
        "auc_per_seed": [round(a, 4) for a in per_seed_auc],
        "auc": round(auc, 4), "n_real": int(len(real)), "n_synth_per_seed": int(args.n_synth),
        "top_features": [[n, round(float(c), 4)] for n, c in coefs],
        "per_feature": {FOOTPRINT_FEATURES[i]: {
            "real": [round(float(rm[i]), 4), round(float(rs[i]), 4)],
            "synth": [round(float(sm[i]), 4), round(float(ss[i]), 4)],
            "gap_sigma": round(float(gap[i]), 4)} for i in range(len(FOOTPRINT_FEATURES))},
    }, indent=2))
    print(f"\nWrote gap report -> {args.output}")


if __name__ == "__main__":
    main()
