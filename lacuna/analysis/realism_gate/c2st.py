"""
lacuna.analysis.realism_gate.c2st

Classifier Two-Sample Test (Lopez-Paz & Oquab 2016) — the L1 footprint gate.

One job: given a real mask sample and a generated mask sample (column-aligned),
train a block-aware classifier to tell them apart and report how separable they
are. Near-chance separability ⇒ the generated masks are realistic.

Outputs:
- ``auc`` / ``accuracy`` : pooled out-of-fold discrimination (effect size). 0.5
  ⇒ indistinguishable. With large n any real difference is "significant", so the
  *effect size* (AUC near 0.5), not the p-value, is the practical gate.
- ``p_value`` : exact finite-sample significance. Under H0 (same distribution)
  each held-out correct/incorrect call is a fair coin, so the OOF correct-count
  is Binomial(n, 1/2); the one-sided tail is the exact C2ST p-value. This is the
  finite-sample anchor that the conformal C2ST (Hu & Lei 2024; Bansal 2025)
  generalises to weak classifiers via conformity scores — we use the exact
  special case for the 0/1 correctness score.
- localisation : per-feature linear-classifier weight magnitude + per-column
  real-vs-generated rate gap, so a failing generator points at *which* columns
  are unrealistic (the actionable C2ST signal).

Block-aware: when >= 2 blocks span both samples we use leave-block-out
(GroupKFold) so separability is not an artefact of within-block memorisation;
otherwise we fall back to stratified K-fold and flag it.

Deterministic given ``seed``. Offline analysis; never on the training path.
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from .mask_stats import check_mask, column_rates, feature_names, row_features


@dataclass(frozen=True)
class C2STResult:
    """Result of one classifier two-sample test (real vs generated masks)."""

    auc: float
    accuracy: float
    n_eval: int
    n_correct: int
    p_value: float
    block_aware: bool
    n_splits: int
    n_real: int
    n_gen: int
    top_features: Tuple[Tuple[str, float], ...]   # (name, |weight|), descending
    top_rate_gaps: Tuple[Tuple[str, float], ...]  # (col, gen_rate - real_rate), |.| desc


def _balance(
    n_real: int, n_gen: int, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """Index arrays subsampling the larger class down to the smaller (deterministic)."""
    k = min(n_real, n_gen)
    if k < 2:
        raise ValueError(f"each sample needs >= 2 rows after balancing, got k={k}")
    ridx = rng.choice(n_real, size=k, replace=False) if n_real > k else np.arange(n_real)
    gidx = rng.choice(n_gen, size=k, replace=False) if n_gen > k else np.arange(n_gen)
    ridx.sort()
    gidx.sort()
    return ridx, gidx


def run_c2st(
    M_real: np.ndarray,
    M_gen: np.ndarray,
    real_blocks: np.ndarray,
    gen_blocks: np.ndarray,
    columns: Tuple[str, ...],
    *,
    seed: int,
    n_splits: int = 5,
    max_iter: int = 500,
) -> C2STResult:
    """Run the block-aware C2ST. See module docstring for the contract.

    Args:
        M_real / M_gen: [n, d] uint8 masks on the SAME d columns.
        real_blocks / gen_blocks: per-row int block ids for each sample.
        columns: the d column names (for localisation reporting).
        seed: RNG seed for class balancing (the only stochastic step).
        n_splits: cross-validation folds (clamped to available blocks/rows).

    Raises:
        ValueError: on shape disagreement or degenerate samples.
    """
    M_real = check_mask(M_real, name="M_real")
    M_gen = check_mask(M_gen, name="M_gen")
    if M_real.shape[1] != M_gen.shape[1]:
        raise ValueError(f"column mismatch: real d={M_real.shape[1]} gen d={M_gen.shape[1]}")
    d = M_real.shape[1]
    if len(columns) != d:
        raise ValueError(f"columns length {len(columns)} != d {d}")
    if len(real_blocks) != M_real.shape[0] or len(gen_blocks) != M_gen.shape[0]:
        raise ValueError("block-id length must match its sample's row count")

    rng = np.random.default_rng(seed)
    ridx, gidx = _balance(M_real.shape[0], M_gen.shape[0], rng)
    Xr, Xg = row_features(M_real[ridx]), row_features(M_gen[gidx])
    X = np.vstack([Xr, Xg])
    y = np.concatenate([np.zeros(len(Xr)), np.ones(len(Xg))]).astype(np.int64)
    groups = np.concatenate([real_blocks[ridx], gen_blocks[gidx]]).astype(np.int64)

    n_groups = len(np.unique(groups))
    use_groups = n_groups >= 2
    splits = max(2, min(n_splits, n_groups if use_groups else len(y) // 2))

    if use_groups:
        splitter = GroupKFold(n_splits=splits)
        fold_iter = splitter.split(X, y, groups)
    else:
        splitter = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)
        fold_iter = splitter.split(X, y)

    oof = np.full(len(y), np.nan)
    coefs: List[np.ndarray] = []
    for tr, te in fold_iter:
        scaler = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=max_iter, random_state=seed)
        clf.fit(scaler.transform(X[tr]), y[tr])
        oof[te] = clf.predict_proba(scaler.transform(X[te]))[:, 1]
        coefs.append(np.abs(clf.coef_.ravel()))

    evaluated = ~np.isnan(oof)
    y_e, s_e = y[evaluated], oof[evaluated]
    auc = float(roc_auc_score(y_e, s_e))
    pred = (s_e >= 0.5).astype(np.int64)
    n_correct = int((pred == y_e).sum())
    n_eval = int(evaluated.sum())
    accuracy = n_correct / n_eval
    p_value = float(binomtest(n_correct, n_eval, 0.5, alternative="greater").pvalue)

    mean_coef = np.mean(np.vstack(coefs), axis=0)
    names = feature_names(d)
    order = np.argsort(mean_coef)[::-1]
    top_features = tuple((names[i], float(mean_coef[i])) for i in order[:5])

    rate_gap = column_rates(M_gen) - column_rates(M_real)
    gorder = np.argsort(np.abs(rate_gap))[::-1]
    top_rate_gaps = tuple((columns[i], float(rate_gap[i])) for i in gorder[:5])

    return C2STResult(
        auc=auc,
        accuracy=accuracy,
        n_eval=n_eval,
        n_correct=n_correct,
        p_value=p_value,
        block_aware=use_groups,
        n_splits=splits,
        n_real=int(M_real.shape[0]),
        n_gen=int(M_gen.shape[0]),
        top_features=top_features,
        top_rate_gaps=top_rate_gaps,
    )
