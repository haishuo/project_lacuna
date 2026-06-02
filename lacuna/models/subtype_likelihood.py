"""
lacuna.models.subtype_likelihood

The DATA (likelihood) channel of the per-column subtype layer (ADR-0008 commitment 6).

A per-column readout over L = {threshold, detection, reject} (`subtype_ontology.LIKELIHOOD_LABELS`)
from the DEPLOYABLE distributional features only (`column_deployable_features` — observed-value
skew/kurtosis/SMD, no oracle), EXCLUDING `missing_rate` (see `n_features` / the __init__ note: a
per-subtype rate is a non-transferable confound). It detects the LOUD MNAR fingerprints (value-threshold and
detection-limit censoring) and REJECTS the non-identifiable region (quiet self-censoring MNAR, MAR,
MCAR), folding the latter into one class — NOT the Stage-5 forced 3-way that collapses.

Why a gradient-boosted detector on the deployable features, and NOT the frozen-encoder + ColumnReadout
probe the mechanism arc used. Stage-Q attribution (scripts/stageQ_feature_attribution.py) measured the
loud-vs-reject ceiling three ways, matched rate:
  - deployable features alone:        ROC-AUC ~0.86-0.89  (the signal)
  - frozen-encoder token reps alone:  ROC-AUC ~0.60        (~chance — the dataset-task encoder does
                                                            NOT carry the per-column loud footprint)
  - features + encoder reps:          ROC-AUC ~0.76        (the 64-d reps DILUTE the 5 features)
So the encoder reps actively hurt (echoing the Stage-C "encoder is the bottleneck" finding). Among
deployable-feature models a random forest / gradient boosting (~0.85 loud-vs-reject AUC) clearly beats
both a small MLP (~0.78) and histogram gradient boosting (~0.79) on these 5 features, so the readout is
a random forest — the honest, best, fully DEPLOYABLE lens (no encoder, no oracle). The loud class is
rare (~10% of columns at matched rate), so `class_weight="balanced"` and a PROBABILITY output (consumed
by the Dirichlet fusion + the risk-coverage commit threshold) are used rather than a hard argmax: the
layer commits on the confidently-loud minority and abstains on the rest.

Determinism (Coding Bible Rule 6): the only stochasticity is the forest's, fixed by an injected `seed`
(sklearn `random_state`). The default `n_jobs=1` makes fit/predict BYTE-reproducible (under `n_jobs>1`
the forest is still identical but `predict_proba`'s parallel summation order can differ at ~1e-15).
Fails loud (Rule 1) on a wrong feature dimension, non-finite features, bad labels, or predict-before-fit.
"""

from typing import Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from lacuna.models.column_deployable_features import N_DEPLOYABLE_FEATURES
from lacuna.priors.subtype_ontology import N_LIKELIHOOD_LABELS


class SubtypeLikelihoodDetector:
    """Deployable-feature per-column subtype detector -> probabilities over L (threshold/detection/reject)."""

    def __init__(self, *, seed: int, n_features: int = N_DEPLOYABLE_FEATURES, n_estimators: int = 400,
                 max_depth: Optional[int] = None, min_samples_leaf: int = 2, n_jobs: int = 1):
        # n_features lets a caller pass a SUBSET of the deployable features. The subtype layer drops
        # `missing_rate` (index 0): the loud subtypes have no characteristic real-world miss rate (a
        # threshold's rate is just where its cutoff sits), so an in-generator loud-vs-reject rate gap is
        # a NON-TRANSFERABLE artifact (Stage-Q rate audit: loud realises ~0.197 vs reject ~0.248 even at
        # "matched" rate) — the Stage-5 confound. Excluding it gives the honest, transferable detector.
        if not 2 <= n_features <= N_DEPLOYABLE_FEATURES:
            raise ValueError(f"n_features must be in [2, {N_DEPLOYABLE_FEATURES}], got {n_features}")
        self.n_features = n_features
        self._clf = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
            class_weight="balanced", random_state=seed, n_jobs=n_jobs,
        )
        self._fitted = False

    def _check_features(self, features: np.ndarray) -> np.ndarray:
        f = np.asarray(features, dtype=float)
        if f.ndim != 2 or f.shape[1] != self.n_features:
            raise ValueError(f"features must be [N, {self.n_features}], got shape {f.shape}")
        if not np.isfinite(f).all():
            raise ValueError("features must be finite")
        return f

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "SubtypeLikelihoodDetector":
        """Fit on per-column (deployable features [N,5], L label [N] in {0,1,2}).

        Raises:
            ValueError: on a bad feature shape, non-finite features, out-of-range labels, or <2 classes.
        """
        f = self._check_features(features)
        y = np.asarray(labels)
        if y.shape != (f.shape[0],):
            raise ValueError(f"labels must be [{f.shape[0]}], got {y.shape}")
        if not np.isin(y, range(N_LIKELIHOOD_LABELS)).all():
            raise ValueError(f"labels must be in [0, {N_LIKELIHOOD_LABELS}); got {np.unique(y)}")
        if len(np.unique(y)) < 2:
            raise ValueError("need at least 2 distinct classes to fit the detector")
        self._clf.fit(f, y)
        self._fitted = True
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Per-column probabilities [N, N_LIKELIHOOD_LABELS] aligned to (threshold, detection, reject).

        Columns for any class absent from training are 0 (the classifier never saw them). Rows sum to 1
        over the classes the classifier did see.

        Raises:
            ValueError: if called before `fit`, or on a bad feature shape / non-finite features.
        """
        if not self._fitted:
            raise ValueError("predict_proba called before fit")
        f = self._check_features(features)
        raw = self._clf.predict_proba(f)                       # [N, n_seen_classes]
        out = np.zeros((f.shape[0], N_LIKELIHOOD_LABELS), dtype=float)
        for col, cls in enumerate(self._clf.classes_):
            out[:, int(cls)] = raw[:, col]
        return out
