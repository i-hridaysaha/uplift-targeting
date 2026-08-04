"""Uplift-driver explainability for the chosen model.

Two honest views of *what drives uplift* (not what drives response):

- **TreeSHAP** for the LightGBM-backed models, read straight from LightGBM's own
  ``pred_contrib`` -- the exact TreeSHAP algorithm ``shap.TreeExplainer`` delegates
  to for LightGBM, so the values are identical with no extra dependency (D26).
  For the class-transformation model the regressor output *is* the uplift, so its
  contributions are per-feature attributions of the uplift in the same units. For
  the S-learner the attribution is the ``T=1`` minus ``T=0`` contribution in the
  base classifier's log-odds margin -- a decomposition of a two-model difference,
  read with the MODELING.md caveat, and the treatment column itself is dropped
  (its difference is the average effect, not a heterogeneity driver).

- **Permutation importance** -- model-agnostic, in Qini units: shuffle a feature
  and measure how far the model's Qini falls. This is the driver view for the
  hand-rolled uplift forest, which no tree explainer can read, and a useful
  cross-check for any model.

Both return a tidy ``(feature, importance)`` table sorted most-important first.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from uplift.data.schema import TREATMENT_COL
from uplift.data.splits import SEED
from uplift.eval.metrics import qini_coefficient
from uplift.models.direct import ClassTransformation
from uplift.models.meta import SLearner


def _features_only(x: pd.DataFrame) -> pd.DataFrame:
    """Drop a stray treatment column so it is never treated as an ordinary feature."""
    return x.drop(columns=[TREATMENT_COL], errors="ignore")


def _contrib(booster: object, frame: pd.DataFrame) -> np.ndarray:
    """Per-feature TreeSHAP contributions (base-value column dropped) via LightGBM."""
    raw = np.asarray(booster.predict(frame, pred_contrib=True))  # type: ignore[attr-defined]
    return raw[:, :-1]  # last column is the base value


def treeshap_uplift(model: SLearner | ClassTransformation, x: pd.DataFrame) -> pd.DataFrame:
    """Mean absolute TreeSHAP contribution to uplift, per feature (most important first).

    ``ClassTransformation`` contributions are in uplift units; ``SLearner``
    contributions are the ``T=1`` minus ``T=0`` difference in the base learner's
    log-odds margin (treatment column excluded). Raises ``TypeError`` for a model
    with no LightGBM booster (e.g. the hand-rolled forest) -- use
    ``permutation_uplift_importance`` there.
    """
    base = _features_only(x)
    if isinstance(model, ClassTransformation):
        contrib = _contrib(model.model.booster_, base)
        features = base.columns.tolist()
    elif isinstance(model, SLearner):
        names = model.feature_names_  # feature order the classifier was fit on
        x1 = base.copy()
        x1[TREATMENT_COL] = np.int8(1)
        x0 = base.copy()
        x0[TREATMENT_COL] = np.int8(0)
        diff = _contrib(model.model.booster_, x1[names]) - _contrib(model.model.booster_, x0[names])
        keep = [i for i, name in enumerate(names) if name != TREATMENT_COL]
        contrib = diff[:, keep]
        features = [names[i] for i in keep]
    else:
        raise TypeError(
            f"{type(model).__name__} has no LightGBM booster; "
            "use permutation_uplift_importance instead."
        )

    importance = np.abs(contrib).mean(axis=0)
    return pd.DataFrame({"feature": features, "mean_abs_shap": importance}).sort_values(
        "mean_abs_shap", ascending=False, ignore_index=True
    )


def permutation_uplift_importance(
    model: object,
    x: pd.DataFrame,
    treatment: np.ndarray,
    y: np.ndarray,
    n_repeats: int = 5,
    seed: int = SEED,
) -> pd.DataFrame:
    """Drop in normalized Qini when each feature is shuffled (most important first).

    Model-agnostic: any object with ``predict_uplift(X)``. For each feature the
    column is permuted ``n_repeats`` times and the fall in Qini from the unshuffled
    baseline is averaged, so a larger value means the model leans on that feature
    more to rank uplift. This is the driver view for the hand-rolled uplift forest.
    """
    base = _features_only(x).reset_index(drop=True)
    t = np.asarray(treatment)
    y = np.asarray(y)
    rng = np.random.default_rng(seed)

    baseline = qini_coefficient(y, model.predict_uplift(base), t).normalized  # type: ignore[attr-defined]

    rows: list[dict[str, object]] = []
    for col in base.columns:
        drops = np.empty(n_repeats, dtype="float64")
        for r in range(n_repeats):
            shuffled = base.copy()
            # index the backing array so a category column stays categorical
            shuffled[col] = base[col].array[rng.permutation(len(base))]
            score = qini_coefficient(y, model.predict_uplift(shuffled), t).normalized  # type: ignore[attr-defined]
            drops[r] = baseline - score
        rows.append({"feature": col, "importance": float(drops.mean()), "std": float(drops.std())})
    return pd.DataFrame(rows).sort_values("importance", ascending=False, ignore_index=True)
