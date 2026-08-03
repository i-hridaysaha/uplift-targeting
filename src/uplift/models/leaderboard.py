"""Cross-fitted out-of-fold scoring and the model leaderboard.

Every model -- the two naive baselines and the S/T/X meta-learners -- is scored
the same way: for each CV fold, fit on the other folds and predict the held-out
fold, so no row is ever scored by a model that trained on it. The hold-out rows
(``fold == -1``) are excluded entirely; they are reserved for Phase 8. The
resulting out-of-fold uplift is fed to the evaluation harness, which reports each
metric as a per-fold mean and spread (the honest confidence band for a
high-variance metric like Qini -- see EVAL.md).

A *scorer* is ``(train, eval_frame, features, outcome, seed) -> uplift`` so a
model needs no shared class; the meta-learners are wrapped into scorers here.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from uplift.data.schema import TREATMENT_COL
from uplift.data.splits import HOLDOUT_FOLD, SEED
from uplift.eval.baselines import response_model_scores, treat_everyone_scores
from uplift.eval.metrics import auuc, fold_metric_band, qini_coefficient
from uplift.models.meta import SLearner, TLearner, XLearner

Scorer = Callable[[pd.DataFrame, pd.DataFrame, list[str], str, int], np.ndarray]

FOLD_COL = "fold"


def _qini_norm(y: np.ndarray, uplift: np.ndarray, treatment: np.ndarray) -> float:
    """Normalized Qini coefficient (the headline number) as a plain float for banding."""
    return qini_coefficient(y, uplift, treatment).normalized


def _auuc_norm(y: np.ndarray, uplift: np.ndarray, treatment: np.ndarray) -> float:
    """Normalized AUUC as a plain float for banding."""
    return auuc(y, uplift, treatment).normalized


def model_scorer(cls: type, **kwargs: object) -> Scorer:
    """Wrap any ``fit(X, t, y)`` / ``predict_uplift(X)`` model into a scorer.

    Extra keyword arguments are passed to the model constructor, so a caller can,
    e.g., cap the forest's per-tree sample size on the large dataset.
    """

    def scorer(
        train: pd.DataFrame, eval_frame: pd.DataFrame, features: list[str], outcome: str, seed: int
    ) -> np.ndarray:
        model = cls(seed=seed, **kwargs)
        model.fit(train[features], train[TREATMENT_COL].to_numpy(), train[outcome].to_numpy())
        return model.predict_uplift(eval_frame[features])

    return scorer


def _response_scorer(
    train: pd.DataFrame, eval_frame: pd.DataFrame, features: list[str], outcome: str, seed: int
) -> np.ndarray:
    """Response-model baseline: rank by predicted ``P(outcome | X)``, ignoring treatment."""
    return response_model_scores(
        train[features], train[outcome].to_numpy(), eval_frame[features], seed=seed
    )


def _treat_everyone_scorer(
    train: pd.DataFrame, eval_frame: pd.DataFrame, features: list[str], outcome: str, seed: int
) -> np.ndarray:
    """Treat-everyone baseline: a constant score, so the ranking is random."""
    return treat_everyone_scores(len(eval_frame))


# Insertion order is the leaderboard's row order: baselines first, then S/T/X.
MODELS: dict[str, Scorer] = {
    "treat_everyone": _treat_everyone_scorer,
    "response_model": _response_scorer,
    "s_learner": model_scorer(SLearner),
    "t_learner": model_scorer(TLearner),
    "x_learner": model_scorer(XLearner),
}


def crossfit_oof(
    dev: pd.DataFrame,
    features: list[str],
    outcome: str,
    scorer: Scorer,
    seed: int = SEED,
) -> np.ndarray:
    """Return out-of-fold uplift for every dev row (aligned to ``dev``'s order).

    ``dev`` must already exclude the hold-out. For each fold the scorer trains on
    the other folds and predicts the held fold, so each row's score is genuinely
    out-of-sample.
    """
    fold = dev[FOLD_COL].to_numpy()
    uplift = np.full(len(dev), np.nan, dtype="float64")
    for f in sorted(int(v) for v in np.unique(fold)):
        eval_mask = fold == f
        preds = scorer(dev[fold != f], dev[eval_mask], features, outcome, seed)
        uplift[eval_mask] = preds
    return uplift


def leaderboard(
    frame: pd.DataFrame,
    features: list[str],
    outcome: str,
    models: dict[str, Scorer] | None = None,
    seed: int = SEED,
) -> pd.DataFrame:
    """Score every model out-of-fold and return their Qini/AUUC bands as a table.

    One row per model with the cross-validated mean and std of the (headline,
    normalized) Qini and AUUC coefficients across folds. Hold-out rows are dropped;
    all bands come from the CV folds only.
    """
    models = MODELS if models is None else models
    dev = frame[frame[FOLD_COL] != HOLDOUT_FOLD].reset_index(drop=True)
    y = dev[outcome].to_numpy()
    t = dev[TREATMENT_COL].to_numpy()
    fold = dev[FOLD_COL].to_numpy()

    rows: list[dict[str, object]] = []
    for name, scorer in models.items():
        uplift = crossfit_oof(dev, features, outcome, scorer, seed=seed)
        qini = fold_metric_band(y, uplift, t, fold, _qini_norm)
        area = fold_metric_band(y, uplift, t, fold, _auuc_norm)
        rows.append(
            {
                "model": name,
                "qini_mean": qini.mean,
                "qini_std": qini.std,
                "auuc_mean": area.mean,
                "auuc_std": area.std,
                "n_folds": len(qini.per_fold),
            }
        )
    return pd.DataFrame(rows)
