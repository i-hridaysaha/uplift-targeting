"""The two naive targeting baselines every uplift model must beat.

Both emit a per-row score the evaluation harness then ranks and measures, so they
plug into the same Qini/AUUC/decile pipeline as a real uplift model:

- ``treat_everyone`` -- a constant score. The ranking carries no information, so
  its Qini curve is the random diagonal (coefficient ~ 0). It stands in for
  "contact the whole list".
- ``response_model`` -- a plain LightGBM classifier of ``P(outcome | X)`` fitted on
  the whole population with the treatment column dropped. It targets the customers
  most likely to respond, ignoring incrementality. Beating this is the bar that
  separates uplift modeling from ordinary response modeling (EVAL.md).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from uplift.data.schema import TREATMENT_COL
from uplift.data.splits import SEED

RESPONSE_MODEL_PARAMS = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "n_jobs": -1,
    "verbose": -1,
}


def treat_everyone_scores(n: int) -> np.ndarray:
    """Return the constant score for the treat-everyone baseline (length ``n``)."""
    return np.zeros(int(n), dtype="float64")


def response_model_scores(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_eval: pd.DataFrame,
    seed: int = SEED,
    **lgbm_params: object,
) -> np.ndarray:
    """Fit ``P(outcome | X)`` on ``x_train`` and return its probabilities for ``x_eval``.

    A plain LightGBM classifier with no notion of treatment: the ``treatment``
    column is dropped from both frames if present, so the score ranks by predicted
    response, not uplift. Pass disjoint train/eval frames (e.g. train fold vs
    validation fold) to keep the evaluation out-of-sample. pandas ``category``
    dtypes are used natively by LightGBM.
    """
    features = x_train.drop(columns=[TREATMENT_COL], errors="ignore")
    eval_features = x_eval.drop(columns=[TREATMENT_COL], errors="ignore")

    params = {**RESPONSE_MODEL_PARAMS, "random_state": seed, **lgbm_params}
    model = LGBMClassifier(**params)
    model.fit(features, np.asarray(y_train))
    return model.predict_proba(eval_features)[:, 1]
