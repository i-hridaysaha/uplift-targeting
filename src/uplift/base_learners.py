"""Base learners the meta-learners and the response baseline are built over.

Every LightGBM-backed model on the board shares one parameter set (``LGBM_PARAMS``)
so the bakeoff compares learner designs, not tuning. ``make_classifier`` and
``make_regressor`` return a fitted-API estimator for a named base learner, which
lets the same designs be re-run over a simpler learner as a robustness check:

- ``"lightgbm"``: the default, gradient-boosted trees, categoricals handled natively.
- ``"logistic"``: a linear model over one-hot categoricals and scaled numerics
  (``LogisticRegression`` for outcomes, ``Ridge`` for the X-learner's imputed
  effects). Deliberately simple; if the model ordering survives it, the ordering
  is a fact about the data rather than about one learner.
"""

from __future__ import annotations

from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from uplift.data.splits import SEED

BASE_LEARNERS = ("lightgbm", "logistic")

# Shared LightGBM base. The response baseline and the meta-learners use exactly
# these, so a model's rank on the board reflects its design, not its budget.
LGBM_PARAMS: dict[str, object] = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "n_jobs": -1,
    "verbose": -1,
}


def _linear_preprocessing():
    """One-hot the pandas ``category`` columns, scale the rest."""
    return make_column_transformer(
        (
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            make_column_selector(dtype_include="category"),
        ),
        (StandardScaler(), make_column_selector(dtype_exclude="category")),
    )


def make_classifier(base: str = "lightgbm", seed: int = SEED, **lgbm_params: object):
    """A binary classifier with ``fit`` / ``predict_proba`` for the named base learner."""
    if base == "lightgbm":
        return LGBMClassifier(**{**LGBM_PARAMS, "random_state": seed, **lgbm_params})
    if base == "logistic":
        return make_pipeline(_linear_preprocessing(), LogisticRegression(max_iter=1000))
    raise ValueError(f"unknown base learner {base!r}; choose from {BASE_LEARNERS}")


def make_regressor(base: str = "lightgbm", seed: int = SEED, **lgbm_params: object):
    """A regressor with ``fit`` / ``predict`` for the named base learner."""
    if base == "lightgbm":
        return LGBMRegressor(**{**LGBM_PARAMS, "random_state": seed, **lgbm_params})
    if base == "logistic":
        return make_pipeline(_linear_preprocessing(), Ridge(alpha=1.0))
    raise ValueError(f"unknown base learner {base!r}; choose from {BASE_LEARNERS}")
