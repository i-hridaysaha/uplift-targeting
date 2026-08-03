"""Meta-learners for uplift over a shared LightGBM base learner: S, T, and X.

Each learner estimates a per-row treatment effect (uplift) by combining ordinary
outcome models; the base learner is the same LightGBM used by the response-model
baseline, so the bakeoff compares *learners*, not hyper-parameters.

- **S-learner** -- one model ``f(X, T)``; uplift = ``f(X, 1) - f(X, 0)``.
- **T-learner** -- two models, one per arm; uplift = ``mu1(X) - mu0(X)``.
- **X-learner** -- T-learner outcome models, then impute treatment effects and fit
  two second-stage regressors, combined by the propensity. On randomized data the
  propensity is the constant marginal treated rate (no confounding), which is the
  empirical treated fraction of the training rows.

Every learner takes ``(X, treatment, y)`` at ``fit`` and returns a 1-D uplift
array from ``predict_uplift``. ``X`` is feature-only; any stray ``treatment``
column is dropped defensively so it never leaks in twice. pandas ``category``
dtypes are consumed natively by LightGBM.
"""

from __future__ import annotations

from typing import Self

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor

from uplift.data.schema import TREATMENT_COL
from uplift.data.splits import SEED

# Shared base learner. Mirrors the response-model baseline's params on purpose so
# the leaderboard measures the learner design, not tuning. Classifier params drive
# the outcome models; the X-learner's second stage regresses continuous imputed
# effects, so it reuses the same core with the regression default objective.
BASE_LGBM_PARAMS: dict[str, object] = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "n_jobs": -1,
    "verbose": -1,
}


def _features_only(x: pd.DataFrame) -> pd.DataFrame:
    """Drop a stray treatment column so it is never used as an ordinary feature."""
    return x.drop(columns=[TREATMENT_COL], errors="ignore")


class SLearner:
    """Single-model learner: one classifier over features + treatment.

    Uplift is the model's predicted response with treatment forced on minus with
    it forced off, so a base learner that ignores the treatment column yields zero
    uplift everywhere.
    """

    def __init__(self, seed: int = SEED, **params: object) -> None:
        self.params = {**BASE_LGBM_PARAMS, "random_state": seed, **params}
        self.model = LGBMClassifier(**self.params)
        self.feature_names_: list[str] = []

    def fit(self, x: pd.DataFrame, treatment: np.ndarray, y: np.ndarray) -> Self:
        """Fit ``f(X, T)`` with the treatment appended as a feature column."""
        features = _features_only(x).copy()
        features[TREATMENT_COL] = np.asarray(treatment).astype("int8")
        self.feature_names_ = features.columns.tolist()
        self.model.fit(features, np.asarray(y))
        return self

    def predict_uplift(self, x: pd.DataFrame) -> np.ndarray:
        """Return ``f(X, T=1) - f(X, T=0)`` per row."""
        base = _features_only(x)
        x1 = base.copy()
        x1[TREATMENT_COL] = np.int8(1)
        x0 = base.copy()
        x0[TREATMENT_COL] = np.int8(0)
        p1 = self.model.predict_proba(x1[self.feature_names_])[:, 1]
        p0 = self.model.predict_proba(x0[self.feature_names_])[:, 1]
        return (p1 - p0).astype("float64")


class TLearner:
    """Two-model learner: separate classifiers for the treated and control arms.

    Uplift is the treated-arm model's predicted response minus the control-arm
    model's, each fitted only on its own arm.
    """

    def __init__(self, seed: int = SEED, **params: object) -> None:
        self.params = {**BASE_LGBM_PARAMS, "random_state": seed, **params}
        self.model_t = LGBMClassifier(**self.params)
        self.model_c = LGBMClassifier(**self.params)

    def fit(self, x: pd.DataFrame, treatment: np.ndarray, y: np.ndarray) -> Self:
        """Fit ``mu1`` on treated rows and ``mu0`` on control rows."""
        features = _features_only(x)
        t = np.asarray(treatment).astype(bool)
        y = np.asarray(y)
        self.model_t.fit(features[t], y[t])
        self.model_c.fit(features[~t], y[~t])
        return self

    def predict_uplift(self, x: pd.DataFrame) -> np.ndarray:
        """Return ``mu1(X) - mu0(X)`` per row."""
        features = _features_only(x)
        p1 = self.model_t.predict_proba(features)[:, 1]
        p0 = self.model_c.predict_proba(features)[:, 1]
        return (p1 - p0).astype("float64")


class XLearner:
    """Cross-fitted learner: T-learner outcome models plus two effect regressors.

    Stage one fits the treated/control outcome models. Stage two imputes each row's
    treatment effect against the *other* arm's model and regresses those imputed
    effects on the features, once per arm. The two effect estimates are blended by
    the propensity ``g``: ``uplift = g * tau_control + (1 - g) * tau_treated``. On
    randomized data ``g`` is the constant marginal treated rate, taken from the
    training rows unless one is passed explicitly.
    """

    def __init__(self, seed: int = SEED, propensity: float | None = None, **params: object) -> None:
        clf_params = {**BASE_LGBM_PARAMS, "random_state": seed, **params}
        reg_params = {**BASE_LGBM_PARAMS, "random_state": seed, **params}
        self.model_t = LGBMClassifier(**clf_params)
        self.model_c = LGBMClassifier(**clf_params)
        self.tau_t = LGBMRegressor(**reg_params)
        self.tau_c = LGBMRegressor(**reg_params)
        self.propensity = propensity
        self.propensity_: float = 0.5

    def fit(self, x: pd.DataFrame, treatment: np.ndarray, y: np.ndarray) -> Self:
        """Fit the outcome models, impute cross-arm effects, and fit the regressors."""
        features = _features_only(x)
        t = np.asarray(treatment).astype(bool)
        y = np.asarray(y, dtype="float64")

        xt, xc = features[t], features[~t]
        self.model_t.fit(xt, y[t])
        self.model_c.fit(xc, y[~t])

        # Impute each arm's treatment effect against the opposite arm's model.
        imputed_t = y[t] - self.model_c.predict_proba(xt)[:, 1]
        imputed_c = self.model_t.predict_proba(xc)[:, 1] - y[~t]
        self.tau_t.fit(xt, imputed_t)
        self.tau_c.fit(xc, imputed_c)

        self.propensity_ = float(t.mean()) if self.propensity is None else self.propensity
        return self

    def predict_uplift(self, x: pd.DataFrame) -> np.ndarray:
        """Return the propensity-weighted blend of the two effect regressors."""
        features = _features_only(x)
        g = self.propensity_
        tau_c = self.tau_c.predict(features)
        tau_t = self.tau_t.predict(features)
        return (g * tau_c + (1.0 - g) * tau_t).astype("float64")
