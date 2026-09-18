"""Unit tests for the base-learner factory and the meta-learners over a linear base."""

import numpy as np
import pytest
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.pipeline import Pipeline

from uplift.base_learners import BASE_LEARNERS, LGBM_PARAMS, make_classifier, make_regressor
from uplift.eval.baselines import response_model_scores
from uplift.eval.metrics import qini_coefficient
from uplift.models.meta import SLearner, TLearner, XLearner

from .test_meta import FEATURES, _uplift_frame

LEARNERS = [SLearner, TLearner, XLearner]


def test_lightgbm_is_the_default_with_the_shared_params():
    clf = make_classifier()
    assert isinstance(clf, LGBMClassifier)
    assert isinstance(make_regressor(), LGBMRegressor)
    params = clf.get_params()
    assert all(params[k] == v for k, v in LGBM_PARAMS.items())


def test_logistic_is_a_preprocessing_pipeline():
    assert isinstance(make_classifier("logistic"), Pipeline)
    assert isinstance(make_regressor("logistic"), Pipeline)


def test_unknown_base_is_rejected():
    with pytest.raises(ValueError, match="unknown base learner"):
        make_classifier("random_forest")
    assert set(BASE_LEARNERS) == {"lightgbm", "logistic"}


@pytest.mark.parametrize("cls", LEARNERS)
def test_meta_learners_over_logistic_detect_the_effect(cls):
    df = _uplift_frame()
    model = cls(seed=42, base="logistic")
    model.fit(df[FEATURES], df["treatment"].to_numpy(), df["outcome"].to_numpy())
    uplift = model.predict_uplift(df[FEATURES])
    assert uplift.shape == (len(df),)
    assert np.all(np.isfinite(uplift))
    normalized = qini_coefficient(
        df["outcome"].to_numpy(), uplift, df["treatment"].to_numpy()
    ).normalized
    assert normalized > 0.05
    # the effect lives above x0 = 0.5; the linear learner should rank those rows higher
    high = df["x0"].to_numpy() > 0.5
    assert uplift[high].mean() > uplift[~high].mean()


def test_response_model_over_logistic_handles_categoricals():
    df = _uplift_frame()
    scores = response_model_scores(
        df[FEATURES], df["outcome"].to_numpy(), df[FEATURES], seed=42, base="logistic"
    )
    assert scores.shape == (len(df),)
    assert np.all((scores >= 0) & (scores <= 1))
    assert len(np.unique(np.round(scores, 6))) > 10
