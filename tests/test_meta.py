"""Unit tests for the S/T/X meta-learners on synthetic data with known uplift."""

import numpy as np
import pandas as pd
import pytest

from uplift.eval.metrics import qini_coefficient
from uplift.models.meta import SLearner, TLearner, XLearner

LEARNERS = [SLearner, TLearner, XLearner]
FEATURES = ["x0", "noise", "channel"]


def _uplift_frame(n: int = 3000, seed: int = 0) -> pd.DataFrame:
    """Synthetic data where only high-``x0`` customers respond to treatment.

    Base response is constant; treatment adds a large effect for ``x0 > 0.5`` and
    none below, so true per-row uplift ranks by ``x0``. A model with uplift signal
    should rank the high-``x0`` rows first.
    """
    rng = np.random.default_rng(seed)
    x0 = rng.random(n)
    df = pd.DataFrame(
        {
            "x0": x0.astype("float32"),
            "noise": rng.normal(size=n).astype("float32"),
            "channel": pd.Categorical(rng.choice(["web", "phone"], n)),
            "treatment": rng.integers(0, 2, n).astype("int8"),
        }
    )
    effect = np.where(x0 > 0.5, 0.4, 0.0)
    prob = np.clip(0.15 + df["treatment"].to_numpy() * effect, 0.0, 1.0)
    df["outcome"] = (rng.random(n) < prob).astype("int8")
    return df


def _fit(cls: type, df: pd.DataFrame, seed: int = 42):
    """Fit a learner on the feature columns, treatment, and outcome."""
    return cls(seed=seed).fit(df[FEATURES], df["treatment"].to_numpy(), df["outcome"].to_numpy())


@pytest.mark.parametrize("cls", LEARNERS)
def test_predict_uplift_shape_and_finite(cls):
    df = _uplift_frame()
    uplift = _fit(cls, df).predict_uplift(df[FEATURES])
    assert uplift.shape == (len(df),)
    assert np.all(np.isfinite(uplift))


@pytest.mark.parametrize("cls", LEARNERS)
def test_beats_random_ranking(cls):
    df = _uplift_frame()
    uplift = _fit(cls, df).predict_uplift(df[FEATURES])
    normalized = qini_coefficient(
        df["outcome"].to_numpy(), uplift, df["treatment"].to_numpy()
    ).normalized
    assert normalized > 0.05  # treat-everyone (constant score) scores 0


@pytest.mark.parametrize("cls", LEARNERS)
def test_detects_where_effect_lives(cls):
    df = _uplift_frame()
    uplift = _fit(cls, df).predict_uplift(df[FEATURES])
    x0 = df["x0"].to_numpy()
    assert uplift[x0 > 0.5].mean() > uplift[x0 <= 0.5].mean()


@pytest.mark.parametrize("cls", LEARNERS)
def test_reproducible(cls):
    df = _uplift_frame()
    a = _fit(cls, df).predict_uplift(df[FEATURES])
    b = _fit(cls, df).predict_uplift(df[FEATURES])
    np.testing.assert_allclose(a, b)


def test_s_learner_uses_treatment_feature():
    model = _fit(SLearner, _uplift_frame())
    assert "treatment" in model.feature_names_


def test_t_learner_fits_one_model_per_arm():
    df = _uplift_frame()
    model = _fit(TLearner, df)
    # Arm models are distinct fits, so their predictions differ on the same rows.
    p_t = model.model_t.predict_proba(df[FEATURES])[:, 1]
    p_c = model.model_c.predict_proba(df[FEATURES])[:, 1]
    assert not np.allclose(p_t, p_c)


def test_x_learner_propensity_is_treated_fraction():
    df = _uplift_frame()
    model = _fit(XLearner, df)
    assert model.propensity_ == pytest.approx(df["treatment"].mean())


def test_x_learner_respects_explicit_propensity():
    df = _uplift_frame()
    model = XLearner(seed=42, propensity=0.5).fit(
        df[FEATURES], df["treatment"].to_numpy(), df["outcome"].to_numpy()
    )
    assert model.propensity_ == 0.5
