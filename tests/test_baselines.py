"""Unit tests for the two naive targeting baselines."""

import numpy as np
import pandas as pd

from uplift.eval.baselines import response_model_scores, treat_everyone_scores
from uplift.eval.metrics import qini_coefficient


def _frame(n: int = 600, seed: int = 2):
    """Features + a signal-carrying outcome + a category column + treatment."""
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=n)
    df = pd.DataFrame(
        {
            "signal": signal.astype("float32"),
            "noise": rng.normal(size=n).astype("float32"),
            "channel": pd.Categorical(rng.choice(["web", "phone"], n)),
            "treatment": rng.integers(0, 2, n).astype("int8"),
        }
    )
    prob = 1.0 / (1.0 + np.exp(-signal))  # outcome tracks `signal`
    df["outcome"] = (rng.random(n) < prob).astype("int8")
    return df


def test_treat_everyone_is_constant():
    scores = treat_everyone_scores(500)
    assert scores.shape == (500,)
    assert np.all(scores == scores[0])


def test_treat_everyone_has_no_ranking_power():
    rng = np.random.default_rng(0)
    n = 800
    treatment = rng.integers(0, 2, n)
    y = rng.integers(0, 2, n)
    normalized = qini_coefficient(y, treat_everyone_scores(n), treatment).normalized
    assert normalized == 0.0 or np.isnan(normalized)


def test_response_model_scores_are_probabilities():
    df = _frame()
    x = df.drop(columns=["outcome"])
    scores = response_model_scores(x, df["outcome"], x)
    assert scores.shape == (len(df),)
    assert scores.min() >= 0.0 and scores.max() <= 1.0


def test_response_model_ranks_by_signal():
    df = _frame(800)
    x = df.drop(columns=["outcome"])
    scores = response_model_scores(x, df["outcome"], x)
    # Outcome rises with `signal`, so predicted response should correlate with it.
    assert np.corrcoef(scores, df["signal"])[0, 1] > 0.3


def test_response_model_ignores_treatment_column():
    df = _frame()
    x = df.drop(columns=["outcome"])
    with_treatment = response_model_scores(x, df["outcome"], x, seed=7)
    without_treatment = response_model_scores(
        x.drop(columns=["treatment"]), df["outcome"], x.drop(columns=["treatment"]), seed=7
    )
    np.testing.assert_allclose(with_treatment, without_treatment)


def test_response_model_is_reproducible():
    df = _frame()
    x = df.drop(columns=["outcome"])
    a = response_model_scores(x, df["outcome"], x, seed=42)
    b = response_model_scores(x, df["outcome"], x, seed=42)
    np.testing.assert_allclose(a, b)
