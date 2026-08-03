"""Unit tests for the uplift metrics, cross-checked against scikit-uplift."""

import numpy as np
import pytest
from sklift.metrics import qini_auc_score, uplift_auc_score, uplift_by_percentile

from uplift.eval.metrics import (
    auuc,
    fold_metric_band,
    qini_coefficient,
    qini_curve,
    uplift_by_decile,
    uplift_curve,
)


def _synthetic(n: int = 1000, seed: int = 0):
    """Outcome whose treatment effect grows with a distinct, continuous score."""
    rng = np.random.default_rng(seed)
    treatment = rng.integers(0, 2, n)
    score = rng.normal(size=n)  # distinct continuous ranker
    effect = 0.25 * (score - score.min()) / (score.max() - score.min())
    prob = 0.1 + treatment * effect
    y = (rng.random(n) < prob).astype(int)
    return y, score, treatment


def test_qini_curve_matches_sklift():
    from sklift.metrics import qini_curve as sk_qini_curve

    y, u, t = _synthetic()
    x_ours, y_ours = qini_curve(y, u, t)
    x_sk, y_sk = sk_qini_curve(y, u, t)
    np.testing.assert_allclose(x_ours, x_sk)
    np.testing.assert_allclose(y_ours, y_sk)


def test_uplift_curve_matches_sklift():
    from sklift.metrics import uplift_curve as sk_uplift_curve

    y, u, t = _synthetic()
    x_ours, y_ours = uplift_curve(y, u, t)
    x_sk, y_sk = sk_uplift_curve(y, u, t)
    np.testing.assert_allclose(x_ours, x_sk)
    np.testing.assert_allclose(y_ours, y_sk)


def test_qini_coefficient_matches_sklift():
    y, u, t = _synthetic()
    assert qini_coefficient(y, u, t).normalized == pytest.approx(qini_auc_score(y, u, t))


def test_auuc_matches_sklift():
    y, u, t = _synthetic()
    assert auuc(y, u, t).normalized == pytest.approx(uplift_auc_score(y, u, t))


def test_perfect_ranking_normalizes_to_one():
    y, _, t = _synthetic()
    # Oracle uplift ranking: treated responders first, control responders last.
    perfect = y * t - y * (1 - t)
    assert qini_coefficient(y, perfect, t).normalized == pytest.approx(1.0)


def test_constant_score_has_zero_coefficient():
    y, _, t = _synthetic()
    flat = np.zeros(len(y))
    score = qini_coefficient(y, flat, t)
    assert score.raw == pytest.approx(0.0, abs=1e-9)
    assert score.normalized == pytest.approx(0.0, abs=1e-9)


def test_good_ranker_beats_random():
    y, u, t = _synthetic()
    assert qini_coefficient(y, u, t).raw > 0.0


def test_uplift_by_decile_shape_and_monotonicity():
    y, u, t = _synthetic(2000)
    table = uplift_by_decile(y, u, t, bins=10)
    assert list(table["decile"]) == list(range(1, 11))
    assert table["count"].sum() == 2000
    # Effect grows with the score, so the top decile should out-lift the bottom.
    assert table.loc[0, "observed_uplift"] > table.loc[9, "observed_uplift"]


def test_uplift_by_decile_matches_sklift_on_distinct_scores():
    y, u, t = _synthetic(1000)
    ours = uplift_by_decile(y, u, t, bins=10)["observed_uplift"].to_numpy()
    theirs = uplift_by_percentile(y, u, t, strategy="overall", bins=10)["uplift"].to_numpy()
    np.testing.assert_allclose(ours, theirs, atol=1e-9)


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="same length"):
        qini_curve(np.array([0, 1]), np.array([0.1]), np.array([1, 0]))


def test_fold_metric_band_excludes_holdout_and_aggregates():
    y, u, t = _synthetic(1000)
    fold = np.tile([0, 1, 2, 3, 4], 200)
    fold[:50] = -1  # hold-out, must be ignored

    band = fold_metric_band(y, u, t, fold, lambda a, b, c: qini_coefficient(a, b, c).normalized)
    assert set(band.per_fold) == {0, 1, 2, 3, 4}
    assert band.mean == pytest.approx(np.mean(list(band.per_fold.values())))
    assert band.std >= 0.0
