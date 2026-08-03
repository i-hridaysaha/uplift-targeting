"""Unit tests for the incremental-value functions."""

import numpy as np
import pytest

from uplift.eval.value import (
    COST_PER_CONTACT,
    VALUE_PER_CONVERSION,
    incremental_conversions_at_k,
    incremental_value_at_k,
    incremental_value_band,
    k_for_budget,
)


def _synthetic(n: int = 1000, seed: int = 1):
    """Conversion outcome whose treatment effect grows with a distinct score."""
    rng = np.random.default_rng(seed)
    treatment = rng.integers(0, 2, n)
    score = rng.normal(size=n)
    effect = 0.2 * (score - score.min()) / (score.max() - score.min())
    prob = 0.05 + treatment * effect
    conversion = (rng.random(n) < prob).astype(int)
    return conversion, score, treatment


def test_k_for_budget_floors():
    assert k_for_budget(100.0, 0.10) == 1000
    assert k_for_budget(9.99, 0.10) == 99


def test_k_for_budget_rejects_nonpositive_cost():
    with pytest.raises(ValueError, match="positive"):
        k_for_budget(100.0, 0.0)


def test_incremental_conversions_at_full_budget_equals_total_uplift():
    conversion, score, treatment = _synthetic()
    n = len(conversion)
    # At k = n the size-corrected lift is the whole-population number.
    y_t = conversion[treatment == 1].sum()
    y_c = conversion[treatment == 0].sum()
    n_t = (treatment == 1).sum()
    n_c = (treatment == 0).sum()
    expected = y_t - y_c * n_t / n_c
    assert incremental_conversions_at_k(conversion, score, treatment, n) == pytest.approx(expected)


def test_incremental_value_at_k_arithmetic():
    conversion, score, treatment = _synthetic()
    k = 300
    inc = incremental_conversions_at_k(conversion, score, treatment, k)
    got = incremental_value_at_k(
        conversion, score, treatment, k, value_per_conversion=100.0, cost_per_contact=0.2
    )
    assert got == pytest.approx(inc * 100.0 - k * 0.2)


def test_incremental_value_at_k_accepts_arrays():
    conversion, score, treatment = _synthetic()
    ks = np.array([100, 300, 600])
    out = incremental_value_at_k(conversion, score, treatment, ks)
    assert out.shape == ks.shape


def test_value_band_orders_and_point_uses_anchors():
    conversion, score, treatment = _synthetic()
    k = 300
    band = incremental_value_band(conversion, score, treatment, k)
    assert band.low <= band.point <= band.high
    assert band.k == k
    inc = incremental_conversions_at_k(conversion, score, treatment, k)
    assert band.incremental_conversions == pytest.approx(inc)
    assert band.point == pytest.approx(inc * VALUE_PER_CONVERSION - k * COST_PER_CONTACT)
