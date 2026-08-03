"""Unit tests for the SMD randomization-check math."""

import numpy as np
import pandas as pd

from uplift.data.balance import covariate_balance, standardized_mean_difference


def test_smd_zero_when_arms_identical() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert standardized_mean_difference(x, x) == 0.0


def test_smd_positive_when_treated_mean_higher() -> None:
    treated = np.array([3.0, 4.0, 5.0])
    control = np.array([0.0, 1.0, 2.0])
    assert standardized_mean_difference(treated, control) > 0


def test_smd_constant_equal_arms_returns_zero() -> None:
    assert standardized_mean_difference(np.ones(3), np.ones(3)) == 0.0


def test_covariate_balance_flags_and_sorts() -> None:
    df = pd.DataFrame(
        {
            "treatment": [1, 1, 1, 0, 0, 0],
            "x": [10.0, 11.0, 12.0, 0.0, 1.0, 2.0],  # separates arms -> flagged
            "z": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],  # balanced
        }
    )
    out = covariate_balance(df, feature_cols=["x", "z"])

    assert bool(out.loc[out.covariate == "x", "imbalanced"].iloc[0]) is True
    assert out["abs_smd"].is_monotonic_decreasing  # sorted worst-first


def test_covariate_balance_expands_categorical_levels() -> None:
    df = pd.DataFrame(
        {
            "treatment": [1, 1, 0, 0],
            "c": pd.Categorical(["a", "b", "a", "b"]),
        }
    )
    out = covariate_balance(df, feature_cols=["c"], categorical_cols=["c"])

    assert set(out["covariate"]) == {"c=a", "c=b"}
