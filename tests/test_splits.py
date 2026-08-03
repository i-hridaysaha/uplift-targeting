"""Unit tests for the arm-balanced split and subsample helpers."""

import numpy as np
import pandas as pd

from uplift.data.splits import assign_splits, subsample_stratified


def _synthetic(n: int = 1000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "treatment": rng.integers(0, 2, n),
            "visit": rng.integers(0, 2, n),
            "conversion": rng.integers(0, 2, n),
            "x": rng.normal(size=n),
        }
    )


def test_assign_splits_holdout_fraction_and_folds() -> None:
    out = assign_splits(_synthetic(), outcome_col="visit", n_folds=5, holdout_frac=0.2, seed=42)

    assert abs(out["holdout"].mean() - 0.2) < 0.02
    dev = out.loc[~out["holdout"]]
    assert set(dev["fold"].unique()) == {0, 1, 2, 3, 4}
    assert (out.loc[out["holdout"], "fold"] == -1).all()
    assert dev["fold"].value_counts().min() > 0


def test_assign_splits_preserves_arm_balance_across_folds() -> None:
    out = assign_splits(_synthetic(2000), outcome_col="visit", seed=42)
    dev = out.loc[~out["holdout"]]
    overall = dev["treatment"].mean()
    for f in range(5):
        assert abs(dev.loc[dev["fold"] == f, "treatment"].mean() - overall) < 0.06


def test_assign_splits_reproducible() -> None:
    df = _synthetic()
    a = assign_splits(df, outcome_col="visit", seed=42)
    b = assign_splits(df, outcome_col="visit", seed=42)
    pd.testing.assert_frame_equal(a, b)


def test_subsample_stratified_size_and_balance() -> None:
    df = _synthetic(5000)
    sub = subsample_stratified(df, strat_cols=["treatment", "conversion"], n=1000, seed=42)

    assert len(sub) == 1000
    assert abs(sub["treatment"].mean() - df["treatment"].mean()) < 0.03


def test_subsample_returns_all_when_n_exceeds_size() -> None:
    df = _synthetic(100)
    sub = subsample_stratified(df, strat_cols=["treatment", "conversion"], n=1000)
    assert len(sub) == 100
