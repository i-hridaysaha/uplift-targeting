"""Unit tests for uplift-driver explainability (TreeSHAP + permutation importance)."""

import numpy as np
import pandas as pd
import pytest

from uplift.eval.explain import permutation_uplift_importance, treeshap_uplift
from uplift.models.direct import ClassTransformation, UpliftForest
from uplift.models.meta import SLearner

FEATURES = ["x0", "noise", "channel"]


def _frame(n: int = 4000, seed: int = 0) -> pd.DataFrame:
    """Synthetic frame where only x0 modulates uplift; noise/channel are irrelevant."""
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
    effect = np.where(x0 > 0.5, 0.5, 0.0)
    prob = np.clip(0.15 + df["treatment"].to_numpy() * effect, 0.0, 1.0)
    df["outcome"] = (rng.random(n) < prob).astype("int8")
    return df


def _fit(model, df):
    return model.fit(df[FEATURES], df["treatment"].to_numpy(), df["outcome"].to_numpy())


def test_treeshap_classtransformation_ranks_the_true_driver_first():
    df = _frame()
    model = _fit(ClassTransformation(), df)
    table = treeshap_uplift(model, df[FEATURES])
    assert list(table.columns) == ["feature", "mean_abs_shap"]
    assert table.iloc[0]["feature"] == "x0"  # the real driver leads
    top = table.set_index("feature")["mean_abs_shap"]
    assert top["x0"] > top["noise"]


def test_treeshap_slearner_excludes_treatment_and_ranks_the_driver():
    df = _frame()
    model = _fit(SLearner(), df)
    table = treeshap_uplift(model, df[FEATURES])
    assert "treatment" not in set(table["feature"])  # its diff is the average effect
    top = table.set_index("feature")["mean_abs_shap"]
    assert top["x0"] > top["noise"]


def test_treeshap_rejects_a_model_without_a_booster():
    df = _frame(n=1500)
    forest = _fit(UpliftForest(n_estimators=5, min_samples_leaf=100, min_arm=25), df)
    with pytest.raises(TypeError, match="no LightGBM booster"):
        treeshap_uplift(forest, df[FEATURES])


def test_permutation_importance_ranks_the_true_driver_first():
    df = _frame()
    forest = _fit(UpliftForest(n_estimators=12, min_samples_leaf=100, min_arm=25), df)
    table = permutation_uplift_importance(
        forest, df[FEATURES], df["treatment"].to_numpy(), df["outcome"].to_numpy(), n_repeats=3
    )
    assert list(table.columns) == ["feature", "importance", "std"]
    assert "treatment" not in set(table["feature"])
    assert table.iloc[0]["feature"] == "x0"  # shuffling x0 hurts Qini most


def test_permutation_importance_is_reproducible():
    df = _frame(n=2000)
    forest = _fit(UpliftForest(n_estimators=8, min_samples_leaf=100, min_arm=25), df)
    args = (df[FEATURES], df["treatment"].to_numpy(), df["outcome"].to_numpy())
    a = permutation_uplift_importance(forest, *args, n_repeats=3, seed=1)
    b = permutation_uplift_importance(forest, *args, n_repeats=3, seed=1)
    pd.testing.assert_frame_equal(a, b)
