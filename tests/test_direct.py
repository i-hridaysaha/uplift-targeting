"""Unit tests for the direct uplift models on synthetic data with known uplift.

The tree/forest tests are the gate: the hand-rolled models earn a place on the
leaderboard only by recovering an effect that is *set* in the data -- ranking the
true high-uplift region first and reproducing roughly the right magnitude. The
class-transformation tests also prove the propensity correction on unequal arms,
where the plain 50/50 transform would be biased.
"""

import numpy as np
import pandas as pd
import pytest

from uplift.eval.metrics import qini_coefficient
from uplift.models.direct import ClassTransformation, UpliftForest, UpliftTree

FEATURES = ["x0", "noise", "channel"]
TREE_MODELS = [UpliftTree, UpliftForest]


def _uplift_frame(n: int = 6000, seed: int = 0, treated_frac: float = 0.5) -> pd.DataFrame:
    """Synthetic data where only high-``x0`` customers respond to treatment.

    Base response is constant; treatment adds ``+0.4`` for ``x0 > 0.5`` and none
    below, so the true per-row uplift is a step in ``x0``. ``treated_frac`` sets the
    arm balance, used to check the propensity correction on unequal arms.
    """
    rng = np.random.default_rng(seed)
    x0 = rng.random(n)
    treatment = (rng.random(n) < treated_frac).astype("int8")
    df = pd.DataFrame(
        {
            "x0": x0.astype("float32"),
            "noise": rng.normal(size=n).astype("float32"),
            "channel": pd.Categorical(rng.choice(["web", "phone"], n)),
            "treatment": treatment,
        }
    )
    effect = np.where(x0 > 0.5, 0.4, 0.0)
    prob = np.clip(0.15 + treatment * effect, 0.0, 1.0)
    df["outcome"] = (rng.random(n) < prob).astype("int8")
    return df


def _constant_uplift_frame(
    n: int = 20000, seed: int = 0, tau: float = 0.2, treated_frac: float = 0.8
) -> pd.DataFrame:
    """Constant uplift ``tau`` on strongly unequal arms (base rate 0.1).

    The true ATE is ``tau`` regardless of ``treated_frac``; a transform that ignores
    the propensity would recover the wrong number on this ~80/20 split.
    """
    rng = np.random.default_rng(seed)
    treatment = (rng.random(n) < treated_frac).astype("int8")
    df = pd.DataFrame(
        {
            "x0": rng.random(n).astype("float32"),
            "noise": rng.normal(size=n).astype("float32"),
            "channel": pd.Categorical(rng.choice(["web", "phone"], n)),
            "treatment": treatment,
        }
    )
    prob = np.clip(0.1 + treatment * tau, 0.0, 1.0)
    df["outcome"] = (rng.random(n) < prob).astype("int8")
    return df


def _fit(model, df: pd.DataFrame):
    """Fit a direct model on the feature columns, treatment, and outcome."""
    return model.fit(df[FEATURES], df["treatment"].to_numpy(), df["outcome"].to_numpy())


def _qini(df: pd.DataFrame, uplift: np.ndarray) -> float:
    """Normalized Qini of an uplift ranking on a frame's outcome/treatment."""
    return qini_coefficient(df["outcome"].to_numpy(), uplift, df["treatment"].to_numpy()).normalized


# --- Tree / forest gate: recover the effect that was set -------------------


@pytest.mark.parametrize("cls", TREE_MODELS)
def test_tree_ranks_true_uplift_region_first(cls):
    df = _uplift_frame()
    model = cls(seed=42) if cls is UpliftTree else cls(n_estimators=20, seed=42)
    uplift = _fit(model, df).predict_uplift(df[FEATURES])
    assert _qini(df, uplift) > 0.10  # treat-everyone (constant score) scores 0


# Raw tree leaves carry the full estimate; a bagged forest shrinks magnitude
# toward the mean (variance for magnitude -- the usual trade), so the recovered
# gap is smaller there. Ranking is gated separately by the Qini test above.
_MIN_GAP = {UpliftTree: 0.20, UpliftForest: 0.15}


@pytest.mark.parametrize("cls", TREE_MODELS)
def test_tree_recovers_effect_magnitude(cls):
    df = _uplift_frame()
    model = cls(seed=42) if cls is UpliftTree else cls(n_estimators=20, seed=42)
    uplift = _fit(model, df).predict_uplift(df[FEATURES])
    x0 = df["x0"].to_numpy()
    gap = uplift[x0 > 0.5].mean() - uplift[x0 <= 0.5].mean()
    assert gap > _MIN_GAP[cls]  # the set gap is 0.4


@pytest.mark.parametrize("cls", TREE_MODELS)
def test_tree_reproducible(cls):
    df = _uplift_frame()
    a = _fit(cls(seed=42) if cls is UpliftTree else cls(n_estimators=10, seed=42), df)
    b = _fit(cls(seed=42) if cls is UpliftTree else cls(n_estimators=10, seed=42), df)
    np.testing.assert_allclose(a.predict_uplift(df[FEATURES]), b.predict_uplift(df[FEATURES]))


def test_tree_leaf_value_is_treated_minus_control():
    # One clean split on x0: each leaf's value must equal its observed p_t - p_c.
    df = _uplift_frame(n=8000, seed=1)
    tree = _fit(UpliftTree(seed=42, max_depth=1), df)
    uplift = tree.predict_uplift(df[FEATURES])
    for val in np.unique(uplift):
        mask = uplift == val
        sub = df[mask]
        t = sub["treatment"].to_numpy().astype(bool)
        obs = sub["outcome"].to_numpy()[t].mean() - sub["outcome"].to_numpy()[~t].mean()
        assert val == pytest.approx(obs, abs=1e-9)


def test_forest_averages_its_trees():
    df = _uplift_frame(n=3000)
    forest = _fit(UpliftForest(n_estimators=8, seed=42), df)
    per_tree = np.mean([t.predict_uplift(df[FEATURES]) for t in forest.trees_], axis=0)
    np.testing.assert_allclose(forest.predict_uplift(df[FEATURES]), per_tree)
    assert len(forest.trees_) == 8


def test_tree_predict_shape_and_finite():
    df = _uplift_frame()
    uplift = _fit(UpliftTree(seed=42), df).predict_uplift(df[FEATURES])
    assert uplift.shape == (len(df),)
    assert np.all(np.isfinite(uplift))


# --- Class transformation: ranking + the propensity correction -------------


def test_class_transformation_beats_random_ranking():
    df = _uplift_frame(treated_frac=0.5)
    uplift = _fit(ClassTransformation(seed=42), df).predict_uplift(df[FEATURES])
    assert _qini(df, uplift) > 0.05


def test_class_transformation_detects_where_effect_lives_on_unequal_arms():
    df = _uplift_frame(treated_frac=0.85)  # Criteo-like imbalance
    uplift = _fit(ClassTransformation(seed=42), df).predict_uplift(df[FEATURES])
    x0 = df["x0"].to_numpy()
    assert uplift[x0 > 0.5].mean() > uplift[x0 <= 0.5].mean()


def test_class_transformation_propensity_is_treated_fraction():
    df = _uplift_frame(treated_frac=0.7)
    model = _fit(ClassTransformation(seed=42), df)
    assert model.propensity_ == pytest.approx(df["treatment"].mean())


def test_class_transformation_correction_beats_uncorrected_on_unequal_arms():
    # On an 80/20 split the propensity-corrected mean uplift lands near the true
    # ATE; forcing p=0.5 (the plain transform) is visibly biased away from it.
    df = _constant_uplift_frame(tau=0.2, treated_frac=0.8)
    tau = 0.2
    corrected = _fit(ClassTransformation(seed=42), df).predict_uplift(df[FEATURES]).mean()
    forced = (
        ClassTransformation(seed=42, propensity=0.5)
        .fit(df[FEATURES], df["treatment"].to_numpy(), df["outcome"].to_numpy())
        .predict_uplift(df[FEATURES])
        .mean()
    )
    assert abs(corrected - tau) < abs(forced - tau)
    assert abs(corrected - tau) < 0.08


def test_class_transformation_reproducible():
    df = _uplift_frame()
    a = _fit(ClassTransformation(seed=42), df).predict_uplift(df[FEATURES])
    b = _fit(ClassTransformation(seed=42), df).predict_uplift(df[FEATURES])
    np.testing.assert_allclose(a, b)


@pytest.mark.parametrize("cls", [ClassTransformation, UpliftTree, UpliftForest])
def test_drops_stray_treatment_column(cls):
    # A leaked treatment column in X must not change predictions.
    df = _uplift_frame(n=3000)
    model = cls(seed=42) if cls is not UpliftForest else cls(n_estimators=8, seed=42)
    _fit(model, df)
    clean = model.predict_uplift(df[FEATURES])
    leaked = model.predict_uplift(df[[*FEATURES, "treatment"]])
    np.testing.assert_allclose(clean, leaked)
