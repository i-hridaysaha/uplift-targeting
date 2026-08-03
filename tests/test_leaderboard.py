"""Unit tests for cross-fit out-of-fold scoring and the leaderboard assembly."""

import numpy as np
import pandas as pd

from uplift.data.splits import assign_splits
from uplift.models.leaderboard import crossfit_oof, leaderboard
from uplift.models.meta import SLearner

FEATURES = ["x0", "noise", "channel"]


def _split_frame(n: int = 1500, seed: int = 0) -> pd.DataFrame:
    """Synthetic uplift frame with holdout/fold columns assigned (see test_meta)."""
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
    return assign_splits(df, outcome_col="outcome")


def _meta_scorer(train, eval_frame, features, outcome, seed):
    model = SLearner(seed=seed).fit(
        train[features], train["treatment"].to_numpy(), train[outcome].to_numpy()
    )
    return model.predict_uplift(eval_frame[features])


def test_crossfit_scores_every_dev_row_once():
    frame = _split_frame()
    dev = frame[frame["fold"] != -1].reset_index(drop=True)
    uplift = crossfit_oof(dev, FEATURES, "outcome", _meta_scorer)
    assert uplift.shape == (len(dev),)
    assert np.all(np.isfinite(uplift))  # every fold got scored, no gaps


def test_leaderboard_has_a_row_per_model():
    frame = _split_frame()
    board = leaderboard(frame, FEATURES, "outcome")
    assert list(board["model"]) == [
        "treat_everyone",
        "response_model",
        "s_learner",
        "t_learner",
        "x_learner",
    ]
    for col in ("qini_mean", "qini_std", "auuc_mean", "auuc_std", "n_folds"):
        assert col in board.columns
    assert (board["n_folds"] == 5).all()


def test_leaderboard_treat_everyone_is_zero_and_meta_beats_it():
    board = leaderboard(_split_frame(), FEATURES, "outcome").set_index("model")
    assert abs(board.loc["treat_everyone", "qini_mean"]) < 1e-9
    best_meta = board.loc[["s_learner", "t_learner", "x_learner"], "qini_mean"].max()
    assert best_meta > board.loc["treat_everyone", "qini_mean"]
