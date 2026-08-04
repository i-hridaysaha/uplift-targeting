"""Unit tests for final selection on the untouched hold-out."""

import numpy as np
import pandas as pd
import pytest

from uplift.data.splits import assign_splits
from uplift.models.leaderboard import MODELS, model_scorer
from uplift.models.meta import SLearner
from uplift.models.selection import holdout_board, holdout_uplift

FEATURES = ["x0", "noise", "channel"]


def _split_frame(n: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Synthetic uplift frame (effect only where x0 > 0.5) with holdout/fold columns."""
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


def test_holdout_uplift_is_aligned_to_the_holdout_rows():
    frame = _split_frame()
    n_hold = int((frame["fold"] == -1).sum())
    uplift = holdout_uplift(frame, FEATURES, "outcome", model_scorer(SLearner))
    assert uplift.shape == (n_hold,)
    assert np.all(np.isfinite(uplift))


def test_holdout_uplift_treat_everyone_is_constant():
    frame = _split_frame()
    uplift = holdout_uplift(frame, FEATURES, "outcome", MODELS["treat_everyone"])
    assert np.allclose(uplift, uplift[0])  # a constant score -> random ranking


def test_holdout_uplift_raises_without_a_holdout():
    frame = _split_frame()
    dev_only = frame[frame["fold"] != -1]
    with pytest.raises(ValueError, match="no hold-out"):
        holdout_uplift(dev_only, FEATURES, "outcome", model_scorer(SLearner))


def test_holdout_board_has_a_row_per_model_and_a_single_number_each():
    frame = _split_frame()
    board = holdout_board(frame, FEATURES, "outcome")
    assert list(board["model"]) == list(MODELS)
    for col in ("model", "qini", "auuc", "holdout_n"):
        assert col in board.columns
    assert (board["holdout_n"] == int((frame["fold"] == -1).sum())).all()


def test_holdout_board_real_model_beats_treat_everyone():
    board = holdout_board(_split_frame(), FEATURES, "outcome").set_index("model")
    assert abs(board.loc["treat_everyone", "qini"]) < 1e-9  # exact diagonal
    assert board.loc["s_learner", "qini"] > board.loc["treat_everyone", "qini"]
