"""Final model selection on the untouched 20% hold-out.

The cross-validated leaderboard (``models.leaderboard``) ranks the field on
out-of-fold folds and gives each model a confidence band. Selection then confirms
the contenders once on the hold-out (``fold == -1``), which no model has seen
during fitting or scoring so far -- a single unbiased number per model, the
tie-break basis when the CV bands overlap (see MODELING.md).

Each model is fit on the whole dev set (all CV folds together) and scored once on
the hold-out. Baselines and every S/T/X and direct model plug in through the same
``Scorer`` used by the CV board, so the hold-out board is directly comparable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from uplift.data.schema import TREATMENT_COL
from uplift.data.splits import HOLDOUT_FOLD, SEED
from uplift.eval.metrics import auuc, qini_coefficient
from uplift.models.leaderboard import FOLD_COL, MODELS, Scorer


def holdout_uplift(
    frame: pd.DataFrame,
    features: list[str],
    outcome: str,
    scorer: Scorer,
    seed: int = SEED,
) -> np.ndarray:
    """Return the scorer's uplift on the hold-out after fitting on all dev rows.

    ``frame`` carries the ``fold`` column: dev rows (``fold != -1``) train the
    model, the hold-out rows (``fold == -1``) are predicted once. The returned
    array is aligned to the hold-out rows in ``frame`` order.
    """
    dev = frame[frame[FOLD_COL] != HOLDOUT_FOLD]
    holdout = frame[frame[FOLD_COL] == HOLDOUT_FOLD]
    if holdout.empty:
        raise ValueError("frame has no hold-out rows (fold == -1); cannot select.")
    return scorer(dev, holdout, features, outcome, seed)


def holdout_board(
    frame: pd.DataFrame,
    features: list[str],
    outcome: str,
    models: dict[str, Scorer] | None = None,
    seed: int = SEED,
) -> pd.DataFrame:
    """Score every model once on the hold-out and return its Qini/AUUC (headline norm).

    One row per model with the normalized Qini and AUUC on the hold-out -- a single
    unbiased number each (no band; the band is the CV board's job). Row order
    follows ``models`` insertion order.
    """
    models = MODELS if models is None else models
    holdout = frame[frame[FOLD_COL] == HOLDOUT_FOLD]
    if holdout.empty:
        raise ValueError("frame has no hold-out rows (fold == -1); cannot select.")
    y = holdout[outcome].to_numpy()
    t = holdout[TREATMENT_COL].to_numpy()

    rows: list[dict[str, object]] = []
    for name, scorer in models.items():
        uplift = holdout_uplift(frame, features, outcome, scorer, seed=seed)
        rows.append(
            {
                "model": name,
                "qini": qini_coefficient(y, uplift, t).normalized,
                "auuc": auuc(y, uplift, t).normalized,
                "holdout_n": int(len(holdout)),
            }
        )
    return pd.DataFrame(rows)
