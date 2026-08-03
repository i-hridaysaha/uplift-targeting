"""Arm-balanced data splits: a stratified hold-out, stratified k-fold labels, and
a stratified subsample. Every split is seeded and reproducible.

Stratification is on ``treatment x outcome`` so each split keeps the arm ratio
and the base rate, which is what makes the fold-to-fold Qini/AUUC spread an
honest confidence band (see DATA.md).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    StratifiedKFold,
    StratifiedShuffleSplit,
    train_test_split,
)

HOLDOUT_FRAC = 0.2
N_FOLDS = 5
SEED = 42
HOLDOUT_FOLD = -1


def _strat_key(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """Build a single string stratification label from several columns."""
    key = df[cols[0]].astype(str)
    for col in cols[1:]:
        key = key.str.cat(df[col].astype(str), sep="_")
    return key.to_numpy()


def assign_splits(
    df: pd.DataFrame,
    outcome_col: str,
    treatment_col: str = "treatment",
    n_folds: int = N_FOLDS,
    holdout_frac: float = HOLDOUT_FRAC,
    seed: int = SEED,
) -> pd.DataFrame:
    """Return a copy of ``df`` with ``holdout`` (bool) and ``fold`` (int8) columns.

    A ``holdout_frac`` stratified hold-out is carved first and marked
    ``fold = -1``; the remaining rows receive ``n_folds`` stratified CV labels
    ``0..n_folds-1``. Stratification is on ``treatment x outcome`` so every split
    preserves arm balance and base rate. The hold-out is never part of a fold.
    """
    key = _strat_key(df, [treatment_col, outcome_col])
    positions = np.arange(len(df))

    sss = StratifiedShuffleSplit(n_splits=1, test_size=holdout_frac, random_state=seed)
    dev_pos, holdout_pos = next(sss.split(positions, key))

    fold = np.full(len(df), HOLDOUT_FOLD, dtype="int8")
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for f, (_, val_local) in enumerate(skf.split(dev_pos, key[dev_pos])):
        fold[dev_pos[val_local]] = f

    out = df.copy()
    out["holdout"] = False
    out.iloc[holdout_pos, out.columns.get_loc("holdout")] = True
    out["fold"] = fold
    return out


def subsample_stratified(
    df: pd.DataFrame,
    strat_cols: list[str],
    n: int,
    seed: int = SEED,
) -> pd.DataFrame:
    """Return an ``n``-row stratified subsample preserving ``strat_cols`` shares.

    Used to draw the Criteo working set while keeping the treated/control ratio
    and the rare positives (stratify on ``treatment x conversion``). If ``n`` is
    at least the frame size, the whole frame is returned.
    """
    if n >= len(df):
        return df.copy()
    key = _strat_key(df, strat_cols)
    sub, _ = train_test_split(df, train_size=n, stratify=key, random_state=seed)
    return sub.reset_index(drop=True)
