"""The full-Criteo run: the three models the verdict turns on, on all 13,979,592 rows.

The headline board uses a one-million-row stratified subsample (D14). This script
scores the response baseline, the S-learner and the uplift forest on the whole
file, with the same protocol: five stratified folds and a 20 percent hold-out at
seed 42, out-of-fold Qini with its band and per-fold values, the hold-out scored
once, and the selection rule re-applied. Fourteen times the rows should shrink
the bands by roughly the square root of fourteen if fold-to-fold noise is
sampling noise; whether it separates the models is what this run measures.

Prerequisite (about five minutes, a few gigabytes of memory):

    uv run python -m uplift.data.ingest --datasets criteo \\
        --criteo-subsample 20000000 --out data/processed_full

Then:

    uv run python scripts/phase8_full_criteo.py

Writes ``reports/phase8_full_criteo.csv`` (the CV board) and
``reports/phase8_full_criteo_holdout.csv``; prints wall time per stage.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from uplift.data.schema import CRITEO_FEATURES
from uplift.data.splits import HOLDOUT_FOLD, SEED
from uplift.models import SLearner, UpliftForest, holdout_board, leaderboard, model_scorer
from uplift.models.leaderboard import FOLD_COL, MODELS, Scorer

REPORTS = Path("reports")
PRIMARY_OUTCOME = "visit"
# Same per-tree cap as the subsample boards (D25): the forest is the same forest,
# bootstrapping 150,000 rows per tree from a far larger pool.
FOREST_MAX_SAMPLES = 150_000


def models() -> dict[str, Scorer]:
    """The bar, the cross-validated leader and the shipped model."""
    return {
        "response_model": MODELS["response_model"],
        "s_learner": model_scorer(SLearner),
        "uplift_forest": model_scorer(UpliftForest, max_samples=FOREST_MAX_SAMPLES),
    }


def verdict(cv: pd.DataFrame, hold: pd.DataFrame) -> str:
    """The MODELING.md rule, re-applied: clear the bar on the mean, tie on the band."""
    table = cv.set_index("model")
    bar = float(table.loc["response_model", "qini_mean"])
    cleared = [m for m in ("s_learner", "uplift_forest") if table.loc[m, "qini_mean"] > bar]
    if not cleared:
        return f"no uplift model clears the bar ({bar:+.4f}); response_model by default"
    top = max(cleared, key=lambda m: table.loc[m, "qini_mean"])
    tied = [
        m
        for m in cleared
        if table.loc[top, "qini_mean"] - table.loc[m, "qini_mean"]
        < max(table.loc[top, "qini_std"], table.loc[m, "qini_std"])
    ]
    winner = min(tied, key=lambda m: table.loc[m, "qini_std"])
    h = hold.set_index("model")["qini"]
    return (
        f"cleared the bar ({bar:+.4f}): {cleared}; tie set {tied}; winner on the tighter band "
        f"{winner} ({table.loc[winner, 'qini_mean']:+.4f} +/- {table.loc[winner, 'qini_std']:.4f}, "
        f"hold-out {h[winner]:+.4f}; response hold-out {h['response_model']:+.4f})"
    )


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: the full-data board for the three models."""
    parser = argparse.ArgumentParser(description="Full-Criteo board for the key models.")
    parser.add_argument("--frame", default="data/processed_full/criteo.parquet")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)

    start = time.perf_counter()
    frame = pd.read_parquet(args.frame)
    n_dev = int((frame[FOLD_COL] != HOLDOUT_FOLD).sum())
    n_hold = int((frame[FOLD_COL] == HOLDOUT_FOLD).sum())
    print(
        f"loaded {len(frame):,} rows ({n_dev:,} dev, {n_hold:,} hold-out) in "
        f"{time.perf_counter() - start:.0f}s"
    )

    field = models()
    t0 = time.perf_counter()
    cv = leaderboard(frame, CRITEO_FEATURES, PRIMARY_OUTCOME, models=field, seed=args.seed)
    cv.insert(0, "dataset", "criteo_full")
    cv["n_dev"] = n_dev
    print(f"cross-validated board in {time.perf_counter() - t0:.0f}s")
    for row in cv.itertuples():
        print(f"  {row.model:<15} {row.qini_mean:+.4f} +/- {row.qini_std:.4f}")

    t0 = time.perf_counter()
    hold = holdout_board(frame, CRITEO_FEATURES, PRIMARY_OUTCOME, models=field, seed=args.seed)
    hold.insert(0, "dataset", "criteo_full")
    print(f"hold-out board in {time.perf_counter() - t0:.0f}s")
    for row in hold.itertuples():
        print(f"  {row.model:<15} {row.qini:+.4f}")

    print("rule re-applied:", verdict(cv, hold))
    REPORTS.mkdir(parents=True, exist_ok=True)
    cv.to_csv(REPORTS / "phase8_full_criteo.csv", index=False)
    hold.to_csv(REPORTS / "phase8_full_criteo_holdout.csv", index=False)
    print(
        f"wrote {REPORTS}/phase8_full_criteo.csv, {REPORTS}/phase8_full_criteo_holdout.csv "
        f"({time.perf_counter() - start:.0f}s total)"
    )


if __name__ == "__main__":
    main()
