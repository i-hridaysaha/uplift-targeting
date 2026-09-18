"""Base-learner swap: is the model ordering a fact about the data, or about LightGBM?

Every LightGBM-backed model on the board (the response baseline and the S, T and X
meta-learners) shares one parameter set, so the board compares learner designs
under one learner. This script re-runs those four designs over a deliberately
simple base, a linear model over one-hot categoricals and scaled numerics
(``uplift.base_learners``), on the same five folds at seed 42, and prints the two
orderings side by side. The LightGBM pass is re-run too, as a machinery check
against ``reports/phase8_leaderboard.csv``. The direct models (class
transformation, tree, forest) have no swappable base and are not on this board.

Writes ``reports/phase8_base_swap.csv``: one row per dataset, base and model, with
the band, the per-fold values and the gap to the response bar under that base.

Run: ``uv run python scripts/phase8_base_swap.py``
     ``uv run python scripts/phase8_base_swap.py --datasets hillstrom``  (fast case)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from uplift.base_learners import BASE_LEARNERS
from uplift.data.schema import CRITEO_FEATURES, HILLSTROM_FEATURES, TREATMENT_COL
from uplift.data.splits import SEED
from uplift.eval.baselines import response_model_scores
from uplift.models import SLearner, TLearner, XLearner, leaderboard
from uplift.models.leaderboard import Scorer

PROCESSED = Path("data/processed")
REPORTS = Path("reports")
PRIMARY_OUTCOME = "visit"
FEATURES: dict[str, list[str]] = {"hillstrom": HILLSTROM_FEATURES, "criteo": CRITEO_FEATURES}
META = {"s_learner": SLearner, "t_learner": TLearner, "x_learner": XLearner}


def field(base: str) -> dict[str, Scorer]:
    """The response bar and the three meta-learners, all over the named base."""

    def response(train, eval_frame, features, outcome, seed):
        return response_model_scores(
            train[features], train[outcome].to_numpy(), eval_frame[features], seed=seed, base=base
        )

    def meta(cls):
        def scorer(train, eval_frame, features, outcome, seed):
            model = cls(seed=seed, base=base)
            model.fit(train[features], train[TREATMENT_COL].to_numpy(), train[outcome].to_numpy())
            return model.predict_uplift(eval_frame[features])

        return scorer

    return {"response_model": response, **{name: meta(cls) for name, cls in META.items()}}


def run(name: str, bases: tuple[str, ...], seed: int) -> pd.DataFrame:
    """One board per base on the same folds; the gap to that base's own bar."""
    frame = pd.read_parquet(PROCESSED / f"{name}.parquet")
    parts = []
    for base in bases:
        start = time.perf_counter()
        board = leaderboard(frame, FEATURES[name], PRIMARY_OUTCOME, models=field(base), seed=seed)
        bar = float(board.set_index("model").loc["response_model", "qini_mean"])
        board.insert(0, "base", base)
        board.insert(0, "dataset", name)
        board["gap_to_bar"] = board["qini_mean"] - bar
        parts.append(board)
        print(f"\n  {name} over {base} ({time.perf_counter() - start:.0f}s)")
        for row in board.itertuples():
            print(
                f"    {row.model:<15} {row.qini_mean:+.4f} +/- {row.qini_std:.4f}   "
                f"gap to bar {row.gap_to_bar:+.4f}"
            )
    return pd.concat(parts, ignore_index=True)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: the four designs under each base learner, per dataset."""
    parser = argparse.ArgumentParser(
        description="Re-run the LightGBM-backed designs over a linear base."
    )
    parser.add_argument("--datasets", default="hillstrom,criteo")
    parser.add_argument("--bases", default=",".join(BASE_LEARNERS))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)

    names = [n.strip() for n in args.datasets.split(",") if n.strip()]
    bases = tuple(b.strip() for b in args.bases.split(",") if b.strip())
    parts = [run(name, bases, args.seed) for name in names]
    out = REPORTS / "phase8_base_swap.csv"
    REPORTS.mkdir(parents=True, exist_ok=True)
    pd.concat(parts, ignore_index=True).to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
