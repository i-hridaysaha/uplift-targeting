"""Phase 7 direct-model bakeoff: class transformation + uplift tree/forest.

Read-only over the tidy processed datasets in ``data/processed/``. For each
dataset it cross-fits every model out-of-fold on the primary ``visit`` outcome,
scores the folds through the evaluation harness, and writes the leaderboard
(cross-validated Qini/AUUC mean +/- std) to ``reports/phase7_leaderboard.csv``.
The two naive baselines are re-scored here too, so this board stands alone and is
directly comparable to the Phase 6 meta-learner board. The hold-out is untouched
(reserved for Phase 8).

The uplift tree/forest are hand-rolled (no causalml). On the large Criteo dataset
each forest tree is bagged over a capped sample so the pure-numpy fit stays inside
the phase's time budget; this is recorded in MODELING.md.

Run: ``uv run python scripts/phase7_direct.py``
     ``uv run python scripts/phase7_direct.py --datasets hillstrom``  (fast case only)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from uplift.data.schema import CRITEO_FEATURES, HILLSTROM_FEATURES
from uplift.data.splits import SEED
from uplift.models.direct import ClassTransformation, UpliftForest, UpliftTree
from uplift.models.leaderboard import MODELS, Scorer, leaderboard, model_scorer

PROCESSED = Path("data/processed")
REPORTS = Path("reports")
PRIMARY_OUTCOME = "visit"

FEATURES: dict[str, list[str]] = {
    "hillstrom": HILLSTROM_FEATURES,
    "criteo": CRITEO_FEATURES,
}

# Per-tree row cap for the forest. None on small Hillstrom (full bootstrap); capped
# on Criteo so 50 pure-numpy trees per fold stay fast without changing the method.
FOREST_MAX_SAMPLES: dict[str, int | None] = {
    "hillstrom": None,
    "criteo": 150_000,
}


def models_for(dataset: str) -> dict[str, Scorer]:
    """Baselines (for context) plus the three direct models, tuned per dataset."""
    return {
        "treat_everyone": MODELS["treat_everyone"],
        "response_model": MODELS["response_model"],
        "class_transformation": model_scorer(ClassTransformation),
        "uplift_tree": model_scorer(UpliftTree),
        "uplift_forest": model_scorer(UpliftForest, max_samples=FOREST_MAX_SAMPLES[dataset]),
    }


def run_dataset(name: str, seed: int = SEED) -> pd.DataFrame:
    """Load one processed dataset, run the direct-model board, tag rows with its name."""
    frame = pd.read_parquet(PROCESSED / f"{name}.parquet")
    board = leaderboard(frame, FEATURES[name], PRIMARY_OUTCOME, models=models_for(name), seed=seed)
    board.insert(0, "dataset", name)
    return board


def _format(board: pd.DataFrame) -> str:
    """Render the leaderboard as an aligned mean +/- std text table."""
    lines = [f"{'dataset':<10} {'model':<21} {'qini':>18} {'auuc':>18}"]
    for row in board.itertuples():
        qini = f"{row.qini_mean:+.4f} +/- {row.qini_std:.4f}"
        auuc = f"{row.auuc_mean:+.4f} +/- {row.auuc_std:.4f}"
        lines.append(f"{row.dataset:<10} {row.model:<21} {qini:>18} {auuc:>18}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: run the direct-model bakeoff and persist the board."""
    parser = argparse.ArgumentParser(description="Phase 7 direct-model leaderboard.")
    parser.add_argument("--datasets", default="hillstrom,criteo")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", default=str(REPORTS / "phase7_leaderboard.csv"))
    args = parser.parse_args(argv)

    names = [n.strip() for n in args.datasets.split(",") if n.strip()]
    boards = []
    for name in names:
        start = time.perf_counter()
        boards.append(run_dataset(name, seed=args.seed))
        print(f"{name}: {time.perf_counter() - start:.1f}s")
    board = pd.concat(boards, ignore_index=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    board.to_csv(out, index=False)

    print(_format(board))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
