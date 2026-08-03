"""Phase 6 meta-learner bakeoff: fit S/T/X (and the baselines), score, log a board.

Read-only over the tidy processed datasets in ``data/processed/``. For each
dataset it cross-fits every model out-of-fold on the primary ``visit`` outcome,
scores the folds through the evaluation harness, and writes the leaderboard
(cross-validated Qini/AUUC mean +/- std) to ``reports/phase6_leaderboard.csv``.
The hold-out is untouched (reserved for Phase 8). Nothing here trains a shipped
model or edits processed data.

Run: ``uv run python scripts/phase6_meta.py``
     ``uv run python scripts/phase6_meta.py --datasets hillstrom``  (fast case only)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from uplift.data.schema import CRITEO_FEATURES, HILLSTROM_FEATURES
from uplift.data.splits import SEED
from uplift.models.leaderboard import leaderboard

PROCESSED = Path("data/processed")
REPORTS = Path("reports")
PRIMARY_OUTCOME = "visit"

FEATURES: dict[str, list[str]] = {
    "hillstrom": HILLSTROM_FEATURES,
    "criteo": CRITEO_FEATURES,
}


def run_dataset(name: str, seed: int = SEED) -> pd.DataFrame:
    """Load one processed dataset, run the leaderboard, and tag rows with its name."""
    frame = pd.read_parquet(PROCESSED / f"{name}.parquet")
    board = leaderboard(frame, FEATURES[name], PRIMARY_OUTCOME, seed=seed)
    board.insert(0, "dataset", name)
    return board


def _format(board: pd.DataFrame) -> str:
    """Render the leaderboard as an aligned mean +/- std text table."""
    lines = [f"{'dataset':<10} {'model':<15} {'qini':>18} {'auuc':>18}"]
    for row in board.itertuples():
        qini = f"{row.qini_mean:+.4f} +/- {row.qini_std:.4f}"
        auuc = f"{row.auuc_mean:+.4f} +/- {row.auuc_std:.4f}"
        lines.append(f"{row.dataset:<10} {row.model:<15} {qini:>18} {auuc:>18}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: run the bakeoff on the requested datasets and persist the board."""
    parser = argparse.ArgumentParser(description="Phase 6 meta-learner leaderboard.")
    parser.add_argument("--datasets", default="hillstrom,criteo")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", default=str(REPORTS / "phase6_leaderboard.csv"))
    args = parser.parse_args(argv)

    names = [n.strip() for n in args.datasets.split(",") if n.strip()]
    boards = [run_dataset(name, seed=args.seed) for name in names]
    board = pd.concat(boards, ignore_index=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    board.to_csv(out, index=False)

    print(_format(board))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
