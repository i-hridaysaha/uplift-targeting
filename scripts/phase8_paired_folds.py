"""Paired fold-by-fold comparison of two rankings from the persisted per-fold Qini.

The leaderboard keeps each model's five out-of-fold Qini values, so any two models
can be compared on the same five draws instead of through two summary bands. For
each pair this script reports the per-fold differences, their mean and spread, how
many folds the first model wins, and a paired t statistic with its two-sided p
value on ``n_folds - 1`` degrees of freedom.

Five paired draws is a small sample, so the p value is a coarse instrument; it is
printed because the one-std tie rule in ``pick_winner`` is a heuristic, and this is
what the same data says when it is read as a test.

Reads ``reports/phase8_leaderboard.csv``; writes ``reports/phase8_paired_folds.csv``.
Any board with ``qini_fold_<k>`` columns works, e.g. the full-Criteo one.

Run: ``uv run python scripts/phase8_paired_folds.py``
     ``uv run python scripts/phase8_paired_folds.py --leaderboard reports/phase8_full_criteo.csv \
         --out reports/phase8_full_criteo_paired.csv``
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPORTS = Path("reports")

# The comparisons the article makes: each contender against the bar, and the two
# contenders against each other.
PAIRS: list[tuple[str, str]] = [
    ("uplift_forest", "response_model"),
    ("s_learner", "response_model"),
    ("s_learner", "uplift_forest"),
]


def fold_columns(board: pd.DataFrame) -> list[str]:
    """The ``qini_fold_<k>`` columns, in fold order."""
    cols = [c for c in board.columns if c.startswith("qini_fold_")]
    if not cols:
        raise ValueError("leaderboard carries no per-fold columns; re-run scripts/phase8_eval.py")
    return sorted(cols, key=lambda c: int(c.rsplit("_", 1)[1]))


def paired(board: pd.DataFrame, a: str, b: str) -> dict[str, object]:
    """Compare model ``a`` against model ``b`` on their shared folds."""
    cols = fold_columns(board)
    table = board.set_index("model")
    diff = table.loc[a, cols].to_numpy(dtype="float64") - table.loc[b, cols].to_numpy(
        dtype="float64"
    )
    n = len(diff)
    std = float(diff.std(ddof=1)) if n > 1 else float("nan")
    t_stat = float(diff.mean() / (std / np.sqrt(n))) if std > 0 else float("nan")
    p = float(2 * stats.t.sf(abs(t_stat), df=n - 1)) if np.isfinite(t_stat) else float("nan")
    return {
        "a": a,
        "b": b,
        "n_folds": n,
        "mean_diff": float(diff.mean()),
        "std_diff": std,
        "wins_a": int((diff > 0).sum()),
        "t": t_stat,
        "p_two_sided": p,
        "per_fold_diff": " ".join(f"{d:+.4f}" for d in diff),
    }


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: one row per dataset and pair."""
    parser = argparse.ArgumentParser(description="Paired per-fold Qini comparison.")
    parser.add_argument("--leaderboard", default=str(REPORTS / "phase8_leaderboard.csv"))
    parser.add_argument("--out", default=str(REPORTS / "phase8_paired_folds.csv"))
    args = parser.parse_args(argv)

    board = pd.read_csv(args.leaderboard)
    rows = []
    for name, part in board.groupby("dataset", sort=False):
        for a, b in PAIRS:
            row: dict[str, object] = {"dataset": name}
            row.update(paired(part, a, b))
            rows.append(row)
            print(
                f"{name:<10} {a:<14} vs {b:<15} mean {row['mean_diff']:+.4f} "
                f"std {row['std_diff']:.4f} wins {row['wins_a']}/{row['n_folds']} "
                f"t {row['t']:+.2f} p {row['p_two_sided']:.2f}   [{row['per_fold_diff']}]"
            )
    out = Path(args.out)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
