"""Repeated cross-validation: more draws of the Qini band for the models that matter.

The headline bands come from one five-fold split at seed 42. Five draws of a
high-variance metric make a wide band and a coarse tie rule, so this script
re-splits the *development* rows (the seed-42 hold-out stays untouched) under
further seeds and scores the response baseline, the S-learner and the uplift
forest out-of-fold on each. Pooled with the seed-42 folds from the leaderboard,
that gives twenty draws per model instead of five.

Writes ``reports/phase8_repeated_cv.csv`` with one row per dataset, seed and model
(mean, std and the five per-fold values), and prints the pooled band plus the
paired fold-by-fold wins against the response baseline.

Run: ``uv run python scripts/phase8_repeated_cv.py``            (seeds 43, 44, 45)
     ``uv run python scripts/phase8_repeated_cv.py --seeds 43``  (one extra draw)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from uplift.data.schema import CRITEO_FEATURES, HILLSTROM_FEATURES, TREATMENT_COL
from uplift.data.splits import HOLDOUT_FOLD, N_FOLDS, SEED
from uplift.models import SLearner, UpliftForest, leaderboard, model_scorer
from uplift.models.leaderboard import FOLD_COL, MODELS, Scorer

PROCESSED = Path("data/processed")
REPORTS = Path("reports")
PRIMARY_OUTCOME = "visit"
FEATURES: dict[str, list[str]] = {"hillstrom": HILLSTROM_FEATURES, "criteo": CRITEO_FEATURES}
FOREST_MAX_SAMPLES: dict[str, int | None] = {"hillstrom": None, "criteo": 150_000}
DEFAULT_SEEDS = (43, 44, 45)


def models_for(name: str) -> dict[str, Scorer]:
    """The bar and the two models that cleared it on Criteo."""
    return {
        "response_model": MODELS["response_model"],
        "s_learner": model_scorer(SLearner),
        "uplift_forest": model_scorer(UpliftForest, max_samples=FOREST_MAX_SAMPLES[name]),
    }


def resplit_dev(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    """The dev rows with fresh stratified folds; the seed-42 hold-out is left out."""
    dev = frame[frame[FOLD_COL] != HOLDOUT_FOLD].reset_index(drop=True)
    key = dev[TREATMENT_COL].astype(str) + "_" + dev[PRIMARY_OUTCOME].astype(str)
    fold = np.empty(len(dev), dtype="int8")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    for f, (_, held) in enumerate(skf.split(np.zeros(len(dev)), key)):
        fold[held] = f
    out = dev.copy()
    out[FOLD_COL] = fold
    return out


def run(name: str, seeds: tuple[int, ...]) -> pd.DataFrame:
    """Score the three models on each re-split and return one row per seed and model."""
    frame = pd.read_parquet(PROCESSED / f"{name}.parquet")
    parts = []
    for seed in seeds:
        start = time.perf_counter()
        dev = resplit_dev(frame, seed)
        board = leaderboard(
            dev, FEATURES[name], PRIMARY_OUTCOME, models=models_for(name), seed=seed
        )
        board.insert(0, "seed", seed)
        board.insert(0, "dataset", name)
        parts.append(board)
        print(f"  {name} seed {seed}: {time.perf_counter() - start:.0f}s")
    return pd.concat(parts, ignore_index=True)


def pooled(name: str, extra: pd.DataFrame, base: pd.DataFrame) -> str:
    """Pool the seed-42 folds with the new ones and read the band and the paired wins."""
    cols = sorted(
        (c for c in extra.columns if c.startswith("qini_fold_")),
        key=lambda c: int(c.rsplit("_", 1)[1]),
    )
    keep = base[(base.dataset == name) & base.model.isin(extra.model.unique())]
    both = pd.concat(
        [keep.assign(seed=SEED)[["seed", "model", *cols]], extra[["seed", "model", *cols]]],
        ignore_index=True,
    )
    wide = {
        m: both[both.model == m].sort_values("seed")[cols].to_numpy().ravel()
        for m in both.model.unique()
    }
    lines = [f"  pooled over {len(wide['response_model'])} folds:"]
    for m, values in wide.items():
        lines.append(f"    {m:<15} {values.mean():+.4f} +/- {values.std(ddof=1):.4f}")
    for m in ("s_learner", "uplift_forest"):
        diff = wide[m] - wide["response_model"]
        lines.append(
            f"    {m} vs response: mean {diff.mean():+.4f}, wins {(diff > 0).sum()}/{len(diff)}"
        )
    diff = wide["s_learner"] - wide["uplift_forest"]
    lines.append(
        f"    s_learner vs forest: mean {diff.mean():+.4f}, wins {(diff > 0).sum()}/{len(diff)}"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: repeated CV per dataset, pooled with the seed-42 board."""
    parser = argparse.ArgumentParser(description="Repeated cross-validation of the key models.")
    parser.add_argument("--datasets", default="hillstrom,criteo")
    parser.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS))
    args = parser.parse_args(argv)

    names = [n.strip() for n in args.datasets.split(",") if n.strip()]
    seeds = tuple(int(s) for s in args.seeds.split(",") if s.strip())
    base = pd.read_csv(REPORTS / "phase8_leaderboard.csv")
    parts = []
    for name in names:
        print(f"\n=== {name} ===")
        extra = run(name, seeds)
        parts.append(extra)
        print(pooled(name, extra, base))
    out = REPORTS / "phase8_repeated_cv.csv"
    pd.concat(parts, ignore_index=True).to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
