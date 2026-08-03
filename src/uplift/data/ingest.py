"""Data ingestion: load both datasets, apply the tidy schema and binary
treatment, draw the Criteo working subsample, assign arm-balanced splits, run the
SMD randomization check, and persist everything to ``data/processed``.

Run it with::

    uv run python -m uplift.data.ingest

Everything is seeded, so a re-run reproduces the same processed frames, splits,
and balance tables. Raw downloads are cached under ``data/raw`` (immutable);
processed outputs are derived and regenerated from raw by this module.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklift.datasets import fetch_hillstrom

from uplift.data.balance import covariate_balance
from uplift.data.schema import (
    CRITEO_FEATURES,
    HILLSTROM_CATEGORICAL,
    HILLSTROM_FEATURES,
    build_criteo_frame,
    build_hillstrom_frame,
)
from uplift.data.splits import SEED, assign_splits, subsample_stratified

DEFAULT_DATA_HOME = "data/raw"
DEFAULT_OUT = "data/processed"
CRITEO_SUBSAMPLE = 1_000_000
PRIMARY_OUTCOME = "visit"

# Criteo Uplift v2.1 raw csv.gz layout: 12 features + treatment + conversion +
# visit + exposure. The dataset's free download URLs are gone, so the loader reads
# a locally provided file, keeps float32/int8 dtypes to stay light on the full
# ~14M rows, and drops the post-randomization `exposure` column.
CRITEO_RAW_FILENAMES = ("criteo-research-uplift-v2.1.csv.gz", "criteo.csv.gz")
CRITEO_USECOLS = [*CRITEO_FEATURES, "treatment", "visit", "conversion"]
CRITEO_READ_DTYPES = {
    **{f: "float32" for f in CRITEO_FEATURES},
    "treatment": "int8",
    "visit": "int8",
    "conversion": "int8",
}


def load_hillstrom(data_home: str = DEFAULT_DATA_HOME) -> pd.DataFrame:
    """Download (if needed) and return the tidy Hillstrom frame."""
    bunch = fetch_hillstrom(target_col="all", data_home=data_home)
    return build_hillstrom_frame(bunch.data, bunch.target, bunch.treatment)


def _resolve_criteo_file(criteo_file: str | None, data_home: str) -> Path:
    """Return the Criteo raw csv.gz path, or raise with drop-in instructions."""
    if criteo_file is not None:
        path = Path(criteo_file)
        if not path.exists():
            raise FileNotFoundError(f"Criteo file not found: {path}")
        return path
    for name in CRITEO_RAW_FILENAMES:
        candidate = Path(data_home) / name
        if candidate.exists():
            return candidate
    searched = ", ".join(str(Path(data_home) / n) for n in CRITEO_RAW_FILENAMES)
    raise FileNotFoundError(
        "Criteo raw file not found. Criteo removed the free download URLs, so the "
        "file must be provided locally. Place 'criteo-research-uplift-v2.1.csv.gz' "
        f"under '{data_home}/' (searched: {searched}), or pass --criteo-file PATH."
    )


def read_criteo_csv(path: str | Path) -> pd.DataFrame:
    """Read the Criteo v2.1 csv.gz from disk into the tidy frame (drops exposure).

    Typed columns (float32 features, int8 flags) keep the full ~14M-row read light.
    """
    raw = pd.read_csv(path, usecols=CRITEO_USECOLS, dtype=CRITEO_READ_DTYPES)
    return build_criteo_frame(raw[CRITEO_FEATURES], raw[["visit", "conversion"]], raw["treatment"])


def load_criteo(
    criteo_file: str | None = None,
    data_home: str = DEFAULT_DATA_HOME,
    subsample: int = CRITEO_SUBSAMPLE,
    seed: int = SEED,
) -> pd.DataFrame:
    """Read the local Criteo file and return the tidy, subsampled frame.

    The full file is read from disk, then reduced to ``subsample`` rows stratified
    by ``treatment x conversion`` to keep the arm ratio and the rare positives.
    """
    path = _resolve_criteo_file(criteo_file, data_home)
    frame = read_criteo_csv(path)
    return subsample_stratified(frame, ["treatment", "conversion"], subsample, seed=seed)


def _persist(frame: pd.DataFrame, balance: pd.DataFrame, name: str, out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out / f"{name}.parquet", index=False)
    balance.to_csv(out / f"{name}_balance.csv", index=False)


def ingest_hillstrom(
    data_home: str = DEFAULT_DATA_HOME, out_dir: str = DEFAULT_OUT, seed: int = SEED
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ingest, split, balance-check, and persist Hillstrom. Returns (frame, balance)."""
    frame = load_hillstrom(data_home)
    frame = assign_splits(frame, outcome_col=PRIMARY_OUTCOME, seed=seed)
    balance = covariate_balance(frame, HILLSTROM_FEATURES, HILLSTROM_CATEGORICAL)
    _persist(frame, balance, "hillstrom", out_dir)
    return frame, balance


def ingest_criteo(
    criteo_file: str | None = None,
    data_home: str = DEFAULT_DATA_HOME,
    out_dir: str = DEFAULT_OUT,
    subsample: int = CRITEO_SUBSAMPLE,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ingest, split, balance-check, and persist Criteo. Returns (frame, balance)."""
    frame = load_criteo(criteo_file, data_home, subsample=subsample, seed=seed)
    frame = assign_splits(frame, outcome_col=PRIMARY_OUTCOME, seed=seed)
    balance = covariate_balance(frame, CRITEO_FEATURES)
    _persist(frame, balance, "criteo", out_dir)
    return frame, balance


def _summarize(name: str, frame: pd.DataFrame, balance: pd.DataFrame) -> str:
    """Build a human-readable one-block summary of a processed dataset."""
    treatment = frame["treatment"]
    treated_n = int(treatment.sum())
    lines = [
        f"[{name}] rows={len(frame):,}  "
        f"treated={treated_n:,} ({treatment.mean():.1%})  "
        f"control={len(frame) - treated_n:,}"
    ]
    for outcome in ("visit", "conversion"):
        if outcome in frame:
            rate_t = frame.loc[treatment == 1, outcome].mean()
            rate_c = frame.loc[treatment == 0, outcome].mean()
            lines.append(
                f"  {outcome}: treated={rate_t:.4f}  control={rate_c:.4f}  "
                f"naive_ATE={rate_t - rate_c:+.4f}"
            )
    if "spend" in frame:
        converters = frame["conversion"] == 1
        lines.append(f"  spend|converter: mean=${frame.loc[converters, 'spend'].mean():.2f}")

    fold_sizes = {int(f): int(n) for f, n in frame.groupby("fold").size().items() if f != -1}
    lines.append(f"  holdout={int(frame['holdout'].sum()):,}  fold_sizes={fold_sizes}")

    n_flagged = int(balance["imbalanced"].sum())
    lines.append(
        f"  balance: {len(balance)} covariates, {n_flagged} imbalanced "
        f"(|SMD|>0.1), max|SMD|={balance['abs_smd'].max():.4f}"
    )
    if n_flagged:
        flagged = balance.loc[balance["imbalanced"], ["covariate", "smd"]]
        lines.append(f"  flagged: {flagged.to_dict('records')}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: ingest the requested datasets and print their summaries."""
    parser = argparse.ArgumentParser(description="Ingest and persist the uplift datasets.")
    parser.add_argument("--datasets", default="hillstrom,criteo")
    parser.add_argument("--data-home", default=DEFAULT_DATA_HOME)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--criteo-subsample", type=int, default=CRITEO_SUBSAMPLE)
    parser.add_argument(
        "--criteo-file",
        default=None,
        help="Path to the Criteo v2.1 csv.gz (defaults to a lookup under --data-home).",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)

    names = [n.strip() for n in args.datasets.split(",") if n.strip()]
    if "hillstrom" in names:
        frame, balance = ingest_hillstrom(args.data_home, args.out, args.seed)
        print(_summarize("hillstrom", frame, balance))
    if "criteo" in names:
        frame, balance = ingest_criteo(
            args.criteo_file,
            args.data_home,
            args.out,
            subsample=args.criteo_subsample,
            seed=args.seed,
        )
        print(_summarize("criteo", frame, balance))


if __name__ == "__main__":
    main()
