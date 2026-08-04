"""Phase 9 persistence -- build the servable policy bundles from processed data.

For each dataset it fits the chosen model on the dev split, scores the untouched
hold-out as the incremental-value reference, and pickles the bundle to
``models/`` (gitignored). The FastAPI app (``uplift.api.app:app``) loads whatever
this writes.

Run: ``uv run python scripts/phase9_persist.py``
     ``uv run python scripts/phase9_persist.py --datasets hillstrom``  (fast case)

Fresh checkout order: ``uv run python -m uplift.data.ingest`` first (this reads
``data/processed/*.parquet``), then this, then serve.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from uplift.api.artifacts import build_bundle, policy, save_bundle
from uplift.data.splits import SEED

PROCESSED = Path("data/processed")
DEFAULT_MODELS_DIR = "models"
SANITY_BUDGET = 100.0


def persist(dataset: str, models_dir: str, seed: int = SEED) -> None:
    """Build, save, and print a one-line sanity summary for one dataset's bundle."""
    start = time.perf_counter()
    frame = pd.read_parquet(PROCESSED / f"{dataset}.parquet")
    bundle = build_bundle(frame, dataset, seed=seed)
    path = save_bundle(bundle, models_dir)

    result = policy(bundle, budget=SANITY_BUDGET)
    print(
        f"[{dataset}] {bundle.model_name} ({bundle.score_type})"
        f"{' -- honest-negative' if bundle.honest_negative else ''}\n"
        f"  wrote {path}  (reference hold-out n={len(bundle.ref_score):,})\n"
        f"  sanity policy @ ${SANITY_BUDGET:.0f}: k={result.k:,} "
        f"({result.fraction_targeted:.1%}), "
        f"inc. conversions={result.band.incremental_conversions:+.1f}, "
        f"value point=${result.band.point:,.0f} "
        f"[${result.band.low:,.0f}, ${result.band.high:,.0f}]  "
        f"({time.perf_counter() - start:.1f}s)"
    )


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: persist the requested datasets' policy bundles."""
    parser = argparse.ArgumentParser(description="Phase 9 -- persist servable policy bundles.")
    parser.add_argument("--datasets", default="hillstrom,criteo")
    parser.add_argument("--models-dir", default=DEFAULT_MODELS_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)

    names = [n.strip() for n in args.datasets.split(",") if n.strip()]
    for name in names:
        persist(name, args.models_dir, seed=args.seed)


if __name__ == "__main__":
    main()
