"""Phase 8 follow-up: compare the shipped model with the response baseline where
the policy actually spends -- the top 10 percent of the hold-out.

The Qini coefficient summarises the whole curve, but a budgeted policy only ever
acts on its head. This script fits each model on the dev split, scores the
untouched hold-out once, and reports, at a fixed targeted fraction:

- the observed visit uplift inside the top decile of the ranking, with the standard
  error of that difference of two rates and the arm counts behind it,
- the cumulative incremental visits captured at that fraction (the size-corrected
  Qini curve read at ``k``) and its share of the visits captured by treating everyone,
- the incremental conversions at ``k``, the raw conversion counts per arm inside the
  top ``k`` (so the reader can see how few events the dollar figure rests on), and the
  priced band under the documented value and cost ranges.

A single hold-out draw has no fold band, so the standard error and the counts are
the honesty here; they are printed, not hidden.

Writes ``reports/phase8_operating_point.csv``, plus the two tables the case-study
figures are drawn from: ``reports/phase8_holdout_curves.csv`` (each ranking's
size-corrected Qini curve on the hold-out, sampled on a fixed grid of targeted
fractions) and ``reports/phase8_deciles.csv`` (observed visit uplift per predicted
decile with the arm counts). Read-only over ``data/processed/``.

Run: ``uv run python scripts/phase8_operating_point.py``
     ``uv run python scripts/phase8_operating_point.py --datasets criteo --fraction 0.1``
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from uplift.data.schema import CRITEO_FEATURES, HILLSTROM_FEATURES, TREATMENT_COL
from uplift.data.splits import HOLDOUT_FOLD, SEED
from uplift.eval.metrics import qini_curve, uplift_by_decile
from uplift.eval.value import incremental_value_band
from uplift.models import UpliftForest, holdout_uplift, model_scorer
from uplift.models.leaderboard import FOLD_COL, MODELS, Scorer

PROCESSED = Path("data/processed")
REPORTS = Path("reports")
PRIMARY_OUTCOME = "visit"
VALUE_OUTCOME = "conversion"

FEATURES: dict[str, list[str]] = {
    "hillstrom": HILLSTROM_FEATURES,
    "criteo": CRITEO_FEATURES,
}
# Same per-tree row cap as the Phase 7/8 boards (D25), so the forest here is the
# forest that was scored there.
FOREST_MAX_SAMPLES: dict[str, int | None] = {"hillstrom": None, "criteo": 150_000}


def scorers(name: str) -> dict[str, Scorer]:
    """The two rankings the deployment decision is between: the bar and the pick."""
    return {
        "response_model": MODELS["response_model"],
        "uplift_forest": model_scorer(UpliftForest, max_samples=FOREST_MAX_SAMPLES[name]),
    }


def operating_point(
    hold: pd.DataFrame, uplift: np.ndarray, fraction: float
) -> dict[str, float | int]:
    """Read one ranking at the top ``fraction`` of the hold-out."""
    y = hold[PRIMARY_OUTCOME].to_numpy()
    t = hold[TREATMENT_COL].to_numpy()
    n = len(hold)
    k = int(round(n * fraction))

    x, curve = qini_curve(y, uplift, t)
    visits_at_k = float(np.interp(k, x, curve))
    total = float(curve[-1])
    top = uplift_by_decile(y, uplift, t).iloc[0]
    p_t, p_c = float(top["response_treated"]), float(top["response_control"])
    n_t, n_c = int(top["n_treated"]), int(top["n_control"])
    top_se = float(np.sqrt(p_t * (1 - p_t) / n_t + p_c * (1 - p_c) / n_c))

    conversion = hold[VALUE_OUTCOME].to_numpy()
    head = np.argsort(uplift, kind="mergesort")[::-1][:k]
    head_t = t[head] == 1
    conv_t = int(conversion[head][head_t].sum())
    conv_c = int(conversion[head][~head_t].sum())
    # Incremental conversions at k are conv_t - ratio * conv_c with ratio the treated
    # to control count inside the top k; with each arm's count taken as Poisson the
    # standard error is sqrt(conv_t + ratio^2 * conv_c), dominated by the scaled
    # control count on an 85/15 dataset.
    ratio = head_t.sum() / max((~head_t).sum(), 1)
    conv_se = float(np.sqrt(conv_t + ratio**2 * conv_c))
    band = incremental_value_band(conversion, uplift, t, k)
    return {
        "fraction": fraction,
        "k": k,
        "top_decile_uplift": float(top["observed_uplift"]),
        "top_decile_se": top_se,
        "top_decile_n_treated": n_t,
        "top_decile_n_control": n_c,
        "incremental_visits": visits_at_k,
        "share_of_treat_everyone_visits": visits_at_k / total if total else float("nan"),
        "incremental_conversions": band.incremental_conversions,
        "incremental_conversions_se": conv_se,
        "conversions_treated_top_k": conv_t,
        "conversions_control_top_k": conv_c,
        "value_point": band.point,
        "value_low": band.low,
        "value_high": band.high,
    }


CURVE_GRID = np.linspace(0.0, 1.0, 201)


def curve_table(hold: pd.DataFrame, uplift: np.ndarray) -> pd.DataFrame:
    """The size-corrected Qini curve of ``visit`` on the hold-out, read on a fixed grid."""
    y = hold[PRIMARY_OUTCOME].to_numpy()
    t = hold[TREATMENT_COL].to_numpy()
    x, curve = qini_curve(y, uplift, t)
    n = len(hold)
    return pd.DataFrame(
        {"fraction": CURVE_GRID, "incremental_visits": np.interp(CURVE_GRID * n, x, curve)}
    )


def decile_table(hold: pd.DataFrame, uplift: np.ndarray) -> pd.DataFrame:
    """Observed visit uplift per predicted decile, with the arm counts behind each bar."""
    table = uplift_by_decile(
        hold[PRIMARY_OUTCOME].to_numpy(), uplift, hold[TREATMENT_COL].to_numpy()
    )
    return table[
        [
            "decile",
            "n_treated",
            "n_control",
            "response_treated",
            "response_control",
            "observed_uplift",
        ]
    ]


def run(
    name: str, fraction: float, seed: int = SEED
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit on dev, score the hold-out once per model, and read the operating point.

    Returns the operating-point rows, the curve samples and the decile tables, each
    tagged with the dataset and model.
    """
    frame = pd.read_parquet(PROCESSED / f"{name}.parquet")
    hold = frame[frame[FOLD_COL] == HOLDOUT_FOLD]
    rows, curves, deciles = [], [], []
    for model_name, scorer in scorers(name).items():
        uplift = holdout_uplift(frame, FEATURES[name], PRIMARY_OUTCOME, scorer, seed=seed)
        row: dict[str, object] = {"dataset": name, "model": model_name}
        row.update(operating_point(hold, uplift, fraction))
        rows.append(row)
        for table, sink in (
            (curve_table(hold, uplift), curves),
            (decile_table(hold, uplift), deciles),
        ):
            table.insert(0, "model", model_name)
            table.insert(0, "dataset", name)
            sink.append(table)
    return (
        pd.DataFrame(rows),
        pd.concat(curves, ignore_index=True),
        pd.concat(deciles, ignore_index=True),
    )


def _format(table: pd.DataFrame) -> str:
    """One aligned line per model: top decile, visits at k, conversions, band."""
    lines = [
        f"{'model':<16} {'top-decile (se)':>20} {'visits@k':>10} {'share':>7} "
        f"{'conv@k (se; t/c)':>22} {'point':>9} {'band':>22}"
    ]
    for r in table.itertuples():
        top = f"{r.top_decile_uplift:+.4f} ({r.top_decile_se:.4f})"
        counts = f"{r.conversions_treated_top_k}/{r.conversions_control_top_k}"
        conv = f"{r.incremental_conversions:+.1f} ({r.incremental_conversions_se:.0f}; {counts})"
        band = f"{r.value_low:+,.0f} to {r.value_high:+,.0f}"
        lines.append(
            f"{r.model:<16} {top:>20} {r.incremental_visits:>10.1f} "
            f"{r.share_of_treat_everyone_visits:>7.1%} {conv:>22} "
            f"{r.value_point:>+9,.0f} {band:>22}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: the operating-point table per dataset."""
    parser = argparse.ArgumentParser(description="Operating-point comparison on the hold-out.")
    parser.add_argument("--datasets", default="hillstrom,criteo")
    parser.add_argument("--fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)

    names = [n.strip() for n in args.datasets.split(",") if n.strip()]
    REPORTS.mkdir(parents=True, exist_ok=True)
    tables, curves, deciles = [], [], []
    for name in names:
        start = time.perf_counter()
        table, curve, decile = run(name, args.fraction, seed=args.seed)
        tables.append(table)
        curves.append(curve)
        deciles.append(decile)
        print(f"\n=== {name} @ top {args.fraction:.0%} ({time.perf_counter() - start:.1f}s) ===")
        print(_format(table))

    outputs = {
        "phase8_operating_point.csv": tables,
        "phase8_holdout_curves.csv": curves,
        "phase8_deciles.csv": deciles,
    }
    for filename, parts in outputs.items():
        pd.concat(parts, ignore_index=True).to_csv(REPORTS / filename, index=False)
    print(f"\nwrote {', '.join(str(REPORTS / f) for f in outputs)}")


if __name__ == "__main__":
    main()
