"""Offline uplift metrics: Qini and uplift curves, their coefficients, and
uplift-by-decile, plus a cross-validation band helper.

Pure array-in / number-out. Curve functions take three aligned 1-D arrays in the
order ``(y_true, uplift, treatment)`` -- the binary outcome, the predicted uplift
used to rank, and the binary treatment flag -- matching the order used across the
uplift ecosystem (e.g. ``scikit-uplift``). Curves are size-corrected for unequal
arms, ``Y_t(k) - Y_c(k) * N_t(k) / N_c(k)``, so they stay honest on Criteo's
~85/15 split (see EVAL.md). ``scikit-uplift`` is the reference implementation;
the tests cross-check these functions against it to machine precision.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import numpy as np
import pandas as pd
from sklearn.metrics import auc


class UpliftScore(NamedTuple):
    """A curve-area coefficient reported two ways.

    ``raw`` is the area between the model curve and the random-targeting diagonal;
    ``normalized`` divides that by the perfect model's area, putting datasets with
    different base rates and arm ratios on one 0-1 scale (the headline number).
    """

    raw: float
    normalized: float


class MetricBand(NamedTuple):
    """A metric's spread across CV folds: the mean, the std, and the per-fold values."""

    mean: float
    std: float
    per_fold: dict[int, float]


def _check(y_true: np.ndarray, uplift: np.ndarray, treatment: np.ndarray) -> None:
    """Guard the three aligned arrays share a length (a silent misalign corrupts every curve)."""
    if not (len(y_true) == len(uplift) == len(treatment)):
        raise ValueError(
            f"y_true, uplift, treatment must be the same length; got "
            f"{len(y_true)}, {len(uplift)}, {len(treatment)}."
        )


def _curve(
    y_true: np.ndarray, uplift: np.ndarray, treatment: np.ndarray, kind: str
) -> tuple[np.ndarray, np.ndarray]:
    """Shared Qini/uplift curve core: sort by uplift desc, evaluate at distinct-score steps.

    ``kind='qini'`` returns cumulative incremental responders (size-corrected);
    ``kind='uplift'`` returns the cumulative response-rate difference times the
    targeted count. Both start at ``(0, 0)``.
    """
    _check(y_true, uplift, treatment)
    y = np.asarray(y_true, dtype="float64")
    u = np.asarray(uplift, dtype="float64")
    t = np.asarray(treatment, dtype="float64")

    order = np.argsort(u, kind="mergesort")[::-1]
    y, u, t = y[order], u[order], t[order]
    y_ctrl = np.where(t == 1, 0.0, y)
    y_trmnt = np.where(t == 0, 0.0, y)

    distinct = np.where(np.diff(u))[0]
    idx = np.r_[distinct, u.size - 1]

    num_all = (idx + 1).astype("float64")
    num_trmnt = np.cumsum(t)[idx]
    num_ctrl = num_all - num_trmnt
    y_t = np.cumsum(y_trmnt)[idx]
    y_c = np.cumsum(y_ctrl)[idx]

    if kind == "qini":
        ratio = np.divide(num_trmnt, num_ctrl, out=np.zeros_like(num_trmnt), where=num_ctrl != 0)
        values = y_t - y_c * ratio
    else:  # uplift
        rate_t = np.divide(y_t, num_trmnt, out=np.zeros_like(y_t), where=num_trmnt != 0)
        rate_c = np.divide(y_c, num_ctrl, out=np.zeros_like(y_c), where=num_ctrl != 0)
        values = (rate_t - rate_c) * num_all

    if num_all.size == 0 or values[0] != 0 or num_all[0] != 0:
        num_all = np.r_[0.0, num_all]
        values = np.r_[0.0, values]
    return num_all, values


def qini_curve(
    y_true: np.ndarray, uplift: np.ndarray, treatment: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return the Qini curve as ``(x, y)``: x = number targeted, y = cumulative uplift.

    y is the size-corrected cumulative incremental responders over the top-ranked
    prefix, ``Y_t(k) - Y_c(k) * N_t(k) / N_c(k)`` (EVAL.md).
    """
    return _curve(y_true, uplift, treatment, "qini")


def uplift_curve(
    y_true: np.ndarray, uplift: np.ndarray, treatment: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return the uplift curve as ``(x, y)``: cumulative response-rate gap times count."""
    return _curve(y_true, uplift, treatment, "uplift")


def _perfect_qini_score(y: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Oracle ranking for the optimum Qini curve (treated responders first, control last)."""
    return y * t - y * (1 - t)


def _perfect_uplift_score(y: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Oracle ranking for the optimum uplift curve (matches scikit-uplift's construction)."""
    control_responders = np.sum((y == 1) & (t == 0))
    treated_nonresponders = np.sum((y == 0) & (t == 1))
    summand = y if control_responders > treated_nonresponders else t
    return 2 * (y == t) + summand


def _coefficient(
    y_true: np.ndarray, uplift: np.ndarray, treatment: np.ndarray, kind: str
) -> UpliftScore:
    """Normalized + raw area for either curve, relative to the diagonal and the perfect curve."""
    y = np.asarray(y_true)
    t = np.asarray(treatment)
    if kind == "qini":
        x_actual, y_actual = qini_curve(y, uplift, t)
        x_perfect, y_perfect = qini_curve(y, _perfect_qini_score(y, t), t)
    else:
        x_actual, y_actual = uplift_curve(y, uplift, t)
        x_perfect, y_perfect = uplift_curve(y, _perfect_uplift_score(y, t), t)

    baseline = auc(np.array([0.0, x_perfect[-1]]), np.array([0.0, y_perfect[-1]]))
    raw = auc(x_actual, y_actual) - baseline
    perfect = auc(x_perfect, y_perfect) - baseline
    normalized = raw / perfect if perfect != 0 else float("nan")
    return UpliftScore(raw=float(raw), normalized=float(normalized))


def qini_coefficient(y_true: np.ndarray, uplift: np.ndarray, treatment: np.ndarray) -> UpliftScore:
    """Qini coefficient: normalized (headline) and raw area under the Qini curve."""
    return _coefficient(y_true, uplift, treatment, "qini")


def auuc(y_true: np.ndarray, uplift: np.ndarray, treatment: np.ndarray) -> UpliftScore:
    """Area under the uplift curve (AUUC): normalized (headline) and raw."""
    return _coefficient(y_true, uplift, treatment, "uplift")


def uplift_by_decile(
    y_true: np.ndarray, uplift: np.ndarray, treatment: np.ndarray, bins: int = 10
) -> pd.DataFrame:
    """Observed uplift within each equal-count bin of predicted uplift (highest first).

    Rows are ranked by predicted uplift descending and split into ``bins``
    equal-count groups; ``observed_uplift = mean(y | treated) - mean(y | control)``
    per group. A good model shows observed uplift decreasing down the deciles.
    Bins with an empty arm yield ``nan`` for that rate. Returns one row per decile.
    """
    _check(y_true, uplift, treatment)
    y = np.asarray(y_true, dtype="float64")
    t = np.asarray(treatment)
    order = np.argsort(np.asarray(uplift), kind="mergesort")[::-1]
    y, t = y[order], t[order]

    rows: list[dict[str, float]] = []
    for i, group in enumerate(np.array_split(np.arange(len(y)), bins), start=1):
        yt, yc = y[group][t[group] == 1], y[group][t[group] == 0]
        rate_t = yt.mean() if yt.size else float("nan")
        rate_c = yc.mean() if yc.size else float("nan")
        rows.append(
            {
                "decile": i,
                "count": len(group),
                "n_treated": int(yt.size),
                "n_control": int(yc.size),
                "response_treated": rate_t,
                "response_control": rate_c,
                "observed_uplift": rate_t - rate_c,
            }
        )
    return pd.DataFrame(rows).astype({"decile": "int64", "count": "int64"})


def fold_metric_band(
    y_true: np.ndarray,
    uplift: np.ndarray,
    treatment: np.ndarray,
    fold: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
) -> MetricBand:
    """Evaluate ``metric_fn`` within each CV fold and return its mean/std across folds.

    The ``fold`` labels come from the processed data (``-1`` marks the hold-out and
    is excluded). ``metric_fn(y_true, uplift, treatment) -> float`` is applied to
    each fold's rows; the fold-to-fold spread is the honest confidence band for a
    high-variance metric like Qini (see EVAL.md). Scores must already be computed
    out-of-fold by the caller; this helper only aggregates.
    """
    y = np.asarray(y_true)
    u = np.asarray(uplift)
    t = np.asarray(treatment)
    f = np.asarray(fold)
    _check(y, u, t)

    per_fold: dict[int, float] = {}
    for label in sorted(int(v) for v in np.unique(f) if v != -1):
        mask = f == label
        per_fold[label] = float(metric_fn(y[mask], u[mask], t[mask]))

    values = np.array(list(per_fold.values()), dtype="float64")
    mean = float(values.mean()) if values.size else float("nan")
    std = float(values.std(ddof=1)) if values.size > 1 else 0.0
    return MetricBand(mean=mean, std=std, per_fold=per_fold)
