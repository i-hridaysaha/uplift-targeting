"""Randomization check: standardized mean difference (SMD) per covariate.

On randomized data the treated and control arms should look the same on every
covariate. The SMD measures how far apart they are in pooled-standard-deviation
units; ``|SMD| > 0.1`` marks an imbalance worth noting (see DATA.md). These are
pure numeric helpers with no I/O.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

IMBALANCE_THRESHOLD = 0.1


def standardized_mean_difference(treated: np.ndarray, control: np.ndarray) -> float:
    """Return the SMD between treated and control samples of one covariate.

    ``SMD = (mean_t - mean_c) / sqrt((var_t + var_c) / 2)`` using sample variance.
    Returns ``0.0`` when both arms are constant with equal means, and ``nan`` when
    the pooled variance is zero but the means differ (a difference with no scale).
    """
    treated = np.asarray(treated, dtype="float64")
    control = np.asarray(control, dtype="float64")
    mean_diff = treated.mean() - control.mean()
    pooled = (treated.var(ddof=1) + control.var(ddof=1)) / 2.0
    if pooled == 0.0:
        return 0.0 if mean_diff == 0.0 else float("nan")
    return float(mean_diff / np.sqrt(pooled))


def covariate_balance(
    df: pd.DataFrame,
    feature_cols: list[str],
    categorical_cols: list[str] | None = None,
    treatment_col: str = "treatment",
    threshold: float = IMBALANCE_THRESHOLD,
) -> pd.DataFrame:
    """Compute the per-covariate SMD between the treated and control arms.

    Numeric covariates yield one row each; each level of a categorical covariate
    yields a row on its 0/1 indicator (the per-category share). Returns a table
    ``[covariate, smd, abs_smd, imbalanced]`` sorted by ``abs_smd`` descending,
    where ``imbalanced`` flags ``|SMD| > threshold``.
    """
    categorical_cols = categorical_cols or []
    is_treated = df[treatment_col].to_numpy() == 1
    rows: list[dict[str, object]] = []

    for col in feature_cols:
        if col in categorical_cols:
            series = df[col].astype("category")
            for level in series.cat.categories:
                indicator = (series == level).to_numpy(dtype="float64")
                smd = standardized_mean_difference(indicator[is_treated], indicator[~is_treated])
                rows.append({"covariate": f"{col}={level}", "smd": smd})
        else:
            values = df[col].to_numpy(dtype="float64")
            smd = standardized_mean_difference(values[is_treated], values[~is_treated])
            rows.append({"covariate": col, "smd": smd})

    out = pd.DataFrame(rows)
    out["abs_smd"] = out["smd"].abs()
    out["imbalanced"] = out["abs_smd"] > threshold
    return out.sort_values("abs_smd", ascending=False, ignore_index=True)
