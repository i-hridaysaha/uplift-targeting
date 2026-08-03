"""Incremental value of a targeting policy at a fixed budget.

Target the top-``k`` customers by predicted uplift; the estimated incremental
conversions over that prefix come straight off the size-corrected Qini curve of
the ``conversion`` outcome. Net value = incremental conversions x
value-per-conversion - k x cost-per-contact.

The two dollar inputs are **modeling assumptions, not measurements**, so every
figure ships as a band swept over the ranges below (see EVAL.md, D17/D20), never
a single confident number.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from uplift.eval.metrics import qini_curve

# value-per-conversion: anchored on Hillstrom mean spend among converters
# ($116.36, 95% CI [$107.57, $125.16]; D20). Point estimate plus a sensitivity
# range that brackets the mean and most of the IQR.
VALUE_PER_CONVERSION = 116.36
VALUE_PER_CONVERSION_RANGE = (50.0, 150.0)

# cost-per-contact: notional (D17), chosen so the budget constraint actually
# bites. NOT a measured email cost -- real email is ~free, which would make
# "treat everyone" win by default.
COST_PER_CONTACT = 0.10
COST_PER_CONTACT_RANGE = (0.05, 0.50)


class ValueBand(NamedTuple):
    """Incremental value at budget ``k``: the point estimate and its assumption band.

    ``incremental_conversions`` is assumption-free (it comes off the Qini curve);
    ``point`` uses the anchor value/cost, and ``low``/``high`` are the worst/best
    corners of the value x cost ranges.
    """

    k: int
    incremental_conversions: float
    point: float
    low: float
    high: float


def k_for_budget(budget: float, cost_per_contact: float = COST_PER_CONTACT) -> int:
    """Number of contacts a dollar ``budget`` buys at ``cost_per_contact`` (floored)."""
    if cost_per_contact <= 0:
        raise ValueError(f"cost_per_contact must be positive; got {cost_per_contact}.")
    return int(np.floor(budget / cost_per_contact))


def incremental_conversions_at_k(
    conversion: np.ndarray,
    uplift: np.ndarray,
    treatment: np.ndarray,
    k: float | np.ndarray,
) -> float | np.ndarray:
    """Estimated incremental conversions from treating the top-``k`` ranked customers.

    Reads the size-corrected Qini curve of the ``conversion`` outcome at position
    ``k`` (linear interpolation between curve steps). ``k`` may be a scalar or an
    array of budgets; values outside ``[0, n]`` clamp to the curve ends.
    """
    x, y = qini_curve(conversion, uplift, treatment)
    return np.interp(k, x, y)


def incremental_value_at_k(
    conversion: np.ndarray,
    uplift: np.ndarray,
    treatment: np.ndarray,
    k: float | np.ndarray,
    value_per_conversion: float = VALUE_PER_CONVERSION,
    cost_per_contact: float = COST_PER_CONTACT,
) -> float | np.ndarray:
    """Net incremental value of targeting the top-``k``: value of lift minus contact cost."""
    inc = incremental_conversions_at_k(conversion, uplift, treatment, k)
    return inc * value_per_conversion - np.asarray(k) * cost_per_contact


def incremental_value_band(
    conversion: np.ndarray,
    uplift: np.ndarray,
    treatment: np.ndarray,
    k: int,
    value_range: tuple[float, float] = VALUE_PER_CONVERSION_RANGE,
    cost_range: tuple[float, float] = COST_PER_CONTACT_RANGE,
    value_point: float = VALUE_PER_CONVERSION,
    cost_point: float = COST_PER_CONTACT,
) -> ValueBand:
    """Incremental value at budget ``k`` as a point estimate plus a min/max band.

    The incremental conversions are computed once (they do not depend on the
    dollar assumptions); the band is the extreme corners of the value x cost
    ranges, so it is correct whether the lift at ``k`` is positive or negative.
    """
    inc = float(incremental_conversions_at_k(conversion, uplift, treatment, k))
    corners = [inc * value - k * cost for value in value_range for cost in cost_range]
    return ValueBand(
        k=int(k),
        incremental_conversions=inc,
        point=inc * value_point - k * cost_point,
        low=min(corners),
        high=max(corners),
    )
