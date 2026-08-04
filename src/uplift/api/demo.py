"""Pure data logic behind the Streamlit demo.

The demo runs *in-process* over the ``uplift`` library (no HTTP call to the running
API): it reads the slim hold-out reference table an ``api.artifacts`` build wrote,
optionally filters it to a readable segment, and turns it into the four demo
views -- uplift score distribution, Qini curve, uplift-by-decile, and the priced
targeting policy -- by reusing the Phase-5 ``eval`` harness. Everything here is a
pure function of a reference frame so it is unit-tested without a browser; the
thin Streamlit script (``app/streamlit_app.py``) only wires these to widgets.

Segments are readable only on Hillstrom (real feature names); Criteo's ``f0..f11``
are anonymized, so it is shown whole-population.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from uplift.api.artifacts import (
    PRIMARY_OUTCOME,
    SCORE_COL,
    VALUE_OUTCOME,
    PolicyResult,
    load_reference_frame,
    reference_path,
    resolve_k,
)
from uplift.data.schema import TREATMENT_COL
from uplift.eval.metrics import UpliftScore, qini_coefficient, qini_curve, uplift_by_decile
from uplift.eval.value import (
    COST_PER_CONTACT,
    COST_PER_CONTACT_RANGE,
    VALUE_PER_CONVERSION,
    VALUE_PER_CONVERSION_RANGE,
    incremental_value_at_k,
    incremental_value_band,
)

# The whole-population choice, shown in the segment dropdowns as the default.
ALL = "All"

# Readable segment features per dataset. Criteo's features are anonymized floats
# (no meaningful segment), so it is served whole-population (empty list).
SEGMENT_FEATURES: dict[str, list[str]] = {
    "hillstrom": ["history_segment", "zip_code", "channel", "newbie", "mens", "womens"],
    "criteo": [],
}

# End-user labels for the segment dropdowns -- the raw column names and 0/1 codes
# are not customer language. Filtering still uses the underlying column and value;
# only the displayed text changes.
SEGMENT_LABELS: dict[str, str] = {
    "history_segment": "Spend History (Last Year)",
    "zip_code": "Location",
    "channel": "Purchase Channel",
    "newbie": "New Customer",
    "mens": "Bought Menswear",
    "womens": "Bought Womenswear",
}
_YES_NO_FEATURES = frozenset({"newbie", "mens", "womens"})
# Display-only fix of a raw-source misspelling (DATA.md keeps "Surburban" verbatim on disk).
_VALUE_DISPLAY: dict[object, str] = {"Surburban": "Suburban"}


def segment_label(feature: str) -> str:
    """Human-readable label for a segment feature (falls back to the raw column name)."""
    return SEGMENT_LABELS.get(feature, feature)


def format_segment_value(feature: str, value: object) -> str:
    """Display text for a dropdown value: ``Yes``/``No`` for 0/1 flags, cleaned labels else."""
    if value == ALL:
        return ALL
    if feature in _YES_NO_FEATURES:
        return "Yes" if int(value) == 1 else "No"
    return _VALUE_DISPLAY.get(value, str(value))


# --- Reference loading (I/O edge) -------------------------------------------


def load_reference(models_dir: str | Path, dataset: str) -> pd.DataFrame:
    """Load the demo hold-out reference table for ``dataset`` from ``models_dir``."""
    return load_reference_frame(reference_path(models_dir, dataset))


# --- Segments ---------------------------------------------------------------


def segment_features(dataset: str) -> list[str]:
    """Return the readable feature columns offered as segment filters for ``dataset``."""
    return SEGMENT_FEATURES.get(dataset, [])


def segment_options(reference: pd.DataFrame, dataset: str) -> dict[str, list[object]]:
    """Map each segment feature to its sorted distinct values (for the dropdowns)."""
    options: dict[str, list[object]] = {}
    for feature in segment_features(dataset):
        if feature in reference.columns:
            values = pd.unique(reference[feature].dropna())
            options[feature] = sorted(values.tolist())
    return options


def apply_segment(reference: pd.DataFrame, selections: dict[str, object]) -> pd.DataFrame:
    """Filter ``reference`` to rows matching every non-``ALL`` selection (values AND-ed)."""
    mask = np.ones(len(reference), dtype=bool)
    for feature, value in selections.items():
        if value == ALL or feature not in reference.columns:
            continue
        mask &= (reference[feature] == value).to_numpy()
    return reference[mask]


# --- Chart views ------------------------------------------------------------


def qini_xy(
    reference: pd.DataFrame, outcome: str = PRIMARY_OUTCOME
) -> tuple[np.ndarray, np.ndarray]:
    """Qini curve ``(x, y)`` over a reference frame, ranked by the stored score."""
    return qini_curve(
        reference[outcome].to_numpy(),
        reference[SCORE_COL].to_numpy(),
        reference[TREATMENT_COL].to_numpy(),
    )


def qini_score(reference: pd.DataFrame, outcome: str = PRIMARY_OUTCOME) -> UpliftScore:
    """Normalized + raw Qini coefficient over a reference frame."""
    return qini_coefficient(
        reference[outcome].to_numpy(),
        reference[SCORE_COL].to_numpy(),
        reference[TREATMENT_COL].to_numpy(),
    )


def decile_table(reference: pd.DataFrame, outcome: str = PRIMARY_OUTCOME) -> pd.DataFrame:
    """Observed uplift by decile of the stored score over a reference frame."""
    return uplift_by_decile(
        reference[outcome].to_numpy(),
        reference[SCORE_COL].to_numpy(),
        reference[TREATMENT_COL].to_numpy(),
    )


# --- Policy -----------------------------------------------------------------


def policy_for_frame(
    reference: pd.DataFrame,
    *,
    budget: float | None = None,
    k: int | None = None,
    fraction: float | None = None,
    value_per_conversion: float | None = None,
    cost_per_contact: float | None = None,
) -> PolicyResult:
    """Price the top-``k`` policy on a (segment-filtered) reference frame as a value band.

    Same contract as ``api.artifacts.policy`` but on an explicit frame instead of a
    bundle: give exactly one of ``budget`` / ``k`` / ``fraction``; the band always
    sweeps the documented value/cost ranges (``eval.value``), so it stays honest
    whatever the point inputs.
    """
    n = len(reference)
    cost_point = COST_PER_CONTACT if cost_per_contact is None else float(cost_per_contact)
    value_point = (
        VALUE_PER_CONVERSION if value_per_conversion is None else float(value_per_conversion)
    )
    resolved_k = resolve_k(n, budget, k, fraction, cost_point)
    band = incremental_value_band(
        reference[VALUE_OUTCOME].to_numpy(),
        reference[SCORE_COL].to_numpy(),
        reference[TREATMENT_COL].to_numpy(),
        resolved_k,
        value_point=value_point,
        cost_point=cost_point,
    )
    return PolicyResult(
        k=resolved_k,
        n=n,
        fraction_targeted=resolved_k / n if n else 0.0,
        band=band,
        value_per_conversion=value_point,
        cost_per_contact=cost_point,
        value_range=VALUE_PER_CONVERSION_RANGE,
        cost_range=COST_PER_CONTACT_RANGE,
    )


def value_sweep(
    reference: pd.DataFrame,
    fractions: np.ndarray,
    value_per_conversion: float = VALUE_PER_CONVERSION,
    cost_per_contact: float = COST_PER_CONTACT,
) -> tuple[np.ndarray, np.ndarray]:
    """Net incremental value across a sweep of targeted fractions (for the policy curve).

    Returns ``(fractions, values)`` where each value is the net incremental value of
    targeting that top fraction by score, under the given point assumptions. Lets the
    demo draw how value responds to the budget slider and mark the current point.
    """
    n = len(reference)
    ks = np.floor(np.asarray(fractions, dtype="float64") * n)
    values = incremental_value_at_k(
        reference[VALUE_OUTCOME].to_numpy(),
        reference[SCORE_COL].to_numpy(),
        reference[TREATMENT_COL].to_numpy(),
        ks,
        value_per_conversion=value_per_conversion,
        cost_per_contact=cost_per_contact,
    )
    return np.asarray(fractions, dtype="float64"), np.asarray(values, dtype="float64")


def honest_note(honest_negative: bool) -> str:
    """One honest sentence on what the served score means (mirrors the API note)."""
    if honest_negative:
        return (
            "Honest-negative pick: on this dataset uplift modeling did not beat plain "
            "response targeting, so customers are ranked by predicted response "
            "probability, not uplift."
        )
    return (
        "Customers are ranked by predicted uplift; the value band is an offline "
        "projection on the untouched hold-out under the stated assumptions."
    )
