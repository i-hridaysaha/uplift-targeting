"""Offline uplift evaluation: Qini, AUUC, uplift-by-decile, incremental value."""

from uplift.eval.baselines import response_model_scores, treat_everyone_scores
from uplift.eval.explain import permutation_uplift_importance, treeshap_uplift
from uplift.eval.metrics import (
    MetricBand,
    UpliftScore,
    auuc,
    fold_metric_band,
    qini_coefficient,
    qini_curve,
    uplift_by_decile,
    uplift_curve,
)
from uplift.eval.value import (
    COST_PER_CONTACT,
    COST_PER_CONTACT_RANGE,
    VALUE_PER_CONVERSION,
    VALUE_PER_CONVERSION_RANGE,
    ValueBand,
    incremental_conversions_at_k,
    incremental_value_at_k,
    incremental_value_band,
    k_for_budget,
)

__all__ = [
    "COST_PER_CONTACT",
    "COST_PER_CONTACT_RANGE",
    "VALUE_PER_CONVERSION",
    "VALUE_PER_CONVERSION_RANGE",
    "MetricBand",
    "UpliftScore",
    "ValueBand",
    "auuc",
    "fold_metric_band",
    "incremental_conversions_at_k",
    "incremental_value_at_k",
    "incremental_value_band",
    "k_for_budget",
    "permutation_uplift_importance",
    "qini_coefficient",
    "qini_curve",
    "response_model_scores",
    "treat_everyone_scores",
    "treeshap_uplift",
    "uplift_by_decile",
    "uplift_curve",
]
