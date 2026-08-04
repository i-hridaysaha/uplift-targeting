"""Persisted policy bundles: the chosen model plus its incremental-value reference.

A *policy bundle* is everything the serving layer needs for one dataset, fixed at
build time so scoring and policy are pure functions of the request:

- the chosen model, fitted on the **dev** split only (all CV folds, hold-out
  excluded), so the hold-out stays an untouched projection set;
- the feature schema (names and dtypes) used to coerce raw request records back
  to the exact columns the model saw at fit time;
- a **reference table** -- the model's score on the hold-out rows together with
  their ``conversion`` and ``treatment`` labels -- off which ``/policy`` reads the
  incremental-value band (``eval.value``). The band is an honest offline
  projection on data the model never trained on.

The Hillstrom winner is the response-model baseline (uplift did not beat plain
response targeting -- the honest-negative Phase 8 verdict), so its score is a
response probability, not an uplift; ``score_type`` labels that openly. The Criteo
winner is the hand-rolled uplift forest.

Persistence is stdlib ``pickle`` (no extra dependency). Bundles live under
``models/`` (gitignored); ``scripts/phase9_persist.py`` builds them from the
processed data.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from uplift.data.schema import (
    CRITEO_FEATURES,
    HILLSTROM_FEATURES,
    TREATMENT_COL,
)
from uplift.data.splits import HOLDOUT_FOLD, SEED
from uplift.eval.baselines import RESPONSE_MODEL_PARAMS
from uplift.eval.value import (
    COST_PER_CONTACT,
    COST_PER_CONTACT_RANGE,
    VALUE_PER_CONVERSION,
    VALUE_PER_CONVERSION_RANGE,
    ValueBand,
    incremental_value_band,
    k_for_budget,
)
from uplift.models.direct import UpliftForest

FOLD_COL = "fold"
PRIMARY_OUTCOME = "visit"
VALUE_OUTCOME = "conversion"
BUNDLE_SUFFIX = "_policy.pkl"
REFERENCE_SUFFIX = "_holdout_reference.parquet"
SCORE_COL = "score"

# The Phase 8 verdict, made servable (MODELING.md, D26): Hillstrom is the
# honest-negative response model; Criteo is the uplift forest.
CHOSEN_MODEL: dict[str, str] = {
    "hillstrom": "response_model",
    "criteo": "uplift_forest",
}
FEATURES: dict[str, list[str]] = {
    "hillstrom": HILLSTROM_FEATURES,
    "criteo": CRITEO_FEATURES,
}
# Per-tree row cap for the forest on the large dataset (D25); full bootstrap on
# small Hillstrom, though Hillstrom is served by the response model.
FOREST_MAX_SAMPLES: dict[str, int | None] = {"hillstrom": None, "criteo": 150_000}


@dataclass
class PolicyBundle:
    """Everything the API needs to score and price a policy for one dataset.

    ``model_kind`` is ``"response"`` (a ``P(outcome | X)`` classifier scored via
    ``predict_proba``) or ``"uplift"`` (a ``predict_uplift`` model). ``ref_*`` are
    the hold-out reference arrays the value band is read off. ``feature_dtypes``
    maps each feature to the pandas dtype used at fit, so request records coerce to
    identical columns (category levels included).
    """

    dataset: str
    model_name: str
    model_kind: str
    score_type: str
    honest_negative: bool
    outcome: str
    features: list[str]
    feature_dtypes: dict[str, object]
    model: object
    ref_score: np.ndarray
    ref_conversion: np.ndarray
    ref_treatment: np.ndarray


@dataclass
class PolicyResult:
    """The priced targeting policy at a chosen budget/size, plus the assumptions used."""

    k: int
    n: int
    fraction_targeted: float
    band: ValueBand
    value_per_conversion: float
    cost_per_contact: float
    value_range: tuple[float, float]
    cost_range: tuple[float, float]


# --- Building ---------------------------------------------------------------


def _fit_chosen(dataset: str, dev: pd.DataFrame, features: list[str], seed: int) -> object:
    """Fit the dataset's chosen model on the dev split and return the fitted object."""
    name = CHOSEN_MODEL[dataset]
    if name == "response_model":
        model = LGBMClassifier(**RESPONSE_MODEL_PARAMS, random_state=seed)
        model.fit(dev[features], dev[PRIMARY_OUTCOME].to_numpy())
        return model
    if name == "uplift_forest":
        model = UpliftForest(seed=seed, max_samples=FOREST_MAX_SAMPLES[dataset])
        model.fit(dev[features], dev[TREATMENT_COL].to_numpy(), dev[PRIMARY_OUTCOME].to_numpy())
        return model
    raise ValueError(f"no serving recipe for chosen model {name!r} (dataset {dataset!r}).")


def score_model(model: object, model_kind: str, x: pd.DataFrame) -> np.ndarray:
    """Return the per-row targeting score: response probability or predicted uplift."""
    if model_kind == "response":
        return model.predict_proba(x)[:, 1].astype("float64")
    return np.asarray(model.predict_uplift(x), dtype="float64")


def build_bundle(frame: pd.DataFrame, dataset: str, seed: int = SEED) -> PolicyBundle:
    """Fit the chosen model on dev and assemble the servable bundle for ``dataset``.

    The model trains on the dev rows (``fold != -1``); the untouched hold-out
    (``fold == -1``) becomes the reference the value band is projected on.
    """
    features = FEATURES[dataset]
    dev = frame[frame[FOLD_COL] != HOLDOUT_FOLD]
    hold = frame[frame[FOLD_COL] == HOLDOUT_FOLD]
    if hold.empty:
        raise ValueError(f"{dataset}: frame has no hold-out rows (fold == -1); cannot build.")

    model = _fit_chosen(dataset, dev, features, seed)
    kind = "response" if CHOSEN_MODEL[dataset] == "response_model" else "uplift"
    ref_score = score_model(model, kind, hold[features])

    return PolicyBundle(
        dataset=dataset,
        model_name=CHOSEN_MODEL[dataset],
        model_kind=kind,
        score_type="response_probability" if kind == "response" else "uplift",
        honest_negative=kind == "response",
        outcome=PRIMARY_OUTCOME,
        features=features,
        feature_dtypes={col: dev[col].dtype for col in features},
        model=model,
        ref_score=ref_score,
        ref_conversion=hold[VALUE_OUTCOME].to_numpy().astype("int8"),
        ref_treatment=hold[TREATMENT_COL].to_numpy().astype("int8"),
    )


def build_reference_frame(frame: pd.DataFrame, dataset: str, bundle: PolicyBundle) -> pd.DataFrame:
    """Slim hold-out reference for the demo: features + labels + the model's score.

    The Streamlit demo reads this table (not the processed data or the running API)
    so every chart is a pure function of an artifact the model never trained on: the
    untouched hold-out rows, their readable feature columns (dtypes preserved, so
    Hillstrom categoricals stay filterable), the ``visit`` and ``conversion``
    outcomes, the ``treatment`` flag, and the per-row ``score`` from the chosen
    model. One row per hold-out customer.
    """
    features = FEATURES[dataset]
    hold = frame[frame[FOLD_COL] == HOLDOUT_FOLD]
    if hold.empty:
        raise ValueError(f"{dataset}: frame has no hold-out rows (fold == -1); cannot build.")
    out = hold[[*features, PRIMARY_OUTCOME, VALUE_OUTCOME, TREATMENT_COL]].copy()
    out[SCORE_COL] = score_model(bundle.model, bundle.model_kind, hold[features])
    return out.reset_index(drop=True)


# --- Scoring ----------------------------------------------------------------


def frame_from_records(records: list[dict[str, object]], bundle: PolicyBundle) -> pd.DataFrame:
    """Coerce raw request records to the bundle's exact feature columns and dtypes.

    Every feature must be present; extra keys are ignored. Categorical columns are
    cast to the fitted category levels (unseen values become missing), so the frame
    matches what the model saw at fit time.
    """
    if not records:
        raise ValueError("no customers provided.")
    raw = pd.DataFrame.from_records(records)
    missing = [c for c in bundle.features if c not in raw.columns]
    if missing:
        raise ValueError(f"missing feature columns: {missing}")
    frame = raw[bundle.features].copy()
    for col in bundle.features:
        frame[col] = frame[col].astype(bundle.feature_dtypes[col])
    return frame


def score_records(records: list[dict[str, object]], bundle: PolicyBundle) -> np.ndarray:
    """Score a batch of customer records with the bundle's model."""
    return score_model(bundle.model, bundle.model_kind, frame_from_records(records, bundle))


# --- Policy -----------------------------------------------------------------


def resolve_k(
    n: int, budget: float | None, k: int | None, fraction: float | None, cost: float
) -> int:
    """Turn exactly one of budget/k/fraction into a contact count ``k`` in ``[0, n]``."""
    given = sum(x is not None for x in (budget, k, fraction))
    if given != 1:
        raise ValueError("provide exactly one of: budget, k, fraction.")
    if k is not None:
        resolved = int(k)
    elif fraction is not None:
        if not 0.0 <= fraction <= 1.0:
            raise ValueError(f"fraction must be in [0, 1]; got {fraction}.")
        resolved = int(np.floor(fraction * n))
    else:
        resolved = k_for_budget(float(budget), cost)
    if resolved < 0:
        raise ValueError(f"k must be non-negative; got {resolved}.")
    return min(resolved, n)


def policy(
    bundle: PolicyBundle,
    *,
    budget: float | None = None,
    k: int | None = None,
    fraction: float | None = None,
    value_per_conversion: float | None = None,
    cost_per_contact: float | None = None,
) -> PolicyResult:
    """Price the top-``k`` targeting policy on the hold-out reference as a value band.

    Give exactly one of ``budget`` (dollars, converted to ``k`` at the point
    cost), ``k`` (contacts), or ``fraction`` (of the population). The dollar point
    inputs default to the documented anchors; the band always sweeps the
    assumption ranges (``eval.value``), so it is honest regardless of the point.
    """
    n = len(bundle.ref_score)
    cost_point = COST_PER_CONTACT if cost_per_contact is None else float(cost_per_contact)
    value_point = (
        VALUE_PER_CONVERSION if value_per_conversion is None else float(value_per_conversion)
    )
    resolved_k = resolve_k(n, budget, k, fraction, cost_point)
    band = incremental_value_band(
        bundle.ref_conversion,
        bundle.ref_score,
        bundle.ref_treatment,
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


# --- Persistence (I/O edge) -------------------------------------------------


def bundle_path(models_dir: str | Path, dataset: str) -> Path:
    """Return the on-disk path for a dataset's bundle under ``models_dir``."""
    return Path(models_dir) / f"{dataset}{BUNDLE_SUFFIX}"


def save_bundle(bundle: PolicyBundle, models_dir: str | Path) -> Path:
    """Pickle a bundle to ``models_dir`` and return the path written."""
    path = bundle_path(models_dir, bundle.dataset)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(bundle, fh)
    return path


def load_bundle(path: str | Path) -> PolicyBundle:
    """Load a single pickled bundle from ``path``."""
    with Path(path).open("rb") as fh:
        return pickle.load(fh)


def load_bundles(models_dir: str | Path) -> dict[str, PolicyBundle]:
    """Load every ``*_policy.pkl`` bundle under ``models_dir``, keyed by dataset."""
    directory = Path(models_dir)
    if not directory.is_dir():
        return {}
    bundles: dict[str, PolicyBundle] = {}
    for path in sorted(directory.glob(f"*{BUNDLE_SUFFIX}")):
        bundle = load_bundle(path)
        bundles[bundle.dataset] = bundle
    return bundles


def reference_path(models_dir: str | Path, dataset: str) -> Path:
    """Return the on-disk path for a dataset's demo reference table under ``models_dir``."""
    return Path(models_dir) / f"{dataset}{REFERENCE_SUFFIX}"


def save_reference_frame(frame: pd.DataFrame, dataset: str, models_dir: str | Path) -> Path:
    """Write a demo reference table to ``models_dir`` as parquet and return the path."""
    path = reference_path(models_dir, dataset)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return path


def load_reference_frame(path: str | Path) -> pd.DataFrame:
    """Load a single demo reference table from ``path`` (category dtypes preserved)."""
    return pd.read_parquet(path)
