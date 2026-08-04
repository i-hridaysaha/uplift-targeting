"""FastAPI serving layer: score customers and price a targeting policy.

Three endpoints over the persisted policy bundles (``api.artifacts``):

- ``GET  /health`` -- liveness plus which datasets and models are loaded.
- ``POST /score``  -- per-customer targeting score (uplift, or response
  probability for the honest-negative Hillstrom pick).
- ``POST /policy`` -- the incremental-value band for a budget / size, projected on
  the untouched hold-out under the stated cost and value assumptions.

``create_app`` builds an app over an explicit bundle dict (used by tests); the
module-level ``app`` loads bundles from ``models/`` (override with
``UPLIFT_MODELS_DIR``) so ``uv run uvicorn uplift.api.app:app`` serves whatever
``scripts/phase9_persist.py`` wrote.
"""

from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from uplift.api.artifacts import (
    PolicyBundle,
    load_bundles,
    policy,
    score_records,
)

DEFAULT_MODELS_DIR = "models"


# --- Request / response schemas ---------------------------------------------


class ScoreRequest(BaseModel):
    """A batch of customer feature records to score for one dataset."""

    dataset: str
    customers: list[dict[str, Any]] = Field(..., min_length=1)


class ScoreResponse(BaseModel):
    """Per-customer targeting scores plus what kind of score they are."""

    dataset: str
    model: str
    score_type: str
    honest_negative: bool
    scores: list[float]


class PolicyRequest(BaseModel):
    """A budget/size to price. Give exactly one of ``budget``, ``k``, ``fraction``."""

    dataset: str
    budget: float | None = None
    k: int | None = None
    fraction: float | None = None
    value_per_conversion: float | None = None
    cost_per_contact: float | None = None


class ValueOut(BaseModel):
    """The net incremental value: point estimate and the assumption-sweep band."""

    point: float
    low: float
    high: float


class AssumptionsOut(BaseModel):
    """The dollar assumptions behind a policy figure (point inputs and their ranges)."""

    value_per_conversion: float
    cost_per_contact: float
    value_range: tuple[float, float]
    cost_range: tuple[float, float]


class PolicyResponse(BaseModel):
    """The priced targeting policy at the chosen budget/size."""

    dataset: str
    model: str
    score_type: str
    honest_negative: bool
    k: int
    n: int
    fraction_targeted: float
    incremental_conversions: float
    value: ValueOut
    assumptions: AssumptionsOut
    note: str


class DatasetInfo(BaseModel):
    """What is loaded and servable for one dataset."""

    model: str
    score_type: str
    honest_negative: bool
    reference_n: int
    features: list[str]


class HealthResponse(BaseModel):
    """Liveness and the loaded model roster."""

    status: str
    datasets: list[str]
    models: dict[str, DatasetInfo]


# --- Helpers ----------------------------------------------------------------


def _note(bundle: PolicyBundle) -> str:
    """One honest sentence on what the served score means for this dataset."""
    if bundle.honest_negative:
        return (
            "Honest-negative pick: on this dataset uplift modeling did not beat plain "
            "response targeting, so the policy ranks by predicted response probability, "
            "not uplift."
        )
    return (
        "Policy ranks customers by predicted uplift; the value band is an offline "
        "projection on the untouched hold-out under the stated assumptions."
    )


def create_app(bundles: dict[str, PolicyBundle]) -> FastAPI:
    """Build the FastAPI app serving the given dataset -> bundle map."""
    app = FastAPI(
        title="Uplift targeting API",
        version="0.1.0",
        description="Score customers by incremental effect and price a targeting policy.",
    )

    def _bundle(dataset: str) -> PolicyBundle:
        bundle = bundles.get(dataset)
        if bundle is None:
            raise HTTPException(
                status_code=404,
                detail=f"unknown dataset {dataset!r}; loaded: {sorted(bundles)}.",
            )
        return bundle

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        """Report liveness and the loaded models."""
        return HealthResponse(
            status="ok",
            datasets=sorted(bundles),
            models={
                name: DatasetInfo(
                    model=b.model_name,
                    score_type=b.score_type,
                    honest_negative=b.honest_negative,
                    reference_n=len(b.ref_score),
                    features=b.features,
                )
                for name, b in bundles.items()
            },
        )

    @app.post("/score", response_model=ScoreResponse)
    def score(request: ScoreRequest) -> ScoreResponse:
        """Return a targeting score per customer for the requested dataset."""
        bundle = _bundle(request.dataset)
        try:
            scores = score_records(request.customers, bundle)
        except ValueError as err:
            raise HTTPException(status_code=422, detail=str(err)) from err
        return ScoreResponse(
            dataset=bundle.dataset,
            model=bundle.model_name,
            score_type=bundle.score_type,
            honest_negative=bundle.honest_negative,
            scores=[float(s) for s in scores],
        )

    @app.post("/policy", response_model=PolicyResponse)
    def policy_endpoint(request: PolicyRequest) -> PolicyResponse:
        """Return the incremental-value band for a budget/size on the reference set."""
        bundle = _bundle(request.dataset)
        try:
            result = policy(
                bundle,
                budget=request.budget,
                k=request.k,
                fraction=request.fraction,
                value_per_conversion=request.value_per_conversion,
                cost_per_contact=request.cost_per_contact,
            )
        except ValueError as err:
            raise HTTPException(status_code=422, detail=str(err)) from err
        return PolicyResponse(
            dataset=bundle.dataset,
            model=bundle.model_name,
            score_type=bundle.score_type,
            honest_negative=bundle.honest_negative,
            k=result.k,
            n=result.n,
            fraction_targeted=result.fraction_targeted,
            incremental_conversions=result.band.incremental_conversions,
            value=ValueOut(point=result.band.point, low=result.band.low, high=result.band.high),
            assumptions=AssumptionsOut(
                value_per_conversion=result.value_per_conversion,
                cost_per_contact=result.cost_per_contact,
                value_range=result.value_range,
                cost_range=result.cost_range,
            ),
            note=_note(bundle),
        )

    return app


app = create_app(load_bundles(os.environ.get("UPLIFT_MODELS_DIR", DEFAULT_MODELS_DIR)))
