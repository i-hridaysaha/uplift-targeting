"""FastAPI serving layer for the chosen model and policy."""

from uplift.api.app import app, create_app
from uplift.api.artifacts import (
    PolicyBundle,
    PolicyResult,
    build_bundle,
    load_bundle,
    load_bundles,
    policy,
    save_bundle,
    score_records,
)

__all__ = [
    "PolicyBundle",
    "PolicyResult",
    "app",
    "build_bundle",
    "create_app",
    "load_bundle",
    "load_bundles",
    "policy",
    "save_bundle",
    "score_records",
]
