"""FastAPI serving layer for the chosen model and policy."""

from uplift.api.app import app, create_app
from uplift.api.artifacts import (
    PolicyBundle,
    PolicyResult,
    build_bundle,
    build_reference_frame,
    load_bundle,
    load_bundles,
    load_reference_frame,
    policy,
    reference_path,
    resolve_k,
    save_bundle,
    save_reference_frame,
    score_records,
)

__all__ = [
    "PolicyBundle",
    "PolicyResult",
    "app",
    "build_bundle",
    "build_reference_frame",
    "create_app",
    "load_bundle",
    "load_bundles",
    "load_reference_frame",
    "policy",
    "reference_path",
    "resolve_k",
    "save_bundle",
    "save_reference_frame",
    "score_records",
]
