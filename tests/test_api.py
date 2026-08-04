"""Serving-layer tests: policy bundles and the FastAPI endpoints.

Bundles are built from small synthetic fixtures shaped like each real dataset, so
the tests exercise the real fit/score/policy recipe (Criteo -> uplift forest,
Hillstrom -> honest-negative response model) with no network and no persisted
artifacts. The API is driven through ``TestClient`` over an app wired to those
in-memory bundles.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from uplift.api.app import create_app
from uplift.api.artifacts import (
    build_bundle,
    frame_from_records,
    load_bundle,
    load_bundles,
    policy,
    save_bundle,
    score_records,
)
from uplift.data.schema import CRITEO_FEATURES, HILLSTROM_FEATURES
from uplift.data.splits import assign_splits
from uplift.eval.value import k_for_budget

N = 1600


def _criteo_frame(n: int = N, seed: int = 0) -> pd.DataFrame:
    """Synthetic Criteo-shaped frame: f0..f11, an f0-driven uplift, splits assigned."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 12)).astype("float32")
    df = pd.DataFrame(x, columns=CRITEO_FEATURES)
    t = rng.integers(0, 2, n).astype("int8")
    p = np.clip(0.12 + 0.20 * (x[:, 0] > 0) + t * 0.15 * (x[:, 0] > 0), 0.0, 1.0)
    visit = (rng.random(n) < p).astype("int8")
    conversion = (visit & (rng.random(n) < 0.3)).astype("int8")
    df["treatment"], df["visit"], df["conversion"] = t, visit, conversion
    return assign_splits(df, outcome_col="visit", seed=seed)


def _hillstrom_frame(n: int = N, seed: int = 0) -> pd.DataFrame:
    """Synthetic Hillstrom-shaped frame with the eight named features (3 categorical)."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "recency": rng.integers(1, 13, n).astype("int16"),
            "history_segment": pd.Categorical(rng.choice(["1) $0 - $100", "2) $100 - $200"], n)),
            "history": rng.uniform(0.0, 500.0, n).astype("float32"),
            "mens": rng.integers(0, 2, n).astype("int8"),
            "womens": rng.integers(0, 2, n).astype("int8"),
            "zip_code": pd.Categorical(rng.choice(["Urban", "Rural", "Surburban"], n)),
            "newbie": rng.integers(0, 2, n).astype("int8"),
            "channel": pd.Categorical(rng.choice(["Phone", "Web", "Multichannel"], n)),
        }
    )
    t = rng.integers(0, 2, n).astype("int8")
    visit = (rng.random(n) < 0.15).astype("int8")
    conversion = (visit & (rng.random(n) < 0.3)).astype("int8")
    df["treatment"], df["visit"], df["conversion"] = t, visit, conversion
    return assign_splits(df, outcome_col="visit", seed=seed)


def _criteo_record(seed: int = 1) -> dict[str, float]:
    """One Criteo customer record (all 12 float features)."""
    rng = np.random.default_rng(seed)
    return {f: float(v) for f, v in zip(CRITEO_FEATURES, rng.normal(size=12), strict=True)}


def _hillstrom_record() -> dict[str, object]:
    """One Hillstrom customer record with the eight named features."""
    return {
        "recency": 6,
        "history_segment": "1) $0 - $100",
        "history": 142.5,
        "mens": 1,
        "womens": 0,
        "zip_code": "Urban",
        "newbie": 0,
        "channel": "Web",
    }


@pytest.fixture(scope="module")
def criteo_bundle():
    return build_bundle(_criteo_frame(), "criteo", seed=0)


@pytest.fixture(scope="module")
def hillstrom_bundle():
    return build_bundle(_hillstrom_frame(), "hillstrom", seed=0)


@pytest.fixture(scope="module")
def client(criteo_bundle, hillstrom_bundle):
    return TestClient(create_app({"criteo": criteo_bundle, "hillstrom": hillstrom_bundle}))


# --- Bundle building --------------------------------------------------------


def test_criteo_bundle_is_uplift(criteo_bundle):
    b = criteo_bundle
    assert (b.model_name, b.model_kind, b.score_type) == ("uplift_forest", "uplift", "uplift")
    assert b.honest_negative is False
    assert b.features == CRITEO_FEATURES
    assert len(b.ref_score) == len(b.ref_conversion) == len(b.ref_treatment) > 0


def test_hillstrom_bundle_is_honest_negative(hillstrom_bundle):
    b = hillstrom_bundle
    assert b.model_name == "response_model"
    assert (b.model_kind, b.score_type) == ("response", "response_probability")
    assert b.honest_negative is True
    assert b.features == HILLSTROM_FEATURES


def test_build_bundle_requires_holdout():
    frame = _criteo_frame()
    frame = frame[frame["fold"] != -1]  # strip the hold-out
    with pytest.raises(ValueError, match="hold-out"):
        build_bundle(frame, "criteo", seed=0)


# --- Scoring ----------------------------------------------------------------


def test_score_records_criteo(criteo_bundle):
    scores = score_records([_criteo_record(1), _criteo_record(2)], criteo_bundle)
    assert scores.shape == (2,)
    assert np.isfinite(scores).all()


def test_score_records_hillstrom_probability(hillstrom_bundle):
    scores = score_records([_hillstrom_record()], hillstrom_bundle)
    assert scores.shape == (1,)
    assert 0.0 <= scores[0] <= 1.0  # a response probability


def test_frame_from_records_coerces_categoricals(hillstrom_bundle):
    frame = frame_from_records([_hillstrom_record()], hillstrom_bundle)
    assert list(frame.columns) == HILLSTROM_FEATURES
    assert isinstance(frame["zip_code"].dtype, pd.CategoricalDtype)
    assert "Surburban" in frame["zip_code"].cat.categories


def test_score_records_missing_feature(criteo_bundle):
    bad = _criteo_record(1)
    del bad["f3"]
    with pytest.raises(ValueError, match="missing feature"):
        score_records([bad], criteo_bundle)


def test_score_records_empty(criteo_bundle):
    with pytest.raises(ValueError, match="no customers"):
        score_records([], criteo_bundle)


# --- Policy -----------------------------------------------------------------


def test_policy_resolves_k(criteo_bundle):
    assert policy(criteo_bundle, k=10).k == 10


def test_policy_resolves_fraction(criteo_bundle):
    n = len(criteo_bundle.ref_score)
    assert policy(criteo_bundle, fraction=0.5).k == int(np.floor(0.5 * n))


def test_policy_resolves_budget(criteo_bundle):
    result = policy(criteo_bundle, budget=100.0)
    assert result.k == min(k_for_budget(100.0), len(criteo_bundle.ref_score))


def test_policy_clamps_k_to_n(criteo_bundle):
    n = len(criteo_bundle.ref_score)
    assert policy(criteo_bundle, k=10 * n).k == n


def test_policy_requires_exactly_one_knob(criteo_bundle):
    with pytest.raises(ValueError, match="exactly one"):
        policy(criteo_bundle)
    with pytest.raises(ValueError, match="exactly one"):
        policy(criteo_bundle, budget=100.0, k=5)


def test_policy_rejects_bad_fraction(criteo_bundle):
    with pytest.raises(ValueError, match="fraction"):
        policy(criteo_bundle, fraction=1.5)


def test_policy_band_brackets_point(criteo_bundle):
    result = policy(criteo_bundle, fraction=0.3)
    assert result.band.low <= result.band.point <= result.band.high


def test_policy_overrides_assumptions(criteo_bundle):
    result = policy(criteo_bundle, k=50, value_per_conversion=200.0, cost_per_contact=1.0)
    assert result.value_per_conversion == 200.0
    assert result.cost_per_contact == 1.0
    assert result.band.point == pytest.approx(
        result.band.incremental_conversions * 200.0 - 50 * 1.0
    )


# --- Persistence ------------------------------------------------------------


def test_save_load_roundtrip(criteo_bundle, tmp_path):
    save_bundle(criteo_bundle, tmp_path)
    loaded = load_bundle(tmp_path / "criteo_policy.pkl")
    records = [_criteo_record(3), _criteo_record(4)]
    assert np.allclose(score_records(records, loaded), score_records(records, criteo_bundle))
    assert set(load_bundles(tmp_path)) == {"criteo"}


def test_load_bundles_missing_dir_is_empty(tmp_path):
    assert load_bundles(tmp_path / "nope") == {}


# --- API endpoints ----------------------------------------------------------


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["datasets"] == ["criteo", "hillstrom"]
    assert body["models"]["criteo"]["model"] == "uplift_forest"
    assert body["models"]["criteo"]["honest_negative"] is False
    assert body["models"]["hillstrom"]["honest_negative"] is True


def test_score_endpoint_criteo(client):
    resp = client.post("/score", json={"dataset": "criteo", "customers": [_criteo_record(1)]})
    assert resp.status_code == 200
    body = resp.json()
    assert body["score_type"] == "uplift"
    assert len(body["scores"]) == 1


def test_score_endpoint_hillstrom_honest_negative(client):
    resp = client.post("/score", json={"dataset": "hillstrom", "customers": [_hillstrom_record()]})
    assert resp.status_code == 200
    body = resp.json()
    assert body["score_type"] == "response_probability"
    assert body["honest_negative"] is True


def test_score_unknown_dataset(client):
    resp = client.post("/score", json={"dataset": "nope", "customers": [_criteo_record(1)]})
    assert resp.status_code == 404


def test_score_missing_feature(client):
    bad = _criteo_record(1)
    del bad["f0"]
    resp = client.post("/score", json={"dataset": "criteo", "customers": [bad]})
    assert resp.status_code == 422


def test_score_empty_customers_rejected(client):
    resp = client.post("/score", json={"dataset": "criteo", "customers": []})
    assert resp.status_code == 422  # pydantic min_length


def test_policy_endpoint(client):
    resp = client.post("/policy", json={"dataset": "criteo", "budget": 100.0})
    assert resp.status_code == 200
    body = resp.json()
    assert body["model"] == "uplift_forest"
    assert 0 <= body["k"] <= body["n"]
    assert set(body["value"]) == {"point", "low", "high"}
    assert body["assumptions"]["value_range"] == [50.0, 150.0]
    assert body["note"]


def test_policy_endpoint_two_knobs_rejected(client):
    resp = client.post("/policy", json={"dataset": "criteo", "budget": 100.0, "k": 5})
    assert resp.status_code == 422


def test_policy_unknown_dataset(client):
    resp = client.post("/policy", json={"dataset": "nope", "budget": 100.0})
    assert resp.status_code == 404
