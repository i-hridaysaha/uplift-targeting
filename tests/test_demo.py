"""Tests for the in-process demo logic (``uplift.api.demo``) and the reference table.

The reference frames are built from the same small synthetic fixtures the serving
tests use -- shaped like each real dataset -- so the demo's segment filtering,
chart aggregations, and policy pricing run against the real fit/score recipe with
no network and no persisted artifacts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from uplift.api.artifacts import (
    SCORE_COL,
    build_bundle,
    build_reference_frame,
    load_reference_frame,
    save_reference_frame,
)
from uplift.api.demo import (
    ALL,
    apply_segment,
    decile_table,
    format_segment_value,
    honest_note,
    load_reference,
    policy_for_frame,
    qini_score,
    qini_xy,
    segment_features,
    segment_label,
    segment_options,
    value_sweep,
)
from uplift.data.schema import CRITEO_FEATURES
from uplift.data.splits import assign_splits
from uplift.eval.metrics import qini_coefficient, qini_curve
from uplift.eval.value import VALUE_PER_CONVERSION_RANGE, incremental_value_at_k

N = 1600
REF_COLS_EXTRA = ["visit", "conversion", "treatment", SCORE_COL]


def _criteo_frame(n: int = N, seed: int = 0) -> pd.DataFrame:
    """Synthetic Criteo-shaped frame: f0..f11 with an f0-driven uplift, splits assigned."""
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


@pytest.fixture(scope="module")
def criteo_ref() -> pd.DataFrame:
    frame = _criteo_frame()
    return build_reference_frame(frame, "criteo", build_bundle(frame, "criteo", seed=0))


@pytest.fixture(scope="module")
def hillstrom_ref() -> pd.DataFrame:
    frame = _hillstrom_frame()
    return build_reference_frame(frame, "hillstrom", build_bundle(frame, "hillstrom", seed=0))


# --- Reference frame --------------------------------------------------------


def test_reference_columns_and_size(criteo_ref):
    assert list(criteo_ref.columns) == [*CRITEO_FEATURES, *REF_COLS_EXTRA]
    assert len(criteo_ref) > 0
    assert np.isfinite(criteo_ref[SCORE_COL].to_numpy()).all()


def test_reference_score_matches_bundle():
    frame = _criteo_frame()
    bundle = build_bundle(frame, "criteo", seed=0)
    ref = build_reference_frame(frame, "criteo", bundle)
    # Same untouched hold-out, same model -> the reference score is the bundle's ref_score.
    assert np.allclose(ref[SCORE_COL].to_numpy(), bundle.ref_score)


def test_build_reference_requires_holdout():
    frame = _criteo_frame()
    bundle = build_bundle(frame, "criteo", seed=0)
    with pytest.raises(ValueError, match="hold-out"):
        build_reference_frame(frame[frame["fold"] != -1], "criteo", bundle)


def test_reference_roundtrip_preserves_categoricals(hillstrom_ref, tmp_path):
    save_reference_frame(hillstrom_ref, "hillstrom", tmp_path)
    loaded = load_reference(tmp_path, "hillstrom")
    assert isinstance(loaded["zip_code"].dtype, pd.CategoricalDtype)
    assert np.allclose(loaded[SCORE_COL].to_numpy(), hillstrom_ref[SCORE_COL].to_numpy())
    # load_reference must resolve the same path save wrote.
    assert (
        load_reference_frame(tmp_path / "hillstrom_holdout_reference.parquet").shape == loaded.shape
    )


# --- Segments ---------------------------------------------------------------


def test_segment_features_readable_hillstrom_empty_criteo():
    feats = segment_features("hillstrom")
    assert {"zip_code", "channel", "history_segment"} <= set(feats)
    assert segment_features("criteo") == []


def test_segment_options_sorted_and_scoped(hillstrom_ref, criteo_ref):
    options = segment_options(hillstrom_ref, "hillstrom")
    assert set(options) <= set(segment_features("hillstrom"))
    assert options["zip_code"] == sorted(options["zip_code"])
    assert set(options["zip_code"]) <= {"Urban", "Rural", "Surburban"}
    assert segment_options(criteo_ref, "criteo") == {}


def test_apply_segment_all_is_passthrough(hillstrom_ref):
    out = apply_segment(hillstrom_ref, {"zip_code": ALL, "channel": ALL})
    assert len(out) == len(hillstrom_ref)


def test_apply_segment_filters_and_ands(hillstrom_ref):
    out = apply_segment(hillstrom_ref, {"zip_code": "Urban", "newbie": 1})
    assert (out["zip_code"] == "Urban").all()
    assert (out["newbie"] == 1).all()
    assert 0 < len(out) < len(hillstrom_ref)


def test_segment_label_humanizes():
    assert segment_label("zip_code") == "Location"
    assert segment_label("newbie") == "New Customer"
    assert segment_label("unknown_feature") == "unknown_feature"  # falls back to the raw name


def test_format_segment_value():
    assert format_segment_value("newbie", 1) == "Yes"
    assert format_segment_value("mens", 0) == "No"
    assert format_segment_value("zip_code", "Surburban") == "Suburban"  # display-only spelling fix
    assert format_segment_value("zip_code", "Urban") == "Urban"
    assert format_segment_value("history_segment", "1) $0 - $100") == "1) $0 - $100"
    assert (
        format_segment_value("zip_code", ALL) == ALL
    )  # the whole-population choice passes through


# --- Chart views ------------------------------------------------------------


def test_qini_xy_matches_metrics(criteo_ref):
    x, y = qini_xy(criteo_ref)
    x2, y2 = qini_curve(
        criteo_ref["visit"].to_numpy(),
        criteo_ref[SCORE_COL].to_numpy(),
        criteo_ref["treatment"].to_numpy(),
    )
    assert np.allclose(x, x2) and np.allclose(y, y2)
    assert x[0] == 0.0 and y[0] == 0.0
    assert x[-1] == len(criteo_ref)


def test_qini_score_matches_metrics(criteo_ref):
    got = qini_score(criteo_ref)
    want = qini_coefficient(
        criteo_ref["visit"].to_numpy(),
        criteo_ref[SCORE_COL].to_numpy(),
        criteo_ref["treatment"].to_numpy(),
    )
    assert got.normalized == pytest.approx(want.normalized)


def test_decile_table_shape(criteo_ref):
    table = decile_table(criteo_ref)
    assert len(table) == 10
    assert "observed_uplift" in table.columns


# --- Policy -----------------------------------------------------------------


def test_policy_for_frame_fraction(criteo_ref):
    n = len(criteo_ref)
    assert policy_for_frame(criteo_ref, fraction=0.5).k == int(np.floor(0.5 * n))


def test_policy_for_frame_clamps(criteo_ref):
    n = len(criteo_ref)
    assert policy_for_frame(criteo_ref, k=10 * n).k == n


def test_policy_for_frame_requires_one_knob(criteo_ref):
    with pytest.raises(ValueError, match="exactly one"):
        policy_for_frame(criteo_ref)
    with pytest.raises(ValueError, match="exactly one"):
        policy_for_frame(criteo_ref, k=5, fraction=0.1)


def test_policy_for_frame_band_brackets_point(criteo_ref):
    result = policy_for_frame(criteo_ref, fraction=0.3)
    assert result.band.low <= result.band.point <= result.band.high
    assert result.value_range == VALUE_PER_CONVERSION_RANGE


def test_policy_for_frame_overrides(criteo_ref):
    result = policy_for_frame(criteo_ref, k=50, value_per_conversion=200.0, cost_per_contact=1.0)
    assert (result.value_per_conversion, result.cost_per_contact) == (200.0, 1.0)
    assert result.band.point == pytest.approx(
        result.band.incremental_conversions * 200.0 - 50 * 1.0
    )


def test_value_sweep_matches_and_zero_at_zero(criteo_ref):
    fractions = np.linspace(0.0, 1.0, 11)
    xs, values = value_sweep(
        criteo_ref, fractions, value_per_conversion=100.0, cost_per_contact=0.1
    )
    assert xs.shape == values.shape == fractions.shape
    assert values[0] == pytest.approx(0.0)  # target no one -> no value, no cost
    ks = np.floor(fractions * len(criteo_ref))
    want = incremental_value_at_k(
        criteo_ref["conversion"].to_numpy(),
        criteo_ref[SCORE_COL].to_numpy(),
        criteo_ref["treatment"].to_numpy(),
        ks,
        value_per_conversion=100.0,
        cost_per_contact=0.1,
    )
    assert np.allclose(values, want)


# --- Note -------------------------------------------------------------------


def test_honest_note_differs():
    assert "response" in honest_note(True).lower()
    assert honest_note(True) != honest_note(False)
