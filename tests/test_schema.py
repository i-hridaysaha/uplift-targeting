"""Unit tests for the tidy-schema builders and the treatment mapping."""

import pandas as pd

from uplift.data.schema import build_hillstrom_frame, map_hillstrom_treatment


def test_map_hillstrom_treatment_collapses_email_arms() -> None:
    seg = pd.Series(["Mens E-Mail", "Womens E-Mail", "No E-Mail", "No E-Mail"])
    t = map_hillstrom_treatment(seg)
    assert t.tolist() == [1, 1, 0, 0]
    assert t.name == "treatment"
    assert str(t.dtype) == "int8"


def test_build_hillstrom_frame_dtypes_and_columns() -> None:
    data = pd.DataFrame(
        {
            "recency": [1, 10],
            "history_segment": ["1) $0 - $100", "2) $100 - $200"],
            "history": [0.0, 150.5],
            "mens": [1, 0],
            "womens": [0, 1],
            "zip_code": ["Urban", "Rural"],
            "newbie": [1, 0],
            "channel": ["Web", "Phone"],
        }
    )
    target = pd.DataFrame({"visit": [1, 0], "conversion": [0, 1], "spend": [0.0, 116.4]})
    treatment = pd.Series(["Mens E-Mail", "No E-Mail"])

    df = build_hillstrom_frame(data, target, treatment)

    assert df["treatment"].tolist() == [1, 0]
    assert str(df["history_segment"].dtype) == "category"
    assert str(df["visit"].dtype) == "int8"
    assert str(df["history"].dtype) == "float32"
    assert {"treatment", "visit", "conversion", "spend"} <= set(df.columns)
