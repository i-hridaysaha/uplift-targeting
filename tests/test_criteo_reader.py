"""Unit test for the Criteo local-file reader (tiny gzipped fixture, no network)."""

import pandas as pd

from uplift.data.ingest import read_criteo_csv
from uplift.data.schema import CRITEO_FEATURES


def test_read_criteo_csv_drops_exposure_and_types(tmp_path) -> None:
    cols = [*CRITEO_FEATURES, "treatment", "conversion", "visit", "exposure"]
    row_a = [float(i) for i in range(12)] + [1, 0, 1, 1]
    row_b = [float(i) + 0.5 for i in range(12)] + [0, 1, 0, 0]
    raw = pd.DataFrame([row_a, row_b], columns=cols)
    path = tmp_path / "criteo-mini.csv.gz"
    raw.to_csv(path, index=False, compression="gzip")

    frame = read_criteo_csv(path)

    assert list(frame.columns) == [*CRITEO_FEATURES, "treatment", "visit", "conversion"]
    assert "exposure" not in frame.columns
    assert str(frame["treatment"].dtype) == "int8"
    assert str(frame["f0"].dtype) == "float32"
    assert frame["treatment"].tolist() == [1, 0]
    assert frame["visit"].tolist() == [1, 0]
    assert frame["conversion"].tolist() == [0, 1]
