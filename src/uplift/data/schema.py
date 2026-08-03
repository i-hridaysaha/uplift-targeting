"""Tidy schema, dtypes, and the binary-treatment mapping for both datasets.

Pure transforms only: functions take in-memory frames/series and return tidy
frames. No dataset download or disk access happens here (that lives in
``ingest``). Categoricals are stored as pandas ``category`` (LightGBM-native),
binary flags/outcomes as ``int8``, and continuous columns as ``float32``.
"""

from __future__ import annotations

import pandas as pd

TREATMENT_COL = "treatment"

# --- Hillstrom -------------------------------------------------------------

HILLSTROM_CATEGORICAL: list[str] = ["history_segment", "zip_code", "channel"]
HILLSTROM_FEATURES: list[str] = [
    "recency",
    "history_segment",
    "history",
    "mens",
    "womens",
    "zip_code",
    "newbie",
    "channel",
]
HILLSTROM_OUTCOMES: list[str] = ["visit", "conversion", "spend"]
_HILLSTROM_TREATED_ARMS = frozenset({"Mens E-Mail", "Womens E-Mail"})

# --- Criteo ----------------------------------------------------------------

CRITEO_FEATURES: list[str] = [f"f{i}" for i in range(12)]
CRITEO_OUTCOMES: list[str] = ["visit", "conversion"]


def map_hillstrom_treatment(segment: pd.Series) -> pd.Series:
    """Map the Hillstrom ``segment`` arm to a binary treatment (1 = any e-mail).

    Both e-mail arms collapse to ``1``; ``No E-Mail`` is ``0`` (v1 is a single
    binary treatment). Returns an ``int8`` series named ``treatment``.
    """
    treated = segment.isin(_HILLSTROM_TREATED_ARMS)
    return treated.astype("int8").rename(TREATMENT_COL)


def build_hillstrom_frame(
    data: pd.DataFrame, target: pd.DataFrame, treatment: pd.Series
) -> pd.DataFrame:
    """Assemble the tidy Hillstrom frame: features + treatment + all outcomes.

    Returns one row per customer with the eight features, the mapped binary
    ``treatment``, and the ``visit``/``conversion``/``spend`` outcomes. Values are
    copied positionally, so ``data``, ``target``, and ``treatment`` must share row
    order (they do, coming from one loader ``Bunch``).
    """
    df = data[HILLSTROM_FEATURES].copy()
    for col in HILLSTROM_CATEGORICAL:
        df[col] = df[col].astype("category")
    df["recency"] = df["recency"].astype("int16")
    for col in ("mens", "womens", "newbie"):
        df[col] = df[col].astype("int8")
    df["history"] = df["history"].astype("float32")

    df[TREATMENT_COL] = map_hillstrom_treatment(treatment).to_numpy()
    df["visit"] = target["visit"].astype("int8").to_numpy()
    df["conversion"] = target["conversion"].astype("int8").to_numpy()
    df["spend"] = target["spend"].astype("float32").to_numpy()
    return df


def build_criteo_frame(
    data: pd.DataFrame, target: pd.DataFrame, treatment: pd.Series
) -> pd.DataFrame:
    """Assemble the tidy Criteo frame: ``f0..f11`` + treatment + visit + conversion.

    Features are ``float32`` and treatment/outcomes ``int8``. The
    post-randomization ``exposure`` flag never enters here: the loader is called
    with ``treatment_col='treatment'``, so only the randomized flag is passed in.
    """
    df = data[CRITEO_FEATURES].astype("float32").copy()
    df[TREATMENT_COL] = treatment.astype("int8").to_numpy()
    df["visit"] = target["visit"].astype("int8").to_numpy()
    df["conversion"] = target["conversion"].astype("int8").to_numpy()
    return df
