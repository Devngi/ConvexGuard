"""
features.py — Optional feature engineering on top of the raw PCA columns.

Two transformations are available but neither is strictly required because
V1..V28 are already PCA-anonymised and well-scaled:

1. **log_amount**: log1p-transform of the ``Amount`` column, which has a
   heavy right tail.  Formula: ``log_amount = log(1 + Amount)``.

2. **cyclic_time**: The ``Time`` column counts seconds since the first
   transaction over ~48 hours.  Mapping it to (sin, cos) of the hour-of-day
   angle encodes periodicity without an arbitrary cut-point.
   Formula:
       hour = (Time mod 86400) / 3600          # [0, 24)
       time_sin = sin(2π * hour / 24)
       time_cos = cos(2π * hour / 24)

Neither transformation is applied by default in the pipeline; pass
``use_log_amount=True`` or ``use_cyclic_time=True`` to the pipeline config
to include them.

Note: Account IDs are absent from this dataset, so graph/cycle features
(e.g. PaySim-style ring detection) cannot be built here.
"""


import numpy as np
import pandas as pd

from convexguard.data import AMOUNT_COL, FEATURE_COLS, TIME_COL


def add_log_amount(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Add a log1p-transformed amount column.

    Parameters
    ----------
    df:
        Dataframe containing the ``Amount`` column.

    Returns
    -------
    df_out : pd.DataFrame
        Copy with ``log_amount`` column appended.
    new_cols : list[str]
        Names of the newly added columns (``["log_amount"]``).
    """
    df_out = df.copy()
    df_out["log_amount"] = np.log1p(df_out[AMOUNT_COL])
    return df_out, ["log_amount"]


def add_cyclic_time(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Add sine/cosine encoding of hour-of-day derived from ``Time``.

    ``Time`` is seconds elapsed since the first transaction in the dataset
    (spans ~48 h).  We map it to a 24-hour clock and encode cyclically.

    Parameters
    ----------
    df:
        Dataframe containing the ``Time`` column.

    Returns
    -------
    df_out : pd.DataFrame
        Copy with ``time_sin`` and ``time_cos`` columns appended.
    new_cols : list[str]
        ``["time_sin", "time_cos"]``.
    """
    df_out = df.copy()
    hour = (df_out[TIME_COL] % 86400) / 3600.0  # fractional hour in [0, 24)
    angle = 2.0 * np.pi * hour / 24.0
    df_out["time_sin"] = np.sin(angle)
    df_out["time_cos"] = np.cos(angle)
    return df_out, ["time_sin", "time_cos"]


def build_feature_list(
    use_log_amount: bool = False,
    use_cyclic_time: bool = False,
) -> list[str]:
    """Return the ordered list of feature column names for the given config.

    Parameters
    ----------
    use_log_amount:
        Include ``log_amount`` in the feature set.
    use_cyclic_time:
        Include ``time_sin`` and ``time_cos`` in the feature set.

    Returns
    -------
    list[str]
        Feature column names.  V1..V28 are always included.
    """
    cols = list(FEATURE_COLS)
    if use_log_amount:
        cols.append("log_amount")
    if use_cyclic_time:
        cols.extend(["time_sin", "time_cos"])
    return cols


def engineer_features(
    df: pd.DataFrame,
    use_log_amount: bool = False,
    use_cyclic_time: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """Apply all requested feature engineering transformations.

    Parameters
    ----------
    df:
        Input dataframe.
    use_log_amount:
        Whether to add the log-amount feature.
    use_cyclic_time:
        Whether to add cyclic time features.

    Returns
    -------
    df_out : pd.DataFrame
        Dataframe with additional engineered columns.
    feature_cols : list[str]
        Ordered list of feature column names to use for modelling.
    """
    df_out = df.copy()
    if use_log_amount:
        df_out, _ = add_log_amount(df_out)
    if use_cyclic_time:
        df_out, _ = add_cyclic_time(df_out)
    feature_cols = build_feature_list(use_log_amount, use_cyclic_time)
    return df_out, feature_cols
