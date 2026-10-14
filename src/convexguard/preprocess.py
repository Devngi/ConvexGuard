"""
preprocess.py — Stratified train/val/test split and feature scaling.

Rules enforced here to prevent data leakage:
  - StandardScaler is fit ONLY on the training split.
  - The same fitted scaler is applied to val and test.
  - Splits are stratified on the Class label with a fixed random seed.

Split ratios (default): 70% train / 15% val / 15% test.
"""


import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from convexguard.data import TARGET_COL

logger = logging.getLogger(__name__)

# Default split sizes
VAL_SIZE = 0.15
TEST_SIZE = 0.15
RANDOM_SEED = 42


def stratified_split(
    df: pd.DataFrame,
    val_size: float = VAL_SIZE,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split *df* into train, val, test, stratified on ``Class``.

    Parameters
    ----------
    df:
        Cleaned dataframe (after deduplication).
    val_size:
        Fraction for validation set (relative to full dataset).
    test_size:
        Fraction for test set (relative to full dataset).
    random_state:
        RNG seed for reproducibility.

    Returns
    -------
    train_df, val_df, test_df : pd.DataFrame
        Three non-overlapping stratified splits.
    """
    y = df[TARGET_COL]
    # First split off the test set
    df_temp, df_test = train_test_split(
        df,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    # Then split val from the remainder
    relative_val = val_size / (1.0 - test_size)
    df_train, df_val = train_test_split(
        df_temp,
        test_size=relative_val,
        stratify=df_temp[TARGET_COL],
        random_state=random_state,
    )
    logger.info(
        "Split sizes — train: %d, val: %d, test: %d",
        len(df_train),
        len(df_val),
        len(df_test),
    )
    for split_name, split_df in [("train", df_train), ("val", df_val), ("test", df_test)]:
        fraud_rate = split_df[TARGET_COL].mean() * 100
        logger.info("  %s fraud rate: %.3f%%", split_name, fraud_rate)
    return df_train, df_val, df_test


def fit_scaler(
    train_df: pd.DataFrame,
    feature_cols: list[str],
) -> StandardScaler:
    """Fit a StandardScaler on *train_df* for the given *feature_cols*.

    The scaler must be fit ONLY on the training set to avoid leakage.

    Parameters
    ----------
    train_df:
        Training split dataframe.
    feature_cols:
        Columns to scale.

    Returns
    -------
    StandardScaler
        Fitted scaler (not yet applied to any split).
    """
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols])
    logger.info("Scaler fit on %d train rows, %d features.", len(train_df), len(feature_cols))
    return scaler


def apply_scaler(
    df: pd.DataFrame,
    scaler: StandardScaler,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Apply a pre-fitted *scaler* to *df*; returns a copy.

    Parameters
    ----------
    df:
        Any split dataframe.
    scaler:
        A scaler that has already been :meth:`fit` on the training set.
    feature_cols:
        Columns to transform (must match those used to fit the scaler).

    Returns
    -------
    pd.DataFrame
        Copy of *df* with *feature_cols* standardised.
    """
    df_out = df.copy()
    df_out[feature_cols] = scaler.transform(df[feature_cols])
    return df_out


def get_Xy(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Extract feature matrix X and label vector y (as +1/-1) from *df*.

    Labels are remapped: Class=1 (fraud) → +1, Class=0 (legit) → -1.
    This sign convention is required by the SVM math.

    Parameters
    ----------
    df:
        Dataframe with feature columns and ``Class`` column.
    feature_cols:
        Ordered list of feature column names.

    Returns
    -------
    X : np.ndarray, shape (n_samples, n_features)
    y : np.ndarray, shape (n_samples,), values in {+1, -1}
    """
    X = df[feature_cols].to_numpy(dtype=np.float64)
    y_raw = df[TARGET_COL].to_numpy(dtype=np.int32)
    y = np.where(y_raw == 1, 1, -1).astype(np.float64)
    return X, y
