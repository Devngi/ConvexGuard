"""
data.py — Dataset loading and deduplication.

Handles:
  - Loading the raw Kaggle credit-card fraud CSV.
  - Dropping exact duplicate rows (1,081 duplicates in the full dataset,
    leaving 473 fraud rows after dedup).
  - Reporting before/after counts for auditability.

Dataset facts (ULB Credit Card Fraud):
  Rows: 284,807  Columns: 31 (Time, V1..V28, Amount, Class)
  Frauds: 492 (0.173%)  Missing values: 0
"""


import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Paths — resolved by walking up from this file to the repo root
# (robust whether the package is installed in-tree or in site-packages)
# ---------------------------------------------------------------------------
def _find_repo_root() -> Path:
    """Walk up from this file until we find a directory containing 'data/'."""
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "data").is_dir() and (parent / "pyproject.toml").is_file():
            return parent
    # Fallback: assume standard src-layout installed in-tree
    return here.parents[3]


_REPO_ROOT = _find_repo_root()
DATA_DIR = _REPO_ROOT / "data"
SAMPLE_CSV = DATA_DIR / "sample_creditcard.csv"
FULL_CSV = DATA_DIR / "creditcard.csv"

# Column names
TARGET_COL = "Class"
TIME_COL = "Time"
AMOUNT_COL = "Amount"
FEATURE_COLS = [f"V{i}" for i in range(1, 29)]
ALL_COLS = [TIME_COL] + FEATURE_COLS + [AMOUNT_COL, TARGET_COL]


def load_raw(path: Path | str | None = None) -> pd.DataFrame:
    """Load the raw CSV from *path*.

    Falls back to the sample CSV if *path* is None or the full file is absent.

    Parameters
    ----------
    path:
        Explicit path to a CSV file.  If None, uses FULL_CSV if it exists,
        otherwise falls back to SAMPLE_CSV.

    Returns
    -------
    pd.DataFrame
        Raw dataframe with columns as delivered by Kaggle.
    """
    if path is None:
        path = FULL_CSV if FULL_CSV.exists() else SAMPLE_CSV
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            "Place creditcard.csv in the data/ folder, then run: "
            "python train/train_and_export.py"
        )
    logger.info("Loading dataset from %s", path)
    df = pd.read_csv(path)
    logger.info("Loaded %d rows, %d columns", len(df), df.shape[1])
    return df


def drop_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Remove exact duplicate rows and return a report dict.

    Parameters
    ----------
    df:
        Raw dataframe (output of :func:`load_raw`).

    Returns
    -------
    df_clean : pd.DataFrame
        Dataframe with duplicates removed.
    report : dict
        Keys: ``rows_before``, ``rows_after``, ``duplicates_dropped``,
        ``fraud_before``, ``fraud_after``.
    """
    rows_before = len(df)
    fraud_before = int(df[TARGET_COL].sum())

    df_clean = df.drop_duplicates()

    rows_after = len(df_clean)
    fraud_after = int(df_clean[TARGET_COL].sum())
    duplicates_dropped = rows_before - rows_after

    report = {
        "rows_before": rows_before,
        "rows_after": rows_after,
        "duplicates_dropped": duplicates_dropped,
        "fraud_before": fraud_before,
        "fraud_after": fraud_after,
    }
    logger.info(
        "Dedup: %d → %d rows (%d dropped). Fraud: %d → %d.",
        rows_before,
        rows_after,
        duplicates_dropped,
        fraud_before,
        fraud_after,
    )
    return df_clean, report
