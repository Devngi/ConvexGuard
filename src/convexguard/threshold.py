"""
threshold.py — Decision threshold tuning on the validation set.

After training, the decision boundary is at score = 0.  Moving the threshold
allows a precision/recall trade-off.  All threshold choices are made on the
VALIDATION set; the chosen threshold is applied to the test set ONCE.

Three strategies:
  1. ``maximize_fbeta``   — maximise F-beta (default beta=2 to weight recall).
  2. ``target_precision`` — find the highest recall threshold that still
     meets a minimum precision requirement.
  3. ``target_recall``    — find the highest precision threshold that still
     meets a minimum recall requirement.

None of these look at the test set.
"""


import logging

import numpy as np
from sklearn.metrics import fbeta_score, precision_recall_curve

logger = logging.getLogger(__name__)


def find_threshold_fbeta(
    y_val: np.ndarray,
    scores_val: np.ndarray,
    beta: float = 2.0,
    n_thresholds: int = 300,
) -> tuple[float, dict]:
    """Find threshold that maximises F-beta on the validation set.

    Parameters
    ----------
    y_val : np.ndarray, values in {0,1} or {-1,+1}
    scores_val : np.ndarray
        Raw decision scores.
    beta : float
        F-beta parameter (2 = weight recall 2× over precision).
    n_thresholds : int
        Number of candidate thresholds to evaluate.

    Returns
    -------
    best_threshold : float
    info : dict with ``best_fbeta``, ``precision_at_threshold``,
                       ``recall_at_threshold``.
    """
    y_bin = (y_val > 0).astype(int)
    lo, hi = float(scores_val.min()), float(scores_val.max())
    thresholds = np.linspace(lo, hi, n_thresholds)

    best_t, best_fb = thresholds[0], -1.0
    best_prec, best_rec = 0.0, 0.0

    for t in thresholds:
        y_pred = (scores_val >= t).astype(int)
        if y_pred.sum() == 0:
            continue
        fb = float(fbeta_score(y_bin, y_pred, beta=beta, zero_division=0))
        if fb > best_fb:
            best_fb = fb
            best_t = t
            from sklearn.metrics import precision_score, recall_score

            best_prec = float(precision_score(y_bin, y_pred, zero_division=0))
            best_rec = float(recall_score(y_bin, y_pred, zero_division=0))

    logger.info(
        "threshold (F%.0f): %.4f  F%.0f=%.4f  P=%.4f  R=%.4f",
        beta,
        best_t,
        beta,
        best_fb,
        best_prec,
        best_rec,
    )
    return best_t, {
        "best_fbeta": best_fb,
        "precision_at_threshold": best_prec,
        "recall_at_threshold": best_rec,
    }


def find_threshold_target_precision(
    y_val: np.ndarray,
    scores_val: np.ndarray,
    min_precision: float = 0.8,
) -> tuple[float, dict]:
    """Find highest-recall threshold achieving at least *min_precision*.

    Parameters
    ----------
    y_val : np.ndarray
    scores_val : np.ndarray
    min_precision : float
        Minimum acceptable precision on the validation set.

    Returns
    -------
    best_threshold : float
    info : dict
    """
    y_bin = (y_val > 0).astype(int)
    precisions, recalls, thresholds = precision_recall_curve(y_bin, scores_val)
    # thresholds has len = len(precisions) - 1
    valid = precisions[:-1] >= min_precision
    if not valid.any():
        logger.warning(
            "No threshold achieves precision >= %.2f; returning maximum score.", min_precision
        )
        return float(scores_val.max()), {"best_recall": 0.0, "achieved_precision": 0.0}

    best_idx = int(np.argmax(recalls[:-1][valid]))
    actual_idx = np.where(valid)[0][best_idx]
    t = float(thresholds[actual_idx])
    info = {
        "achieved_precision": float(precisions[actual_idx]),
        "best_recall": float(recalls[actual_idx]),
    }
    logger.info(
        "threshold (prec≥%.2f): %.4f  P=%.4f  R=%.4f",
        min_precision,
        t,
        info["achieved_precision"],
        info["best_recall"],
    )
    return t, info


def find_threshold_target_recall(
    y_val: np.ndarray,
    scores_val: np.ndarray,
    min_recall: float = 0.8,
) -> tuple[float, dict]:
    """Find highest-precision threshold achieving at least *min_recall*.

    Parameters
    ----------
    y_val : np.ndarray
    scores_val : np.ndarray
    min_recall : float
        Minimum acceptable recall.

    Returns
    -------
    best_threshold : float
    info : dict
    """
    y_bin = (y_val > 0).astype(int)
    precisions, recalls, thresholds = precision_recall_curve(y_bin, scores_val)
    valid = recalls[:-1] >= min_recall
    if not valid.any():
        logger.warning("No threshold achieves recall >= %.2f; returning minimum score.", min_recall)
        return float(scores_val.min()), {"best_precision": 0.0, "achieved_recall": 0.0}

    best_idx = int(np.argmax(precisions[:-1][valid]))
    actual_idx = np.where(valid)[0][best_idx]
    t = float(thresholds[actual_idx])
    info = {
        "achieved_recall": float(recalls[actual_idx]),
        "best_precision": float(precisions[actual_idx]),
    }
    logger.info(
        "threshold (rec≥%.2f): %.4f  P=%.4f  R=%.4f",
        min_recall,
        t,
        info["best_precision"],
        info["achieved_recall"],
    )
    return t, info
