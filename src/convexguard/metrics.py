"""
metrics.py — Evaluation metrics for imbalanced fraud detection.

Primary metric: PR-AUC (Precision-Recall Area Under Curve).
Accuracy and ROC-AUC are NOT reported as headline metrics because they are
misleading under extreme class imbalance (0.17% fraud rate).

Bootstrap confidence intervals are reported because the test set contains
only ~70 fraud samples, making point estimates noisy.

Cost-based metric:
    total_cost = FN * cost_fn + FP * cost_fp
where ``cost_fn`` is the cost of a missed fraud and ``cost_fp`` the cost of
a false alarm.  Both are user-configurable in the web app.
"""


import logging

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)


def evaluate(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.0,
    beta: float = 2.0,
    cost_fn: float = 10.0,
    cost_fp: float = 1.0,
) -> dict:
    """Compute a full evaluation report.

    Parameters
    ----------
    y_true : np.ndarray, shape (n_samples,), values in {0, 1} or {-1, +1}
    y_score : np.ndarray, shape (n_samples,)
        Raw decision scores (not probabilities).
    threshold : float
        Decision threshold applied to y_score.
    beta : float
        Beta for F-beta score (default 2 = weight recall over precision).
    cost_fn : float
        Cost of a false negative (missed fraud).
    cost_fp : float
        Cost of a false positive (false alarm).

    Returns
    -------
    dict with keys: pr_auc, precision, recall, f1, fbeta, tn, fp, fn, tp,
                    total_cost, threshold.
    """
    # Normalise labels to {0, 1}
    y_bin = (y_true > 0).astype(int)

    pr_auc = float(average_precision_score(y_bin, y_score))
    y_pred = (y_score >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_bin, y_pred, labels=[0, 1]).ravel()
    prec = float(precision_score(y_bin, y_pred, zero_division=0))
    rec = float(recall_score(y_bin, y_pred, zero_division=0))
    f1 = float(f1_score(y_bin, y_pred, zero_division=0))
    fb = float(fbeta_score(y_bin, y_pred, beta=beta, zero_division=0))
    total_cost = int(fn) * cost_fn + int(fp) * cost_fp

    return {
        "pr_auc": pr_auc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        f"f{beta:.0f}": fb,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "total_cost": total_cost,
        "threshold": threshold,
    }


def bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    random_state: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap confidence interval for a scalar metric.

    Parameters
    ----------
    y_true : np.ndarray
    y_score : np.ndarray
    metric_fn : callable
        Function(y_true_sub, y_score_sub) → float.
    n_bootstrap : int
        Number of bootstrap resamples.
    ci : float
        Coverage probability (e.g., 0.95).
    random_state : int

    Returns
    -------
    point_est, lower, upper : float
    """
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    estimates = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        try:
            val = metric_fn(y_true[idx], y_score[idx])
            estimates.append(val)
        except Exception:
            pass
    estimates = np.array(estimates)
    alpha = (1.0 - ci) / 2.0
    lower = float(np.nanpercentile(estimates, 100 * alpha))
    upper = float(np.nanpercentile(estimates, 100 * (1 - alpha)))
    point_est = float(metric_fn(y_true, y_score))
    return point_est, lower, upper


def pr_auc_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    random_state: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap CI for PR-AUC. Returns (point_estimate, lower, upper)."""
    y_bin = (y_true > 0).astype(int)

    def _metric(yt, ys):
        # Guard against degenerate resample (all one class)
        if yt.sum() == 0 or yt.sum() == len(yt):
            return float("nan")
        return average_precision_score(yt, ys)

    return bootstrap_ci(y_bin, y_score, _metric, n_bootstrap, ci, random_state)
