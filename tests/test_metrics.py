"""Unit tests for metrics and PR-AUC calculations."""

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, precision_recall_curve

from convexguard.metrics import (
    evaluate,
    safe_average_precision_score,
    safe_precision_recall_curve,
)


def test_safe_pr_matches_sklearn_on_small_n():
    """Verify exact equivalence to scikit-learn on datasets where N < 32,768."""
    rng = np.random.default_rng(42)
    for n in [50, 500, 2000]:
        y = rng.binomial(1, 0.05, n)
        if y.sum() == 0:
            y[0] = 1
        s = rng.standard_normal(n)

        sk_p, sk_r, sk_th = precision_recall_curve(y, s)
        sk_ap = average_precision_score(y, s)

        my_p, my_r, my_th = safe_precision_recall_curve(y, s)
        my_ap = safe_average_precision_score(y, s)

        np.testing.assert_allclose(my_ap, sk_ap, atol=1e-10)
        np.testing.assert_allclose(my_p, sk_p, atol=1e-10)
        np.testing.assert_allclose(my_r, sk_r, atol=1e-10)
        np.testing.assert_allclose(my_th, sk_th, atol=1e-10)


def test_large_n_buffer_reuse_immunity():
    """Verify that datasets with N >= 32,768 do not suffer from the Python 3.14 buffer-reuse bug.

    In Python 3.14 + NumPy, scikit-learn's `precision_recall_curve` mutates `tps` in-place
    during `ps = tps + fps`, producing PR-AUC = 1.0 even for random noise.
    """
    n = 35000
    y = np.zeros(n, dtype=int)
    y[:70] = 1  # 0.2% positive rate
    rng = np.random.default_rng(42)
    s = rng.standard_normal(n)

    ap = safe_average_precision_score(y, s)
    # Random guess AP should be approximately the prevalence (0.002), definitely not 1.0
    assert ap < 0.05, f"Expected random AP near 0.002, got {ap}"


def test_evaluate_returns_realistic_pr_auc():
    """Verify evaluate dict has valid non-trivial PR-AUC."""
    n = 1000
    y = np.zeros(n, dtype=int)
    y[:50] = 1
    # Partially correlated scores
    rng = np.random.default_rng(42)
    s = y * 2.0 + rng.standard_normal(n)

    res = evaluate(y, s, threshold=0.0)
    assert 0.0 < res["pr_auc"] < 1.0
    assert "precision" in res
    assert "recall" in res
