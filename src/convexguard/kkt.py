"""
kkt.py — KKT condition verification for the soft-margin SVM.

The KKT conditions for the soft-margin SVM are:

  Stationarity:
    (S1)  w = Σ_i α_i y_i x_i
    (S2)  Σ_i α_i y_i = 0
    (S3)  C_i = α_i + μ_i  (and μ_i ≥ 0, so 0 ≤ α_i ≤ C_i)

  Primal feasibility:
    (P1)  y_i(w·x_i + b) ≥ 1 − ξ_i
    (P2)  ξ_i ≥ 0

  Dual feasibility:
    (D1)  α_i ≥ 0
    (D2)  μ_i = C_i − α_i ≥ 0

  Complementary slackness:
    (CS1)  α_i [ y_i(w·x_i + b) − 1 + ξ_i ] = 0
    (CS2)  μ_i ξ_i = 0

Support vector interpretation:
  α_i = 0                 →  correctly classified beyond the margin (non-SV)
  0 < α_i < C_i           →  on the margin (free / margin SV)
  α_i = C_i               →  inside margin or misclassified (bound SV)

This module is also used in the web app Optimisation Insights page.
"""


import numpy as np

SV_NON = "non_sv"  # alpha = 0
SV_MARGIN = "margin_sv"  # 0 < alpha < C
SV_BOUND = "bound_sv"  # alpha = C


def classify_support_vectors(
    alpha: np.ndarray,
    C: np.ndarray,
    tol: float = 1e-4,
) -> np.ndarray:
    """Classify each sample as non-SV, margin SV, or bound SV.

    Parameters
    ----------
    alpha : np.ndarray, shape (n_samples,)
        Dual multipliers.
    C : np.ndarray, shape (n_samples,)
        Per-sample penalty (C_i).
    tol : float
        Numerical tolerance for boundary comparisons.

    Returns
    -------
    np.ndarray of str, shape (n_samples,)
        Each element is one of ``SV_NON``, ``SV_MARGIN``, ``SV_BOUND``.
    """
    labels = np.full(len(alpha), SV_NON, dtype=object)
    labels[alpha > tol] = SV_MARGIN
    labels[alpha >= C - tol] = SV_BOUND
    return labels


def kkt_violations(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    b: float,
    alpha: np.ndarray,
    C: np.ndarray,
    tol: float = 1e-3,
) -> dict:
    """Compute per-sample KKT violations and overall pass/fail summary.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
    y : np.ndarray, shape (n_samples,), values in {+1, −1}
    w : np.ndarray, shape (n_features,)
        Primal weight vector.
    b : float
        Bias term.
    alpha : np.ndarray, shape (n_samples,)
        Dual multipliers.
    C : np.ndarray, shape (n_samples,)
        Per-sample penalties.
    tol : float
        Violation tolerance.

    Returns
    -------
    dict with keys:
        ``stationarity_w_err`` : float   ‖w − Σ α_i y_i x_i‖ (S1 violation)
        ``stationarity_sum_ay`` : float  |Σ α_i y_i| (S2 violation)
        ``dual_feasibility_violations`` : int  count of α < 0 or α > C
        ``complementary_slackness`` : np.ndarray  per-sample |CS1| violation
        ``sv_labels`` : np.ndarray  per-sample SV type string
        ``all_pass`` : bool  True if all violations < tol
    """
    f = X @ w + b  # decision values
    margins = y * f  # functional margins

    # Compute slack ξ_i = max(0, 1 - y_i f_i)
    xi = np.maximum(0.0, 1.0 - margins)

    # S1: w = Σ α_i y_i x_i
    w_reconstructed = (alpha * y) @ X
    s1_err = float(np.linalg.norm(w - w_reconstructed))

    # S2: Σ α_i y_i = 0
    s2_err = float(abs(np.dot(alpha, y)))

    # D1 + S3: 0 ≤ α_i ≤ C_i
    df_violations = int(np.sum((alpha < -tol) | (alpha > C + tol)))

    # CS1: α_i (y_i f_i - 1 + ξ_i) = 0
    cs1 = np.abs(alpha * (margins - 1.0 + xi))

    sv_labels = classify_support_vectors(alpha, C, tol=tol)

    all_pass = s1_err < tol and s2_err < tol and df_violations == 0 and float(cs1.max()) < tol

    return {
        "stationarity_w_err": s1_err,
        "stationarity_sum_ay": s2_err,
        "dual_feasibility_violations": df_violations,
        "complementary_slackness": cs1,
        "sv_labels": sv_labels,
        "all_pass": all_pass,
    }
