"""
duality.py — Primal / dual objective computation and duality-gap verification.

Strong duality holds for the soft-margin SVM because:
  1. The primal objective is convex (quadratic with PSD Hessian in w, plus
     a positive-weighted sum of hinge losses which are convex).
  2. The constraints are affine (half-spaces and non-negativity), so the
     feasible set is convex.
  3. Slater's condition is satisfied: the interior of the feasible set is
     non-empty (e.g., w=0, b=0, ξ_i=1 for all i is strictly feasible).

Therefore the duality gap P(w,b,ξ) − D(α) = 0 at optimality, and any
positive gap is a rigorous upper bound on sub-optimality.

Functions
---------
primal_objective(X, y, w, b, C)  →  float
dual_objective(alpha, y, K)       →  float
duality_gap(X, y, w, b, alpha, C) →  dict
"""


import numpy as np


def primal_objective(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    b: float,
    C: np.ndarray,
) -> float:
    """Compute the primal SVM objective value.

    P(w, b, ξ*) = 0.5‖w‖² + Σ_i C_i ξ_i*
    where ξ_i* = max(0, 1 − y_i(w·x_i + b)) are the implied slack values.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
    y : np.ndarray, shape (n_samples,), values in {+1, −1}
    w : np.ndarray, shape (n_features,)
    b : float
    C : np.ndarray, shape (n_samples,)
        Per-sample penalty.

    Returns
    -------
    float
        Primal objective value.
    """
    margins = y * (X @ w + b)
    xi = np.maximum(0.0, 1.0 - margins)
    return float(0.5 * np.dot(w, w) + np.dot(C, xi))


def dual_objective(
    alpha: np.ndarray,
    y: np.ndarray,
    K: np.ndarray,
) -> float:
    """Compute the dual SVM objective value.

    D(α) = Σ_i α_i − 0.5 Σ_{ij} α_i α_j y_i y_j K(x_i, x_j)
         = Σ α − 0.5 (α⊙y)ᵀ K (α⊙y)

    Parameters
    ----------
    alpha : np.ndarray, shape (n_samples,)
    y : np.ndarray, shape (n_samples,)
    K : np.ndarray, shape (n_samples, n_samples)
        Kernel matrix.

    Returns
    -------
    float
        Dual objective value.
    """
    ya = alpha * y
    return float(alpha.sum() - 0.5 * ya @ K @ ya)


def duality_gap(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    b: float,
    alpha: np.ndarray,
    C: np.ndarray,
    kernel: str = "linear",
    gamma: float | None = None,
) -> dict:
    """Compute primal obj P, dual obj D, and the gap P − D.

    By weak duality, P ≥ D always.  At optimality, gap = 0.

    Parameters
    ----------
    X, y, w, b, alpha, C :
        As in :func:`primal_objective` / :func:`dual_objective`.
    kernel : str
        ``"linear"`` (default) or ``"rbf"``.
    gamma : float | None
        RBF bandwidth (required if kernel == "rbf").

    Returns
    -------
    dict with keys:
        ``primal``  : float — P(w, b, ξ*)
        ``dual``    : float — D(α)
        ``gap``     : float — P − D  (≥ 0)
        ``gap_pct`` : float — gap / |P| × 100
    """
    if kernel == "linear":
        K = X @ X.T
    elif kernel == "rbf":
        if gamma is None:
            gamma = 1.0 / (X.shape[1] * X.var())
        sq = np.sum(X**2, axis=1, keepdims=True) + np.sum(X**2, axis=1) - 2.0 * (X @ X.T)
        K = np.exp(-gamma * np.maximum(sq, 0.0))
    else:
        raise ValueError(f"Unknown kernel: {kernel!r}")

    P = primal_objective(X, y, w, b, C)
    D = dual_objective(alpha, y, K)
    gap = P - D
    gap_pct = 100.0 * gap / (abs(P) + 1e-12)

    return {
        "primal": P,
        "dual": D,
        "gap": gap,
        "gap_pct": gap_pct,
    }
