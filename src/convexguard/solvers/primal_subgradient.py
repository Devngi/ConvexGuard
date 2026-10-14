"""
primal_subgradient.py — Pegasos-style primal subgradient SVM.

Minimises the regularised hinge loss (equivalent to the soft-margin primal):

    f(w, b) = 0.5 λ ‖w‖² + (1/n) Σ_i C_i · max(0, 1 − y_i(w·x_i + b))   ... (1)

where λ = lambda_reg controls regularisation strength (separate from C).

**Subgradient of the hinge term** with respect to w:
    If  y_i(w·x_i + b) < 1  (sample inside or beyond the margin):
        ∂/∂w  C_i · max(0, 1 − y_i f_i) = −C_i y_i x_i
    Otherwise:
        ∂/∂w = 0

Full mini-batch subgradient (averaged over batch):
    g_w = λ w − (1/|B|) Σ_{i∈B: margin violated} C_i y_i x_i
    g_b =     − (1/|B|) Σ_{i∈B: margin violated} C_i y_i

Step size schedule (Pegasos, Shalev-Shwartz et al. 2007):
    η_t = 1 / (λ · t)

The bias b is NOT regularised (standard practice).

Note on parameter mapping to the spec's primal (P):
    Spec uses:  min 0.5‖w‖² + C_pos Σ xi_i + C_neg Σ xi_j
    This file:  min 0.5 λ ‖w‖² + (1/n) Σ C_i max(0, 1 - y_i f_i)
    With λ = 1 and C_i scaled by n, both are equivalent.
    Default: lambda_reg=1.0, C_pos/C_neg are the spec's values.

References
----------
Shalev-Shwartz, S., Singer, Y., Srebro, N., & Cotter, A. (2011).
Pegasos: Primal estimated sub-gradient solver for SVM.
Mathematical Programming, 127(1), 3–30.
"""


import logging

import numpy as np

logger = logging.getLogger(__name__)


class PrimalSubgradientSVM:
    """Soft-margin SVM trained via mini-batch subgradient descent (Pegasos).

    Scales to the full ~284 k row dataset (linear kernel, O(nd) per step).

    Parameters
    ----------
    C_pos : float
        Penalty weight for fraud samples (y = +1).
    C_neg : float
        Penalty weight for legit samples (y = −1).
    lambda_reg : float
        L2 regularisation coefficient (controls margin width).
        Default 1.0 — increase to reduce overfitting.
    max_epochs : int
        Number of passes over the training data.
    batch_size : int
        Mini-batch size; 1 = pure stochastic, n = full-batch.
    tol : float
        Convergence tolerance on relative objective change between epochs.
    random_state : int
        RNG seed for reproducibility.
    """

    def __init__(
        self,
        C_pos: float = 1.0,
        C_neg: float = 1.0,
        lambda_reg: float = 1.0,
        max_epochs: int = 100,
        batch_size: int = 256,
        tol: float = 1e-5,
        random_state: int = 42,
    ) -> None:
        self.C_pos = C_pos
        self.C_neg = C_neg
        self.lambda_reg = lambda_reg
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.tol = tol
        self.random_state = random_state

        # Set after fit()
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self.history_: list[float] = []  # primal objective per epoch
        self.n_iter_: int = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _C_vector(self, y: np.ndarray) -> np.ndarray:
        """Build per-sample C array from label vector y ∈ {+1, −1}."""
        return np.where(y == 1, self.C_pos, self.C_neg)

    def _primal_objective(
        self,
        X: np.ndarray,
        y: np.ndarray,
        C: np.ndarray,
        w: np.ndarray,
        b: float,
    ) -> float:
        """Compute primal objective.

        P(w,b) = 0.5 λ ‖w‖² + (1/n) Σ_i C_i max(0, 1 − y_i(w·x_i + b))
        """
        margins = y * (X @ w + b)
        hinge = np.maximum(0.0, 1.0 - margins)
        n = len(y)
        return float(0.5 * self.lambda_reg * np.dot(w, w) + np.dot(C, hinge) / n)

    # ------------------------------------------------------------------
    # Public API (sklearn-compatible)
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> PrimalSubgradientSVM:
        """Train the SVM via mini-batch subgradient descent.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Feature matrix (must be standardised before calling).
        y : np.ndarray, shape (n_samples,), values in {+1, −1}
            Labels.

        Returns
        -------
        self
        """
        rng = np.random.default_rng(self.random_state)
        n, d = X.shape
        C = self._C_vector(y)
        lam = float(self.lambda_reg)

        w = np.zeros(d)
        b = 0.0
        global_t = 1  # Pegasos step counter (starts at 1)

        self.history_ = []
        prev_obj = np.inf

        for epoch in range(self.max_epochs):
            idx = rng.permutation(n)

            for start in range(0, n, self.batch_size):
                batch = idx[start : start + self.batch_size]
                X_b = X[batch]
                y_b = y[batch]
                C_b = C[batch]
                m = len(batch)  # actual batch size (last batch may be smaller)

                # Pegasos step size: η_t = 1 / (λ · t)
                eta = 1.0 / (lam * global_t)

                # Identify margin violations: y_i(w·x_i + b) < 1
                margins = y_b * (X_b @ w + b)
                violated = margins < 1.0

                # Subgradient of (1/n) Σ C_i hinge:
                #   g_w = λ w − (1/m) Σ_{violated} C_i y_i x_i
                #   g_b =     − (1/m) Σ_{violated} C_i y_i
                g_w = lam * w
                g_b = 0.0
                if violated.any():
                    scale = 1.0 / m
                    g_w -= scale * ((C_b[violated] * y_b[violated]) @ X_b[violated])
                    g_b -= scale * float(np.dot(C_b[violated], y_b[violated]))

                w = w - eta * g_w
                b = b - eta * g_b
                global_t += 1

            # Full-batch objective at end of epoch
            obj = self._primal_objective(X, y, C, w, b)
            self.history_.append(obj)

            # Convergence check on relative change
            rel_change = abs(prev_obj - obj) / (abs(prev_obj) + 1e-12)
            if rel_change < self.tol and epoch > 5:
                logger.info(
                    "Converged at epoch %d, obj=%.6f (rel_change=%.2e)", epoch, obj, rel_change
                )
                break
            prev_obj = obj

            if epoch % 10 == 0:
                logger.debug("Epoch %3d / %d  obj=%.6f", epoch, self.max_epochs, obj)

        self.coef_ = w
        self.intercept_ = b
        self.n_iter_ = epoch + 1
        logger.info(
            "PrimalSubgradientSVM done. n_iter=%d, final_obj=%.6f",
            self.n_iter_,
            self.history_[-1],
        )
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Signed distance to the decision boundary: w·x + b.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples,)
            Positive values → fraud side, negative → legit side.
        """
        if self.coef_ is None:
            raise RuntimeError("Call fit() before decision_function().")
        return X @ self.coef_ + self.intercept_

    def predict(self, X: np.ndarray, threshold: float = 0.0) -> np.ndarray:
        """Predict class labels in {+1, −1}.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        threshold : float
            Tune on validation set to trade precision vs recall.

        Returns
        -------
        np.ndarray, shape (n_samples,), values in {+1, −1}
        """
        return np.where(self.decision_function(X) >= threshold, 1.0, -1.0)
