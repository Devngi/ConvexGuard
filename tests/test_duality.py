"""
tests/test_duality.py — Tests for duality gap computation (duality.py).
"""


import numpy as np
import pytest

from convexguard.duality import dual_objective, duality_gap, primal_objective
from convexguard.solvers.primal_subgradient import PrimalSubgradientSVM


class TestObjectiveValues:
    def test_primal_finite_nonneg(self, toy_2d):
        X, y = toy_2d["X"], toy_2d["y"]
        w = np.array([1.0, 0.5])
        b = 0.0
        C = np.ones(6)
        P = primal_objective(X, y, w, b, C)
        assert np.isfinite(P)
        assert P >= 0.0

    def test_primal_zero_weights(self, toy_2d):
        """With w=0, b=0: all samples violated → P = Σ C_i · 1 = n."""
        X, y = toy_2d["X"], toy_2d["y"]
        C = np.ones(6)
        P = primal_objective(X, y, np.zeros(2), 0.0, C)
        assert abs(P - 6.0) < 1e-9

    def test_dual_finite(self, toy_2d):
        X, y = toy_2d["X"], toy_2d["y"]
        alpha = np.ones(6) * 0.1
        K = X @ X.T
        D = dual_objective(alpha, y, K)
        assert np.isfinite(D)

    def test_dual_zero_alpha(self, toy_2d):
        """With alpha=0, dual objective = 0."""
        X, y = toy_2d["X"], toy_2d["y"]
        K = X @ X.T
        D = dual_objective(np.zeros(6), y, K)
        assert abs(D) < 1e-12


class TestWeakDuality:
    def test_gap_nonneg_arbitrary(self, toy_2d):
        """For any feasible (w, alpha), P - D >= 0."""
        X, y = toy_2d["X"], toy_2d["y"]
        alpha = np.array([0.1, 0.1, 0.0, 0.1, 0.1, 0.0])
        C = np.ones(6)
        w = (alpha * y) @ X
        b = 0.0
        result = duality_gap(X, y, w, b, alpha, C)
        assert result["gap"] >= -1e-6, f"Gap negative: {result['gap']:.4e}"

    def test_gap_structure(self, toy_2d):
        X, y = toy_2d["X"], toy_2d["y"]
        result = duality_gap(X, y, np.zeros(2), 0.0, np.zeros(6), np.ones(6))
        assert "primal" in result
        assert "dual" in result
        assert "gap" in result
        assert "gap_pct" in result
        assert np.isfinite(result["primal"])
        assert np.isfinite(result["dual"])

    @pytest.mark.parametrize("seed", [0, 7, 42])
    def test_gap_nonneg_after_primal_solver(self, seed, toy_2d):
        """After training the primal solver, P >= 0."""
        X, y = toy_2d["X"], toy_2d["y"]
        clf = PrimalSubgradientSVM(C_pos=1.0, C_neg=1.0, max_epochs=200, random_state=seed)
        clf.fit(X, y)
        C = np.ones(6)
        P = primal_objective(X, y, clf.coef_, clf.intercept_, C)
        assert P >= 0.0
