"""
tests/test_solvers.py — Tests for the from-scratch PrimalSubgradientSVM solver.
"""


import numpy as np
import pytest
from sklearn.svm import LinearSVC

from convexguard.solvers.primal_subgradient import PrimalSubgradientSVM


class TestPrimalSubgradient:
    def test_fit_toy(self, toy_2d):
        """Solver fits without error and returns self (sklearn API)."""
        clf = PrimalSubgradientSVM(C_pos=1.0, C_neg=1.0, max_epochs=200, random_state=0)
        ret = clf.fit(toy_2d["X"], toy_2d["y"])
        assert ret is clf, "fit() must return self"
        assert clf.coef_ is not None
        assert clf.coef_.shape == (2,)
        assert isinstance(clf.intercept_, float)

    def test_history_non_empty(self, toy_2d):
        clf = PrimalSubgradientSVM(max_epochs=50, random_state=0)
        clf.fit(toy_2d["X"], toy_2d["y"])
        assert len(clf.history_) > 0, "history_ must be populated after fit"

    def test_objective_decreases(self, toy_2d):
        """Primal objective should decrease (or converge) over epochs."""
        clf = PrimalSubgradientSVM(
            C_pos=1.0, C_neg=1.0, max_epochs=200, batch_size=6, random_state=0
        )
        clf.fit(toy_2d["X"], toy_2d["y"])
        first_half = np.mean(clf.history_[: len(clf.history_) // 2])
        second_half = np.mean(clf.history_[len(clf.history_) // 2 :])
        assert (
            second_half <= first_half + 1.0
        ), f"Objective not decreasing: first_half={first_half:.4f}, second_half={second_half:.4f}"

    def test_correct_predictions_toy(self, toy_2d):
        """On a linearly separable toy dataset all predictions should be correct."""
        clf = PrimalSubgradientSVM(
            C_pos=10.0, C_neg=1.0, max_epochs=500, batch_size=6, random_state=0
        )
        clf.fit(toy_2d["X"], toy_2d["y"])
        preds = clf.predict(toy_2d["X"])
        acc = (preds == toy_2d["y"]).mean()
        assert acc == 1.0, f"Expected 100% accuracy on separable toy set, got {acc:.2f}"

    def test_same_objective_different_seeds(self, toy_2d):
        """Multiple random seeds should converge to similar objectives (empirical convexity)."""
        objs = []
        for seed in [0, 7, 42]:
            clf = PrimalSubgradientSVM(
                C_pos=1.0, C_neg=1.0, max_epochs=300, batch_size=6, random_state=seed
            )
            clf.fit(toy_2d["X"], toy_2d["y"])
            objs.append(clf.history_[-1])
        assert max(objs) - min(objs) < 1.0, f"Objectives across seeds diverge too much: {objs}"

    def test_class_dependent_c(self, toy_2d):
        """C_pos > C_neg should run without error and produce finite weights."""
        clf_eq = PrimalSubgradientSVM(C_pos=1.0, C_neg=1.0, max_epochs=300, random_state=0)
        clf_asym = PrimalSubgradientSVM(C_pos=50.0, C_neg=1.0, max_epochs=300, random_state=0)
        clf_eq.fit(toy_2d["X"], toy_2d["y"])
        clf_asym.fit(toy_2d["X"], toy_2d["y"])
        assert np.isfinite(clf_eq.coef_).all()
        assert np.isfinite(clf_asym.coef_).all()
        assert (clf_eq.predict(toy_2d["X"]) == toy_2d["y"]).all()

    def test_decision_function_shape(self, toy_2d):
        clf = PrimalSubgradientSVM(max_epochs=50, random_state=0)
        clf.fit(toy_2d["X"], toy_2d["y"])
        scores = clf.decision_function(toy_2d["X"])
        assert scores.shape == (6,)

    def test_weight_direction_matches_sklearn(self, toy_2d):
        """Weight vector direction should agree with sklearn LinearSVC (cosine > 0)."""
        X, y = toy_2d["X"], toy_2d["y"]
        y_sk = (y + 1) // 2

        clf = PrimalSubgradientSVM(C_pos=1.0, C_neg=1.0, max_epochs=500, random_state=0)
        clf.fit(X, y)

        sk = LinearSVC(C=1.0, max_iter=5000, random_state=0)
        sk.fit(X, y_sk)

        w_ours = clf.coef_ / (np.linalg.norm(clf.coef_) + 1e-12)
        w_sk = sk.coef_[0] / (np.linalg.norm(sk.coef_[0]) + 1e-12)
        cosine = float(np.dot(w_ours, w_sk))
        assert cosine > 0.0, f"Weight directions are anti-correlated: cosine={cosine:.4f}"
