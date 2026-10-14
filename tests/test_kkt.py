"""
tests/test_kkt.py — Tests for KKT condition verification (kkt.py).
"""


import numpy as np

from convexguard.kkt import (
    SV_BOUND,
    SV_MARGIN,
    SV_NON,
    classify_support_vectors,
    kkt_violations,
)


class TestClassifySupportVectors:
    def test_all_non_sv(self):
        alpha = np.zeros(5)
        C = np.ones(5)
        labels = classify_support_vectors(alpha, C)
        assert (labels == SV_NON).all()

    def test_all_bound_sv(self):
        C = np.array([1.0, 2.0, 0.5])
        alpha = C.copy()
        labels = classify_support_vectors(alpha, C)
        assert (labels == SV_BOUND).all()

    def test_all_margin_sv(self):
        C = np.ones(4) * 2.0
        alpha = np.ones(4) * 1.0  # 0 < alpha < C
        labels = classify_support_vectors(alpha, C)
        assert (labels == SV_MARGIN).all()

    def test_mixed(self):
        C = np.array([1.0, 1.0, 1.0])
        alpha = np.array([0.0, 0.5, 1.0])
        labels = classify_support_vectors(alpha, C)
        assert labels[0] == SV_NON
        assert labels[1] == SV_MARGIN
        assert labels[2] == SV_BOUND

    def test_class_dependent_c(self):
        """With different C per sample, bound is determined per-sample."""
        C = np.array([2.0, 1.0])
        alpha = np.array([1.0, 1.0])  # first is margin SV, second is bound SV
        labels = classify_support_vectors(alpha, C)
        assert labels[0] == SV_MARGIN  # alpha=1 < C=2
        assert labels[1] == SV_BOUND  # alpha=1 = C=1


class TestKktViolationsStructure:
    def test_return_keys(self, toy_2d):
        """kkt_violations must return all required keys."""
        X, y = toy_2d["X"], toy_2d["y"]
        w = np.ones(2)
        b = 0.0
        alpha = np.zeros(6)
        C = np.ones(6)

        result = kkt_violations(X, y, w, b, alpha, C)
        required_keys = {
            "stationarity_w_err",
            "stationarity_sum_ay",
            "dual_feasibility_violations",
            "complementary_slackness",
            "sv_labels",
            "all_pass",
        }
        assert required_keys <= set(result.keys())

    def test_complementary_slackness_shape(self, toy_2d):
        X, y = toy_2d["X"], toy_2d["y"]
        result = kkt_violations(X, y, np.zeros(2), 0.0, np.zeros(6), np.ones(6))
        assert result["complementary_slackness"].shape == (6,)

    def test_sv_labels_shape(self, toy_2d):
        X, y = toy_2d["X"], toy_2d["y"]
        result = kkt_violations(X, y, np.zeros(2), 0.0, np.zeros(6), np.ones(6))
        assert result["sv_labels"].shape == (6,)

    def test_all_pass_is_bool(self, toy_2d):
        X, y = toy_2d["X"], toy_2d["y"]
        result = kkt_violations(X, y, np.zeros(2), 0.0, np.zeros(6), np.ones(6))
        assert isinstance(result["all_pass"], bool)
