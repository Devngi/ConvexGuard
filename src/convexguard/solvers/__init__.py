"""
solvers/ — From-scratch SVM solver implementations.

All solvers implement the soft-margin SVM with class-dependent penalties:

    Primal:
        min_{w,b,ξ}  0.5‖w‖² + C_pos Σ_{i∈fraud} ξ_i + C_neg Σ_{i∈legit} ξ_i
        s.t.  y_i(w·x_i + b) ≥ 1 − ξ_i,   ξ_i ≥ 0

Labels y ∈ {+1 (fraud), −1 (legit)}.

Available solvers
-----------------
PrimalSubgradientSVM   primal_subgradient.py
"""

from convexguard.solvers.primal_subgradient import PrimalSubgradientSVM

__all__ = [
    "PrimalSubgradientSVM",
]
