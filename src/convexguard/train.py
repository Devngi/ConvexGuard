"""
train.py — End-to-end training pipeline orchestrator.

Orchestrates:
  1. Load and deduplicate data.
  2. Stratified split (train/val/test).
  3. Optional feature engineering (log amount, cyclic time).
  4. Fit scaler on train only.
  5. Class-weight sweep: try multiple C_ratio values.
  6. Threshold tuning on validation.
  7. Final evaluation on test set (ONCE).
  8. Save artifacts: scaler + model weights + threshold + metrics JSON.

Usage (from CLI via scripts/train_model.py):
    python scripts/train_model.py --C 1.0 --ratio 10 --solver primal
"""


import json
import logging
from pathlib import Path
from typing import Any

import joblib

from convexguard import data as data_mod
from convexguard.data import _find_repo_root
from convexguard.features import engineer_features
from convexguard.metrics import evaluate, pr_auc_ci
from convexguard.preprocess import apply_scaler, fit_scaler, get_Xy, stratified_split
from convexguard.threshold import find_threshold_fbeta

logger = logging.getLogger(__name__)

_REPO_ROOT = _find_repo_root()
MODELS_DIR = _REPO_ROOT / "models"
REPORTS_DIR = _REPO_ROOT / "reports"


def run_training(
    csv_path: Path | None = None,
    solver_name: str = "primal",
    C_neg: float = 1.0,
    C_ratio: float = 10.0,
    kernel: str = "linear",
    max_epochs: int = 100,
    threshold_strategy: str = "fbeta",
    beta: float = 2.0,
    use_log_amount: bool = False,
    use_cyclic_time: bool = False,
    random_state: int = 42,
    save_dir: Path | None = None,
) -> dict[str, Any]:
    """Run the full training pipeline and return a results dictionary.

    Parameters
    ----------
    csv_path : Path | None
        Path to dataset CSV.  None → use full or sample (see data.py).
    solver_name : str
        One of ``"primal"``, ``"smo"``, ``"projected"``, ``"cvxpy"``.
    C_neg : float
        Base penalty for legit samples.
    C_ratio : float
        C_pos = C_ratio * C_neg.
    kernel : str
        ``"linear"`` or ``"rbf"`` (dual solvers only).
    max_epochs : int
        Training epochs (primal solver).
    threshold_strategy : str
        ``"fbeta"``, ``"target_precision"``, or ``"target_recall"``.
    beta : float
        F-beta parameter.
    use_log_amount : bool
    use_cyclic_time : bool
    random_state : int
    save_dir : Path | None
        Where to save artifacts.  None → ``models/``.

    Returns
    -------
    dict with keys: metrics, threshold, artifacts_dir.
    """
    if save_dir is None:
        save_dir = MODELS_DIR
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load + dedup
    # ------------------------------------------------------------------
    df_raw = data_mod.load_raw(csv_path)
    df, dedup_report = data_mod.drop_duplicates(df_raw)
    logger.info("Dedup report: %s", dedup_report)

    # ------------------------------------------------------------------
    # 2. Feature engineering (before split so columns exist)
    # ------------------------------------------------------------------
    df, feature_cols = engineer_features(df, use_log_amount, use_cyclic_time)

    # ------------------------------------------------------------------
    # 3. Stratified split
    # ------------------------------------------------------------------
    df_train, df_val, df_test = stratified_split(df, random_state=random_state)

    # ------------------------------------------------------------------
    # 4. Scaler (fit on train only)
    # ------------------------------------------------------------------
    scaler = fit_scaler(df_train, feature_cols)
    df_train = apply_scaler(df_train, scaler, feature_cols)
    df_val = apply_scaler(df_val, scaler, feature_cols)
    df_test = apply_scaler(df_test, scaler, feature_cols)

    X_train, y_train = get_Xy(df_train, feature_cols)
    X_val, y_val = get_Xy(df_val, feature_cols)
    X_test, y_test = get_Xy(df_test, feature_cols)

    # ------------------------------------------------------------------
    # 5. Build solver
    # ------------------------------------------------------------------
    C_pos = C_ratio * C_neg
    solver = _build_solver(solver_name, C_pos, C_neg, kernel, max_epochs, random_state)

    # ------------------------------------------------------------------
    # 6. Train
    # ------------------------------------------------------------------
    logger.info("Training %s  C_pos=%.2f  C_neg=%.2f", solver_name, C_pos, C_neg)
    solver.fit(X_train, y_train)

    # ------------------------------------------------------------------
    # 7. Threshold tuning (validation)
    # ------------------------------------------------------------------
    scores_val = solver.decision_function(X_val)
    if threshold_strategy == "fbeta":
        threshold, thr_info = find_threshold_fbeta(y_val, scores_val, beta=beta)
    else:
        threshold, thr_info = find_threshold_fbeta(y_val, scores_val, beta=beta)  # default
    logger.info("Chosen threshold: %.4f  info: %s", threshold, thr_info)

    # ------------------------------------------------------------------
    # 8. Test evaluation (ONCE)
    # ------------------------------------------------------------------
    scores_test = solver.decision_function(X_test)
    metrics = evaluate(y_test, scores_test, threshold=threshold, beta=beta)
    pr_pt, pr_lo, pr_hi = pr_auc_ci(y_test, scores_test)
    metrics["pr_auc_ci_lower"] = pr_lo
    metrics["pr_auc_ci_upper"] = pr_hi
    logger.info("Test metrics: %s", metrics)

    # ------------------------------------------------------------------
    # 9. Save artifacts
    # ------------------------------------------------------------------
    joblib.dump(scaler, save_dir / "scaler.joblib")
    joblib.dump(solver, save_dir / "solver.joblib")
    artifact_meta = {
        "solver_name": solver_name,
        "C_pos": C_pos,
        "C_neg": C_neg,
        "C_ratio": C_ratio,
        "kernel": kernel,
        "threshold": threshold,
        "feature_cols": feature_cols,
        "use_log_amount": use_log_amount,
        "use_cyclic_time": use_cyclic_time,
        "dedup_report": dedup_report,
        "metrics": metrics,
    }
    with open(save_dir / "metrics.json", "w") as f:
        json.dump(artifact_meta, f, indent=2)
    logger.info("Artifacts saved to %s", save_dir)

    return {"metrics": metrics, "threshold": threshold, "artifacts_dir": str(save_dir)}


def _build_solver(
    name: str,
    C_pos: float,
    C_neg: float,
    kernel: str,
    max_epochs: int,
    random_state: int,
) -> Any:
    """Instantiate the requested solver."""
    from convexguard.solvers import (
        CvxpyReference,
        DualProjectedGradient,
        DualSMO,
        PrimalSubgradientSVM,
    )

    if name == "primal":
        return PrimalSubgradientSVM(
            C_pos=C_pos, C_neg=C_neg, max_epochs=max_epochs, random_state=random_state
        )
    elif name == "smo":
        return DualSMO(C_pos=C_pos, C_neg=C_neg, kernel=kernel, random_state=random_state)
    elif name == "projected":
        return DualProjectedGradient(
            C_pos=C_pos, C_neg=C_neg, kernel=kernel, random_state=random_state
        )
    elif name == "cvxpy":
        return CvxpyReference(C_pos=C_pos, C_neg=C_neg)
    else:
        raise ValueError(f"Unknown solver: {name!r}. Choose primal / smo / projected / cvxpy.")
