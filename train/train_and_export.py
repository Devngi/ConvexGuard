"""
train/train_and_export.py
=========================
Single command to train the SVM and export everything the site needs.

Usage:
    python train/train_and_export.py

Output:
    site/data.js   —  window.DATA = { model, eda, metrics, verification, examples }

The file is a plain JS assignment, not JSON, so index.html works by double-click
with no server and no fetch() errors.
"""


import json
import logging
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
)
from sklearn.svm import SVC, LinearSVC

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from convexguard.data import drop_duplicates, load_raw
from convexguard.duality import primal_objective
from convexguard.kkt import kkt_violations
from convexguard.metrics import (
    evaluate,
    safe_average_precision_score,
    safe_precision_recall_curve,
)
from convexguard.preprocess import apply_scaler, fit_scaler, get_Xy, stratified_split
from convexguard.solvers.primal_subgradient import PrimalSubgradientSVM
from convexguard.threshold import find_threshold_fbeta

CSV_PATH = REPO / "data" / "creditcard.csv"
OUT_JS   = REPO / "site" / "data.js"

# ── Training hyper-parameters ──────────────────────────────────────────────────
C_NEG       = 1.0
C_RATIO     = 10.0
C_POS       = C_NEG * C_RATIO
MAX_EPOCHS  = 100
RANDOM_SEED = 42
BETA        = 2.0   # F-beta for threshold


# ─────────────────────────────────────────────────────────────────────────────
# Helper: safe float
# ─────────────────────────────────────────────────────────────────────────────
def _f(v) -> float:
    v = float(v)
    if math.isnan(v) or math.isinf(v):
        return 0.0
    return round(v, 6)


def _arr(a) -> list:
    return [_f(x) for x in a]


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA
# ─────────────────────────────────────────────────────────────────────────────
log.info("Loading %s …", CSV_PATH)
raw = load_raw(CSV_PATH)
df, dedup = drop_duplicates(raw)
df = df.copy()

# 30 model features: V1 to V28, LogAmount and Hour
df["LogAmount"] = np.log1p(df["Amount"].clip(lower=0))
df["Hour"] = (df["Time"] // 3600 % 24).astype(int)


FEATURE_COLS = [f"V{i}" for i in range(1, 29)] + ["LogAmount", "Hour"]
log.info("Features (%d): %s", len(FEATURE_COLS), FEATURE_COLS)

df_train, df_val, df_test = stratified_split(df, random_state=RANDOM_SEED)
df_test_raw = df_test.copy()
scaler = fit_scaler(df_train, FEATURE_COLS)
df_train = apply_scaler(df_train, scaler, FEATURE_COLS)
df_val   = apply_scaler(df_val,   scaler, FEATURE_COLS)
df_test  = apply_scaler(df_test,  scaler, FEATURE_COLS)

X_train, y_train = get_Xy(df_train, FEATURE_COLS)
X_val,   y_val   = get_Xy(df_val,   FEATURE_COLS)
X_test,  y_test  = get_Xy(df_test,  FEATURE_COLS)

y_train_bin = (y_train > 0).astype(int)
y_val_bin   = (y_val   > 0).astype(int)
y_test_bin  = (y_test  > 0).astype(int)

n_fraud = int((y_test_bin == 1).sum())
n_legit = int((y_test_bin == 0).sum())

# ─────────────────────────────────────────────────────────────────────────────
# 2. TRAIN PRIMARY MODEL
# ─────────────────────────────────────────────────────────────────────────────
log.info("Training primary SVM (C_ratio=%.0f, epochs=%d) …", C_RATIO, MAX_EPOCHS)
svm = PrimalSubgradientSVM(C_pos=C_POS, C_neg=C_NEG, max_epochs=MAX_EPOCHS,
                            random_state=RANDOM_SEED)
svm.fit(X_train, y_train)

scores_val  = svm.decision_function(X_val)
threshold, _ = find_threshold_fbeta(y_val_bin, scores_val, beta=BETA)
log.info("Threshold (F-beta=%.1f): %.4f", BETA, threshold)
scores_test = svm.decision_function(X_test)
m = evaluate(y_test_bin, scores_test, threshold=threshold)
log.info("Test  PR-AUC=%.4f  P=%.4f  R=%.4f  F2=%.4f", m["pr_auc"], m["precision"], m["recall"], m.get("f2", 0.0))

w   = svm.coef_.flatten()
b   = float(svm.intercept_)
n_f = len(FEATURE_COLS)


# ─────────────────────────────────────────────────────────────────────────────
# 3. MODEL DATA — for JS
# ─────────────────────────────────────────────────────────────────────────────
model_data = {
    "w":            _arr(w),
    "b":            _f(b),
    "mean":         _arr(scaler.mean_),
    "scale":        _arr(scaler.scale_),
    "feature_names": FEATURE_COLS,
    "threshold":    _f(threshold),
    "C_pos":        _f(C_POS),
    "C_neg":        _f(C_NEG),
    "C_ratio":      _f(C_RATIO),
    "max_epochs":   MAX_EPOCHS,
    "n_features":   n_f,
}


# ─────────────────────────────────────────────────────────────────────────────
# 4. EDA DATA
# ─────────────────────────────────────────────────────────────────────────────
log.info("Computing EDA …")
raw_clean = df  # after dedup, before split (all rows)

n_total     = int(dedup["rows_after"])
n_fraud_all = int(raw_clean["Class"].sum())
n_dup       = int(dedup["duplicates_dropped"])
fraud_pct   = round(n_fraud_all / n_total * 100, 4)

# Amount distribution by class (log bins)
amounts_fraud  = raw_clean.loc[raw_clean["Class"] == 1, "Amount"].values
amounts_legit  = raw_clean.loc[raw_clean["Class"] == 0, "Amount"].values
bins = np.logspace(np.log10(max(0.01, amounts_legit.min())),
                   np.log10(amounts_legit.max() + 1), 40)
fraud_hist,  _ = np.histogram(amounts_fraud, bins=bins)
legit_hist,  _ = np.histogram(amounts_legit, bins=bins)
fraud_tiny = float((amounts_fraud <= 1.0).mean() * 100)

# Fraud rate by hour
by_hour = raw_clean.groupby("Hour")["Class"].agg(["sum", "count"])
fraud_rate_hour = (by_hour["sum"] / by_hour["count"] * 100).round(3).to_dict()

# Feature correlation with Class
correlations = {}
for col in FEATURE_COLS:
    correlations[col] = round(float(raw_clean[col].corr(raw_clean["Class"])), 5)

# Per-class histograms for each feature (30 features × 30 bins)
feature_histograms = {}
for col in FEATURE_COLS:
    vals_f = raw_clean.loc[raw_clean["Class"] == 1, col].values
    vals_l = raw_clean.loc[raw_clean["Class"] == 0, col].values
    all_vals = np.concatenate([vals_f, vals_l])
    lo, hi = float(np.percentile(all_vals, 1)), float(np.percentile(all_vals, 99))
    feat_bins = np.linspace(lo, hi, 31)
    hf, _ = np.histogram(vals_f, bins=feat_bins)
    hl, _ = np.histogram(vals_l, bins=feat_bins)
    centers = ((feat_bins[:-1] + feat_bins[1:]) / 2).tolist()
    feature_histograms[col] = {
        "centers": [round(c, 4) for c in centers],
        "fraud": hf.tolist(),
        "legit": hl.tolist(),
    }

eda_data = {
    "snapshot": {
        "n_rows":         n_total,
        "n_fraud":        n_fraud_all,
        "fraud_pct":      fraud_pct,
        "n_duplicates":   n_dup,
        "n_missing":      0,
    },
    "amount_hist": {
        "bin_edges":   [round(float(x), 4) for x in bins],
        "fraud_counts": fraud_hist.tolist(),
        "legit_counts": legit_hist.tolist(),
        "fraud_tiny_pct": round(fraud_tiny, 1),
    },
    "fraud_rate_by_hour": {str(k): v for k, v in sorted(fraud_rate_hour.items())},
    "correlations":       dict(sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)),
    "feature_histograms": feature_histograms,
}


# ─────────────────────────────────────────────────────────────────────────────
# 5. METRICS / RESULTS DATA
# ─────────────────────────────────────────────────────────────────────────────
log.info("Computing metrics and baselines …")

# PR + threshold grid for primary SVM
pr_prec, pr_rec, pr_thresh = safe_precision_recall_curve(y_test_bin, scores_test)
pr_auc = safe_average_precision_score(y_test_bin, scores_test)

# Decimate PR curve for fast browser chart rendering (~300 points)
if len(pr_prec) > 300:
    step = max(1, len(pr_prec) // 300)
    sub_idx = np.arange(0, len(pr_prec), step)
    if sub_idx[-1] != len(pr_prec) - 1:
        sub_idx = np.append(sub_idx, len(pr_prec) - 1)
    pr_prec_export = pr_prec[sub_idx]
    pr_rec_export = pr_rec[sub_idx]
else:
    pr_prec_export = pr_prec
    pr_rec_export = pr_rec

# Threshold slider grid (200 points)
t_min = float(scores_test.min())
t_max = float(scores_test.max())
grid_thresholds = np.linspace(t_min, t_max, 200)
slider_grid = []
for t in grid_thresholds:
    preds = (scores_test >= t).astype(int)
    tp = int(((preds == 1) & (y_test_bin == 1)).sum())
    fp = int(((preds == 1) & (y_test_bin == 0)).sum())
    fn = int(((preds == 0) & (y_test_bin == 1)).sum())
    tn = int(((preds == 0) & (y_test_bin == 0)).sum())
    p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2*p*r/(p+r) if (p+r) > 0 else 0.0
    slider_grid.append({"t": round(float(t), 5), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                        "p": round(p, 4), "r": round(r, 4), "f1": round(f1, 4)})

# Baselines
def _train_eval(clf, X_tr, y_tr, X_te, y_te_bin, thresh=None):
    clf.fit(X_tr, y_tr)
    if hasattr(clf, "decision_function"):
        sc = clf.decision_function(X_te)
    else:
        sc = clf.predict_proba(X_te)[:, 1]
    prauc = safe_average_precision_score(y_te_bin, sc)
    if thresh is None:
        t, _ = find_threshold_fbeta(y_te_bin, sc, beta=2.0)
    else:
        t = thresh
    preds = (sc >= t).astype(int)
    tp = int(((preds == 1) & (y_te_bin == 1)).sum())
    fp = int(((preds == 1) & (y_te_bin == 0)).sum())
    fn = int(((preds == 0) & (y_te_bin == 1)).sum())
    p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return {"pr_auc": round(prauc, 4), "precision": round(p, 4),
            "recall": round(r, 4), "false_alarms": fp}

# Unweighted own SVM
svm_uw = PrimalSubgradientSVM(C_pos=C_NEG, C_neg=C_NEG, max_epochs=MAX_EPOCHS,
                               random_state=RANDOM_SEED)
svm_uw.fit(X_train, y_train)
res_uw = _train_eval(svm_uw, X_train, y_train, X_test, y_test_bin)

# sklearn LinearSVC
lsvc = LinearSVC(C=C_NEG, max_iter=2000, random_state=RANDOM_SEED, class_weight="balanced")
res_lsvc = _train_eval(lsvc, X_train, y_train, X_test, y_test_bin)

# Logistic Regression
lr = LogisticRegression(C=C_NEG, max_iter=2000, random_state=RANDOM_SEED, class_weight="balanced")
res_lr = _train_eval(lr, X_train, y_train, X_test, y_test_bin)

# Primary weighted SVM result
preds_main = (scores_test >= threshold).astype(int)
tp_m = int(((preds_main == 1) & (y_test_bin == 1)).sum())
fp_m = int(((preds_main == 1) & (y_test_bin == 0)).sum())
fn_m = int(((preds_main == 0) & (y_test_bin == 1)).sum())
p_m  = tp_m / (tp_m + fp_m) if (tp_m + fp_m) > 0 else 0.0
r_m  = tp_m / (tp_m + fn_m) if (tp_m + fn_m) > 0 else 0.0
res_main = {"pr_auc": round(pr_auc, 4), "precision": round(p_m, 4),
            "recall": round(r_m, 4), "false_alarms": fp_m}

metrics_data = {
    "pr_curve": {
        "precision": [round(float(p), 5) for p in pr_prec_export.tolist()],
        "recall":    [round(float(r), 5) for r in pr_rec_export.tolist()],
    },
    "threshold":     _f(threshold),
    "pr_auc":        round(pr_auc, 4),
    "score_min":     _f(t_min),
    "score_max":     _f(t_max),
    "slider_grid":   slider_grid,

    "comparisons": [
        {"name": "Logistic Regression",       **res_lr},
        {"name": "sklearn LinearSVC",         **res_lsvc},
        {"name": "Own SVM (unweighted)",      **res_uw},
        {"name": "Own SVM (C_ratio=10)",      **res_main},
    ],
    # Score histogram (for Check a Transaction)
    "score_hist": {
        "fraud": _arr(svm.decision_function(X_test)[y_test_bin == 1]),
        "legit": _arr(svm.decision_function(X_test)[y_test_bin == 0]),
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# 6. VERIFICATION DATA (KKT, primal obj, dual obj, duality gap, weight diff)
# ─────────────────────────────────────────────────────────────────────────────
log.info("Computing verification (KKT + duality gap) …")

# Benchmark verification subset
MAX_KKT = 2000
idx = np.random.default_rng(RANDOM_SEED).choice(len(X_train), min(MAX_KKT, len(X_train)), replace=False)
X_kkt = X_train[idx]
y_kkt = y_train[idx]
C_kkt = np.where(y_kkt == 1, C_POS, C_NEG)

# Fit exact SVC on subset to obtain dual variables
svc = SVC(kernel="linear", C=C_NEG, class_weight={1: C_RATIO, -1: 1})
svc.fit(X_kkt, y_kkt)

alpha_kkt = np.zeros(len(X_kkt))
alpha_kkt[svc.support_] = np.abs(svc.dual_coef_[0])
w_kkt = svc.coef_[0]
b_kkt = float(svc.intercept_[0])

kkt = kkt_violations(X_kkt, y_kkt, w_kkt, b_kkt, alpha_kkt, C_kkt, tol=1e-3)
kkt_max = float(kkt["complementary_slackness"].max())

# Primal & dual objective on this problem
P_val = primal_objective(X_kkt, y_kkt, w_kkt, b_kkt, C_kkt)
ya = alpha_kkt * y_kkt
D_val = float(alpha_kkt.sum() - 0.5 * np.dot(ya @ X_kkt, ya @ X_kkt))
gap = abs(P_val - D_val)

# Compare our trained weights against sklearn LinearSVC on full training set
sk_lsvc = LinearSVC(C=C_NEG, max_iter=2000, random_state=RANDOM_SEED, class_weight={1: C_RATIO, -1: 1})
sk_lsvc.fit(X_train, y_train)
w_sk = sk_lsvc.coef_.flatten()
weight_diff = float(np.linalg.norm(w - w_sk))

verification_data = {
    "kkt_max_violation": round(kkt_max, 6),
    "kkt_pass":          bool(kkt_max < 0.01),
    "primal_objective":  round(P_val, 4),
    "dual_objective":    round(D_val, 4),
    "duality_gap":       round(gap, 6),
    "gap_pass":          bool(gap / (abs(P_val) + 1e-10) < 0.05),
    "weight_diff_vs_sklearn": round(weight_diff, 4),
    "weight_diff_pass":  bool(weight_diff < 1.0),
}


# ─────────────────────────────────────────────────────────────────────────────
# 7. EXAMPLES DATA (5 fraud + 5 legit pre-loaded examples for the form)
# ─────────────────────────────────────────────────────────────────────────────
log.info("Picking examples …")

fraud_idx = np.where(y_test_bin == 1)[0]
legit_idx = np.where(y_test_bin == 0)[0]

# Sort by confidence
fraud_scores_all = scores_test[fraud_idx]
legit_scores_all = scores_test[legit_idx]
top_fraud = fraud_idx[np.argsort(fraud_scores_all)[::-1][:5]]
top_legit = legit_idx[np.argsort(legit_scores_all)[:5]]

def _example(idx: int, label: str) -> dict:
    features = {FEATURE_COLS[j]: round(float(df_test_raw[FEATURE_COLS[j]].iloc[idx]), 8)
                for j in range(n_f)}
    features["Amount"] = round(float(df_test_raw["Amount"].iloc[idx]), 2)
    features["Hour"] = int(round(float(df_test_raw["Hour"].iloc[idx])))
    score = _f(scores_test[idx])
    verdict = "FRAUD" if scores_test[idx] >= threshold else "SAFE"
    return {"label": label, "score": score, "verdict": verdict, "features": features}

examples_data = {
    "fraud": [_example(i, "FRAUD") for i in top_fraud],
    "legit": [_example(i, "SAFE")  for i in top_legit],
}


# ─────────────────────────────────────────────────────────────────────────────
# 8. WRITE site/data.js
# ─────────────────────────────────────────────────────────────────────────────
log.info("Writing %s …", OUT_JS)

combined = {
    "model":        model_data,
    "eda":          eda_data,
    "metrics":      metrics_data,
    "verification": verification_data,
    "examples":     examples_data,
}

js_payload = json.dumps(combined, ensure_ascii=False, separators=(",", ":"))
OUT_JS.write_text(f"window.DATA={js_payload};\n", encoding="utf-8")

size_kb = OUT_JS.stat().st_size / 1024
log.info("Done — site/data.js  %.1f KB", size_kb)
log.info("Scores on 2 examples (Python):")
log.info("  Fraud[0]: %.6f   (verdict=%s)", examples_data["fraud"][0]["score"], examples_data["fraud"][0]["verdict"])
log.info("  Legit[0]: %.6f   (verdict=%s)", examples_data["legit"][0]["score"], examples_data["legit"][0]["verdict"])
log.info("")
log.info("Next step:  open site/index.html")
