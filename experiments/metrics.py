"""
metrics.py
Person 4 — core metric functions. Pure functions: (labels, predictions) -> numbers.
No training logic here — this file must work identically whether fed mock or real predictions.
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_curve, precision_recall_curve
)


def to_binary(predictions, threshold=0.5):
    """Convert probability scores to binary predictions at a given threshold."""
    return (np.asarray(predictions) >= threshold).astype(int)


def compute_confusion(labels, predictions, threshold=0.5):
    """Returns TP, FP, TN, FN counts."""
    preds_bin = to_binary(predictions, threshold)
    tn, fp, fn, tp = confusion_matrix(labels, preds_bin, labels=[0, 1]).ravel()
    return {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)}


def fpr_fnr(labels, predictions, threshold=0.5):
    """False Positive Rate and False Negative Rate."""
    c = compute_confusion(labels, predictions, threshold)
    fpr = c["fp"] / (c["fp"] + c["tn"]) if (c["fp"] + c["tn"]) > 0 else 0.0
    fnr = c["fn"] / (c["fn"] + c["tp"]) if (c["fn"] + c["tp"]) > 0 else 0.0
    return fpr, fnr


def compute_all_metrics(labels, predictions, threshold=0.5):
    """
    Master function: given ground-truth binary labels (0=honest, 1=cheating)
    and continuous prediction scores in [0,1], return every metric required
    by the project (AUC, F1, Precision, Recall, FPR, FNR, confusion matrix).
    """
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    preds_bin = to_binary(predictions, threshold)

    # AUC requires both classes present; guard against degenerate mock data
    try:
        auc = roc_auc_score(labels, predictions)
    except ValueError:
        auc = float("nan")

    f1 = f1_score(labels, preds_bin, zero_division=0)
    precision = precision_score(labels, preds_bin, zero_division=0)
    recall = recall_score(labels, preds_bin, zero_division=0)
    fpr, fnr = fpr_fnr(labels, predictions, threshold)
    cm = compute_confusion(labels, predictions, threshold)

    return {
        "AUC": round(float(auc), 4),
        "F1": round(float(f1), 4),
        "Precision": round(float(precision), 4),
        "Recall": round(float(recall), 4),
        "FPR": round(float(fpr), 4),
        "FNR": round(float(fnr), 4),
        "TP": cm["tp"], "FP": cm["fp"], "TN": cm["tn"], "FN": cm["fn"],
    }


def get_roc_curve(labels, predictions):
    """Returns fpr array, tpr array, thresholds — for plotting later."""
    fpr, tpr, thresholds = roc_curve(labels, predictions)
    return fpr, tpr, thresholds


def get_pr_curve(labels, predictions):
    """Returns precision array, recall array, thresholds — for plotting later."""
    precision, recall, thresholds = precision_recall_curve(labels, predictions)
    return precision, recall, thresholds


if __name__ == "__main__":
    # Quick self-test with dummy data so this file can be verified standalone
    rng = np.random.default_rng(42)
    labels = rng.integers(0, 2, 200)
    scores = np.clip(labels * 0.6 + rng.normal(0.3, 0.2, 200), 0, 1)
    result = compute_all_metrics(labels, scores)
    print("Self-test metrics:", result)
