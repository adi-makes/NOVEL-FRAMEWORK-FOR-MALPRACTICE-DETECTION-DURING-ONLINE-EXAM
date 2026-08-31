from typing import Dict, Any, Tuple
import numpy as np

def calculate_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    num_bins: int = 10,
) -> Dict[str, Any]:
    """
    Calculate full set of evaluation metrics cleanly handling edge cases.
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    y_prob = np.asarray(y_prob, dtype=np.float32)

    # Sanity checks for ROC-AUC / PR-AUC
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)

    if n_pos == 0 or n_neg == 0:
        auc_roc = float("nan")
        pr_auc = float("nan")
    else:
        from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
        auc_roc = float(roc_auc_score(y_true, y_prob))
        prec_arr, rec_arr, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = float(auc(rec_arr, prec_arr))

    # Binary thresholding metrics
    y_pred = (y_prob >= threshold).astype(np.int32)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    # Calibration metrics
    brier = float(np.mean((y_prob - y_true) ** 2))

    # ECE
    bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    n_samples = len(y_true)
    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper) if i < num_bins - 1 else (y_prob >= bin_lower) & (y_prob <= bin_upper)
        bin_size = np.sum(in_bin)
        if bin_size > 0:
            bin_acc = np.mean(y_true[in_bin])
            bin_conf = np.mean(y_prob[in_bin])
            ece += (bin_size / n_samples) * np.abs(bin_acc - bin_conf)

    return {
        "auc_roc": auc_roc,
        "pr_auc": pr_auc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
        "fnr": fnr,
        "brier": brier,
        "ece": float(ece),
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "f1",
) -> float:
    best_thresh = 0.5
    best_score = -1.0
    thresholds = np.linspace(0.05, 0.95, 91)
    for t in thresholds:
        m = calculate_metrics(y_true, y_prob, threshold=t)
        score = m.get(metric, 0.0)
        if not np.isnan(score) and score > best_score:
            best_score = score
            best_thresh = float(t)
    return best_thresh
