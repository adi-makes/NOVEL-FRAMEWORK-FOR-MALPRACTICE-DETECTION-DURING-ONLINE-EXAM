import os
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.exam_proctoring.data.dataset import get_project_root
from src.exam_proctoring.evaluation.metrics import calculate_metrics, find_optimal_threshold

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

@torch.no_grad()
def get_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: str,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    model.eval()
    all_logits = []
    all_labels = []
    all_attentions = []

    for batch in loader:
        gaze, interaction, environment, labels = batch
        gaze = gaze.to(device)
        interaction = interaction.to(device)
        environment = environment.to(device)

        res = model(gaze, interaction, environment, return_attention=True)
        if isinstance(res, tuple):
            logits, attn = res[0], res[1]
            if attn is not None:
                all_attentions.append(attn.cpu().numpy())
        else:
            logits = res

        all_logits.append(logits.cpu().numpy())
        all_labels.append(labels.numpy())

    logits_np = np.concatenate(all_logits)
    labels_np = np.concatenate(all_labels)
    probs_np = 1.0 / (1.0 + np.exp(-logits_np))

    attn_np = np.concatenate(all_attentions, axis=0) if all_attentions else None
    return probs_np, labels_np, attn_np

def evaluate_hypotheses(results_dict: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    att_auc = results_dict["attention_fusion"]["auc_roc"]

    single_aucs = [
        results_dict["gaze_only"]["auc_roc"],
        results_dict["interaction_only"]["auc_roc"],
        results_dict["environment_only"]["auc_roc"],
    ]
    max_single = max(single_aucs)
    
    # H1 check
    if att_auc > max_single:
        h1_status = "SUPPORTED"
    elif att_auc == max_single:
        h1_status = "INCONCLUSIVE"
    else:
        h1_status = "NOT SUPPORTED"

    # H2 check: Attention vs Early and Late
    fusion_aucs = [
        results_dict["early_fusion"]["auc_roc"],
        results_dict["late_fusion"]["auc_roc"],
    ]
    max_fusion = max(fusion_aucs)

    if att_auc > max_fusion:
        h2_status = "SUPPORTED"
    elif att_auc == max_fusion:
        h2_status = "INCONCLUSIVE"
    else:
        h2_status = "NOT SUPPORTED"

    # H3 check: Removing any single stream should reduce performance (competing against leave-one-out models)
    pairwise_aucs = [
        results_dict["gaze_interaction"]["auc_roc"],
        results_dict["gaze_environment"]["auc_roc"],
        results_dict["interaction_environment"]["auc_roc"],
    ]
    max_pairwise = max(pairwise_aucs)

    if att_auc > max_pairwise:
        h3_status = "SUPPORTED"
    elif att_auc == max_pairwise:
        h3_status = "INCONCLUSIVE"
    else:
        h3_status = "NOT SUPPORTED"

    return {
        "H1": {
            "status": h1_status,
            "attention_auc": float(att_auc),
            "single_stream_best_auc": float(max_single),
            "difference": float(att_auc - max_single),
        },
        "H2": {
            "status": h2_status,
            "attention_auc": float(att_auc),
            "fusion_best_auc": float(max_fusion),
            "difference": float(att_auc - max_fusion),
        },
        "H3": {
            "status": h3_status,
            "attention_auc": float(att_auc),
            "pairwise_best_auc": float(max_pairwise),
            "difference": float(att_auc - max_pairwise),
        },
    }

def generate_visualizations(
    results_df: pd.DataFrame,
    predictions_dict: Dict[str, pd.DataFrame],
    stress_results_df: pd.DataFrame,
    figures_dir: Path,
):
    figures_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 1. AUC Comparison Bar Plot
    plt.figure(figsize=(10, 6))
    bars = plt.barh(results_df["model"], results_df["auc_roc"], color=sns.color_palette("viridis", len(results_df)))
    plt.xlabel("ROC-AUC Score")
    plt.title("Model Performance Comparison (ROC-AUC)")
    plt.xlim(0.0, 1.05)
    for bar in bars:
        w = bar.get_width()
        plt.text(w + 0.01, bar.get_y() + bar.get_height()/2, f"{w:.4f}", va='center')
    plt.tight_layout()
    plt.savefig(figures_dir / "auc_comparison.png", dpi=300)
    plt.close()

    # 2. ROC Curves
    plt.figure(figsize=(8, 6))
    for model_key, pred_df in predictions_dict.items():
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, _ = roc_curve(pred_df["label"], pred_df["probability"])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{model_key} (AUC={roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves across Models")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(figures_dir / "roc_curves.png", dpi=300)
    plt.close()

    # 3. PR Curves
    plt.figure(figsize=(8, 6))
    for model_key, pred_df in predictions_dict.items():
        from sklearn.metrics import precision_recall_curve, auc
        p, r, _ = precision_recall_curve(pred_df["label"], pred_df["probability"])
        pr_auc = auc(r, p)
        plt.plot(r, p, label=f"{model_key} (PR-AUC={pr_auc:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(figures_dir / "pr_curves.png", dpi=300)
    plt.close()

    # 4. Confusion Matrix for Attention Fusion
    if "attention_fusion" in predictions_dict:
        att_preds = predictions_dict["attention_fusion"]
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(att_preds["label"], att_preds["prediction"])
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Honest', 'Cheating'], yticklabels=['Honest', 'Cheating'])
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix - Attention Fusion")
        plt.tight_layout()
        plt.savefig(figures_dir / "confusion_matrix_attention.png", dpi=300)
        plt.close()

    # 5. Ablation Comparison Plot
    ablation_keys = ["gaze_interaction", "gaze_environment", "interaction_environment", "attention_fusion"]
    ablation_df = results_df[results_df["model_key"].isin(ablation_keys)]
    plt.figure(figsize=(8, 5))
    sns.barplot(data=ablation_df, x="model", y="auc_roc", palette="magma")
    plt.ylabel("ROC-AUC")
    plt.title("Ablation Study (Stream Combinations vs Attention Fusion)")
    plt.ylim(0.0, 1.05)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(figures_dir / "ablation_comparison.png", dpi=300)
    plt.close()

    # 6. Stress Test Degradation
    if not stress_results_df.empty:
        plt.figure(figsize=(10, 6))
        sns.barplot(data=stress_results_df, x="scenario", y="auc_roc", hue="model")
        plt.title("Stress-Test Performance (ROC-AUC under Distribution Shift)")
        plt.ylabel("ROC-AUC")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(figures_dir / "stress_test_degradation.png", dpi=300)
        plt.close()

    # 7. Calibration Plot for Attention Fusion
    if "attention_fusion" in predictions_dict:
        att_preds = predictions_dict["attention_fusion"]
        y_true = att_preds["label"].values
        y_prob = att_preds["probability"].values
        
        bin_boundaries = np.linspace(0.0, 1.0, 11)
        bin_accs = []
        bin_confs = []
        for i in range(10):
            bl, bu = bin_boundaries[i], bin_boundaries[i+1]
            in_b = (y_prob >= bl) & (y_prob < bu) if i < 9 else (y_prob >= bl) & (y_prob <= bu)
            if np.sum(in_b) > 0:
                bin_accs.append(np.mean(y_true[in_b]))
                bin_confs.append(np.mean(y_prob[in_b]))

        plt.figure(figsize=(6, 6))
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        plt.plot(bin_confs, bin_accs, 's-', label='Attention Fusion')
        plt.xlabel("Mean Predicted Probability")
        plt.ylabel("Fraction of Positives")
        plt.title("Reliability Diagram (Calibration)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figures_dir / "calibration_attention.png", dpi=300)
        plt.close()
