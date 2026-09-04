"""
visualize.py
Person 4 — generates the required figures: ROC graph, PR graph, confusion matrix,
ablation bar graph, model comparison/AUC graph.
Uses matplotlib only (no seaborn dependency) to keep the environment lightweight.
"""

import matplotlib
matplotlib.use("Agg")  # headless — safe for servers/CI
import matplotlib.pyplot as plt
import pandas as pd

from metrics import get_roc_curve, get_pr_curve, compute_confusion


def plot_roc_all_models(df, out_path="figures/roc_curves.png"):
    plt.figure(figsize=(7, 6))
    for model_name, group in df.groupby("model"):
        fpr, tpr, _ = get_roc_curve(group["label"].values, group["prediction"].values)
        plt.plot(fpr, tpr, label=model_name)
    plt.plot([0, 1], [0, 1], "k--", alpha=0.3)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — All Models")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_pr_all_models(df, out_path="figures/pr_curves.png"):
    plt.figure(figsize=(7, 6))
    for model_name, group in df.groupby("model"):
        precision, recall, _ = get_pr_curve(group["label"].values, group["prediction"].values)
        plt.plot(recall, precision, label=model_name)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves — All Models")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_confusion_matrix(df, model_name, out_path=None):
    sub = df[df["model"] == model_name]
    cm = compute_confusion(sub["label"].values, sub["prediction"].values)
    matrix = [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]]

    plt.figure(figsize=(4, 4))
    plt.imshow(matrix, cmap="Blues")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, matrix[i][j], ha="center", va="center")
    plt.xticks([0, 1], ["Pred Honest", "Pred Cheating"])
    plt.yticks([0, 1], ["True Honest", "True Cheating"])
    plt.title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()
    out_path = out_path or f"figures/confusion_{model_name}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_model_comparison(results_table_path="results/results_table.csv", out_path="figures/model_comparison_auc.png"):
    table = pd.read_csv(results_table_path)
    plt.figure(figsize=(8, 5))
    plt.barh(table["Model"], table["AUC"])
    plt.xlabel("AUC")
    plt.title("Model Comparison — AUC")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


if __name__ == "__main__":
    df = pd.read_csv("results/fake_predictions.csv")
    plot_roc_all_models(df)
    plot_pr_all_models(df)
    plot_confusion_matrix(df, "attention_fusion")
    plot_model_comparison()
    print("Figures written to figures/")
