"""
evaluator.py
Person 4 — the core pipeline: load predictions.csv -> compute metrics per model -> save results.
Usage:
    python evaluator.py --predictions results/fake_predictions.csv --out results/results.json
On Day 3, Person 1/4 simply point --predictions at the real model output file (same schema:
model,label,prediction) and this script needs ZERO changes.
"""

import argparse
import json
import pandas as pd

from metrics import compute_all_metrics


def load_predictions(path):
    df = pd.read_csv(path)
    required_cols = {"model", "label", "prediction"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"predictions file must contain columns {required_cols}, got {df.columns.tolist()}")
    return df


def evaluate_all_models(df, threshold=0.5):
    """Returns a dict: {model_name: metrics_dict}"""
    results = {}
    for model_name, group in df.groupby("model"):
        metrics = compute_all_metrics(group["label"].values, group["prediction"].values, threshold)
        results[model_name] = metrics
    return results


def results_to_table(results):
    """Converts nested dict to a flat pandas DataFrame matching the paper's Table 2 format."""
    rows = []
    for model_name, m in results.items():
        row = {"Model": model_name}
        row.update(m)
        rows.append(row)
    table = pd.DataFrame(rows)
    # Order columns to match project's required table: Model | AUC | F1 | Precision | Recall | FPR | FNR
    ordered_cols = ["Model", "AUC", "F1", "Precision", "Recall", "FPR", "FNR", "TP", "FP", "TN", "FN"]
    table = table[[c for c in ordered_cols if c in table.columns]]
    table = table.sort_values("AUC", ascending=False).reset_index(drop=True)
    return table


def extract_errors(df, model_name, n=5):
    """
    E6 — Error analysis. Returns up to n false positives and n false negatives
    for a given model, for manual inspection/classification of WHY they happened.
    """
    sub = df[df["model"] == model_name].copy()
    sub["pred_bin"] = (sub["prediction"] >= 0.5).astype(int)
    false_positives = sub[(sub["label"] == 0) & (sub["pred_bin"] == 1)].head(n)
    false_negatives = sub[(sub["label"] == 1) & (sub["pred_bin"] == 0)].head(n)
    return false_positives, false_negatives


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default="results/fake_predictions.csv")
    parser.add_argument("--out", default="results/results.json")
    parser.add_argument("--table_out", default="results/results_table.csv")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    df = load_predictions(args.predictions)
    results = evaluate_all_models(df, args.threshold)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    table = results_to_table(results)
    table.to_csv(args.table_out, index=False)

    print("Results table (sorted by AUC):\n")
    print(table.to_string(index=False))
    print(f"\nSaved: {args.out}")
    print(f"Saved: {args.table_out}")


if __name__ == "__main__":
    main()
