"""
ablation.py
Person 4 — E3/E5: does every stream contribute? Leave-one-stream-out comparison.
Takes the already-computed results table (from evaluator.py) and reshapes it
into the ablation view: single streams -> pairs -> all three.
"""

import pandas as pd
from baselines import SINGLE_MODALITY_MODELS, ABLATION_MODELS


def build_ablation_table(results_table_path="results/results_table.csv"):
    df = pd.read_csv(results_table_path)
    order = SINGLE_MODALITY_MODELS + ABLATION_MODELS
    subset = df[df["Model"].isin(order)].copy()
    subset["Model"] = pd.Categorical(subset["Model"], categories=order, ordered=True)
    subset = subset.sort_values("Model")
    return subset


def check_hypothesis_h3(ablation_table):
    """
    H3: Removing any one modality reduces performance vs all_three.
    Returns a dict of {comparison: auc_drop} — positive = all_three is better (H3 supported).
    """
    if "all_three" not in ablation_table["Model"].values:
        return {"error": "all_three not found in results — run evaluator.py first"}

    all_three_auc = ablation_table.loc[ablation_table["Model"] == "all_three", "AUC"].values[0]
    drops = {}
    for pair_model in ["gaze_mouse", "gaze_environment", "mouse_environment"]:
        if pair_model in ablation_table["Model"].values:
            pair_auc = ablation_table.loc[ablation_table["Model"] == pair_model, "AUC"].values[0]
            drops[f"all_three_vs_{pair_model}"] = round(float(all_three_auc - pair_auc), 4)
    return drops


if __name__ == "__main__":
    table = build_ablation_table()
    print("Ablation table:\n", table.to_string(index=False))
    print("\nH3 check (positive = all_three wins, supports hypothesis):")
    print(check_hypothesis_h3(table))
