import os
import pandas as pd
import numpy as np

def run_correlation_audit(csv_path=None):
    if csv_path is None:
        # Resolves path dynamically relative to this script's directory
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        csv_path = os.path.join(base_dir, "data", "synthetic", "dataset.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    df = pd.read_csv(csv_path)
    
    # Exclude non-feature metadata columns
    ignore_cols = ["session_id", "window_id", "timestamp_start", "timestamp_end", "cheating_type", "split", "label"]
    feature_cols = [c for c in df.columns if c not in ignore_cols and np.issubdtype(df[c].dtype, np.number)]

    correlations = {}
    for col in feature_cols:
        corr = df[col].corr(df["label"])
        correlations[col] = float(corr)

    corr_df = pd.DataFrame(list(correlations.items()), columns=["feature", "correlation_with_label"])
    corr_df["abs_correlation"] = corr_df["correlation_with_label"].abs()
    corr_df = corr_df.sort_values(by="abs_correlation", ascending=False).reset_index(drop=True)

    print("=== Feature-Label Correlation Audit ===")
    print(corr_df[["feature", "correlation_with_label"]].to_string(index=False))
    
    # Check for shortcut learning risk
    high_corr = corr_df[corr_df["abs_correlation"] >= 0.85]
    if len(high_corr) > 0:
        print("\n[WARNING] Features with near-perfect separation detected (|r| >= 0.85):")
        for _, row in high_corr.iterrows():
            print(f" - {row['feature']}: r = {row['correlation_with_label']:.4f}")
    else:
        print("\n[PASS] No single feature trivially separates the label (|r| < 0.85 for all features).")

    return corr_df

if __name__ == "__main__":
    run_correlation_audit()