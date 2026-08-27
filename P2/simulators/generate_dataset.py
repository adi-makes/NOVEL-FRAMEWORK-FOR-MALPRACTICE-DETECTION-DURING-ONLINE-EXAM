import os
import json
import pandas as pd
import numpy as np
from simulators.exam_simulator import ExamSimulator

def generate_primary_dataset(output_dir="data/synthetic", seed=42):
    """Generates the primary 200-session dataset for Day 2 with audit reporting."""
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    simulator = ExamSimulator()
    all_data = []

    # Exactly 200 sessions: 100 Honest, 25 Phone, 25 Notes, 25 Copy/Paste, 25 External Assistance
    distribution = (
        ["none"] * 100 +
        ["phone"] * 25 +
        ["notes"] * 25 +
        ["copy_paste"] * 25 +
        ["external_assistance"] * 25
    )

    for i, cheat_type in enumerate(distribution):
        s_id = f"sess_{i:03d}"
        session_rows = simulator.generate_session(session_id=s_id, duration_windows=18, cheating_type=cheat_type)
        all_data.extend(session_rows)

    df = pd.DataFrame(all_data)

    # Leakage-free session split (60% Train / 20% Val / 20% Test)
    session_ids = list(df["session_id"].unique())
    np.random.shuffle(session_ids)

    n_total = len(session_ids)
    train_ids = session_ids[:int(0.6 * n_total)]
    val_ids = session_ids[int(0.6 * n_total):int(0.8 * n_total)]
    test_ids = session_ids[int(0.8 * n_total):]

    df["split"] = "train"
    df.loc[df["session_id"].isin(val_ids), "split"] = "val"
    df.loc[df["session_id"].isin(test_ids), "split"] = "test"

    # 1. Export CSV
    csv_path = os.path.join(output_dir, "dataset.csv")
    df.to_csv(csv_path, index=False)

    # 2. Export metadata.json
    metadata = {
        "dataset_version": "1.0",
        "total_sessions": len(session_ids),
        "total_windows": len(df),
        "window_duration_seconds": 10,
        "splits": {
            "train_sessions": len(train_ids),
            "val_sessions": len(val_ids),
            "test_sessions": len(test_ids)
        },
        "session_distribution": {
            "honest": 100,
            "phone": 25,
            "notes": 25,
            "copy_paste": 25,
            "external_assistance": 25
        },
        "window_label_counts": {str(k): int(v) for k, v in df["label"].value_counts().items()}
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # 3. Export data_quality_report.md
    report_md = f"""# Synthetic Dataset Quality Report (Day 2 Deliverable)

## 1. Summary Statistics
* **Total Sessions:** {len(session_ids)}
* **Total 10-Second Windows:** {len(df)}
* **Session Splits:** Train ({len(train_ids)} sessions), Val ({len(val_ids)} sessions), Test ({len(test_ids)} sessions)
* **Missing / NaN Values:** {int(df.isna().sum().sum())}

## 2. Window Class Balance
* **Honest Windows (label=0):** {int((df['label'] == 0).sum())}
* **Cheating Windows (label=1):** {int((df['label'] == 1).sum())}

## 3. Scenario Distribution
| Modality Scenario | Session Count | Window Count |
|---|---|---|
| Honest (`none`) | 100 | {int((df['cheating_type'] == 'none').sum())} |
| Phone | 25 | {int((df['cheating_type'] == 'phone').sum())} |
| Notes | 25 | {int((df['cheating_type'] == 'notes').sum())} |
| Copy/Paste | 25 | {int((df['cheating_type'] == 'copy_paste').sum())} |
| External Assistance | 25 | {int((df['cheating_type'] == 'external_assistance').sum())} |

## 4. Leakage Verification
* Distinct Train-Val Session Overlap: {len(set(train_ids).intersection(set(val_ids)))}
* Distinct Train-Test Session Overlap: {len(set(train_ids).intersection(set(test_ids)))}
* Distinct Val-Test Session Overlap: {len(set(val_ids).intersection(set(test_ids)))}
"""
    with open(os.path.join(output_dir, "data_quality_report.md"), "w") as f:
        f.write(report_md)

    print(f"Generated 200 sessions ({len(df)} windows) across all modalities.")
    print(f"Artifacts successfully written to '{output_dir}/'.")
    print("\nClass distribution by split:")
    print(df.groupby(["split", "label"]).size())

if __name__ == "__main__":
    generate_primary_dataset()