import pandas as pd
import numpy as np
from simulators.exam_simulator import ExamSimulator

def generate_day1_test_dataset(num_sessions=20, seed=42):
    """Generates synthetic exam sessions split strictly at the session level."""
    np.random.seed(seed)
    simulator = ExamSimulator()
    all_data = []

    # Cheating types pool
    types_pool = ["none", "phone", "notes", "copy_paste", "external_assistance"]
    # 50% honest, 50% distributed cheating
    types_weights = [0.5, 0.15, 0.15, 0.1, 0.1]

    session_types = np.random.choice(types_pool, size=num_sessions, p=types_weights)

    for i, cheat_type in enumerate(session_types):
        s_id = f"sess_{i:03d}"
        session_rows = simulator.generate_session(session_id=s_id, duration_windows=18, cheating_type=cheat_type)
        all_data.extend(session_rows)

    df = pd.DataFrame(all_data)

    # Session-level train/val/test split (60% / 20% / 20%)
    session_ids = list(df["session_id"].unique())
    np.random.shuffle(session_ids)

    n_total = len(session_ids)
    train_ids = session_ids[:int(0.6 * n_total)]
    val_ids = session_ids[int(0.6 * n_total):int(0.8 * n_total)]
    test_ids = session_ids[int(0.8 * n_total):]

    df["split"] = "train"
    df.loc[df["session_id"].isin(val_ids), "split"] = "val"
    df.loc[df["session_id"].isin(test_ids), "split"] = "test"

    output_path = "data/synthetic/dataset_day1_sample.csv"
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} 10-second windows across {num_sessions} sessions saved to: {output_path}")
    print(df.groupby(["split", "label"]).size())

if __name__ == "__main__":
    generate_day1_test_dataset()