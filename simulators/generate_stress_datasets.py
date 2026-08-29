import os
import json
import pandas as pd
import numpy as np
from .exam_simulator import ExamSimulator

def create_stress_test_suites(output_dir=None, seed=42):
    np.random.seed(seed)
    
    if output_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(base_dir, "data", "stress_tests")

    os.makedirs(output_dir, exist_ok=True)
    simulator = ExamSimulator()

    # Base test cohort: 40 sessions (20 honest, 20 cheating across modalities)
    base_scenarios = (
        ["none"] * 20 +
        ["phone"] * 5 +
        ["notes"] * 5 +
        ["copy_paste"] * 5 +
        ["external_assistance"] * 5
    )

    def generate_base_sessions():
        records = []
        for i, sc in enumerate(base_scenarios):
            s_id = f"stress_sess_{i:03d}"
            rows = simulator.generate_session(session_id=s_id, duration_windows=18, cheating_type=sc)
            records.extend(rows)
        return pd.DataFrame(records)

    # --- Test A: Noisy Gaze (Degraded lighting / low confidence) ---
    df_a = generate_base_sessions()
    df_a["gaze_confidence"] = np.clip(df_a["gaze_confidence"] * np.random.uniform(0.3, 0.6, len(df_a)), 0.1, 1.0)
    df_a["gaze_deviation"] = np.clip(df_a["gaze_deviation"] + np.random.normal(0.25, 0.15, len(df_a)), 0.0, 1.0)
    df_a["head_yaw"] = np.clip(df_a["head_yaw"] + np.random.normal(0.0, 18.0, len(df_a)), -60.0, 60.0)
    df_a.to_csv(os.path.join(output_dir, "test_a_noisy_gaze.csv"), index=False)

    # --- Test B: Mouse Noise (Erratic movements / jitter) ---
    df_b = generate_base_sessions()
    df_b["cursor_velocity_mean"] = np.clip(df_b["cursor_velocity_mean"] + np.random.uniform(100, 250, len(df_b)), 10.0, 800.0)
    df_b["cursor_velocity_std"] = np.clip(df_b["cursor_velocity_std"] + np.random.uniform(40, 90, len(df_b)), 5.0, 300.0)
    df_b["velocity_spike_ratio"] = np.clip(df_b["velocity_spike_ratio"] + np.random.uniform(0.15, 0.35, len(df_b)), 0.0, 1.0)
    df_b.to_csv(os.path.join(output_dir, "test_b_mouse_noise.csv"), index=False)

    # --- Test C: Environment Sensor Failure (Total camera drop) ---
    df_c = generate_base_sessions()
    df_c["phone_detected"] = 0
    df_c["phone_confidence"] = 0.0
    df_c["notes_detected"] = 0
    df_c["extra_person_count"] = 0
    df_c["suspicious_objects_count"] = 0
    df_c.to_csv(os.path.join(output_dir, "test_c_environment_failure.csv"), index=False)

    # --- Test D: Single-Modality Cheating (Isolated signals) ---
    records_d = []
    # Phone detected, but gaze and mouse are completely normal
    for i in range(40):
        s_id = f"single_mod_{i:03d}"
        is_cheat = (i >= 20)
        for w in range(18):
            gaze_feats = simulator.gaze_sim.simulate_window(is_cheating=False)
            mouse_feats = simulator.mouse_sim.simulate_window(is_cheating=False)
            if is_cheat:
                env_feats = {
                    "phone_detected": 1,
                    "phone_confidence": float(np.random.uniform(0.85, 0.98)),
                    "notes_detected": 0,
                    "extra_person_count": 0,
                    "suspicious_objects_count": 1
                }
            else:
                env_feats = simulator.env_sim.simulate_window(is_cheating=False)
            
            row = {
                "session_id": s_id,
                "window_id": f"{s_id}_w{w:03d}",
                "timestamp_start": w * 10,
                "timestamp_end": (w + 1) * 10,
                **gaze_feats,
                **mouse_feats,
                **env_feats,
                "label": 1 if is_cheat else 0,
                "cheating_type": "phone" if is_cheat else "none",
                "split": "test"
            }
            records_d.append(row)
    df_d = pd.DataFrame(records_d)
    df_d.to_csv(os.path.join(output_dir, "test_d_single_modality.csv"), index=False)

    # --- Test E: Silent Cheating (Subtle off-screen cues, zero tab switch/mouse change) ---
    records_e = []
    for i in range(40):
        s_id = f"silent_sess_{i:03d}"
        is_cheat = (i >= 20)
        for w in range(18):
            if is_cheat:
                # Slight gaze deflection only; mouse and env appear honest
                gaze_feats = {
                    "fixation_duration_mean": float(np.random.normal(2.2, 0.4)),
                    "fixation_count": 5,
                    "saccade_velocity_mean": float(np.random.normal(210, 30)),
                    "gaze_deviation": float(np.random.normal(0.24, 0.05)),
                    "gaze_confidence": 0.88,
                    "head_yaw": float(np.random.normal(8.0, 3.0)),
                    "head_pitch": float(np.random.normal(-4.0, 2.0))
                }
            else:
                gaze_feats = simulator.gaze_sim.simulate_window(is_cheating=False)

            mouse_feats = simulator.mouse_sim.simulate_window(is_cheating=False)
            env_feats = simulator.env_sim.simulate_window(is_cheating=False)

            row = {
                "session_id": s_id,
                "window_id": f"{s_id}_w{w:03d}",
                "timestamp_start": w * 10,
                "timestamp_end": (w + 1) * 10,
                **gaze_feats,
                **mouse_feats,
                **env_feats,
                "label": 1 if is_cheat else 0,
                "cheating_type": "external_assistance" if is_cheat else "none",
                "split": "test"
            }
            records_e.append(row)
    df_e = pd.DataFrame(records_e)
    df_e.to_csv(os.path.join(output_dir, "test_e_silent_cheating.csv"), index=False)

    print("=== Successfully Generated Day 4 Stress-Test Datasets ===")
    print(f"Directory: {output_dir}")
    print(" - test_a_noisy_gaze.csv (720 windows)")
    print(" - test_b_mouse_noise.csv (720 windows)")
    print(" - test_c_environment_failure.csv (720 windows)")
    print(" - test_d_single_modality.csv (720 windows)")
    print(" - test_e_silent_cheating.csv (720 windows)")

if __name__ == "__main__":
    create_stress_test_suites()