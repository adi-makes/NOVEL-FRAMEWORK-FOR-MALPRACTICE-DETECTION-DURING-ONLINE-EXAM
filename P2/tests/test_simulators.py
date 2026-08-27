import os
import json
import pandas as pd
from ..simulators.gaze_simulator import GazeSimulator
from ..simulators.mouse_simulator import MouseSimulator
from ..simulators.environment_simulator import EnvironmentSimulator
from ..simulators.exam_simulator import ExamSimulator

def test_gaze_simulator():
    gaze = GazeSimulator()
    res_honest = gaze.simulate_window(is_cheating=False)
    res_cheat = gaze.simulate_window(is_cheating=True, cheating_type="phone")
    
    assert "gaze_deviation" in res_honest
    assert "head_yaw" in res_cheat
    assert 0.0 <= res_honest["gaze_deviation"] <= 1.0

def test_mouse_simulator():
    mouse = MouseSimulator()
    res_honest = mouse.simulate_window(is_cheating=False)
    res_cheat = mouse.simulate_window(is_cheating=True, cheating_type="copy_paste")
    
    assert "cursor_velocity_mean" in res_honest
    assert "tab_switch_count" in res_cheat
    assert res_cheat["tab_switch_count"] >= 0

def test_environment_simulator():
    env = EnvironmentSimulator()
    res_honest = env.simulate_window(is_cheating=False)
    res_cheat = env.simulate_window(is_cheating=True, cheating_type="phone")
    
    assert "phone_detected" in res_honest
    assert "suspicious_objects_count" in res_cheat

def test_exam_simulator():
    exam = ExamSimulator()
    session = exam.generate_session(session_id="test_001", duration_windows=10, cheating_type="phone")
    
    assert len(session) == 10
    assert session[0]["window_id"] == "test_001_w000"
    assert "label" in session[0]

def test_primary_dataset_integrity():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, "data", "synthetic", "dataset.csv")
    meta_path = os.path.join(base_dir, "data", "synthetic", "metadata.json")
    report_path = os.path.join(base_dir, "data", "synthetic", "data_quality_report.md")
    config_path = os.path.join(base_dir, "data", "synthetic", "generation_config.json")
    feature_desc_path = os.path.join(base_dir, "data", "synthetic", "feature_description.md")
    
    assert os.path.exists(csv_path), "dataset.csv missing!"
    assert os.path.exists(meta_path), "metadata.json missing!"
    assert os.path.exists(report_path), "data_quality_report.md missing!"
    assert os.path.exists(config_path), "generation_config.json missing!"
    assert os.path.exists(feature_desc_path), "feature_description.md missing!"
    
    df = pd.read_csv(csv_path)
    assert len(df) == 3600, f"Expected 3600 rows, found {len(df)}"
    assert df.isnull().sum().sum() == 0, "Found NaNs in dataset"
    assert len(df["session_id"].unique()) == 200, "Expected 200 unique sessions"
    assert set(df["split"].unique()) == {"train", "val", "test"}
    assert set(df["label"].unique()) == {0, 1}
    
    # Check session-level zero leakage
    train_sessions = set(df[df["split"] == "train"]["session_id"])
    val_sessions = set(df[df["split"] == "val"]["session_id"])
    test_sessions = set(df[df["split"] == "test"]["session_id"])
    
    assert len(train_sessions.intersection(val_sessions)) == 0, "Leakage between train and val"
    assert len(train_sessions.intersection(test_sessions)) == 0, "Leakage between train and test"
    assert len(val_sessions.intersection(test_sessions)) == 0, "Leakage between val and test"

def test_stress_test_suites():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    stress_dir = os.path.join(base_dir, "data", "stress_tests")
    expected_files = [
        "test_a_noisy_gaze.csv",
        "test_b_mouse_noise.csv",
        "test_c_environment_failure.csv",
        "test_d_single_modality.csv",
        "test_e_silent_cheating.csv"
    ]
    for fname in expected_files:
        fpath = os.path.join(stress_dir, fname)
        assert os.path.exists(fpath), f"Missing stress dataset: {fname}"
        df = pd.read_csv(fpath)
        assert len(df) == 720, f"Expected 720 rows in {fname}, found {len(df)}"
        assert df.isnull().sum().sum() == 0, f"Found NaNs in {fname}"

if __name__ == "__main__":
    test_gaze_simulator()
    test_mouse_simulator()
    test_environment_simulator()
    test_exam_simulator()
    test_primary_dataset_integrity()
    test_stress_test_suites()
    print("\n-----------------------------------------------------------")
    print(">>> ALL DAY 5 INTEGRITY AUDITS PASSED: DATASET FROZEN! <<<")
    print("-----------------------------------------------------------\n")