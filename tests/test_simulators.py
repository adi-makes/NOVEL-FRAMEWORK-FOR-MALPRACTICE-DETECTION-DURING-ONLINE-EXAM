import os
import pandas as pd
from simulators.gaze_simulator import GazeSimulator
from simulators.mouse_simulator import MouseSimulator
from simulators.environment_simulator import EnvironmentSimulator
from simulators.exam_simulator import ExamSimulator
from simulators.generate_dataset import generate_day1_test_dataset

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
    
    assert res_honest["phone_detected"] == 0
    assert res_cheat["phone_detected"] == 1

def test_exam_simulator():
    exam = ExamSimulator()
    session = exam.generate_session(session_id="test_001", duration_windows=10, cheating_type="phone")
    
    assert len(session) == 10
    assert session[0]["window_id"] == "test_001_w000"
    assert "label" in session[0]

def test_dataset_generation():
    output_path = "data/synthetic/dataset_day1_sample.csv"
    generate_day1_test_dataset(num_sessions=5, seed=42)
    
    assert os.path.exists(output_path)
    df = pd.read_csv(output_path)
    assert len(df) == 5 * 18
    assert set(df["split"].unique()).issubset({"train", "val", "test"})
    assert df.isnull().sum().sum() == 0  # No missing values

if __name__ == "__main__":
    test_gaze_simulator()
    test_mouse_simulator()
    test_environment_simulator()
    test_exam_simulator()
    test_dataset_generation()
    print("\n------------------------------------")
    print(">>> ALL SIMULATOR TESTS PASSED! <<<")
    print("------------------------------------\n")