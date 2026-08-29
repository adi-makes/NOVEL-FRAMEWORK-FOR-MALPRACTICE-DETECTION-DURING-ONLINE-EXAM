import numpy as np

class GazeSimulator:
    """Simulates eye-gaze and head-pose features over a 10-second window."""

    def simulate_window(self, is_cheating=False, cheating_type="none"):
        if not is_cheating or cheating_type in ["copy_paste"]:
            # Honest baseline: looking mostly at the screen
            return {
                "fixation_duration_mean": float(np.clip(np.random.normal(1.8, 0.3), 0.2, 5.0)),
                "fixation_count": int(np.clip(np.random.poisson(6), 1, 15)),
                "saccade_velocity_mean": float(np.clip(np.random.normal(180, 25), 50, 400)),
                "gaze_deviation": float(np.clip(np.random.normal(0.08, 0.03), 0.0, 1.0)),
                "gaze_confidence": float(np.clip(np.random.normal(0.95, 0.03), 0.7, 1.0)),
                "head_yaw": float(np.clip(np.random.normal(0.0, 4.0), -45.0, 45.0)),
                "head_pitch": float(np.clip(np.random.normal(-2.0, 3.0), -45.0, 45.0)),
            }
        else:
            # Cheating: off-screen glance, note/phone inspection
            return {
                "fixation_duration_mean": float(np.clip(np.random.normal(3.2, 0.8), 0.5, 8.0)),
                "fixation_count": int(np.clip(np.random.poisson(3), 0, 10)),
                "saccade_velocity_mean": float(np.clip(np.random.normal(320, 60), 100, 600)),
                "gaze_deviation": float(np.clip(np.random.normal(0.55, 0.15), 0.1, 1.0)),
                "gaze_confidence": float(np.clip(np.random.normal(0.82, 0.08), 0.4, 1.0)),
                "head_yaw": float(np.clip(np.random.normal(25.0, 10.0), -60.0, 60.0)),
                "head_pitch": float(np.clip(np.random.normal(-15.0, 8.0), -60.0, 60.0)),
            }