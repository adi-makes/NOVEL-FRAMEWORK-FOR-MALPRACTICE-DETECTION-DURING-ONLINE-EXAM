from .gaze_simulator import GazeSimulator
from .mouse_simulator import MouseSimulator
from .environment_simulator import EnvironmentSimulator

class ExamSimulator:
    """Combines Gaze, Mouse, and Environment streams into 10-second exam windows."""

    def __init__(self):
        self.gaze_sim = GazeSimulator()
        self.mouse_sim = MouseSimulator()
        self.env_sim = EnvironmentSimulator()

    def generate_session(self, session_id, duration_windows=18, cheating_type="none"):
        """Generates a sequence of 10-second windows for a full exam session."""
        session_records = []
        is_cheating_session = (cheating_type != "none")

        for w_idx in range(duration_windows):
            # Cheating occurs actively in the middle windows (windows 6 to 14)
            is_active_cheating = is_cheating_session and (6 <= w_idx <= 14)
            active_type = cheating_type if is_active_cheating else "none"

            record = {
                "window_id": f"{session_id}_w{w_idx:03d}",
                "session_id": session_id,
                "timestamp_start": w_idx * 10,
                "timestamp_end": (w_idx + 1) * 10,
            }
            record.update(self.gaze_sim.simulate_window(is_active_cheating, active_type))
            record.update(self.mouse_sim.simulate_window(is_active_cheating, active_type))
            record.update(self.env_sim.simulate_window(is_active_cheating, active_type))
            record["label"] = 1 if is_active_cheating else 0
            record["cheating_type"] = active_type

            session_records.append(record)
        return session_records