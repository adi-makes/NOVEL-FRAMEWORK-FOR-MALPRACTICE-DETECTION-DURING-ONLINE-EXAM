import numpy as np

class MouseSimulator:
    """Simulates interaction and mouse behavioral features over 10-second exam windows."""

    def simulate_window(self, is_cheating=False, cheating_type="none"):
        if not is_cheating:
            # Steady activity
            return {
                "cursor_velocity_mean": float(np.clip(np.random.normal(120, 30), 0, 500)),
                "cursor_velocity_std": float(np.clip(np.random.normal(40, 10), 0, 150)),
                "click_frequency": float(np.clip(np.random.normal(0.4, 0.15), 0.0, 2.0)),
                "keystroke_frequency": float(np.clip(np.random.normal(1.2, 0.4), 0.0, 5.0)),
                "idle_fraction": float(np.clip(np.random.normal(0.15, 0.08), 0.0, 1.0)),
                "tab_switch_count": int(np.random.choice([0, 1], p=[0.97, 0.03])),
                "velocity_spike_ratio": float(np.clip(np.random.normal(0.04, 0.02), 0.0, 1.0))
            }
        elif cheating_type == "copy_paste":
            # Web searching / rapid copy-pasting
            return {
                "cursor_velocity_mean": float(np.clip(np.random.normal(310, 60), 50, 800)),
                "cursor_velocity_std": float(np.clip(np.random.normal(110, 25), 20, 300)),
                "click_frequency": float(np.clip(np.random.normal(1.4, 0.3), 0.1, 4.0)),
                "keystroke_frequency": float(np.clip(np.random.normal(2.8, 0.6), 0.0, 8.0)),
                "idle_fraction": float(np.clip(np.random.normal(0.08, 0.04), 0.0, 0.5)),
                "tab_switch_count": int(np.random.randint(2, 6)),
                "velocity_spike_ratio": float(np.clip(np.random.normal(0.35, 0.08), 0.05, 1.0))
            }
        else:
            # Silent or external assistance: long idle periods
            return {
                "cursor_velocity_mean": float(np.clip(np.random.normal(30, 15), 0, 200)),
                "cursor_velocity_std": float(np.clip(np.random.normal(15, 8), 0, 80)),
                "click_frequency": float(np.clip(np.random.normal(0.1, 0.05), 0.0, 0.5)),
                "keystroke_frequency": float(np.clip(np.random.normal(0.2, 0.1), 0.0, 1.0)),
                "idle_fraction": float(np.clip(np.random.normal(0.70, 0.15), 0.2, 1.0)),
                "tab_switch_count": 0,
                "velocity_spike_ratio": float(np.clip(np.random.normal(0.02, 0.01), 0.0, 0.5))
            }