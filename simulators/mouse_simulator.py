import numpy as np

class MouseSimulator:
    """Simulates mouse and interaction behaviors with realistic benign anomalies."""

    def simulate_window(self, is_cheating=False, cheating_type="none"):
        # Benign tab switch: 4% chance honest candidate has an OS popup/accidental tab change
        benign_tab_switch = (not is_cheating) and (np.random.rand() < 0.04)
        # Thinking pause: 12% chance of complete idle period while reading/thinking
        thinking_pause = (not is_cheating) and (np.random.rand() < 0.12)

        if not is_cheating and not benign_tab_switch and not thinking_pause:
            return {
                "cursor_velocity_mean": float(np.clip(np.random.normal(120, 30), 0, 500)),
                "cursor_velocity_std": float(np.clip(np.random.normal(40, 10), 0, 150)),
                "click_frequency": float(np.clip(np.random.normal(0.4, 0.15), 0.0, 2.0)),
                "keystroke_frequency": float(np.clip(np.random.normal(1.2, 0.4), 0.0, 5.0)),
                "idle_fraction": float(np.clip(np.random.normal(0.15, 0.08), 0.0, 0.6)),
                "tab_switch_count": 0,
                "velocity_spike_ratio": float(np.clip(np.random.normal(0.04, 0.02), 0.0, 0.3))
            }
        elif benign_tab_switch:
            # Hard negative: single tab switch caused by system notification
            return {
                "cursor_velocity_mean": float(np.clip(np.random.normal(160, 40), 10, 450)),
                "cursor_velocity_std": float(np.clip(np.random.normal(55, 15), 5, 180)),
                "click_frequency": float(np.clip(np.random.normal(0.5, 0.2), 0.0, 2.0)),
                "keystroke_frequency": float(np.clip(np.random.normal(0.8, 0.3), 0.0, 3.0)),
                "idle_fraction": float(np.clip(np.random.normal(0.20, 0.10), 0.0, 0.6)),
                "tab_switch_count": 1,
                "velocity_spike_ratio": float(np.clip(np.random.normal(0.12, 0.04), 0.0, 0.4))
            }
        elif thinking_pause:
            return {
                "cursor_velocity_mean": float(np.clip(np.random.normal(15, 8), 0, 80)),
                "cursor_velocity_std": float(np.clip(np.random.normal(8, 4), 0, 40)),
                "click_frequency": 0.0,
                "keystroke_frequency": 0.0,
                "idle_fraction": float(np.clip(np.random.normal(0.88, 0.08), 0.6, 1.0)),
                "tab_switch_count": 0,
                "velocity_spike_ratio": 0.0
            }
        elif cheating_type == "copy_paste":
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
            return {
                "cursor_velocity_mean": float(np.clip(np.random.normal(30, 15), 0, 200)),
                "cursor_velocity_std": float(np.clip(np.random.normal(15, 8), 0, 80)),
                "click_frequency": float(np.clip(np.random.normal(0.1, 0.05), 0.0, 0.5)),
                "keystroke_frequency": float(np.clip(np.random.normal(0.2, 0.1), 0.0, 1.0)),
                "idle_fraction": float(np.clip(np.random.normal(0.70, 0.15), 0.2, 1.0)),
                "tab_switch_count": 0,
                "velocity_spike_ratio": float(np.clip(np.random.normal(0.02, 0.01), 0.0, 0.5))
            }