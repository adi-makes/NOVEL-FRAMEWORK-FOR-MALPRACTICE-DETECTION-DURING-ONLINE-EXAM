import numpy as np

class EnvironmentSimulator:
    """Simulates environment detections with realistic sensor noise and false alarms."""

    def simulate_window(self, is_cheating=False, cheating_type="none"):
        # Sensor false-alarm: 4% chance a harmless object triggers a low-confidence detection
        false_alarm = (not is_cheating) and (np.random.rand() < 0.04)

        if not is_cheating and not false_alarm:
            return {
                "phone_detected": 0,
                "phone_confidence": float(np.clip(np.random.normal(0.02, 0.01), 0.0, 0.10)),
                "notes_detected": 0,
                "extra_person_count": 0,
                "suspicious_objects_count": 0
            }
        elif false_alarm:
            return {
                "phone_detected": 0,
                "phone_confidence": float(np.clip(np.random.normal(0.35, 0.08), 0.15, 0.55)),
                "notes_detected": int(np.random.choice([0, 1], p=[0.7, 0.3])),
                "extra_person_count": 0,
                "suspicious_objects_count": 1
            }

        # Cheating: non-perfect camera coverage (85% detection probability)
        missed_detection = np.random.rand() < 0.15
        if missed_detection:
            return {
                "phone_detected": 0,
                "phone_confidence": float(np.clip(np.random.normal(0.25, 0.10), 0.0, 0.50)),
                "notes_detected": 0,
                "extra_person_count": 0,
                "suspicious_objects_count": 0
            }

        return {
            "phone_detected": 1 if cheating_type == "phone" else 0,
            "phone_confidence": float(np.clip(np.random.normal(0.88, 0.06), 0.6, 0.99)) if cheating_type == "phone" else float(np.clip(np.random.normal(0.05, 0.02), 0.0, 0.15)),
            "notes_detected": 1 if cheating_type == "notes" else 0,
            "extra_person_count": 1 if cheating_type == "external_assistance" else 0,
            "suspicious_objects_count": 1 if cheating_type in ["phone", "notes", "external_assistance"] else 0
        }