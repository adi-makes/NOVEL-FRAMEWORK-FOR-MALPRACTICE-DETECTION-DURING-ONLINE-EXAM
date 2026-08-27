import numpy as np

class EnvironmentSimulator:
    """Simulates environment camera detections (phones, notes, extra people)."""

    def simulate_window(self, is_cheating=False, cheating_type="none"):
        if not is_cheating:
            return {
                "phone_detected": 0,
                "phone_confidence": float(np.clip(np.random.normal(0.02, 0.01), 0.0, 0.15)),
                "notes_detected": 0,
                "extra_person_count": 0,
                "suspicious_objects_count": 0
            }

        return {
            "phone_detected": 1 if cheating_type == "phone" else 0,
            "phone_confidence": float(np.clip(np.random.normal(0.88, 0.06), 0.6, 0.99)) if cheating_type == "phone" else 0.0,
            "notes_detected": 1 if cheating_type == "notes" else 0,
            "extra_person_count": 1 if cheating_type == "external_assistance" else 0,
            "suspicious_objects_count": 1 if cheating_type in ["phone", "notes"] else 0
        }