import numpy as np

class EnvironmentSimulator:
    """Simulates environment detections with hard negatives and sensor noise."""

    def simulate_window(self, is_cheating=False, cheating_type="none"):
        # Hard Negative: 7% chance honest candidate holds a phone-like object (calculator, wallet)
        phone_lookalike = (not is_cheating) and (np.random.rand() < 0.07)
        
        # Hard Negative: 5% chance honest candidate has permitted scratch paper
        scratch_paper = (not is_cheating) and (np.random.rand() < 0.05)

        if not is_cheating and not phone_lookalike and not scratch_paper:
            return {
                "phone_detected": 0,
                "phone_confidence": float(np.clip(np.random.normal(0.02, 0.01), 0.0, 0.10)),
                "notes_detected": 0,
                "extra_person_count": 0,
                "suspicious_objects_count": 0
            }
        elif phone_lookalike:
            return {
                "phone_detected": 1,  # Object detector trigger on honest candidate
                "phone_confidence": float(np.clip(np.random.normal(0.62, 0.10), 0.45, 0.85)),
                "notes_detected": 0,
                "extra_person_count": 0,
                "suspicious_objects_count": 1
            }
        elif scratch_paper:
            return {
                "phone_detected": 0,
                "phone_confidence": float(np.clip(np.random.normal(0.05, 0.02), 0.0, 0.15)),
                "notes_detected": 1,  # Permitted scratch paper trigger
                "extra_person_count": 0,
                "suspicious_objects_count": 1
            }

        # Cheating scenarios with 15% sensor occlusion/missed coverage
        if np.random.rand() < 0.15:
            return {
                "phone_detected": 0,
                "phone_confidence": float(np.clip(np.random.normal(0.20, 0.08), 0.0, 0.40)),
                "notes_detected": 0,
                "extra_person_count": 0,
                "suspicious_objects_count": 0
            }

        return {
            "phone_detected": 1 if cheating_type == "phone" else 0,
            "phone_confidence": float(np.clip(np.random.normal(0.88, 0.06), 0.60, 0.99)) if cheating_type == "phone" else float(np.clip(np.random.normal(0.05, 0.02), 0.0, 0.15)),
            "notes_detected": 1 if cheating_type == "notes" else 0,
            "extra_person_count": 1 if cheating_type == "external_assistance" else 0,
            "suspicious_objects_count": 1 if cheating_type in ["phone", "notes", "external_assistance"] else 0
        }