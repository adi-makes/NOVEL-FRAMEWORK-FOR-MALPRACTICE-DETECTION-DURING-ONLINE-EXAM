from collections import deque
import time


class EnvironmentFeatureExtractor:
    """
    Extracts higher-level environment evidence from object detection output.
    Maintains a sliding temporal window to verify observation persistence
    and filter out single-frame false-positive noise.
    """

    def __init__(self, history_window_size: int = 10, stability_threshold_ratio: float = 0.5):
        """
        Args:
            history_window_size: Number of recent frames to retain for temporal tracking.
            stability_threshold_ratio: Ratio of frames in window required to mark an observation as temporally stable.
        """
        self.history_window_size = history_window_size
        self.stability_threshold_ratio = stability_threshold_ratio

        # Sliding window history buffers for key environmental signals
        self.phone_history = deque(maxlen=self.history_window_size)
        self.book_history = deque(maxlen=self.history_window_size)
        self.multi_person_history = deque(maxlen=self.history_window_size)

    def extract_features(self, detection_output: dict) -> dict:
        """
        Process raw detection dict and produce environment evidence summary with temporal persistence.
        """
        timestamp = detection_output.get("timestamp", time.time())
        frame_index = detection_output.get("frame_index", 0)
        detections = detection_output.get("detections", [])

        person_count = 0
        phone_detected = False
        phone_max_conf = 0.0
        book_detected = False
        book_max_conf = 0.0
        laptop_detected = False
        laptop_max_conf = 0.0
        relevant_object_count = 0

        formatted_objects = []

        for det in detections:
            cls_name = det["class_name"].lower()
            conf = det["confidence"]

            formatted_objects.append(
                {
                    "class_name": det["class_name"],
                    "confidence": conf,
                    "is_relevant": det["is_relevant"],
                    "bbox": det["bbox"],
                }
            )

            if det["is_relevant"]:
                relevant_object_count += 1

            if cls_name == "person":
                person_count += 1
            elif cls_name == "cell phone":
                phone_detected = True
                phone_max_conf = max(phone_max_conf, conf)
            elif cls_name in ["book", "notebook"]:
                book_detected = True
                book_max_conf = max(book_max_conf, conf)
            elif cls_name == "laptop":
                laptop_detected = True
                laptop_max_conf = max(laptop_max_conf, conf)

        # Update sliding window temporal buffers
        self.phone_history.append(1 if phone_detected else 0)
        self.book_history.append(1 if book_detected else 0)
        self.multi_person_history.append(1 if person_count > 1 else 0)

        # Calculate temporal persistence scores
        phone_score = (
            sum(self.phone_history) / len(self.phone_history) if self.phone_history else 0.0
        )
        book_score = sum(self.book_history) / len(self.book_history) if self.book_history else 0.0
        multi_person_score = (
            sum(self.multi_person_history) / len(self.multi_person_history)
            if self.multi_person_history
            else 0.0
        )

        phone_stable = phone_score >= self.stability_threshold_ratio
        book_stable = book_score >= self.stability_threshold_ratio
        multi_person_stable = multi_person_score >= self.stability_threshold_ratio

        return {
            "timestamp": timestamp,
            "frame_index": frame_index,
            "person_count": person_count,
            "phone_detected": phone_detected,
            "phone_confidence": round(phone_max_conf, 4),
            "book_detected": book_detected,
            "book_confidence": round(book_max_conf, 4),
            "laptop_detected": laptop_detected,
            "laptop_confidence": round(laptop_max_conf, 4),
            "relevant_object_count": relevant_object_count,
            "total_objects_detected": len(detections),
            "objects": formatted_objects,
            "temporal_stability": {
                "window_size": len(self.phone_history),
                "phone_stable": phone_stable,
                "book_stable": book_stable,
                "multiple_persons_stable": multi_person_stable,
                "persistence_scores": {
                    "phone": round(phone_score, 2),
                    "book": round(book_score, 2),
                    "multiple_persons": round(multi_person_score, 2),
                },
            },
        }

    def reset_history(self):
        """Reset temporal history buffers."""
        self.phone_history.clear()
        self.book_history.clear()
        self.multi_person_history.clear()
