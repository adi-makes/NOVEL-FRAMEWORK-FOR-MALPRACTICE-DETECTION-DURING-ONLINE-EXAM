import pytest
import numpy as np
from exam_proctoring.environment.environment_features import EnvironmentFeatureExtractor


def test_environment_feature_extractor():
    extractor = EnvironmentFeatureExtractor(history_window_size=10, stability_threshold_ratio=0.5)

    fake_detection = {
        "timestamp": 100.0,
        "frame_index": 1,
        "inference_time_ms": 15.0,
        "detections": [
            {
                "class_id": 0,
                "class_name": "person",
                "confidence": 0.95,
                "bbox": [10, 10, 100, 200],
                "is_relevant": True,
            },
            {
                "class_id": 67,
                "class_name": "cell phone",
                "confidence": 0.88,
                "bbox": [50, 50, 80, 120],
                "is_relevant": True,
            },
        ],
    }

    res = extractor.extract_features(fake_detection)

    assert res["person_count"] == 1
    assert bool(res["phone_detected"]) is True
    assert res["phone_confidence"] == 0.88
    assert res["book_detected"] is False
    assert res["relevant_object_count"] == 2
    assert "temporal_stability" in res
