"""
Unit tests for GazeFeatureExtractor module.
"""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "code")))

from gaze.feature_extractor import GazeFeatureExtractor


def test_insufficient_data():
    """Test feature extractor behavior when buffer has zero or insufficient samples."""
    extractor = GazeFeatureExtractor(window_seconds=10.0, min_valid_samples=3)
    res = extractor.get_features()

    assert res["valid_samples"] == 0
    assert len(res["feature_vector"]) == 7
    assert res["feature_vector"] == [0.0] * 7
    assert res["features"]["gaze_confidence"] == 0.0


def test_synthetic_gaze_stream_7_features():
    """Test feature extraction on synthetic stream over a 10-second window."""
    extractor = GazeFeatureExtractor(window_seconds=10.0)

    start_ts = 1000.0
    # Simulate 300 frames (30 FPS over 10 seconds)
    for i in range(300):
        t = start_ts + i * (1.0 / 30.0)
        # Looking near center with small movement
        gx = 0.5 + 0.02 * np.sin(i * 0.1)
        gy = 0.5 + 0.02 * np.cos(i * 0.1)
        yaw = 2.0 * np.sin(i * 0.05)
        pitch = -1.0 * np.cos(i * 0.05)

        frame_result = {
            "timestamp": t,
            "gaze_x": float(gx),
            "gaze_y": float(gy),
            "raw_gaze_x": float(gx),
            "raw_gaze_y": float(gy),
            "gaze_confidence": 0.95,
            "head_yaw": float(yaw),
            "head_pitch": float(pitch),
            "face_detected": True
        }

        res = extractor.update(frame_result)

    assert res["valid_samples"] == 300
    vec = res["feature_vector"]
    assert len(vec) == 7

    feats = res["features"]
    assert "gaze_deviation" in feats
    assert "fixation_duration" in feats
    assert "fixation_count" in feats
    assert "saccade_velocity" in feats
    assert "gaze_confidence" in feats
    assert "head_yaw" in feats
    assert "head_pitch" in feats

    # Dev should be small since looking near center (0.5, 0.5)
    assert 0.0 <= feats["gaze_deviation"] <= 0.2
    assert feats["gaze_confidence"] > 0.9


def test_pytorch_tensor_output():
    """Test conversion of 7D feature vector to PyTorch tensor or exception when torch is missing."""
    from gaze.feature_extractor import TORCH_AVAILABLE
    extractor = GazeFeatureExtractor()
    extractor.update({
        "timestamp": 1.0,
        "gaze_x": 0.5, "gaze_y": 0.5,
        "gaze_confidence": 0.9,
        "head_yaw": 0.0, "head_pitch": 0.0,
        "face_detected": True
    })
    if TORCH_AVAILABLE:
        tensor = extractor.to_tensor()
        assert tensor.shape == (1, 7)
    else:
        with pytest.raises(RuntimeError):
            extractor.to_tensor()


def test_json_output_format():
    """Test machine-readable JSON output string."""
    import json
    extractor = GazeFeatureExtractor()
    extractor.update({
        "timestamp": 1.0,
        "gaze_x": 0.5, "gaze_y": 0.5,
        "gaze_confidence": 0.9,
        "head_yaw": 0.0, "head_pitch": 0.0,
        "face_detected": True
    })
    json_str = extractor.to_json()
    parsed = json.loads(json_str)

    assert "timestamp" in parsed
    assert "features" in parsed
    assert len(parsed["features"]) == 7
