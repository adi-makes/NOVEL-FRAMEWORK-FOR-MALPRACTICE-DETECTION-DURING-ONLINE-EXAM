"""
Unit tests for GazeEstimator module.
"""

import sys
import os
import pytest
import numpy as np

# Ensure code directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "code")))

from gaze.gaze_estimator import GazeEstimator, get_default_model_path, ensure_model_file


def test_model_file_exists():
    """Test that model file path resolution and auto-download works."""
    model_path = get_default_model_path()
    path = ensure_model_file(model_path)
    assert os.path.exists(path)


def test_gaze_estimator_initialization():
    """Test that GazeEstimator initializes cleanly without raising exceptions."""
    estimator = GazeEstimator()
    assert estimator is not None
    assert estimator.landmarker is not None


def test_process_empty_frame():
    """Test process_frame on None, empty, or zero-sized frames."""
    estimator = GazeEstimator()

    res_none = estimator.process_frame(None)
    assert res_none["face_detected"] is False
    assert res_none["gaze_x"] is None
    assert res_none["gaze_y"] is None
    assert res_none["gaze_confidence"] == 0.0

    empty_frame = np.zeros((0, 0, 3), dtype=np.uint8)
    res_empty = estimator.process_frame(empty_frame)
    assert res_empty["face_detected"] is False
    assert res_empty["gaze_confidence"] == 0.0


def test_process_blank_black_frame():
    """Test process_frame on a solid black frame without a face."""
    estimator = GazeEstimator()
    black_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    res = estimator.process_frame(black_frame, timestamp=100.0)
    assert res["timestamp"] == 100.0
    assert res["face_detected"] is False
    assert res["gaze_x"] is None
    assert res["gaze_y"] is None
    assert res["gaze_confidence"] == 0.0
    assert res["head_yaw"] is None
    assert res["head_pitch"] is None


def test_result_schema_keys():
    """Test that process_frame returns all required dictionary keys."""
    estimator = GazeEstimator()
    dummy_frame = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    res = estimator.process_frame(dummy_frame)

    required_keys = [
        "timestamp", "gaze_x", "gaze_y", "raw_gaze_x", "raw_gaze_y",
        "gaze_confidence", "head_yaw", "head_pitch", "face_detected"
    ]
    for key in required_keys:
        assert key in res
