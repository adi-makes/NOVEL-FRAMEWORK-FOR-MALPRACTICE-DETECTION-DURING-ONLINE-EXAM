"""
Unit tests for GazeCalibrator module.
"""

import sys
import os
import tempfile
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "code")))

from gaze.calibration import GazeCalibrator, GRID_POINTS_3X3


def test_calibrator_filter_samples():
    """Test outlier and confidence filtering."""
    calibrator = GazeCalibrator(calib_file=None)
    calibrator.is_fitted = False

    samples = [
        {"face_detected": True, "gaze_confidence": 0.9, "raw_gaze_x": 0.5, "raw_gaze_y": 0.5},
        {"face_detected": True, "gaze_confidence": 0.85, "raw_gaze_x": 0.51, "raw_gaze_y": 0.49},
        {"face_detected": False, "gaze_confidence": 0.9, "raw_gaze_x": 0.5, "raw_gaze_y": 0.5},  # no face
        {"face_detected": True, "gaze_confidence": 0.1, "raw_gaze_x": 0.5, "raw_gaze_y": 0.5},  # low conf
        {"face_detected": True, "gaze_confidence": 0.9, "raw_gaze_x": 5.0, "raw_gaze_y": 5.0},  # outlier
    ]

    filtered = calibrator.filter_samples(samples, min_confidence=0.3)
    assert len(filtered) == 2
    assert filtered[0]["raw_gaze_x"] == 0.5
    assert filtered[1]["raw_gaze_x"] == 0.51


def test_calibrator_fit_and_predict():
    """Test model fitting on synthetic 3x3 grid observations and prediction."""
    calibrator = GazeCalibrator(calib_file=None)
    calibrator.is_fitted = False

    observations = []
    target_points = []

    # Generate synthetic observations with small linear offset
    for tgt_x, tgt_y in GRID_POINTS_3X3:
        for _ in range(10):
            raw_x = tgt_x * 0.8 + 0.1 + np.random.normal(0, 0.01)
            raw_y = tgt_y * 0.8 + 0.1 + np.random.normal(0, 0.01)
            observations.append({
                "raw_gaze_x": float(raw_x),
                "raw_gaze_y": float(raw_y),
                "head_yaw": 0.0,
                "head_pitch": 0.0
            })
            target_points.append((tgt_x, tgt_y))

    success = calibrator.fit(observations, target_points)
    assert success is True
    assert calibrator.is_fitted is True

    # Test prediction bounds
    pred_x, pred_y = calibrator.predict(0.5, 0.5, yaw=0.0, pitch=0.0)
    assert 0.0 <= pred_x <= 1.0
    assert 0.0 <= pred_y <= 1.0


def test_calibrator_save_and_load():
    """Test saving calibration model to file and reloading it."""
    with tempfile.TemporaryDirectory() as tmpdir:
        calib_file = os.path.join(tmpdir, "test_calibration.json")
        calibrator = GazeCalibrator(calib_file=calib_file)

        observations = []
        target_points = []
        for tgt_x, tgt_y in GRID_POINTS_3X3:
            for _ in range(5):
                observations.append({
                    "raw_gaze_x": tgt_x,
                    "raw_gaze_y": tgt_y,
                    "head_yaw": 0.0,
                    "head_pitch": 0.0
                })
                target_points.append((tgt_x, tgt_y))

        calibrator.fit(observations, target_points)
        calibrator.save()

        assert os.path.exists(calib_file)

        # Load into a new calibrator instance
        new_calibrator = GazeCalibrator(calib_file=calib_file)
        assert new_calibrator.is_fitted is True

        p1_x, p1_y = calibrator.predict(0.3, 0.7, yaw=2.0, pitch=-1.0)
        p2_x, p2_y = new_calibrator.predict(0.3, 0.7, yaw=2.0, pitch=-1.0)

        assert abs(p1_x - p2_x) < 1e-5
        assert abs(p1_y - p2_y) < 1e-5
