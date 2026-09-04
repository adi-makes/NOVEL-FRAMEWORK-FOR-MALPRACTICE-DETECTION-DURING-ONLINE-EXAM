"""
Person 3 — Unit Tests for TemporalWindowAggregator / EnvironmentFeatureExtractor
==================================================================================

All tests use synthetic detection dicts — no physical webcam required.
Mocked timestamps are used where 10-second behaviour must be verified
without actually waiting 10 seconds.

Run with:
    cd /home/agnivesh/novel
    source venv/bin/activate
    pytest p3/tests/test_environment_features.py -v
"""

import sys
import os
import time
import pytest

# Allow importing from p3/src/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.environment_features import (
    TemporalWindowAggregator,
    EnvironmentFeatureExtractor,
    SUSPICIOUS_CLASSES,
    NOTES_PROXY_CLASSES,
)
from src.env_fusion_adapter import EnvironmentFusionAdapter, ENVIRONMENT_FEATURE_ORDER

try:
    import torch
except ImportError:
    torch = None



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_detection(class_name: str, confidence: float = 0.80) -> dict:
    """Build a minimal detection dict compatible with TemporalWindowAggregator."""
    return {
        "class_name": class_name,
        "confidence": confidence,
        "bbox": [0, 0, 100, 100],
        "is_relevant": True,
    }


def _make_detection_output(detections: list, timestamp: float = None) -> dict:
    """Wrap detections into the format returned by ObjectDetector.detect()."""
    return {
        "timestamp": timestamp if timestamp is not None else time.time(),
        "frame_index": 0,
        "inference_time_ms": 10.0,
        "detections": detections,
    }


def _feed_frames(
    aggregator: TemporalWindowAggregator,
    detections: list,
    n_frames: int,
    base_timestamp: float = 0.0,
    interval: float = 0.1,
) -> dict:
    """
    Feed `n_frames` identical detection frames to the aggregator.
    Returns the feature dict from the last update.
    """
    result = {}
    for i in range(n_frames):
        ts = base_timestamp + i * interval
        output = _make_detection_output(detections, timestamp=ts)
        result = aggregator.update(output)
    return result


# ---------------------------------------------------------------------------
# Test 1 — Empty detections → all zeros
# ---------------------------------------------------------------------------

class TestEmptyDetections:
    def test_empty_input(self):
        """No detections → all 5 primary features are zero / falsy."""
        agg = TemporalWindowAggregator()
        result = _feed_frames(agg, [], n_frames=5)

        assert result["phone_detected"] == 0
        assert result["phone_confidence"] == 0.0
        assert result["notes_detected"] == 0
        assert result["extra_person_count"] == 0
        assert result["suspicious_objects_count"] == 0

    def test_get_features_before_any_update(self):
        """Before any frame is processed, get_features() returns all zeros."""
        agg = TemporalWindowAggregator()
        result = agg.get_features()
        assert result["phone_detected"] == 0
        assert result["phone_confidence"] == 0.0

    def test_feature_vector_length(self):
        """Feature vector must always be exactly 5 elements."""
        agg = TemporalWindowAggregator()
        assert len(agg.get_feature_vector()) == 5

        _feed_frames(agg, [], n_frames=3)
        assert len(agg.get_feature_vector()) == 5


# ---------------------------------------------------------------------------
# Test 2 — One person → no anomaly
# ---------------------------------------------------------------------------

class TestOnePersonNoAnomaly:
    def test_single_person_no_anomaly(self):
        """Exactly 1 person present → extra_person_count=0, person_count_anomaly=0."""
        agg = TemporalWindowAggregator(persistence_ratio=0.30)
        result = _feed_frames(
            agg,
            [_make_detection("person")],
            n_frames=20,
        )
        assert result["extra_person_count"] == 0
        assert result["person_count_anomaly"] == 0
        assert result["raw_person_count"] == 1


# ---------------------------------------------------------------------------
# Test 3 — Multiple people → anomaly
# ---------------------------------------------------------------------------

class TestMultiplePeopleAnomaly:
    def test_two_people_anomaly(self):
        """2 people stably detected → extra_person_count=1, person_count_anomaly=1."""
        agg = TemporalWindowAggregator(persistence_ratio=0.30)
        result = _feed_frames(
            agg,
            [_make_detection("person"), _make_detection("person")],
            n_frames=20,
        )
        assert result["extra_person_count"] == 1
        assert result["person_count_anomaly"] == 1
        assert result["raw_person_count"] == 2

    def test_three_people(self):
        """3 people → extra_person_count=2."""
        agg = TemporalWindowAggregator(persistence_ratio=0.30)
        dets = [_make_detection("person")] * 3
        result = _feed_frames(agg, dets, n_frames=20)
        assert result["extra_person_count"] == 2
        assert result["person_count_anomaly"] == 1


# ---------------------------------------------------------------------------
# Test 4 — Phone above threshold → phone_detected = 1
# ---------------------------------------------------------------------------

class TestPhoneDetectedAboveThreshold:
    def test_phone_above_threshold_stable(self):
        """Phone confidently and persistently detected → phone_detected=1."""
        agg = TemporalWindowAggregator(
            confidence_threshold=0.40,
            persistence_ratio=0.30,
        )
        result = _feed_frames(
            agg,
            [_make_detection("cell phone", confidence=0.85)],
            n_frames=20,
        )
        assert result["phone_detected"] == 1
        assert result["phone_confidence"] > 0.0

    def test_phone_confidence_value(self):
        """phone_confidence should reflect the max observed confidence."""
        agg = TemporalWindowAggregator(confidence_threshold=0.40)
        _feed_frames(agg, [_make_detection("cell phone", confidence=0.72)], n_frames=10)
        result = agg.get_features()
        assert result["phone_confidence"] == pytest.approx(0.72, abs=0.01)


# ---------------------------------------------------------------------------
# Test 5 — Phone below confidence threshold → phone_detected = 0
# ---------------------------------------------------------------------------

class TestPhoneBelowThreshold:
    def test_low_confidence_phone_ignored(self):
        """
        A phone detected at 0.20 confidence (below 0.40 threshold) must
        NOT trigger phone_detected even if seen in every frame.
        """
        agg = TemporalWindowAggregator(
            confidence_threshold=0.40,
            persistence_ratio=0.30,
        )
        result = _feed_frames(
            agg,
            [_make_detection("cell phone", confidence=0.20)],
            n_frames=20,
        )
        assert result["phone_detected"] == 0
        assert result["phone_confidence"] == 0.0


# ---------------------------------------------------------------------------
# Test 6 — Suspicious object counted
# ---------------------------------------------------------------------------

class TestSuspiciousObjectCount:
    def test_configured_suspicious_class_counted(self):
        """
        A class that is in SUSPICIOUS_CLASSES and seen persistently
        must increment suspicious_objects_count.
        """
        # Pick an actual suspicious class (not keyboard/mouse)
        suspicious_cls = next(iter(SUSPICIOUS_CLASSES))  # e.g. "cell phone"
        agg = TemporalWindowAggregator(
            confidence_threshold=0.40,
            persistence_ratio=0.30,
        )
        result = _feed_frames(
            agg,
            [_make_detection(suspicious_cls, confidence=0.80)],
            n_frames=20,
        )
        assert result["suspicious_objects_count"] >= 1

    def test_non_suspicious_class_not_counted(self):
        """keyboard and mouse are NOT suspicious by design."""
        agg = TemporalWindowAggregator(
            confidence_threshold=0.40,
            persistence_ratio=0.30,
        )
        result = _feed_frames(
            agg,
            [_make_detection("keyboard", confidence=0.90),
             _make_detection("mouse", confidence=0.90)],
            n_frames=20,
        )
        assert result["suspicious_objects_count"] == 0

    def test_multiple_suspicious_classes_each_counted(self):
        """
        Two distinct suspicious classes present → suspicious_objects_count = 2.
        (phone + remote both in SUSPICIOUS_CLASSES)
        """
        agg = TemporalWindowAggregator(
            confidence_threshold=0.40,
            persistence_ratio=0.30,
        )
        result = _feed_frames(
            agg,
            [
                _make_detection("cell phone", confidence=0.80),
                _make_detection("remote", confidence=0.80),
            ],
            n_frames=20,
        )
        assert result["suspicious_objects_count"] == 2


# ---------------------------------------------------------------------------
# Test — Notes detection proxy and custom override
# ---------------------------------------------------------------------------

class TestNotesDetection:
    def test_book_proxy_sets_notes_detected(self):
        """COCO book is the documented proxy for notes/reference material."""
        agg = TemporalWindowAggregator(
            confidence_threshold=0.40,
            persistence_ratio=0.30,
        )
        result = _feed_frames(
            agg,
            [_make_detection("book", confidence=0.80)],
            n_frames=20,
        )
        assert "book" in NOTES_PROXY_CLASSES
        assert result["notes_detected"] == 1

    def test_custom_notes_detector_can_override_book_proxy(self):
        """A custom notes detector returning False bypasses the COCO book proxy."""
        class NoNotesDetector:
            def detect(self, frame_bgr):
                return False

        agg = TemporalWindowAggregator(
            confidence_threshold=0.40,
            persistence_ratio=0.30,
            custom_notes_detector=NoNotesDetector(),
        )
        result = _feed_frames(
            agg,
            [_make_detection("book", confidence=0.80)],
            n_frames=20,
        )
        assert result["notes_detected"] == 0


# ---------------------------------------------------------------------------
# Test 7 — Temporal persistence (single vs persistent detection)
# ---------------------------------------------------------------------------

class TestTemporalPersistence:
    def test_single_isolated_detection_does_not_trigger(self):
        """
        A phone seen in only 1 out of 30 frames (3.3 % ratio < 30 % threshold)
        must NOT trigger phone_detected.
        """
        agg = TemporalWindowAggregator(
            confidence_threshold=0.40,
            persistence_ratio=0.30,
        )
        base_ts = 0.0
        interval = 0.1

        # 29 empty frames
        for i in range(29):
            ts = base_ts + i * interval
            agg.update(_make_detection_output([], timestamp=ts))

        # 1 phone frame
        agg.update(_make_detection_output(
            [_make_detection("cell phone", confidence=0.90)],
            timestamp=base_ts + 29 * interval,
        ))

        result = agg.get_features()
        assert result["phone_detected"] == 0, (
            f"Expected phone_detected=0 for single isolated detection, "
            f"got {result['phone_detected']}. "
            f"persistence_score={result['persistence_scores'].get('phone')}"
        )

    def test_persistent_detection_triggers(self):
        """
        A phone seen in 15 out of 20 frames (75 % ratio > 30 % threshold)
        MUST trigger phone_detected.
        """
        agg = TemporalWindowAggregator(
            confidence_threshold=0.40,
            persistence_ratio=0.30,
        )
        base_ts = 0.0
        interval = 0.1

        # 15 frames with phone
        for i in range(15):
            agg.update(_make_detection_output(
                [_make_detection("cell phone", confidence=0.80)],
                timestamp=base_ts + i * interval,
            ))

        # 5 empty frames
        for i in range(5):
            agg.update(_make_detection_output(
                [], timestamp=base_ts + (15 + i) * interval
            ))

        result = agg.get_features()
        assert result["phone_detected"] == 1, (
            f"Expected phone_detected=1, got {result['phone_detected']}. "
            f"persistence_score={result['persistence_scores'].get('phone')}"
        )


# ---------------------------------------------------------------------------
# Test 8 — 10-second aggregation window (mocked timestamps)
# ---------------------------------------------------------------------------

class TestTenSecondWindow:
    def test_window_does_not_finalize_early(self):
        """
        After 5 seconds of observations (< 10s window), the aggregator
        should still have all observations in the buffer.
        """
        agg = TemporalWindowAggregator(window_seconds=10.0)

        # Feed 50 frames across 5 seconds
        for i in range(50):
            ts = 0.0 + i * 0.1   # 0.0 → 4.9 s
            agg.update(_make_detection_output([], timestamp=ts))

        # All 50 observations should still be in window
        result = agg.get_features()
        assert result["window_obs_count"] == 50

    def test_old_observations_pruned_after_window(self):
        """
        Observations older than 10 seconds must be pruned from the buffer.
        After feeding frames from t=0 to t=15, frames at t<5 should be gone.
        """
        agg = TemporalWindowAggregator(window_seconds=10.0)

        # Feed 151 frames from t=0 to t=15.0 (interval 0.1s)
        for i in range(151):
            ts = 0.0 + i * 0.1
            agg.update(_make_detection_output([], timestamp=ts))

        result = agg.get_features()
        # At t=15.0, frames older than t=5.0 are pruned.
        # Remaining frames: t=5.0..15.0 = 101 frames (indices 50..150)
        # Allow ±2 for boundary rounding
        assert result["window_obs_count"] <= 105, (
            f"Expected ≤105 obs in 10s window, got {result['window_obs_count']}"
        )
        assert result["window_obs_count"] >= 99, (
            f"Expected ≥99 obs, got {result['window_obs_count']}"
        )

    def test_phone_only_in_expired_frames_is_not_detected(self):
        """
        Phone seen only in frames that are now outside the 10-second window
        must NOT trigger phone_detected.
        """
        agg = TemporalWindowAggregator(
            window_seconds=10.0,
            confidence_threshold=0.40,
            persistence_ratio=0.30,
        )

        # 50 frames with phone at t=0.0..4.9 (will be outside window at t=15.0)
        for i in range(50):
            ts = 0.0 + i * 0.1
            agg.update(_make_detection_output(
                [_make_detection("cell phone", confidence=0.85)],
                timestamp=ts,
            ))

        # 101 empty frames at t=5.0..15.0 (push old frames out of window)
        for i in range(101):
            ts = 5.0 + i * 0.1
            agg.update(_make_detection_output([], timestamp=ts))

        result = agg.get_features()
        assert result["phone_detected"] == 0, (
            "Phone frames outside the 10s window must not contribute to detection. "
            f"Got phone_detected={result['phone_detected']}, "
            f"window_obs_count={result['window_obs_count']}"
        )


# ---------------------------------------------------------------------------
# Test — Fusion adapter correctness
# ---------------------------------------------------------------------------

class TestFusionAdapter:
    def test_vector_length_always_5(self):
        """EnvironmentFusionAdapter.to_vector() always returns exactly 5 elements."""
        adapter = EnvironmentFusionAdapter()
        agg = TemporalWindowAggregator()
        _feed_frames(agg, [], n_frames=5)
        vec = adapter.to_vector(agg.get_features())
        assert len(vec) == 5

    def test_vector_order(self):
        """to_vector() returns values in ENVIRONMENT_FEATURE_ORDER."""
        adapter = EnvironmentFusionAdapter()
        assert adapter.feature_names() == list(ENVIRONMENT_FEATURE_ORDER)

    def test_tensor_shape(self):
        """to_tensor() produces shape (1, 5)."""
        if torch is None:
            pytest.skip("PyTorch tensor conversion requires torch")
        adapter = EnvironmentFusionAdapter()
        agg = TemporalWindowAggregator()
        _feed_frames(agg, [], n_frames=5)
        tensor = adapter.to_tensor(agg.get_features())
        assert tensor.shape == (1, 5)
        assert tensor.dtype == torch.float32

    def test_tensor_matches_vector(self):
        """Tensor values match the to_vector() values."""
        if torch is None:
            pytest.skip("PyTorch tensor conversion requires torch")
        adapter = EnvironmentFusionAdapter()
        agg = TemporalWindowAggregator()
        _feed_frames(agg, [_make_detection("cell phone", 0.85)], n_frames=20)
        feats = agg.get_features()
        vec = adapter.to_vector(feats)
        tensor = adapter.to_tensor(feats)
        for i, v in enumerate(vec):
            assert abs(tensor[0, i].item() - v) < 1e-5


# ---------------------------------------------------------------------------
# Test — Backward-compat EnvironmentFeatureExtractor shim
# ---------------------------------------------------------------------------

class TestBackwardCompatShim:
    def test_extract_features_api(self):
        """Old extract_features() call site still works."""
        old = EnvironmentFeatureExtractor(
            history_window_size=10,
            stability_threshold_ratio=0.5,
        )
        result = old.extract_features({"timestamp": 0.0, "detections": []})
        assert "phone_detected" in result
        assert "extra_person_count" in result

    def test_reset_history_clears_buffer(self):
        """reset_history() should clear all observations."""
        old = EnvironmentFeatureExtractor()
        _feed_frames(old, [_make_detection("cell phone", 0.90)], n_frames=10)
        old.reset_history()
        result = old.get_features()
        assert result["window_obs_count"] == 0
        assert result["phone_detected"] == 0
