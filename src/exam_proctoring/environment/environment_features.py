"""
Person 3 — Environment Feature Extractor
=========================================

Aggregates YOLO object-detection observations over a real-time 10-second window
and produces the 5-dimensional environment feature vector required by the
AttentionFusionModel.

Primary Output (matches data_schema.md & AttentionFusionModel environment_dim=5):
---------------------------------------------------------------------------
    1. phone_detected          (int : 0 or 1)
    2. phone_confidence        (float : [0, 1])
    3. notes_detected          (int : 0 or 1)
    4. extra_person_count      (int : max(0, stable_person_count - 1))
    5. suspicious_objects_count (int : count of stable suspicious-class detections)

Notes-detection limitation:
    YOLOv8n (COCO) does NOT have a "notes" or "notebook" class.
    We map COCO class "book" → notes_detected.
    This produces false positives (textbooks, tablet covers) and
    false negatives (handwritten notes on paper).
    A custom notes-detection model can be plugged in via the
    `CustomNotesDetector` hook without rewriting the pipeline.

Temporal design:
    - Uses detector timestamps, falling back to time.monotonic() when absent.
    - A rolling deque keeps observations within the last `window_seconds` seconds.
    - Detection is marked "stable" when its observation ratio within the window
      exceeds `persistence_ratio` (default 0.30 = 30 % of observations).
    - This tolerates variable FPS (10–30+ fps) correctly.
"""

import time
import collections
from typing import Optional

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Centralized class definitions
# ---------------------------------------------------------------------------

# Objects that are directly suspicious in an online exam context.
# Deliberately conservative — keyboards, mice, and laptops are normal exam tools.
SUSPICIOUS_CLASSES = {
    "cell phone",   # Primary cheating tool
    "remote",       # Indicates a TV/media device nearby
    "book",         # Notes/reference material (also drives notes_detected)
}

# COCO class(es) we interpret as potential notes/written-reference material.
# A future custom model can replace this mapping entirely.
NOTES_PROXY_CLASSES = {
    "book",
}

# Person class name in COCO.
PERSON_CLASS = "person"


# ---------------------------------------------------------------------------
# Optional hook for a custom notes detector
# ---------------------------------------------------------------------------

class CustomNotesDetector:
    """
    Stub / hook for a future custom notes-detection model.

    Override `detect(frame_bgr) -> bool` in a subclass and pass the instance
    to TemporalWindowAggregator to replace the COCO-book proxy.
    The stub always returns None (unavailable), keeping existing behaviour.
    """

    def detect(self, frame_bgr) -> Optional[bool]:
        """
        Returns:
            True  — custom model says notes are present
            False — custom model says no notes
            None  — no custom model available; caller falls back to COCO proxy
        """
        return None  # Not yet implemented


# ---------------------------------------------------------------------------
# Per-frame observation (lightweight named container)
# ---------------------------------------------------------------------------

class _Observation:
    """Lightweight container for one frame's relevant detections."""

    __slots__ = (
        "timestamp",
        "phone_seen", "phone_conf",
        "notes_seen",
        "person_count",
        "suspicious_set",
    )

    def __init__(
        self,
        timestamp: float,
        phone_seen: bool,
        phone_conf: float,
        notes_seen: bool,
        person_count: int,
        suspicious_set: set,
    ):
        self.timestamp = timestamp
        self.phone_seen = phone_seen
        self.phone_conf = phone_conf
        self.notes_seen = notes_seen
        self.person_count = person_count
        self.suspicious_set = suspicious_set   # set of class names seen in this frame


# ---------------------------------------------------------------------------
# Temporal window aggregator
# ---------------------------------------------------------------------------

class TemporalWindowAggregator:
    """
    Aggregates per-frame object detections over a real-time sliding window
    and produces a stable 5-dimensional environment feature vector.

    Parameters
    ----------
    window_seconds : float
        Duration of the rolling observation window (default 10.0 s).
    persistence_ratio : float
        Fraction of observations within the window that must contain an object
        for it to be counted as "stably detected" (default 0.30).
    confidence_threshold : float
        Minimum per-frame detection confidence to register an observation.
        Should match ObjectDetector.confidence_threshold for consistency.
    custom_notes_detector : CustomNotesDetector or None
        Optional custom model for notes detection. If provided and its
        detect() returns non-None, the COCO-book proxy is bypassed.
    """

    def __init__(
        self,
        window_seconds: float = 10.0,
        persistence_ratio: float = 0.30,
        confidence_threshold: float = 0.40,
        custom_notes_detector: Optional[CustomNotesDetector] = None,
    ):
        self.window_seconds = window_seconds
        self.persistence_ratio = persistence_ratio
        self.confidence_threshold = confidence_threshold
        self.custom_notes_detector = custom_notes_detector

        # Rolling buffer of _Observation objects
        self._buffer: collections.deque = collections.deque()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(self, detection_output: dict) -> dict:
        """
        Ingest one frame's detection output from ObjectDetector.detect().

        Parameters
        ----------
        detection_output : dict
            Must contain:
                "timestamp"  : float  (wall-clock time of this frame)
                "detections" : list of detection dicts, each with:
                    "class_name"  : str
                    "confidence"  : float
                    "is_relevant" : bool

        Returns
        -------
        dict
            Full result dict from get_features() — primary + diagnostic.
        """
        obs = self._parse_detections(detection_output)
        self._buffer.append(obs)
        self._prune_old_observations(obs.timestamp)
        return self.get_features()

    def get_features(self) -> dict:
        """
        Compute environment features from current window contents.

        Returns a dict with:
            Primary (5-dim fusion vector):
                phone_detected          : int
                phone_confidence        : float
                notes_detected          : int
                extra_person_count      : int
                suspicious_objects_count: int

            Convenience:
                person_count_anomaly    : int  (1 if extra_person_count > 0)

            Diagnostic:
                raw_person_count        : int
                window_obs_count        : int
                window_duration_seconds : float
                persistence_scores      : dict
                object_confidences      : dict
        """
        if not self._buffer:
            return self._empty_features()

        n = len(self._buffer)
        window_duration = (
            self._buffer[-1].timestamp - self._buffer[0].timestamp
            if n > 1
            else 0.0
        )

        # --- Phone ---
        phone_obs = [o for o in self._buffer if o.phone_seen]
        phone_ratio = len(phone_obs) / n
        phone_stable = phone_ratio >= self.persistence_ratio
        phone_detected = 1 if phone_stable else 0
        phone_confidence = (
            round(max(o.phone_conf for o in phone_obs), 4) if phone_obs else 0.0
        )

        # --- Notes (COCO book proxy) ---
        notes_obs_count = sum(1 for o in self._buffer if o.notes_seen)
        notes_ratio = notes_obs_count / n
        notes_stable = notes_ratio >= self.persistence_ratio
        notes_detected = 1 if notes_stable else 0

        # --- Person count ---
        # Stable person count = mode (most frequent) value across window
        person_counts = [o.person_count for o in self._buffer]
        stable_person_count = _mode(person_counts)
        extra_person_count = max(0, stable_person_count - 1)
        person_count_anomaly = 1 if extra_person_count > 0 else 0

        # --- Suspicious objects ---
        # Count how many suspicious class names appear stably
        suspicious_objects_count = 0
        suspicious_ratios = {}
        for cls in SUSPICIOUS_CLASSES:
            cls_obs = sum(1 for o in self._buffer if cls in o.suspicious_set)
            ratio = cls_obs / n
            suspicious_ratios[cls] = round(ratio, 3)
            if ratio >= self.persistence_ratio:
                suspicious_objects_count += 1

        # Diagnostic
        persistence_scores = {
            "phone": round(phone_ratio, 3),
            "notes": round(notes_ratio, 3),
            "multi_person": round(
                sum(1 for o in self._buffer if o.person_count > 1) / n, 3
            ),
            **{f"suspicious_{k.replace(' ', '_')}": v for k, v in suspicious_ratios.items()},
        }

        return {
            # --- Primary 5-dim fusion vector ---
            "phone_detected": phone_detected,
            "phone_confidence": phone_confidence,
            "notes_detected": notes_detected,
            "extra_person_count": extra_person_count,
            "suspicious_objects_count": suspicious_objects_count,
            # --- Convenience alias ---
            "person_count_anomaly": person_count_anomaly,
            # --- Diagnostic ---
            "raw_person_count": stable_person_count,
            "window_obs_count": n,
            "window_duration_seconds": round(window_duration, 2),
            "persistence_scores": persistence_scores,
        }

    def get_feature_vector(self) -> list:
        """
        Returns the ordered 5-element list for the fusion model:
            [phone_detected, phone_confidence, notes_detected,
             extra_person_count, suspicious_objects_count]
        """
        f = self.get_features()
        return [
            f["phone_detected"],
            f["phone_confidence"],
            f["notes_detected"],
            f["extra_person_count"],
            f["suspicious_objects_count"],
        ]

    def to_tensor(self):
        """
        Returns a PyTorch FloatTensor of shape (1, 5) for direct fusion model input.
        Raises RuntimeError if PyTorch is not installed.
        """
        if not _TORCH_AVAILABLE:
            raise RuntimeError(
                "PyTorch is not installed. Cannot produce fusion tensor."
            )
        vec = self.get_feature_vector()
        return torch.tensor([vec], dtype=torch.float32)

    def reset(self):
        """Clear all observations (start a fresh window)."""
        self._buffer.clear()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse_detections(self, detection_output: dict) -> "_Observation":
        """Convert a raw ObjectDetector output dict into an _Observation."""
        timestamp = detection_output.get("timestamp", time.monotonic())
        detections = detection_output.get("detections", [])

        phone_seen = False
        phone_conf = 0.0
        notes_seen = False
        person_count = 0
        suspicious_set = set()

        for det in detections:
            cls_name = det.get("class_name", "").lower()
            conf = det.get("confidence", 0.0)

            if conf < self.confidence_threshold:
                continue

            if cls_name == PERSON_CLASS:
                person_count += 1

            elif cls_name == "cell phone":
                phone_seen = True
                phone_conf = max(phone_conf, conf)

            if cls_name in NOTES_PROXY_CLASSES:
                notes_seen = True

            if cls_name in SUSPICIOUS_CLASSES:
                suspicious_set.add(cls_name)

        if self.custom_notes_detector is not None:
            custom_notes_result = self.custom_notes_detector.detect(
                detection_output.get("frame_bgr")
            )
            if custom_notes_result is not None:
                notes_seen = bool(custom_notes_result)

        return _Observation(
            timestamp=timestamp,
            phone_seen=phone_seen,
            phone_conf=phone_conf,
            notes_seen=notes_seen,
            person_count=person_count,
            suspicious_set=suspicious_set,
        )

    def _prune_old_observations(self, current_timestamp: float) -> None:
        """Remove observations older than window_seconds from the buffer."""
        cutoff = current_timestamp - self.window_seconds
        while self._buffer and self._buffer[0].timestamp < cutoff:
            self._buffer.popleft()

    @staticmethod
    def _empty_features() -> dict:
        return {
            "phone_detected": 0,
            "phone_confidence": 0.0,
            "notes_detected": 0,
            "extra_person_count": 0,
            "suspicious_objects_count": 0,
            "person_count_anomaly": 0,
            "raw_person_count": 0,
            "window_obs_count": 0,
            "window_duration_seconds": 0.0,
            "persistence_scores": {},
        }


# ---------------------------------------------------------------------------
# Backward-compatibility shim
# ---------------------------------------------------------------------------

class EnvironmentFeatureExtractor(TemporalWindowAggregator):
    """
    Backward-compatible alias for TemporalWindowAggregator.

    Accepts the old `history_window_size` and `stability_threshold_ratio`
    keyword arguments and maps them to the new time-based interface.

    NOTE:  `history_window_size` was previously a *frame count*; it is
    ignored in the new implementation because the window is now time-based
    (window_seconds=10.0 by default).  Pass `window_seconds` directly to
    TemporalWindowAggregator for explicit control.
    """

    def __init__(
        self,
        history_window_size: int = 10,       # kept for API compat, ignored
        stability_threshold_ratio: float = 0.30,
        window_seconds: float = 10.0,
        confidence_threshold: float = 0.40,
        **kwargs,
    ):
        super().__init__(
            window_seconds=window_seconds,
            persistence_ratio=stability_threshold_ratio,
            confidence_threshold=confidence_threshold,
            **kwargs,
        )

    def extract_features(self, detection_output: dict) -> dict:
        """
        Drop-in replacement for the old extract_features() API.
        Delegates to update() and returns the full feature dict.
        """
        return self.update(detection_output)

    def reset_history(self):
        """Alias for reset() — backward compat."""
        self.reset()


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _mode(values: list) -> int:
    """Return the most frequent integer value in a list, or 0 if empty."""
    if not values:
        return 0
    counts = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    return max(counts, key=counts.get)
