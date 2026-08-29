"""
Temporal Gaze Feature Extractor.

Processes continuous frame-level gaze predictions over a configurable time window (default 10s)
and computes a stable 7-dimensional feature vector required by AttentionFusionModel.
"""

import math
import json
import collections
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class GazeFeatureExtractor:
    """
    Rolling window temporal feature extractor for gaze stream.

    Computes 7 gaze features:
        1. gaze_deviation       (mean normalized distance from screen center [0.5, 0.5])
        2. fixation_duration    (mean duration of stable fixation episodes in seconds)
        3. fixation_count       (number of fixation episodes in temporal window)
        4. saccade_velocity    (mean velocity of rapid gaze movements in screen-units/sec)
        5. gaze_confidence     (mean gaze confidence in temporal window)
        6. head_yaw            (mean absolute head yaw angle in degrees)
        7. head_pitch          (mean absolute head pitch angle in degrees)
    """

    def __init__(
        self,
        window_seconds: float = 10.0,
        fixation_velocity_threshold: float = 0.2,
        min_fixation_duration: float = 0.1,
        min_valid_samples: int = 3
    ):
        self.window_seconds = window_seconds
        self.fixation_velocity_threshold = fixation_velocity_threshold
        self.min_fixation_duration = min_fixation_duration
        self.min_valid_samples = min_valid_samples

        self.buffer = collections.deque()

    def reset(self):
        """Clears the temporal frame buffer."""
        self.buffer.clear()

    def update(self, frame_result: dict) -> dict:
        """
        Appends a new frame-level prediction dictionary to the temporal buffer,
        purges expired frames outside window_seconds, and computes the 7 features.
        """
        ts = frame_result.get("timestamp")
        if ts is None:
            import time
            ts = time.time()
            frame_result["timestamp"] = ts

        self.buffer.append(frame_result)

        # Purge frames older than current timestamp - window_seconds
        cutoff = ts - self.window_seconds
        while self.buffer and self.buffer[0]["timestamp"] < cutoff:
            self.buffer.popleft()

        return self.get_features()

    def get_features(self) -> dict:
        """
        Computes 7 temporal features over current window.

        Returns dictionary:
        {
            "timestamp": float,
            "features": {
                "gaze_deviation": float,
                "fixation_duration": float,
                "fixation_count": float,
                "saccade_velocity": float,
                "gaze_confidence": float,
                "head_yaw": float,
                "head_pitch": float
            },
            "feature_vector": list of 7 floats,
            "valid_samples": int
        }
        """
        current_ts = self.buffer[-1]["timestamp"] if self.buffer else 0.0

        # Filter valid samples where gaze coordinates are present
        valid_samples = [
            f for f in self.buffer
            if f.get("face_detected", False)
            and f.get("gaze_x") is not None
            and f.get("gaze_y") is not None
        ]

        if len(valid_samples) < self.min_valid_samples:
            # Default / insufficient data fallback
            default_feats = {
                "gaze_deviation": 0.0,
                "fixation_duration": 0.0,
                "fixation_count": 0.0,
                "saccade_velocity": 0.0,
                "gaze_confidence": 0.0,
                "head_yaw": 0.0,
                "head_pitch": 0.0
            }
            return {
                "timestamp": current_ts,
                "features": default_feats,
                "feature_vector": [0.0] * 7,
                "valid_samples": len(valid_samples)
            }

        # 1. Gaze Deviation
        # Max distance from (0.5, 0.5) in screen coordinates is sqrt(0.5^2 + 0.5^2) = ~0.7071
        max_dist = math.sqrt(0.5 ** 2 + 0.5 ** 2)
        deviations = [
            math.sqrt((s["gaze_x"] - 0.5) ** 2 + (s["gaze_y"] - 0.5) ** 2) / max_dist
            for s in valid_samples
        ]
        gaze_deviation = float(np.mean(deviations))

        # 2, 3, 4: Velocities, Fixations, Saccades
        velocities = []
        fixation_durations = []
        current_fixation_start = None
        fixation_count = 0

        for i in range(1, len(valid_samples)):
            dt = valid_samples[i]["timestamp"] - valid_samples[i - 1]["timestamp"]
            if dt <= 0.0 or dt > 1.0:
                continue

            dx = valid_samples[i]["gaze_x"] - valid_samples[i - 1]["gaze_x"]
            dy = valid_samples[i]["gaze_y"] - valid_samples[i - 1]["gaze_y"]
            dist = math.sqrt(dx ** 2 + dy ** 2)
            velocity = dist / dt
            velocities.append(velocity)

            if velocity <= self.fixation_velocity_threshold:
                if current_fixation_start is None:
                    current_fixation_start = valid_samples[i - 1]["timestamp"]
            else:
                if current_fixation_start is not None:
                    duration = valid_samples[i - 1]["timestamp"] - current_fixation_start
                    if duration >= self.min_fixation_duration:
                        fixation_durations.append(duration)
                        fixation_count += 1
                    current_fixation_start = None

        if current_fixation_start is not None:
            duration = valid_samples[-1]["timestamp"] - current_fixation_start
            if duration >= self.min_fixation_duration:
                fixation_durations.append(duration)
                fixation_count += 1

        fixation_duration_mean = float(np.mean(fixation_durations)) if fixation_durations else 0.0

        # Saccade velocities (rapid movements > threshold)
        saccade_vels = [v for v in velocities if v > self.fixation_velocity_threshold]
        saccade_velocity_mean = float(np.mean(saccade_vels)) if saccade_vels else (float(np.mean(velocities)) if velocities else 0.0)

        # 5. Gaze Confidence
        confidences = [s.get("gaze_confidence", 0.0) for s in valid_samples]
        gaze_confidence = float(np.mean(confidences))

        # 6 & 7. Head Yaw & Pitch (mean absolute degrees)
        yaws = [abs(s.get("head_yaw", 0.0)) for s in valid_samples if s.get("head_yaw") is not None]
        pitches = [abs(s.get("head_pitch", 0.0)) for s in valid_samples if s.get("head_pitch") is not None]

        head_yaw_mean = float(np.mean(yaws)) if yaws else 0.0
        head_pitch_mean = float(np.mean(pitches)) if pitches else 0.0

        feats = {
            "gaze_deviation": float(np.clip(gaze_deviation, 0.0, 1.0)),
            "fixation_duration": float(fixation_duration_mean),
            "fixation_count": float(fixation_count),
            "saccade_velocity": float(saccade_velocity_mean),
            "gaze_confidence": float(np.clip(gaze_confidence, 0.0, 1.0)),
            "head_yaw": float(head_yaw_mean),
            "head_pitch": float(head_pitch_mean)
        }

        vector = [
            feats["gaze_deviation"],
            feats["fixation_duration"],
            feats["fixation_count"],
            feats["saccade_velocity"],
            feats["gaze_confidence"],
            feats["head_yaw"],
            feats["head_pitch"]
        ]

        return {
            "timestamp": current_ts,
            "features": feats,
            "feature_vector": vector,
            "valid_samples": len(valid_samples)
        }

    def get_feature_vector(self) -> list:
        """Returns ordered list of 7 feature values."""
        res = self.get_features()
        return res["feature_vector"]

    def to_tensor(self):
        """Returns PyTorch tensor of shape (1, 7) for direct fusion model input."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not installed in the environment.")
        vec = self.get_feature_vector()
        return torch.tensor([vec], dtype=torch.float32)

    def to_json(self) -> str:
        """Returns machine-readable JSON string of current window features."""
        res = self.get_features()
        return json.dumps(res, indent=4)
