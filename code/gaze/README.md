# NOVA Gaze Estimation Component

## 1. Overview
The **Gaze Estimation Component** is the vision sensing stream of the NOVA Online Exam Malpractice Detection Framework. It extracts 3D facial mesh and iris landmarks using a pretrained MediaPipe Face Landmarker model, computes relative iris displacement vectors, estimates head orientation (yaw and pitch), calibrates raw gaze observations to normalized screen coordinates, and extracts a stable **7-dimensional temporal gaze feature vector** over a rolling window for consumption by the `AttentionFusionModel`.

---

## 2. Sensing Architecture & Research Context
In this framework, **MediaPipe is treated as a pretrained sensing backbone** for facial geometric landmark extraction rather than a novel gaze network trained from scratch. 

The novelty of the research resides in the **multimodal fusion architecture** combining gaze dynamics, computer interaction behavior, and environment observations. The gaze module extracts geometric features and temporal statistics to produce machine-readable vectors without reliance on third-party cloud APIs.

```
       Webcam / Frame Input
                │
                ▼
    MediaPipe Face Landmarker Tasks API
                │
                ▼
     478 3D Mesh + Iris Landmarks
                │
                ▼
   Normalized Eye / Iris Geometry + Head Pose (solvePnP)
                │
                ▼
      Raw Observation (raw_x, raw_y)
                │
                ▼
   3x3 Polynomial Calibration Mapping
                │
                ▼
   Screen Coordinates (gaze_x, gaze_y) ∈ [0, 1] x [0, 1]
                │
                ▼
    Rolling Temporal Feature Extractor (10s window)
                │
                ▼
  7-Dimensional Gaze Feature Vector -> AttentionFusionModel
```

---

## 3. Installation & Dependencies

### Environment Requirements:
- Python 3.10
- OpenCV (`opencv-python`)
- MediaPipe (`mediapipe >= 1.0.0`)
- NumPy (`numpy < 2.0.0`)
- PyTorch (`torch`, optional for direct tensor output)

```bash
pip install opencv-python mediapipe "numpy<2.0.0" torch
```

---

## 4. Model Download & Setup
The component relies on the official Google MediaPipe Face Landmarker model asset:
`models/gaze/face_landmarker.task`

If the file is absent, `GazeEstimator` automatically downloads it from the official Google Storage endpoint on first initialization.

To manually trigger download:
```python
from gaze.gaze_estimator import ensure_model_file
ensure_model_file()
```

---

## 5. Mathematical Definitions of the 7 Gaze Features

The temporal feature extractor converts frame-level predictions into a 7-dimensional vector $(x_1, x_2, x_3, x_4, x_5, x_6, x_7)$ over a rolling window (default 10 seconds):

1. **`gaze_deviation`**: Mean normalized Euclidean distance of calibrated gaze $(x_i, y_i)$ from screen center $(0.5, 0.5)$:
   $$\text{gaze\_deviation} = \frac{1}{N} \sum_{i=1}^N \frac{\sqrt{(x_i - 0.5)^2 + (y_i - 0.5)^2}}{\sqrt{0.5^2 + 0.5^2}} \in [0.0, 1.0]$$

2. **`fixation_duration`**: Average duration (in seconds) of stable fixation episodes where gaze velocity is below `fixation_velocity_threshold` (0.2 screen-units/s).

3. **`fixation_count`**: Total count of distinct gaze fixation episodes detected within the temporal window.

4. **`saccade_velocity`**: Average velocity (screen units per second) of rapid eye movements (saccades) exceeding the fixation threshold.

5. **`gaze_confidence`**: Average reliability score across valid frames in the temporal window, accounting for face presence, eye aspect ratio (EAR), and extreme head pose angles.

6. **`head_yaw`**: Mean absolute head rotation angle around vertical Y-axis (in degrees).

7. **`head_pitch`**: Mean absolute head rotation angle around horizontal X-axis (in degrees).

---

## 6. How Calibration Works
Calibration fits a 2nd-order polynomial mapping from raw iris/head observations $(x_{raw}, y_{raw}, \text{yaw}, \text{pitch})$ to normalized screen coordinates $(x_{target}, y_{target}) \in [0, 1] \times [0, 1]$.

- **Grid**: 3x3 uniform grid points on screen space.
- **Sample Collection**: Default 30 valid samples collected per grid point.
- **Outlier Filtering**: Samples are filtered using Median Absolute Deviation (MAD) to reject blinks, face movement artifacts, and dropped frames.
- **Regression Fit**: Ridge regression with L2 regularization ($\alpha=10^{-3}$) fits polynomial feature matrix $X$:
  $$X = [1, x, y, \text{yaw}, \text{pitch}, x^2, y^2, xy, x\cdot\text{yaw}, y\cdot\text{pitch}]$$
- **Storage**: Fitted coefficients are saved to `models/gaze/calibration.json`.

---

## 7. Commands & Execution

### Run Automated Unit & Integration Tests:
```bash
python3.10 -m pytest tests/ -v
```

### Run Performance Benchmark:
```bash
python3.10 code/gaze/benchmark.py
```

### Run Interactive Calibration UI:
```bash
python3.10 code/gaze/calibration.py --samples 30 --camera 0
```

### Run Real-time Webcam Demo:
```bash
python3.10 code/gaze/demo.py --camera 0 --calibration models/gaze/calibration.json
```

---

## 8. JSON Output Format

### Frame-Level Prediction Schema:
```json
{
    "timestamp": 1787901733.105,
    "gaze_x": 0.512,
    "gaze_y": 0.488,
    "raw_gaze_x": 0.505,
    "raw_gaze_y": 0.491,
    "gaze_confidence": 0.945,
    "head_yaw": -2.3,
    "head_pitch": 1.1,
    "face_detected": true
}
```

### Window-Level Feature Output Schema:
```json
{
    "timestamp": 1787901733.105,
    "features": {
        "gaze_deviation": 0.034,
        "fixation_duration": 1.820,
        "fixation_count": 6.0,
        "saccade_velocity": 0.125,
        "gaze_confidence": 0.940,
        "head_yaw": 2.10,
        "head_pitch": 1.05
    },
    "feature_vector": [0.034, 1.820, 6.0, 0.125, 0.940, 2.10, 1.05],
    "valid_samples": 300
}
```

---

## 9. Integration with AttentionFusionModel
To feed the gaze stream output into the `AttentionFusionModel` (`exam-proctoring/fusion/attention_fusion.py`):

```python
from gaze import GazeEstimator, GazeFeatureExtractor
from fusion.attention_fusion import AttentionFusionModel

estimator = GazeEstimator()
feature_extractor = GazeFeatureExtractor(window_seconds=10.0)
fusion_model = AttentionFusionModel(gaze_dim=7, interaction_dim=7, environment_dim=5)

# Per-frame loop:
result = estimator.process_frame(frame)
temporal_feats = feature_extractor.update(result)

# Convert to PyTorch tensor (shape 1x7)
gaze_tensor = feature_extractor.to_tensor()

# Pass to multi-modal fusion model
risk_score, attention_weights = fusion_model(gaze_tensor, interaction_tensor, environment_tensor)
```

---

## 10. Limitations & Edge Cases
- **No Face Detected**: When the user turns completely away or blocks the camera, `face_detected` is set to `False`, `gaze_confidence` drops to `0.0`, and coordinates return `None` (no false coordinate hallucination).
- **Extreme Lighting / Reflection**: Reflective glasses or extreme backlighting can reduce iris landmark confidence; the confidence score degrades smoothly.
- **CPU Performance**: Benchmark achieves **~161 FPS (6.19 ms/frame latency)** on standard x86 CPU, making it suitable for resource-constrained examination terminals.
