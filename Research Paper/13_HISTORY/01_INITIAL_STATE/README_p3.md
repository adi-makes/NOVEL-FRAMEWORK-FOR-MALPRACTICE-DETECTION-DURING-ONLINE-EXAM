# Person 3 — Computer Vision Workspace (`p3/`)

This directory is the **Person 3** contribution to the online exam malpractice detection system.
It implements the **environment/object-detection stream** of the three-stream fusion architecture.

---

## 1. Architecture

```
Laptop Webcam
      ↓
Camera  (p3/src/camera.py — OpenCV VideoCapture)
      ↓
ObjectDetector  (p3/src/object_detector.py — YOLOv8n, COCO)
      ↓
Raw Detections  [class_name, confidence, bbox, timestamp]
      ↓
TemporalWindowAggregator  (p3/src/environment_features.py)
      10-second real-time rolling window
      Temporal persistence filtering (≥30% observation ratio)
      ↓
5 Environment Features  (fusion-compatible)
      ↓
EnvironmentFusionAdapter  (p3/src/env_fusion_adapter.py)
      ↓
torch.FloatTensor  shape (1, 5)
      ↓
AttentionFusionModel  (exam-proctoring/fusion/attention_fusion.py)
```

Face/iris landmark pipeline (Task 2, `face_landmarks.py`) is a **separate pipeline** and is not part of the environment stream.

---

## 2. Five Environment Features

These are the **primary outputs** of the environment stream. They form the 5-dimensional environment vector consumed by `AttentionFusionModel(environment_dim=5)`.

The order in the fusion vector is fixed and must not be changed:

| Index | Feature | Type | Description |
|-------|---------|------|-------------|
| 0 | `phone_detected` | `int` (0 or 1) | Phone stably observed in current window |
| 1 | `phone_confidence` | `float` [0, 1] | Max detection confidence across phone observations |
| 2 | `notes_detected` | `int` (0 or 1) | COCO `book` class stably observed (see limitation §6) |
| 3 | `extra_person_count` | `int` ≥ 0 | Number of people beyond the expected 1 person |
| 4 | `suspicious_objects_count` | `int` ≥ 0 | Count of distinct stably-detected suspicious classes |

Additional diagnostic metadata is available (not in the fusion vector):

```python
{
    "person_count_anomaly":    int,   # 1 if extra_person_count > 0, else 0
    "raw_person_count":        int,   # mode of person counts in window
    "window_obs_count":        int,   # frames currently in the window
    "window_duration_seconds": float, # elapsed time of current window
    "persistence_scores":      dict,  # per-class observation ratios
}
```

---

## 3. Object Detector (`p3/src/object_detector.py`)

- **Model**: `YOLOv8n` — nano variant, 6.3 MB
- **Source**: Ultralytics pretrained on COCO 80-class dataset
- **Confidence threshold**: `0.40` (configurable, shared with aggregator)
- **Device**: auto-selects CUDA if available; CPU fallback guaranteed
- **Output per frame**: list of detection dicts:
  ```python
  {
      "class_id":   int,
      "class_name": str,
      "confidence": float,
      "bbox":       [x1, y1, x2, y2],
      "is_relevant": bool,
      "timestamp":  float,
      "frame_index": int
  }
  ```
- Detection and camera capture are **kept separate** (`camera.py` owns `cv2.VideoCapture`)

### Relevant COCO classes (broad filter for `is_relevant` flag)
`cell phone`, `book`, `laptop`, `person`, `backpack`, `keyboard`, `mouse`, `remote`, `tv`, `bottle`

---

## 4. Temporal Window Aggregator (`p3/src/environment_features.py`)

### Time-based window (not frame-count-based)

The aggregator uses `time.monotonic()` / detection timestamps to maintain a **real-time rolling 10-second window**. This works correctly regardless of camera FPS (10, 20, 30 fps).

```
At each frame:
    1. Parse detections from ObjectDetector output
    2. Append _Observation to buffer
    3. Prune observations older than window_seconds
    4. Compute 5 features from current buffer
```

### Persistence filtering

An object must be observed in **≥ 30%** of frames within the window to be counted as stable. A single noisy detection is ignored.

Example: phone seen in 9/30 frames (30%) → `phone_detected = 1`
Example: phone seen in 1/30 frames (3.3%) → `phone_detected = 0`

Configurable via `persistence_ratio` (default `0.30`).

### Suspicious classes (centralized, conservative)

```python
SUSPICIOUS_CLASSES = {
    "cell phone",   # direct cheating tool
    "remote",       # indicates TV/media device nearby
    "book",         # notes/reference material proxy
}
```

**Not classified as suspicious**: `keyboard`, `mouse`, `laptop`, `bottle` — normal exam equipment.

---

## 5. Fusion Adapter (`p3/src/env_fusion_adapter.py`)

Thin adapter between `TemporalWindowAggregator` and `AttentionFusionModel`.

```python
from p3.src.env_fusion_adapter import EnvironmentFusionAdapter

adapter = EnvironmentFusionAdapter()
tensor = adapter.to_tensor(env_feature_dict)   # shape (1, 5), dtype float32
vector = adapter.to_vector(env_feature_dict)   # list of 5 floats
```

The fusion model can also be fed directly from the aggregator:

```python
tensor = aggregator.to_tensor()   # shape (1, 5)
```

---

## 6. Notes Detection — Limitation

> **IMPORTANT: Notes detection is approximate.**

The COCO-pretrained YOLOv8n model **does not have a `notes` class**.
It has a `book` class.

We map: `COCO "book"` → `notes_detected`

**False positives**: textbooks, tablet covers, large notebooks, binders  
**False negatives**: handwritten notes on paper, folded sheets, notes out of frame

This is a known limitation. The architecture provides a `CustomNotesDetector` hook in `environment_features.py`:

```python
class CustomNotesDetector:
    def detect(self, frame_bgr) -> Optional[bool]:
        return None  # stub — override in subclass
```

Pass a `CustomNotesDetector` instance to `TemporalWindowAggregator(custom_notes_detector=...)` to replace the COCO proxy without rewriting the pipeline.

---

## 7. Directory Structure

```
p3/
├── src/
│   ├── camera.py                # OpenCV webcam interface
│   ├── object_detector.py       # YOLOv8n wrapper (detection only)
│   ├── environment_features.py  # Temporal window aggregator (main logic)
│   ├── env_fusion_adapter.py    # Fusion vector adapter
│   ├── face_landmarks.py        # MediaPipe face/iris (separate pipeline)
│   └── benchmark.py             # Inference latency & FPS measurement
├── tests/
│   ├── test_environment_features.py  # ★ Unit tests (no webcam needed)
│   ├── test_object_detection.py      # Integration demo (webcam)
│   ├── test_camera.py                # Camera capture test
│   └── test_face_landmarks.py        # Face/iris landmark test
├── models/
│   └── yolov8n.pt               # Pre-downloaded YOLO nano weights
├── run_tests.sh                 # P3-only pytest runner
├── requirements.txt
└── README.md
```

---

## 8. Requirements & Setup

Dependencies (`p3/requirements.txt`):
```
opencv-python
numpy
mediapipe==0.10.14
ultralytics
torch
torchvision
pyyaml
psutil
requests
```

Activate the project virtual environment (at repo root):
```bash
source venv/bin/activate
```

Install:
```bash
pip install -r p3/requirements.txt
```

---

## 9. Running

### Unit Tests (recommended — no webcam required)
```bash
cd /home/agnivesh/novel
./p3/run_tests.sh -q
```

This machine has ROS pytest plugins installed globally, so the runner disables
external pytest plugin autoload for P3 tests. Equivalent explicit command:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MPLCONFIGDIR=/tmp/matplotlib-cache \
  venv/bin/python -m pytest p3/tests -q
```

### Webcam Integration Demo
```bash
source venv/bin/activate
python p3/tests/test_object_detection.py
python p3/tests/test_object_detection.py --conf 0.40 --max-frames 200
```

### Benchmark (synthetic frames — no webcam needed)
```bash
source venv/bin/activate
python p3/src/benchmark.py --no-camera --frames 100
```

### Benchmark (with webcam)
```bash
source venv/bin/activate
python p3/src/benchmark.py --frames 100 --warmup 10
```

### Camera Test
```bash
python p3/tests/test_camera.py
```

---

## 10. Performance

Measured locally with:

```bash
env MPLCONFIGDIR=/tmp/matplotlib-cache venv/bin/python p3/src/benchmark.py --frames 20 --warmup 5
```

| Metric | Value |
|--------|-------|
| Python | 3.12.3 |
| OpenCV | 5.0.0 |
| Ultralytics | 8.4.131 |
| PyTorch | 2.13.0+cu130 |
| YOLO model | yolov8n (6.3 MB) |
| Device | CPU |
| Input resolution | 640x480 webcam frames |
| YOLO inference latency | mean 87.26 ms, p50 85.79 ms, p95 106.77 ms |
| Camera capture latency | mean 0.38 ms |
| Pipeline FPS (end-to-end) | 11.4 FPS |

The detector still auto-selects CUDA when available and otherwise falls back to CPU.

---

## 11. Limitations

| Limitation | Details |
|-----------|---------|
| **Notes false positives** | COCO `book` ≠ exam notes; textbooks trigger `notes_detected` |
| **Notes false negatives** | Handwritten notes on loose paper are not detected |
| **Phone occlusion** | Phone hidden under desk or behind back not visible to camera |
| **Lighting sensitivity** | Poor lighting degrades YOLO confidence significantly |
| **Camera angle** | Detection quality depends on camera placement and FoV |
| **Person count** | Multiple overlapping people may be undercounted (occlusion) |
| **Pretrained model** | No fine-tuning on exam environments; general COCO classes only |
| **No custom notes model** | `CustomNotesDetector` hook is a stub pending future training |
| **No active phone-usage inference** | Detecting a phone on a desk ≠ active use during exam |

---

## 12. Interface Contract

This stream outputs a 5-dim vector matching `data_schema.md §5 Environment Features`:

```json
{
  "phone_detected":           0,
  "phone_confidence":         0.0,
  "notes_detected":           0,
  "extra_person_count":       0,
  "suspicious_objects_count": 0
}
```

The fusion model receives `environment_tensor` of shape `(batch, 5)`.
No malpractice/cheating labels are produced by this stream — evidence only.

---

## 13. Remaining Work

| Item | Status |
|------|--------|
| Custom notes detection model (fine-tuned) | ❌ Not yet — stub hook provided |
| Custom suspicious-object dataset | ❌ Not yet |
| Model fine-tuning on exam environments | ❌ Not yet |
| Fusion integration end-to-end test | ⚠️ Partial — adapter + tensor shape verified |
| CCTV / fixed-camera support | ❌ Out of scope for 1-week sprint |
