# Person 3 — Computer Vision Workspace (p3/)

This directory contains the Computer Vision component of the online exam malpractice detection system, covering camera streaming, face/iris landmark tracking, and real-time environment & object detection.

---

## 1. Architecture & Pipelines

### Face & Iris Landmark Pipeline (Task 2)
```
Laptop Webcam → OpenCV (camera.py) → MediaPipe FaceMesh → Face (478 3D) & Iris Landmarks
```

### Environment & Object Detection Pipeline (Task 3)
```
Laptop Webcam
      ↓
OpenCV Camera (p3/src/camera.py)
      ↓
YOLO Object Detector (p3/src/object_detector.py)
      ↓
Raw Detections [class, confidence, bbox]
      ↓
Environment Feature Extractor (p3/src/environment_features.py)
      ↓
Environment Evidence & Temporal Persistence
```

---

## 2. Component Details

### Object Detection (`p3/src/object_detector.py`)
- **Model**: `YOLOv8n` (`yolov8n.pt` downloaded into `p3/models/`).
- **Rationale**: Chosen for its lightweight architecture (6.3 MB) enabling real-time CPU inference on standard laptop hardware without requiring dedicated CUDA GPUs.
- **Hardware Target**: CPU mode via PyTorch CPU backend (`device='cpu'`), with automatic GPU CUDA detection.
- **Confidence Threshold**: Configurable parameter (default `0.40`).
- **Structured Output**:
  ```json
  {
      "class_id": 67,
      "class_name": "cell phone",
      "confidence": 0.8700,
      "bbox": [420, 210, 530, 390],
      "is_relevant": true,
      "timestamp": 1787902821.5,
      "frame_index": 42
  }
  ```

### Project-Relevant Classes
Configured in `DEFAULT_RELEVANT_CLASSES`:
- `cell phone`
- `book`
- `laptop`
- `person`
- `backpack`
- `keyboard`
- `mouse`
- `remote`
- `bottle`
- `tv`

### Environment Features & Temporal Persistence (`p3/src/environment_features.py`)
- **Feature Extraction**: Aggregates `person_count`, `phone_detected`, `book_detected`, `laptop_detected`, and `relevant_object_count`.
- **Temporal Persistence**: Uses a 10-frame sliding window with a 50%+ presence ratio requirement to filter out single-frame false positives before marking evidence as `stable`.
- **No Malpractice Decisions**: Does not compute a hard `cheating = true` flag. It provides evidence metrics for downstream feature fusion.

---

## 3. Directory Structure

```
p3/
├── src/
│   ├── camera.py
│   ├── face_landmarks.py
│   ├── object_detector.py
│   └── environment_features.py
├── tests/
│   ├── test_camera.py
│   ├── test_face_landmarks.py
│   └── test_object_detection.py
├── models/
│   └── yolov8n.pt
├── docs/
├── requirements.txt
└── README.md
```

---

## 4. Requirements & Setup

1. Dependencies in `p3/requirements.txt`:
   - `opencv-python`
   - `numpy`
   - `mediapipe==0.10.14`
   - `ultralytics`
   - `torch`
   - `torchvision`
   - `pyyaml`
   - `psutil`
   - `requests`

2. Activate virtual environment (created at repo root):
   ```bash
   source ../venv/bin/activate
   ```

3. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

---

## 5. Running Tests

- **Run Camera Test**:
  ```bash
  python3 tests/test_camera.py
  ```

- **Run Face & Iris Landmark Detection Test**:
  ```bash
  python3 tests/test_face_landmarks.py
  ```

- **Run Object & Environment Detection Test**:
  ```bash
  python3 tests/test_object_detection.py
  ```
  *(Options: `--conf 0.40`, `--max-frames 50`)*

---

## 6. Performance & Limitations

- **YOLO Inference Latency**: `~127.10 ms` on CPU (`~7.9 FPS`).
- **Complete Pipeline Rate**: `~7.6 FPS` (Webcam Capture + YOLO + Feature Extraction + Render).
- **False-Positive & Environment Limitations**:
  - Presence of a phone on a desk does not indicate active usage.
  - Partial occlusions, heavy shadows, or book-like rectangular objects (e.g., tablet cases) can trigger false positives.
