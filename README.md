# Novel Framework for Malpractice Detection During Online Exam

Research repository for multi-stream attention fusion malpractice detection during online examinations.

## Architectural Streams

The framework ingests three synchronized 10-second window sensing streams:
1. **Gaze Stream**: Eye tracking, fixation duration/count, saccade velocity, gaze deviation, confidence, head yaw/pitch (7 features).
2. **Interaction Stream**: Mouse velocity mean/std, click/keystroke frequency, idle fraction, tab switches, velocity spikes (7 features).
3. **Environment Stream**: Computer vision detection for phone, notes, extra persons, and suspicious objects (5 features).

Total Domain Features: **19 features**.

## Repository Structure

```
.
├── docs/                     # Schemas, architecture diagrams, and protocols
├── src/
│   └── exam_proctoring/      # Primary importable package
│       ├── gaze/             # Gaze estimation & feature extraction
│       ├── interaction/      # Mouse/keyboard interaction modules
│       ├── environment/      # OpenCV & YOLO environment detection
│       ├── data/             # Dataset abstractions
│       ├── models/           # Single-stream, pairwise, & fusion models
│       ├── training/         # Training pipeline (trainer, loss, checkpoints)
│       ├── evaluation/       # Metrics & evaluation scripts
│       └── explainability/   # Model interpretability
├── scripts/                  # Runnable scripts (benchmarks, demos, training)
├── data/                     # Canonical frozen dataset and stress suites
├── simulators/               # Data simulators for reproducibility
├── models/                   # Pretrained weights (MediaPipe, YOLO)
├── configs/                  # Experiment YAML configs
├── tests/                    # Unit, integration, & data integrity tests
└── archive/                  # Legacy artifacts and documentation
```

## Quick Start

### Installation

Install in editable mode:
```bash
pip install -e .
```

### Running Tests

Run the full consolidated test suite:
```bash
python3 -m pytest tests/ -v
```

### Verification & Demos

- **Benchmark Gaze Pipeline**:
  ```bash
  python3 scripts/benchmark_gaze.py
  ```
- **Gaze Tracking Demo**:
  ```bash
  python3 scripts/run_gaze_demo.py
  ```