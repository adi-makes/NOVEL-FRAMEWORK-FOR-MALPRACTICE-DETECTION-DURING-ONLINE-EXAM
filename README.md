# Novel Framework for Malpractice Detection During Online Exam

Research repository for multi-stream attention fusion malpractice detection during online examinations.

## Overview

This project implements an **explainable multi-modal exam proctoring system** that detects suspicious behavioral patterns during online examinations using three synchronized sensing streams. The framework is designed to support informed decision-making by educators while respecting student privacy and accessibility needs.

## Core Architecture

The system fuses evidence from three synchronized 10-second window sensing streams:

### 1. Gaze Stream (7 features)
- **fixation_duration_mean**: Average duration of eye fixations
- **fixation_count**: Number of eye fixation events
- **saccade_velocity_mean**: Average speed of eye movements
- **gaze_deviation**: Degree of eye movement away from target
- **gaze_confidence**: Confidence in gaze estimation
- **head_yaw**: Horizontal head rotation (degrees)
- **head_pitch**: Vertical head rotation (degrees)

### 2. Interaction Stream (7 features)
- **cursor_velocity_mean**: Average mouse cursor speed
- **cursor_velocity_std**: Variability in cursor movement
- **click_frequency**: Number of clicks per second
- **keystroke_frequency**: Keyboard input rate
- **idle_fraction**: Proportion of time with no activity
- **tab_switch_count**: Number of window focus changes
- **velocity_spike_ratio**: Ratio of maximum velocity to mean velocity

### 3. Environment Stream (5 features)
- **phone_detected**: Whether a phone is detected in camera view
- **phone_confidence**: Confidence in phone detection (0-1)
- **notes_detected**: Whether notes are visible
- **extra_person_count**: Number of additional people detected
- **suspicious_objects_count**: Count of suspicious objects detected

**Total Domain Features: 19 features** per synchronized 10-second analysis window.

## System Philosophy

### What We Are NOT
- Continuous background monitoring
- Biometric identification without consent
- Binary "catch cheaters" decisions
- Invasive surveillance systems

### What We ARE
- **Calibrated risk scoring** for suspicious patterns
- **Evidence surfacing** for human review
- **Explainable alerts** with clear reasons
- **Privacy-respecting** design with local processing
- **Accessibility-aware** implementation

## Repository Structure

```
.
├── docs/                     # Documentation & specifications
├── src/                      # Main package (exam_proctoring)
│   ├── gaze/                # Gaze estimation & feature extraction
│   ├── interaction/         # Mouse/keyboard interaction modules
│   ├── environment/         # OpenCV & YOLO environment detection
│   ├── data/                # Dataset abstractions
│   ├── models/              # Single-stream, pairwise, & fusion models
│   ├── training/            # Training pipeline (trainer, loss, checkpoints)
│   ├── evaluation/          # Metrics & evaluation scripts
│   └── explainability/      # Model interpretability
├── scripts/                 # Runnable scripts (benchmarks, demos, training)
├── data/                    # Canonical frozen dataset and stress suites
├── simulators/              # Data simulators for reproducibility
├── models/                  # Pretrained weights (MediaPipe, YOLO)
├── configs/                 # Experiment YAML configs
├── tests/                   # Unit, integration, & data integrity tests
├── results/                 # Experiment results and metrics
├── paper/                   # Research paper documentation
├── notebooks/               # Jupyter tutorials and exploration
├── training/                # Training scripts and utilities
├── archive/                 # Legacy artifacts and documentation
└── .claude/                 # Configuration for Claude Code integration
```

## Installation & Setup

### Quick Start

```bash
# Install in editable mode
pip install -e .

# Verify installation
python -c "import exam_proctoring; print('Import successful')"

# Run the full test suite
python3 -m pytest tests/ -v
```

### Development Dependencies

The project uses:
- **Python**: >=3.9
- **Torch**: For deep learning models
- **OpenCV**: Computer vision tasks
- **MediaPipe**: Gaze estimation
- **Ultralytics/YOLO**: Object detection
- **scikit-learn**: Machine learning utilities

## Model Architecture

### Three Fusion Strategies

1. **Early Fusion** (`EarlyFusionModel`)
   - Concatenates all 19 raw features → 128 → 64 → 1 logit
   - Simple but ignores stream importance

2. **Late Fusion** (`LateFusionModel`)
   - Independent per-stream heads with learnable softmax-normalized weights
   - Interpretable but misses cross-stream patterns

3. **Attention Fusion** (`AttentionFusionModel`) - *The proposed architecture*
   - Each stream encoded to 128 dimensions
   - Cross-modal self-attention across 3 modality tokens [Gaze, Interaction, Environment]
   - 384→64→1 classification head with sigmoid output

### Model Registry

The system includes 9 pre-defined model configurations:

| Key | Streams | Name |
|-----|---------|------|
| `gaze_only` | gaze | Gaze-only |
| `interaction_only` | interaction | Interaction-only |
| `environment_only` | environment | Environment-only |
| `gaze_interaction` | gaze + interaction | Gaze + Interaction |
| `gaze_environment` | gaze + environment | Gaze + Environment |
| `interaction_environment` | interaction + environment | Interaction + Environment |
| `early_fusion` | all 3 | Early Fusion |
| `late_fusion` | all 3 | Late Fusion |
| `attention_fusion` | all 3 | Three-stream Attention Fusion (Proposed) |

**Build a model:**
```python
from exam_proctoring.models import build_model

# Build the proposed attention fusion model
model = build_model("attention_fusion", dropout=0.2)
```

## Data Schema & Format

### Exam Window Structure

Each 10-second analysis window contains synchronized data from all three streams:

- **Gaze**: Screen coordinates, head pose (yaw/pitch), confidence, aggregated features
- **Interaction**: Cursor velocity, click/keystroke frequency, idle time, tab switches
- **Environment**: Phone/notes/person detection, confidence scores, object counts
- **Label**: Ground truth ("honest" or "cheating"), optional cheating type

### Critical Design: Session-Level Splitting

**Windows from the same session must never appear in both training and testing splits.** This prevents temporal/session leakage. Use `verify_dataset_integrity()` to validate.

## Training Pipeline

### Experiment Configuration

See `configs/experiment_config.yaml`:

```yaml
seed: 42
learning_rate: 0.001
batch_size: 32
epochs: 50
patience: 8  # For early stopping
dropout: 0.2
```

### Running the Full Experiment Pipeline

```bash
# Master orchestrator runs 6 stages:
python3 scripts/run_experiments.py

# Stage 1: Dataset & DataLoaders
# Stage 2: Build & Train Models
# Stage 3: Test Evaluation
# Stage 4: Hypothesis Evaluation
# Stage 5: Stress Testing
# Stage 6: Figures & Visualization
```

## Key Files & Their Purposes

| Path | Purpose |
|------|---------|
| `src/exam_proctoring/models/registry.py` | Model registry with 9 model keys; `build_model()` constructor |
| `src/exam_proctoring/models/attention_fusion.py` | The proposed attention fusion architecture |
| `src/exam_proctoring/data/dataset.py` | `ExamDataset` and `create_dataloaders()` |
| `src/exam_proctoring/training/trainer.py` | `Trainer` class with training loop |
| `src/exam_proctoring/evaluation/metrics.py` | `calculate_metrics()` and `find_optimal_threshold()` |
| `scripts/run_experiments.py` | Master orchestrator for full pipeline |
| `configs/experiment_config.yaml` | Experiment hyperparameters |
| `docs/fusion_architecture.md` | Detailed architecture documentation |
| `docs/data_schema.md` | Data schema contract |

## Testing Infrastructure

```bash
# Run all tests
python3 -m pytest tests/ -v

# Unit tests
python3 -m pytest tests/unit/ -v

# Integration tests
python3 -m pytest tests/integration/ -v
```

### Test Categories
- **Gaze**: `test_gaze_estimator.py`, `test_gaze_calibration.py`, `test_gaze_features.py`
- **Environment**: `test_environment.py`, `test_object_detection.py`
- **Models**: `test_models_stack.py`
- **Integration**: `test_attention_fusion.py`, `test_gaze_fusion_integration.py`

## Scripts & Utilities

- **Benchmark Gaze**: `scripts/benchmark_gaze.py`
- **Gaze Tracking Demo**: `scripts/run_gaze_demo.py`
- **Environment Demo**: `scripts/run_environment_demo.py`
- **Experiments**: `scripts/run_experiments.py`

## Simulators

For reproducible research:

- **Gaze Simulator**: `simulators/gaze_simulator.py`
- **Mouse Simulator**: `simulators/mouse_simulator.py`
- **Environment Simulator**: `simulators/environment_simulator.py`
- **Exam Simulator**: `simulators/exam_simulator.py`
- **Dataset Generator**: `simulators/generate_dataset.py`

## Documentation

- **README.md** - High-level overview and quick start
- **CLAUDE.md** - Guidance for Claude Code integration
- **docs/fusion_architecture.md** - Detailed architecture with diagrams
- **docs/data_schema.md** - Data schema contract and specifications
- **Complete_Plan.md** - Comprehensive 18-month project plan
- **ONE_WEEK_SPRINT_PLAN.md** - Weekly sprint planning

## Results & Artifacts

- **results/metrics/**: Evaluation metrics and hypothesis tests
- **results/predictions/**: Model prediction outputs
- **results/stress_tests/**: Robustness testing results
- **checkpoints/**: Pre-trained model weights

## Ethical & Responsible AI

### Core Principles
1. **Explainability**: Every alert has a clear, human-readable reason
2. **Fairness**: No bias against accessibility needs (glasses, neurodivergence)
3. **Privacy**: Minimal data collection, local processing, consent requirements
4. **Human-in-the-Loop**: AI surfaces evidence, humans make final decisions

### Guardrails
- No continuous monitoring
- No biometric identification without consent
- No punishment orientation
- Transparency for student appeals
- IRB approval required for real participant studies

## Acknowledgements

This project builds upon:
- **Hu et al. (2024)**: Three-camera gaze+object fusion
- **Li et al. (2021)**: Video + mouse analytics
- **Atoum et al. (2017)**: Multimodal baseline
- **MediaPipe**: Gaze estimation
- **Ultralytics/YOLO**: Object detection

---

**Last Updated:** August 30, 2026
**Version:** 0.1.0
