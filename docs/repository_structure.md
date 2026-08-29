# Exam Proctoring Research Repository Architecture

This document describes the directory structure and layout of the consolidated exam proctoring research repository.

## Directory Layout

```
NOVEL-FRAMEWORK-FOR-MALPRACTICE-DETECTION-DURING-ONLINE-EXAM/
├── docs/                     # Technical documentation, schemas, and architecture
├── src/                      # Production and research Python packages
│   └── exam_proctoring/      # Core Python library
│       ├── gaze/             # Stream 1: Gaze tracking & estimation
│       ├── interaction/      # Stream 2: Mouse & keyboard interaction tracking
│       ├── environment/      # Stream 3: Environment camera feature extraction
│       ├── data/             # Dataset loaders & preprocessing abstractions
│       ├── models/           # Deep learning models (Single-stream, Pairwise, Fusion)
│       ├── training/         # Training loop, loss functions, & checkpointing
│       ├── evaluation/       # Metrics, ablation studies, & evaluation scripts
│       └── explainability/   # Model attribution & interpretation (e.g. attention/SHAP)
├── scripts/                  # Executable entry points for training, evaluation, & demos
├── data/                     # Canonical synthetic dataset & stress test suites
│   ├── synthetic/            # Canonical 3,600-window frozen dataset
│   └── stress_tests/         # Noisy and failure-case evaluation datasets
├── simulators/               # Data generation & simulation tools for reproducibility
├── models/                   # External pretrained model assets (e.g. MediaPipe tasks)
│   └── pretrained/gaze/
├── configs/                  # Experiment and model configuration YAML files
├── checkpoints/              # Model checkpoints saved during training
├── results/                  # Generated predictions, metrics, and explanations
├── figures/                  # Publication-ready figures and charts
├── tests/                    # Consolidated test suite
│   ├── unit/                 # Unit tests for individual components
│   ├── integration/          # Integration tests across streams & models
│   └── data/                 # Dataset and simulator integrity tests
├── notebooks/                # Jupyter notebooks for exploratory data analysis
├── paper/                    # Paper draft, TeX sources, and bib files
└── archive/                  # Archived legacy files and old documentation
```

## Key Modules & Locations

- **Source Code**: `src/exam_proctoring/`
- **Data Schemas**: `docs/data_schema.md`
- **Canonical Dataset**: `data/synthetic/dataset.csv` (Frozen)
- **Stress Test Datasets**: `data/stress_tests/`
- **Simulators**: `simulators/`
- **Attention Fusion Model**: `src/exam_proctoring/models/attention_fusion.py`
- **Pretrained Assets**: `models/pretrained/gaze/face_landmarker.task`
- **Test Suite**: `tests/` (Run via `python3 -m pytest tests/ -v`)
