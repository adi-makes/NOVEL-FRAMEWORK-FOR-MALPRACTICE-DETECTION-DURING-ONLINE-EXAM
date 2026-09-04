# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Working Conventions

- **Branch policy**: Work on the current working branch by default. Create a new branch only if the user explicitly says so. Never push to `main`/`master`, force-push, or merge without explicit approval.
- **Read before editing**: Always read a file (or use the search tools) before modifying it — don't assume contents.
- **Match surrounding code**: Match the comment density, naming, and idioms of the code you're changing.
- **Confirm before outward actions**: For actions that are hard to reverse or outward-facing (deletes, overwrites, pushes, publishing), confirm with the user first unless durably authorized.

## Commands

### Build / Install

```bash
pip install -e .
```

This installs the `exam-proctoring` package in editable mode from the `src/` directory.

### Lint

```bash
# Check with ruff (if configured) or use general Python linting
ruff check src/
```

No dedicated lint config exists; run `ruff` or `py_qa` if available.

### Test

```bash
# Run the full consolidated test suite
python3 -m pytest tests/ -v
```

Run a single test file:

```bash
python3 -m pytest tests/unit/test_gaze_estimator.py -v
python3 -m pytest tests/integration/test_attention_fusion.py -v
```

## High-Level Architecture

This repository implements a **three-stream fusion framework** for malpractice detection during online examinations. The system fuses evidence from three synchronized sensing streams:

### Input Streams (19 total features, 10-second windows)

| Stream | Features | Dimension |
|---|---|---|
| **Gaze** | fixation_duration_mean, fixation_count, saccade_velocity_mean, gaze_deviation, gaze_confidence, head_yaw, head_pitch | 7 |
| **Interaction** | cursor_velocity_mean, cursor_velocity_std, click_frequency, keystroke_frequency, idle_fraction, tab_switch_count, velocity_spike_ratio | 7 |
| **Environment** | phone_detected, phone_confidence, notes_detected, extra_person_count, suspicious_objects_count | 5 |

### Model Architecture

Three fusion strategies are available, with **Attention Fusion** as the proposed architecture:

1. **Early Fusion** (`EarlyFusionModel`): Concatenates all 19 raw features → 128 → 64 → 1 logit
2. **Late Fusion** (`LateFusionModel`): Independent per-stream heads with learnable softmax-normalized weights
3. **Attention Fusion** (`AttentionFusionModel`): The proposed model — each stream is encoded to 128 dim, then cross-modal self-attention is applied across the 3 modality tokens [Gaze, Interaction, Environment], followed by a 384→64→1 classification head with sigmoid output

All models go through `ModelAdapter` (`src/exam_proctoring/models/adapter.py`) to ensure consistent `(gaze, interaction, environment)` forward signatures and logit-only output.

### Key Files

| Path | Purpose |
|---|---|
| `src/exam_proctoring/models/registry.py` | Model registry with 9 model keys; `build_model()` constructor |
| `src/exam_proctoring/models/attention_fusion.py` | The proposed attention fusion architecture |
| `src/exam_proctoring/models/common.py` (project root) | Feature constant definitions: `GAZE_FEATURES`, `INTERACTION_FEATURES`, `ENVIRONMENT_FEATURES`, `ALL_FEATURES`, dims, `MLPBlock` |
| `src/exam_proctoring/data/dataset.py` | `ExamDataset` returning `(gaze, interaction, environment, label)` tuples; `create_dataloaders()` |
| `src/exam_proctoring/training/trainer.py` | `Trainer` class with PGD-trained `Trainer.fit()` loop, checkpointing, early stopping |
| `src/exam_proctoring/evaluation/metrics.py` | `calculate_metrics()` and `find_optimal_threshold()` returning ROC-AUC, PR-AUC, F1, precision, recall, ECE, Brier |
| `scripts/run_experiments.py` | Master orchestrator: loads config, trains/all models, evaluates on test/stress sets, generates figures, hypothesis tests |
| `configs/experiment_config.yaml` | Training config: seed=42, lr=0.001, batch_size=32, epochs=50, patience=8, dropout=0.2 |
| `docs/fusion_architecture.md` | Full architecture documentation with diagrams and ablation requirements |
| `docs/data_schema.md` | Data schema contract (the "four people" agreement) |

### Data Splitting Rule (Critical)

**Sessions are the unit of split — never individual windows.** Windows from the same session must never appear in both training and testing splits. This prevents temporal/session leakage (`src/exam_proctoring/data/dataset.py:verify_dataset_integrity()`).

### Models Registry

The 9 model keys in `MODEL_REGISTRY`:

| Key | Streams | Name |
|---|---|---|
| `gaze_only` | gaze | Gaze-only |
| `interaction_only` | interaction | Interaction-only |
| `environment_only` | environment | Environment-only |
| `gaze_interaction` | gaze + interaction | Gaze + Interaction |
| `gaze_environment` | gaze + environment | Gaze + Environment |
| `interaction_environment` | interaction + environment | Interaction + Environment |
| `early_fusion` | all 3 | Early Fusion |
| `late_fusion` | all 3 | Late Fusion |
| `attention_fusion` | all 3 | Three-stream Attention Fusion |

Build a model: `build_model("attention_fusion", dropout=0.2)`.

### Evaluation Pipeline (run_experiments.py)

The 6-stage pipeline:

1. **Dataset & DataLoaders** — loads synthetic dataset, validates integrity, fits scaler
2. **Build & Train Models** — trains/all models (or loads checkpoints), tunes thresholds on val set via F1
3. **Test Evaluation** — runs test-set inference, calculates metrics, saves predictions, records attention diagnostics
4. **Hypothesis Evaluation** — evaluates H1 (attention vs single-stream), H2 (attention vs early/late), H3 (attention vs pairwise)
5. **Stress Testing** — evaluates AUC degradation on 5 scenarios (noisy gaze, mouse noise, environment failure, single-modality, silent cheating)
6. **Figures & Visualization** — generates publication-quality figures

### Development Tips

- Feature constants are in `models/common.py`; all model files in `src/exam_proctoring/models/` import from it
- Data features are: gaze (7) + interaction (7) + environment (5) = 19 total
- Session-level splitting is enforced — check `verify_dataset_integrity()` for violations
- The `ModelAdapter` wraps all models for consistent API; its `forward(gaze, interaction, environment)` is what `Trainer.train_epoch()` calls
- To add a new model: add its config to `MODEL_REGISTRY` in `src/exam_proctoring/models/registry.py`, create the model class, and ensure it accepts `(gaze, interaction, environment)` via `ModelAdapter`

### Important External References

- `README.md` — high-level overview, quick start, installation, demo scripts
- `docs/fusion_architecture.md` — detailed architecture, fusion strategy comparison, ablation requirements
- `docs/data_schema.md` — the data schema contract between data generators and model evaluators
- `configs/experiment_config.yaml` — experiment hyperparameters
- `.claude/settings.local.json` — environment variables and permission settings for this session
