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

### Claude Code Setup (Optional)

If you use Claude Code with this repo, copy the settings template and fill in any API keys you need:
```bash
cp .claude/settings.local.json.template .claude/settings.local.json
# then edit .claude/settings.local.json and paste your keys
```

`settings.local.json` is in `.gitignore` and will **never** be committed. Any field you leave as `""` is simply ignored, so you can ship the template untouched if you don't need a given integration.

#### Free-tier setup (no payment required)

You can run Claude Code with zero paid keys. The defaults work out of the box on a free Anthropic account:

1. **Skip the API keys entirely.** Open `.claude/settings.local.json` and leave every value in `"env"` as `""`. The template ships this way.
2. **Sign in for free.** When you first launch `claude` in the terminal, choose **"Sign in with Anthropic"** and create a free account. Free accounts include enough usage for individual development on this repo.
3. **Use the default model.** Don't set `ANTHROPIC_API_KEY` unless you want to bypass rate limits. Without an API key, Claude Code uses your login session and falls back to the free-tier model automatically.
4. **Disable optional integrations.** Leave `WANDB_API_KEY`, `HUGGINGFACE_API_KEY`, and `OPENAI_API_KEY` blank — none of them are required for training or evaluation in this repo.
5. **Optional — free W&B.** If you want experiment tracking, create a free W&B account at https://wandb.ai and paste the API key into `WANDB_API_KEY`. The free tier is sufficient for solo academic experiments.

The repo runs end-to-end on a free Anthropic login + a free Hugging Face account (for any pretrained weights). No paid API access is required.

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