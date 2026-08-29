# P2: Data Simulation & Synthetic Dataset Generation

## Overview
This directory contains the synthetic behavioral streams and exam generator for the multimodal exam proctoring framework.

## Structure
* `simulators/`: Behavioral stream generators (gaze, mouse/interaction, environment, and exam orchestrator).
* `data/synthetic/`: Output directory containing `dataset.csv`, `metadata.json`, and `data_quality_report.md`.
* `docs/`: Dataset schema specifications (`data_schema.md`).
* `tests/`: Integrity test suite for zero-leakage splits and stream validity.

## Quick Reproduction
```powershell
python -m simulators.generate_dataset
python -m tests.test_simulators