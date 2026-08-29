# Comprehensive audit of repository status and plan of action

## 1. Executive Summary
This document provides a thorough audit of the NOVEL-FRAMEWORK-FOR-MALPRACTICE-DETECTION-DURING-ONLINE-EXAM repository prior to executing final fixes, model training, evaluation, stress testing, and master runner integration.

## 2. Codebase & Directory Structure Audit
- `data/synthetic/dataset.csv`: Verified canonical dataset location.
  - Rows: 3600 total (2160 train, 720 val, 720 test).
  - 17 model features: 7 Gaze, 7 Interaction, 5 Environment.
  - Metadata columns: `window_id`, `session_id`, `timestamp_start`, `timestamp_end`, `label`, `cheating_type`, `split`.
- `models/`: Contains model definitions (`single_stream.py`, `pairwise.py`, `early_fusion.py`, `late_fusion.py`, `attention_fusion.py`, `common.py`).
- `src/exam_proctoring/`: Module tree for exports.
- `tests/`: Contains test files; `tests/gpu_pytorch_test.py` has a bash string syntax error causing pytest collection failure.

## 3. Findings & Deficiencies
1. **Broken Pytest Collection**: `tests/gpu_pytorch_test.py` contains raw bash syntax `python -c "..."` instead of valid Python code.
2. **Path Handling & Working Directory Dependence**: Several imports and data loaders resolve paths relative to cwd or hardcoded strings. `pathlib` with `PROJECT_ROOT` auto-resolution must be strictly enforced.
3. **Model Interfaces & Adapter Consistency**: Single-stream, pairwise, early, late, and attention models take varying positional inputs. Standard adapter/forward wrapper is needed for clean multi-model evaluation.
4. **Scaler Leakage Prevention**: Must ensure `StandardScaler` is fitted *only* on `split == 'train'` and persisted as `checkpoints/scaler.joblib`.
5. **Loss & Class Imbalance Weighting**: `pos_weight` in `BCEWithLogitsLoss` must be dynamically computed from train set `(label == 0) / (label == 1)` rather than hardcoded to 3.0.
6. **Threshold Protocol**: Thresholds must be tuned on `val` set (e.g. max F1) and frozen for `test` and `stress_tests`.
7. **Attention Explainability Integrity**: Ensure non-causal language is used; attention self-attention matrix outputted for diagnostics without false causal claims.

## 4. Plan of Action
1. Fix `tests/gpu_pytorch_test.py` and write complete test suite covering all 9 models, dataset loader, scaler, training, evaluation, and stress-tests.
2. Update/create standard dataset loading module in `src/exam_proctoring/data/dataset.py` (and `data/dataset.py`).
3. Audit and standardise model constructors & forward methods across all 9 configurations (M1 - M9).
4. Implement standard evaluation and metric calculator (AUC, PR-AUC, F1, Precision, Recall, FPR, FNR, Brier, ECE, Confusion Matrix).
5. Implement hypothesis evaluation engine (H1, H2, H3) and stress testing suite.
6. Build `run_experiments.py` root CLI script orchestrating full pipeline reproducibly (Seed=42).
7. Validate end-to-end with smoke test, full model stack run, and pytest suite.
