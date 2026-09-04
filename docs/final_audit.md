# Phase 0: Complete Repository Forensics Audit Report

**Project**: Explainable Three-Stream Fusion for Exam Malpractice Detection  
**Repository**: `NOVEL-FRAMEWORK-FOR-MALPRACTICE-DETECTION-DURING-ONLINE-EXAM`  
**Date**: 2026-09-04  

---

## 1. Executive Summary & Repository State

The repository represents a multimodal ML research framework designed for online exam proctoring across three sensing streams:
1. **Gaze** (7 features)
2. **Computer Interaction** (7 features)
3. **Environment Camera / Object Detections** (5 features)

Total canonical feature dimension: **19 features**.

### Git & Branch State
- **Current Branch**: `main` (clean working tree, up to date with `origin/main`).
- **Head Commit**: `ea95f5a fix(tests): update dataset_1_initial path in dataset integrity unit tests`
- **Other Branches & Remotes**:
  - `origin/agni`
  - `origin/amrutha`
  - `origin/feature/data-simulators`
  - `worktree-claude-work`
  - `worktree-logical-wiggling-goblet`
  - `worktree-readme-update`

---

## 2. Dataset Inventory

1. **Dataset 1 (Initial Synthetic Dataset)**
   - Location: `data/dataset_1_initial/synthetic/dataset.csv`
   - Size: 1,077,067 bytes (~2,400 rows)
   - Status: Active historical baseline dataset for initial unit testing and sanity checks.

2. **Dataset 2 (Large-Scale Dataset)**
   - Location: `data/dataset_2_new/large_scale/large_scale_dataset.csv`
   - Rows: **94,190 rows**, 37 columns
   - Class Balance: 81,661 honest (0) vs 12,529 cheating (1) (~6.5:1 imbalance ratio)
   - Groups: 350 unique sessions, 125 unique participants
   - Metadata files: `metadata.json`, `distribution_audit.json`
   - Status: **Preserved as immutable historical dataset** per Rule 2.

3. **Stress Test Datasets**
   - Location: `data/stress_tests/`
   - Files: `test_a_noisy_gaze.csv`, `test_b_mouse_noise.csv`, `test_c_environment_failure.csv`, `test_d_single_modality.csv`, `test_e_silent_cheating.csv`
   - Status: Pre-existing synthetic stress test evaluation sets.

---

## 3. Model Inventory

The framework defines 9 distinct model architectures across single-stream, pairwise, early, late, and proposed attention fusion:

1. `gaze_only` (Single-Stream MLP)
2. `interaction_only` (Single-Stream MLP)
3. `environment_only` (Single-Stream MLP)
4. `gaze_interaction` (Pairwise Concatenation MLP)
5. `gaze_environment` (Pairwise Concatenation MLP)
6. `interaction_environment` (Pairwise Concatenation MLP)
7. `early_fusion` (19-dim Concatenation MLP)
8. `late_fusion` (Per-stream encoders + Logit averaging / weighting)
9. `attention_fusion` (Proposed 3-stream projection + Multi-Head Self-Attention + LayerNorm + MLP)

- Registry Location: `src/exam_proctoring/models/registry.py` & `src/exam_proctoring/models/adapter.py`
- Core Model Implementations: `models/` directory (`common.py`, `single_stream.py`, `pairwise.py`, `early_fusion.py`, `late_fusion.py`, `attention_fusion.py`).

---

## 4. Pipeline & Training Inventory

- **Orchestrator**: `run_experiments.py` (Main execution pipeline supporting end-to-end dataset loading, model training, optimal threshold tuning on validation set, test inference, stress test evaluation, hypothesis testing, and figure generation).
- **Trainer**: `src/exam_proctoring/training/trainer.py` (PyTorch trainer with Adam optimizer, BCEWithLogitsLoss, EarlyStopping, best weight saving).
- **DataLoader & Preprocessing**: `src/exam_proctoring/data/dataset.py` (Handles StandardScaler fitting **only on train set**, group leakage verification, PyTorch DataLoader construction).

---

## 5. Evaluation, Testing & Figure Inventory

- **Metrics Module**: `src/exam_proctoring/evaluation/metrics.py` (Calculates ROC-AUC, PR-AUC, F1, Precision, Recall, FPR, FNR, Accuracy).
- **Visualization Suite**: `src/exam_proctoring/evaluation/utils.py` (Generates ROC curves, PR curves, Confusion Matrices, Calibration curves, Ablation bar charts, Stress test degradation plots).
- **Test Suite**: `tests/` directory containing 32 passing unit/integration tests (`tests/data/test_simulators.py`, `tests/integration/test_attention_fusion.py`, `tests/integration/test_gaze_fusion_integration.py`, `tests/unit/test_models_stack.py`, `tests/unit/test_environment.py`, `tests/unit/test_gaze_features.py`, etc.).

---

## 6. Identified Gaps & Inconsistencies

1. **Dataset Class Imbalance & Lack of Final Research Dataset (Dataset 3)**:
   - Dataset 2 has a 6.5:1 imbalance (81,661 honest : 12,529 cheating).
   - Final research requirement: Need **Dataset 3 (`dataset_final_balanced`)** with ~50:50 class balance created by generating new realistic, temporal cheating sessions (not row duplication).
2. **Group-Level Isolation Enforcement**:
   - Data generation must ensure strict participant/session level isolation for train/val/test splits (60/20/20) with seed tracking.
3. **Explicit Stream Gating in Attention Fusion**:
   - Verify and enhance Attention Fusion to output explicit stream weights/gating for explainability and human evidence generation.
4. **Post-Hoc Attribution / Stream-Level Evidence**:
   - Current explainability output is preliminary; need explicit post-hoc feature/stream perturbation/attribution and human-readable evidence summaries without claiming raw self-attention is causal.
5. **Canonical Results & Artifact Reorganization**:
   - Establish dedicated `results/final/` and `figures/final/` directories for Dataset 3 experiments.

---

## 7. Proposed Final Target Structure

```
.
├── configs/
│   └── experiment_config.yaml
├── data/
│   ├── dataset_1_initial/
│   ├── dataset_2_existing/ (Archived immutable Dataset 2)
│   └── dataset_final_balanced/ (NEW Dataset 3: ~50:50 session-split)
├── docs/
│   ├── final_audit.md
│   ├── dataset_final.md
│   └── claim_evidence_matrix.md
├── models/
├── results/
│   └── final/
├── figures/
│   └── final/
├── simulators/
├── src/
│   └── exam_proctoring/
├── tests/
├── run_experiments.py
├── REPRODUCTION.md
└── FINAL_REPORT.md
```
