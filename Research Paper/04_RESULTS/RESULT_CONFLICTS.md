# Result Conflicts & Variance Forensic Analysis

**Project**: `NOVEL-FRAMEWORK-FOR-MALPRACTICE-DETECTION-DURING-ONLINE-EXAM`  
**Date**: 2026-09-04  

---

## 1. Overview of Experimental Runs & Conflicts

During the forensic audit of the repository, multiple result files were identified across different experimental generations:
1. `results/model_1_initial_dataset/run1/metrics/results.json` (First Experiment Generation)
2. `experiments/results/results.json` (Intermediate / Prototype Sanity Run)
3. `results/final/results.json` (Current / Final Experiment Generation on Dataset 3)

---

## 2. Detailed Conflict Breakdown

### Conflict 1: `attention_fusion` Test ROC-AUC

- **Result Generation A (First Experiment)**:
  - **Source**: `results/model_1_initial_dataset/run1/metrics/results.json`
  - **Value**: ROC-AUC = **0.8654**, F1 = **0.6842**
  - **Dataset**: `dataset_1_initial` (3,600 rows, synthetic, 3:1 imbalance)
  - **Context**: Preliminary validation of early architecture design on small synthetic dataset.

- **Result Generation B (Current / Final Experiment)**:
  - **Source**: `results/final/results.json`
  - **Value**: ROC-AUC = **0.9924**, F1 = **0.9572**
  - **Dataset**: `dataset_final_balanced` (175,190 rows, 600 sessions, 180 participants, ~52:48 class ratio)
  - **Context**: **Authoritative Final Result** for the research paper. Trained on full multi-session balanced temporal dataset with participant-isolated train/val/test splits.

- **Reason for Difference**:
  - The dataset scale increased by >48x (from 3.6k rows to 175k rows).
  - Feature representation in Dataset 3 incorporates complete temporal sequence dynamics, realistic gaze calibration offsets, and multi-object environment bounding boxes.
  - Model capacity was fully leveraged under participant-level isolation.

---

### Conflict 2: Single-Stream `gaze_only` Performance

- **Result Generation A (First Experiment)**:
  - **Source**: `results/model_1_initial_dataset/run1/metrics/results.json`
  - **Value**: ROC-AUC = **0.7123**, F1 = **0.4851**

- **Result Generation B (Current / Final Experiment)**:
  - **Source**: `results/final/results.json`
  - **Value**: ROC-AUC = **0.9372**, F1 = **0.8624**

- **Reason for Difference**:
  - Dataset 1 contained uncalibrated raw gaze vectors with Gaussian noise.
  - Dataset 3 incorporates standardized gaze direction angles (`gaze_pitch`, `gaze_yaw`), fixations (`gaze_fixation_duration`), and pupil dilation features, allowing single-stream MLPs to perform significantly better.

---

## 3. Conflict Resolution & Paper Selection Rule

> [!IMPORTANT]
> **Authoritative Selection**: `results/final/results.json` (Dataset 3) is the primary authoritative source for all paper claims, tables, and figures in the final publication.
> 
> `results/model_1_initial_dataset/run1/metrics/results.json` must be cited exclusively in **Section 01_RESEARCH_HISTORY / FIRST_VS_CURRENT_COMPARISON** as empirical evidence of model evolution and dataset scaling impact.
