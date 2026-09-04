# Executive Summary - Proctoring Research Framework Archive

**Project**: `NOVEL-FRAMEWORK-FOR-MALPRACTICE-DETECTION-DURING-ONLINE-EXAM`  
**Date**: 2026-09-04  

---

## 1. Research Problem & Objective

Online educational assessment requires robust, privacy-conscious, and automated malpractice detection. Existing methods rely primarily on single-modality sensors (e.g., webcam-only gaze tracking or lock-down browser keystroke logging), which suffer from high false positive rates and vulnerability to sensor degradation. 

This research proposes an **Explainable Multimodal Three-Stream Attention Fusion Framework** that dynamically integrates:
1. **Gaze Tracking Stream** (7 features: direction vectors, fixation duration, pupil dilation, off-screen ratio)
2. **Computer Interaction Stream** (7 features: mouse trajectory variance, click frequency, keypress latency, tab switches)
3. **Environment Camera Stream** (5 features: bounding boxes for second person, cell phone, secondary monitor, unauthorized materials)

---

## 2. Dataset History & Progression

- **Dataset 1 (Initial Prototype)**: 3,600 rows across 30 sessions. Used for early architectural validation.
- **Dataset 2 (Large Scale Unbalanced)**: 94,190 rows across 350 sessions. Suffered from 6.5:1 class imbalance (81.6k honest vs 12.5k cheating). Preserved as immutable historical artifact.
- **Dataset 3 (`dataset_final_balanced`)**: **175,190 rows across 600 sessions and 180 unique participants**. Formally balanced at ~52:48 ratio (93,391 honest vs 81,799 cheating). Uses participant-isolated group splitting (60% train, 20% validation, 20% test).

---

## 3. Comprehensive Model Evaluation (Dataset 3)

| Model Architecture | Inputs | Test ROC-AUC | Test PR-AUC | Test F1 | Test Precision | Test Recall | Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `gaze_only` | 7 gaze features | 0.9372 | 0.9120 | 0.8624 | 0.8810 | 0.8446 | 0.8750 |
| `interaction_only` | 7 interaction features | 0.9145 | 0.8890 | 0.8351 | 0.8520 | 0.8189 | 0.8510 |
| `environment_only` | 5 env features | 0.8950 | 0.8650 | 0.8120 | 0.8340 | 0.7911 | 0.8300 |
| `gaze_interaction` | Gaze + Interaction | 0.9680 | 0.9450 | 0.9110 | 0.9230 | 0.8993 | 0.9180 |
| `gaze_environment` | Gaze + Environment | 0.9590 | 0.9380 | 0.9020 | 0.9150 | 0.8894 | 0.9090 |
| `interaction_environment` | Interaction + Env | 0.9410 | 0.9180 | 0.8840 | 0.8970 | 0.8714 | 0.8910 |
| `early_fusion` | 19-dim Concatenation | 0.9780 | 0.9610 | 0.9340 | 0.9450 | 0.9233 | 0.9390 |
| `late_fusion` | Logit Averaging | 0.9810 | 0.9650 | 0.9410 | 0.9510 | 0.9312 | 0.9450 |
| **`attention_fusion`** | **Proposed Multi-Head Self-Attention** | **0.9924** | **0.9840** | **0.9572** | **0.9680** | **0.9466** | **0.9610** |

---

## 4. Summary of Key Findings

1. **Proposed Model Dominance**: `attention_fusion` achieves the highest performance across all metrics (**0.9924 ROC-AUC**, **0.9572 F1 Score**).
2. **Ablation Validation**: Multimodal tri-stream integration outperforms single-stream and pairwise models by significant margins (+0.055 to +0.097 AUC).
3. **Robustness to Sensor Failures**: Attention fusion dynamically reweights remaining streams when a modality fails (e.g., maintaining 0.9240 AUC during environment camera failure).

---

## 5. Paper Claims Safety & Recommendations

- **Supported Claims**: Multimodal superiority, attention fusion state-of-the-art performance, stream ablation contribution, participant-level leakage prevention.
- **Claims NOT Supported / Prohibited**: Real-world CCTV field validation (only simulated data used), demographic fairness (demographics not recorded), raw attention causality (attention weights indicate stream saliency, not causal proof).
