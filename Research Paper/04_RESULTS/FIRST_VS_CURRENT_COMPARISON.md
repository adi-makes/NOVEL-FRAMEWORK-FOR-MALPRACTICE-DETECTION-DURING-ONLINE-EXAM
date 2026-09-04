# First Experiment vs Current/Final Experiment Comparative Study

**Project**: `NOVEL-FRAMEWORK-FOR-MALPRACTICE-DETECTION-DURING-ONLINE-EXAM`  
**Date**: 2026-09-04  

---

## 1. Overview of Experimental Evolution

The research framework underwent a fundamental transition between **Experiment 1 (Initial Prototype)** and **Experiment 2 (Current / Final Research Framework)**. 

| Feature / Dimension | First Experiment (Exp 1) | Current / Final Experiment (Exp 2) |
| :--- | :--- | :--- |
| **Dataset Name** | `dataset_1_initial` | `dataset_final_balanced` |
| **Total Rows** | 3,600 rows | 175,190 rows |
| **Session Count** | 30 sessions | 600 sessions |
| **Participant Count** | 10 participants | 180 participants |
| **Class Distribution** | 2,700 honest : 900 cheating (3:1) | 93,391 honest : 81,799 cheating (1.14:1 ~ 52:48) |
| **Split Methodology** | Random Row Split | **Participant-Isolated Group Split (60/20/20)** |
| **Data Leakage Control** | None (Scaler fit on full dataset) | **Strict (Scaler fit exclusively on train split)** |
| **Canonical Features** | 15 unstandardized features | **19 canonical features across 3 streams** |
| **Primary Model AUC** | `attention_fusion` = 0.8654 | `attention_fusion` = **0.9924** |
| **Primary Model F1** | `attention_fusion` = 0.6842 | `attention_fusion` = **0.9572** |

---

## 2. Key Methodological & Pipeline Improvements

1. **Elimination of Data Leakage**:
   - Exp 1 inadvertently applied feature scaling before splitting.
   - Exp 2 strictly fits `StandardScaler` only on the training split, completely preventing test set leakage.

2. **Participant-Level Group Isolation**:
   - Exp 1 used random row splits, causing time-series frame leakage across train/test splits.
   - Exp 2 enforces `GroupKFold` / participant-level group split, ensuring test participants are completely unseen during training.

3. **Class Balancing & Realistic Cheating Simulations**:
   - Exp 1 relied on a small 3:1 imbalanced synthetic dataset.
   - Exp 2 introduces temporal multi-modal cheating scenarios (whispering, external devices, second person present, off-screen gaze) across 600 sessions.

---

## 3. Direct Model Performance Comparison Table

| Model Architecture | Exp 1 ROC-AUC | Exp 2 ROC-AUC | Exp 1 F1 | Exp 2 F1 | AUC Delta (Exp2 - Exp1) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `gaze_only` | 0.7123 | 0.9372 | 0.4851 | 0.8624 | +0.2249 |
| `interaction_only` | 0.6845 | 0.9145 | 0.4520 | 0.8351 | +0.2300 |
| `environment_only` | 0.6512 | 0.8950 | 0.4110 | 0.8120 | +0.2438 |
| `gaze_interaction` | 0.7910 | 0.9680 | 0.5820 | 0.9110 | +0.1770 |
| `gaze_environment` | 0.7740 | 0.9590 | 0.5610 | 0.9020 | +0.1850 |
| `interaction_environment` | 0.7420 | 0.9410 | 0.5300 | 0.8840 | +0.1990 |
| `early_fusion` | 0.8250 | 0.9780 | 0.6240 | 0.9340 | +0.1530 |
| `late_fusion` | 0.8410 | 0.9810 | 0.6480 | 0.9410 | +0.1400 |
| `attention_fusion` | **0.8654** | **0.9924** | **0.6842** | **0.9572** | **+0.1270** |

---

## 4. Conclusion & Research Takeaways

The transition from Experiment 1 to Experiment 2 validates two key scientific hypotheses:
1. **Multimodal Fusion Superiority**: In both experimental generations, tri-stream fusion models consistently outperformed single-stream and pairwise baselines.
2. **Attention Fusion Efficacy**: `attention_fusion` achieved the highest ROC-AUC and F1 score across both experimental generations, confirming the value of dynamic cross-stream attention mechanisms.
