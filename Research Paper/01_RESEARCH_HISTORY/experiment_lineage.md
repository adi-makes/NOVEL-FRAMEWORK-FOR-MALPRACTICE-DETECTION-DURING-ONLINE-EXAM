# Research Experiment Lineage & Historical Evolution

**Project**: `NOVEL-FRAMEWORK-FOR-MALPRACTICE-DETECTION-DURING-ONLINE-EXAM`  
**Author**: Research Archivist & Documentation Engineer  
**Date**: 2026-09-04  

---

## Technical Lineage Flowchart

```
[EXPERIMENT 1: INITIAL PROTOTYPE]
   │  - Dataset: dataset_1_initial (3,600 rows, 30 sessions, 10 participants, 3:1 ratio)
   │  - Limitations: Small synthetic sample size, time-series frame leakage across row splits, uncalibrated gaze noise.
   │  - Result: attention_fusion AUC = 0.8654, F1 = 0.6842.
   ▼
[DATASET ITERATION 2: LARGE SCALE UNBALANCED]
   │  - Dataset: dataset_2_new (94,190 rows, 350 sessions, 125 participants, 6.5:1 ratio)
   │  - Problem: Severe class imbalance (81.6k honest vs 12.5k cheating) causing high false negative rate.
   ▼
[EXPERIMENT 2: CURRENT / FINAL RESEARCH FRAMEWORK]
      - Dataset: dataset_final_balanced (175,190 rows, 600 sessions, 180 participants, 1.14:1 ratio)
      - Improvements: Participant-isolated group split, strict zero-leakage StandardScaler pipeline, 19 canonical features.
      - Result: attention_fusion AUC = 0.9924, F1 = 0.9572.
```

---

## Detailed Narrative

### 1. First Experiment (Phase 1 Initial Prototype)
The initial experiment validated the technical feasibility of multimodal proctoring. Using a lightweight synthetic generator (`simulators/gaze_simulator.py`, `mouse_simulator.py`), 3,600 samples were generated across 30 simulated exam sessions. Models were evaluated using early prototype PyTorch scripts. Although `attention_fusion` outperformed single-stream models (0.8654 vs 0.7123 AUC), the dataset suffered from random split temporal leakage and uncalibrated gaze distributions.

### 2. Intermediate Dataset Development (Dataset 2)
To address sample size limitations, a large-scale dataset was synthesized with 94,190 rows across 350 sessions. Forensic audit revealed a 6.5:1 class imbalance ratio, which biased classifier decision thresholds toward predicting honest behavior, yielding unacceptably high False Negative Rates for subtle cheating modalities.

### 3. Current / Final Experiment (Dataset 3 Research Framework)
The current research framework resolved all prior limitations by creating `dataset_final_balanced` (Dataset 3). It incorporates 175,190 rows across 600 exam sessions and 180 unique participants with ~52:48 class balance. The training pipeline enforces strict participant-isolated group splits (60% train, 20% validation, 20% test) and zero-leakage normalization. All 9 model architectures were trained and evaluated, confirming `attention_fusion` as the state-of-the-art model with 0.9924 ROC-AUC and 0.9572 F1 score.
