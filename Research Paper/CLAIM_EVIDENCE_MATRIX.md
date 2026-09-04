# Claim vs Evidence Verification Matrix

**Project**: `NOVEL-FRAMEWORK-FOR-MALPRACTICE-DETECTION-DURING-ONLINE-EXAM`  
**Date**: 2026-09-04  

---

This matrix evaluates scientific claims made by the paper against actual empirical evidence present in the repository.

| Claim ID | Scientific Claim | Empirical Evidence Source File | Experiment | Status | Verification Summary |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **C001** | Multimodal fusion outperforms single-stream proctoring baselines. | `04_RESULTS/02_CURRENT_FINAL_EXPERIMENT/01_RAW_RESULTS/current_final_results.json` | Exp 2 | **SUPPORTED** | `attention_fusion` AUC (0.9924) > `gaze_only` (0.9372), `interaction_only` (0.9145), `environment_only` (0.8950). |
| **C002** | Proposed Attention Fusion outperforms Early and Late Fusion architectures. | `04_RESULTS/MASTER_RESULTS_TABLE.csv` | Exp 2 | **SUPPORTED** | `attention_fusion` (0.9924 AUC / 0.9572 F1) > `early_fusion` (0.9780 AUC / 0.9340 F1) > `late_fusion` (0.9810 AUC / 0.9410 F1). |
| **C003** | Each modality stream makes a distinct, positive contribution to detection performance. | `05_ABLATION/01_RESULTS/current_final_ablation_results.csv` | Exp 2 | **SUPPORTED** | Removing any single stream reduces AUC by 2.4% to 5.1%. |
| **C004** | The system is robust to sensor degradation and missing stream failures. | `06_ROBUSTNESS/01_RESULTS/current_final_stress_results.csv` | Exp 2 | **SUPPORTED** | Under environment failure, `attention_fusion` maintains 0.9240 AUC, whereas `early_fusion` degrades to 0.8120 AUC. |
| **C005** | Attention weights provide human-interpretable evidence for proctoring review. | `08_EXPLAINABILITY/01_RESULTS/current_final_attention_diagnostics.csv` | Exp 2 | **PARTIALLY SUPPORTED** | Stream-level attention weights reflect stream activation; however, attention weights alone do not constitute causal explanations. |
| **C006** | Real-world CCTV video validation demonstrates deployability. | N/A (No external CCTV video dataset) | Exp 2 | **NOT SUPPORTED / UNTESTED** | All experiments were conducted on simulated temporal datasets (`dataset_final_balanced`). No live CCTV field trials were performed. |
| **C007** | Demographically fair across diverse student populations. | N/A (No demographic metadata) | Exp 2 | **NOT TESTED** | Dataset metadata does not track demographic attributes (gender, ethnicity, age). Demographic fairness cannot be claimed. |
| **C008** | Zero participant data leakage between training and testing sets. | `09_TRAINING_AND_REPRODUCIBILITY/06_REPRODUCTION/REPRODUCTION.md` | Exp 2 | **SUPPORTED** | Data loader enforces strict participant-isolated group splitting (`GroupKFold` on participant ID). |
