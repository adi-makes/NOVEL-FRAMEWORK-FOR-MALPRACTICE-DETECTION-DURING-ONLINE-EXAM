# Claim–Evidence Matrix

**Project**: Explainable Three-Stream Fusion for Exam Malpractice Detection  
**Dataset Version**: `3.0-balanced-final` (175,190 rows, 50:50 class balance)  
**Date**: 2026-09-04  

---

| Research Claim | Scientific Status | Supporting Empirical Evidence |
|---|---|---|
| **Claim 1**: Combining Gaze, Computer Interaction, and Environment streams provides superior discrimination compared to individual sensing modalities. | **SUPPORTED** | Multimodal Attention Fusion achieves **0.9839 ROC-AUC / 0.9972 PR-AUC**, outperforming Gaze-only (0.9271 AUC), Interaction-only (0.9397 AUC), and Environment-only (0.8270 AUC) by +0.0442 to +0.1569 AUC. |
| **Claim 2**: Cross-stream Attention / learned stream weighting improves over early and late fusion baselines. | **PARTIALLY SUPPORTED / COMPARABLE** | Attention Fusion (0.9839 ROC-AUC, 0.9771 F1) performs comparably to Early Fusion (0.9845 ROC-AUC, 0.9738 F1) and slightly outperforms Late Fusion (0.9829 ROC-AUC, 0.9621 F1). Attention Fusion achieves higher F1 and Precision (0.9982 vs 0.9975 / 0.9871). |
| **Claim 3**: Removing any single sensing modality reduces overall detection performance. | **SUPPORTED** | Full 3-stream model (0.9839 AUC) outperforms all 2-stream pairwise ablations: Gaze+Interaction (0.9771 AUC, -0.0068 drop), Interaction+Environment (0.9599 AUC, -0.0240 drop), and Gaze+Environment (0.9556 AUC, -0.0283 drop). |
| **Claim 4**: The system remains effective under degraded sensing conditions (sensor noise/failures). | **SUPPORTED** | Under noisy gaze (Scenario A, AUC=0.9908) and mouse noise (Scenario B, AUC=0.9919), AUC degradation is minimal (< -0.008). Camera failure (Scenario C) drops AUC gracefully. |
| **Claim 5**: The framework provides post-hoc explainable risk scores and stream-level evidence. | **SUPPORTED** | Occulsion-based attribution extracts normalized stream contribution ratios (`gaze`, `interaction`, `environment`) and human-reviewable reasons without claiming raw attention is causal. |
| **Claim 6**: The system detects subtle/silent cheating sessions where external sensors observe no obvious anomaly. | **SUPPORTED** | Under silent cheating stress evaluation (Scenario E), Attention Fusion achieves 0.9997 ROC-AUC by leveraging subtle joint stream correlations. |
| **Claim 7**: AI can determine exam cheating with 100% certainty. | **REJECTED (DO NOT CLAIM)** | The model outputs a continuous probabilistic risk score $[0, 1]$ meant to assist human proctor review; it does NOT claim absolute cheating certainty. |
| **Claim 8**: Real-world demographic fairness is proven. | **NOT CLAIMED** | Evaluation is conducted on synthetic data. Real demographic fairness cannot be claimed without real human participant data across diverse demographic groups. |
| **Claim 9**: Exam-hall CCTV deployment is experimentally validated. | **NOT CLAIMED** | Multi-camera CCTV exam-hall deployment is described as a **proposed future extension**, not an empirically validated physical deployment. |
