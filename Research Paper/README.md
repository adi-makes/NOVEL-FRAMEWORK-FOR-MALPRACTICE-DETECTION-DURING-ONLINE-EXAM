# Research Paper Forensic Archive & Evidence Package

**Project**: `NOVEL-FRAMEWORK-FOR-MALPRACTICE-DETECTION-DURING-ONLINE-EXAM`  
**Framework**: Explainable Three-Stream Fusion for Exam Malpractice Detection  
**Archivist**: Research Archivist & Documentation Engineer  
**Archive Created**: 2026-09-04  

---

## 1. Executive Summary & Purpose

This archive contains the complete forensic collection of every file, dataset, experiment record, model architecture, result table, figure, log, literature assessment, and reproduction configuration for the research project.

It provides **100% complete traceability** from raw source code and data files to final paper metrics, allowing any researcher to audit, reproduce, or extend the work without prior familiarity with the repository.

---

## 2. Directory Structure Overview

```
Research Paper/
├── README.md                                # Master archive guide (THIS FILE)
├── 00_EXECUTIVE_SUMMARY.md                  # Comprehensive executive summary
├── EVIDENCE_MAP.md                          # Paper section-to-file evidence map
├── CLAIM_EVIDENCE_MATRIX.md                 # Scientific claims vs empirical support matrix
├── SOURCE_INDEX.csv                         # Full artifact registry with SHA-256 hashes
├── TRACEABILITY_MATRIX.csv                  # Traceability from paper metrics to raw source code/data
├── DUPLICATE_MAP.csv                        # Deduplication mapping record
├── ARCHIVE_MANIFEST.json                    # Machine-readable archive metadata
├── ARCHIVE_FINAL_REPORT.md                  # Complete audit report
│
├── 01_RESEARCH_HISTORY/                     # Historical evolution & experiment lineage
│   ├── experiment_lineage.md                # Flowchart & technical lineage breakdown
│   ├── first_experiment/                    # Experiment 1 initial prototype records
│   └── current_experiment/                  # Experiment 2 final research framework records
│
├── 02_RESEARCH_DESIGN/                     # Plans, literature, novelty, hypotheses
│   ├── 01_RESEARCH_PLAN/
│   ├── 02_SPRINT_PLAN/
│   ├── 03_LITERATURE_REVIEW/
│   ├── 04_NOVELTY/
│   ├── 05_RESEARCH_QUESTIONS/
│   ├── 06_HYPOTHESES/
│   ├── 07_EXPERIMENT_DESIGN/
│   └── 08_ETHICS_LIMITATIONS/
│
├── 03_SYSTEM_AND_METHOD/                   # Source code for network architectures & pipelines
│   ├── 01_ARCHITECTURE/
│   ├── 02_GAZE/
│   ├── 03_INTERACTION/
│   ├── 04_ENVIRONMENT/
│   ├── 05_FUSION/
│   ├── 06_ATTENTION/
│   ├── 07_PREPROCESSING/
│   ├── 08_INFERENCE/
│   ├── 09_EXPLAINABILITY/
│   └── 10_DIAGRAMS/
│
├── 04_RESULTS/                             # Results tables, raw metrics, conflict analysis
│   ├── MASTER_RESULTS_TABLE.csv             # Primary master metrics table
│   ├── MASTER_RESULTS_TABLE.md              # Markdown formatted master table
│   ├── RESULT_CONFLICTS.md                  # Detailed metric conflict resolution
│   ├── FIRST_VS_CURRENT_COMPARISON.md       # Empirical comparison between Exp 1 and Exp 2
│   ├── 01_FIRST_EXPERIMENT/                 # Exp 1 raw JSON, CSVs, predictions, plots
│   └── 02_CURRENT_FINAL_EXPERIMENT/         # Exp 2 raw JSON, CSVs, predictions, plots
│
├── 05_ABLATION/                             # 9-model ablation study metrics & figures
├── 06_ROBUSTNESS/                           # Stress-testing & degradation under noise
├── 07_ERROR_ANALYSIS/                       # False positive / false negative breakdown
├── 08_EXPLAINABILITY/                       # Stream attributions & attention diagnostics
├── 09_TRAINING_AND_REPRODUCIBILITY/         # Training scripts, configs, seeds, reproduction guide
├── 10_FIGURES/                              # High-resolution figures (ROC, PR, confusion matrices)
├── 11_TABLES/                               # Standardized CSV tables by category
├── 12_DATASETS/                             # Dataset metadata, audits, schemas, and generators
├── 13_HISTORY/                              # Development record & commit logs
└── 14_PAPER_READY/                          # Curated folder structure mapping directly to paper sections
```

---

## 3. Experimental Generations

This research contains two distinct experimental generations:

1. **First Experiment (Exp 1)**:
   - **Dataset**: `dataset_1_initial` (3,600 rows, 30 sessions, 10 participants, 3:1 imbalance).
   - **Main Model AUC**: `attention_fusion` = **0.8654**.
   - **Location**: `04_RESULTS/01_FIRST_EXPERIMENT/`

2. **Current / Final Experiment (Exp 2)**:
   - **Dataset**: `dataset_final_balanced` (175,190 rows, 600 sessions, 180 participants, ~52:48 balance).
   - **Main Model AUC**: `attention_fusion` = **0.9924** (F1 = **0.9572**).
   - **Location**: `04_RESULTS/02_CURRENT_FINAL_EXPERIMENT/`

---

## 4. Key Paper Findings

- **Proposed Attention Fusion**: Achieves state-of-the-art performance (**0.9924 ROC-AUC**, **0.9572 F1 Score**), outperforming single-stream (`gaze_only` 0.9372 AUC) and traditional fusion methods (`early_fusion` 0.9780 AUC, `late_fusion` 0.9810 AUC).
- **Ablation Study**: Removing any modality stream degrades AUC by 2.4% to 5.1%, confirming that gaze, computer interaction, and environment streams provide complementary information.
- **Robustness**: Attention fusion maintains 0.9240 AUC even under complete environment camera failure.

---

## 5. Provenance & Non-Destructive Guarantee

All files in `Research Paper/` are exact copies or derived analytical summaries of the primary workspace. **No original files were modified or deleted.** Refer to `SOURCE_INDEX.csv` for SHA-256 file hashes and original paths.
