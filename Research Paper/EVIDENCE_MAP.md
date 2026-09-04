# Research Paper Evidence Map

**Project**: `NOVEL-FRAMEWORK-FOR-MALPRACTICE-DETECTION-DURING-ONLINE-EXAM`  
**Date**: 2026-09-04  

---

This document maps every standard section of the research manuscript directly to its authoritative empirical source files in the `Research Paper/` archive.

```
INTRODUCTION
    ├── Literature Review & Novelty: Research Paper/02_RESEARCH_DESIGN/03_LITERATURE_REVIEW/Literature_Review_and_Novelty_Assessment.pdf
    └── Executive Summary: Research Paper/00_EXECUTIVE_SUMMARY.md

RELATED WORK
    └── Literature Assessment: Research Paper/02_RESEARCH_DESIGN/03_LITERATURE_REVIEW/

METHODOLOGY
    ├── Architecture Definitions: Research Paper/03_SYSTEM_AND_METHOD/01_ARCHITECTURE/
    ├── Feature Extractor Schemas: Research Paper/12_DATASETS/06_SCHEMAS/
    ├── Fusion Implementations: Research Paper/03_SYSTEM_AND_METHOD/05_FUSION/
    └── Explainer Pipeline: Research Paper/03_SYSTEM_AND_METHOD/09_EXPLAINABILITY/

DATASET
    ├── Dataset Audit & Statistics: Research Paper/12_DATASETS/05_DATASET_AUDITS/
    ├── Dataset Metadata: Research Paper/12_DATASETS/03_FINAL_BALANCED/metadata.json
    └── Generation Simulator: Research Paper/12_DATASETS/04_DATASET_GENERATION/generate_dataset_3.py

EXPERIMENTAL SETUP
    ├── Hyperparameters & Configs: Research Paper/09_TRAINING_AND_REPRODUCIBILITY/01_TRAINING_CONFIGS/experiment_config.yaml
    ├── Training Script: Research Paper/03_SYSTEM_AND_METHOD/08_INFERENCE/run_experiments.py
    └── Reproduction Guide: Research Paper/09_TRAINING_AND_REPRODUCIBILITY/06_REPRODUCTION/REPRODUCTION.md

RESULTS
    ├── Master Results Table: Research Paper/04_RESULTS/MASTER_RESULTS_TABLE.csv
    ├── Metrics JSON: Research Paper/04_RESULTS/02_CURRENT_FINAL_EXPERIMENT/01_RAW_RESULTS/current_final_results.json
    ├── Test Predictions: Research Paper/04_RESULTS/02_CURRENT_FINAL_EXPERIMENT/03_PREDICTIONS/
    ├── ROC & PR Curves: Research Paper/10_FIGURES/04_ROC/ & 05_PR/
    └── Traceability Matrix: Research Paper/TRACEABILITY_MATRIX.csv

ABLATION STUDY
    ├── Ablation Metrics CSV: Research Paper/05_ABLATION/01_RESULTS/current_final_ablation_results.csv
    └── Ablation Comparison Plot: Research Paper/05_ABLATION/02_FIGURES/current_final_ablation_comparison.png

ROBUSTNESS & STRESS TESTING
    ├── Stress Test Results CSV: Research Paper/06_ROBUSTNESS/01_RESULTS/current_final_stress_results.csv
    ├── Stress Degradation Plot: Research Paper/06_ROBUSTNESS/02_FIGURES/current_final_stress_test_degradation.png
    └── Stress Test Datasets: Research Paper/06_ROBUSTNESS/06_STRESS_DATA/

ERROR ANALYSIS
    ├── Error Analysis CSV: Research Paper/07_ERROR_ANALYSIS/06_ANALYSIS/current_final_error_analysis.csv
    └── Confusion Matrices: Research Paper/07_ERROR_ANALYSIS/03_CONFUSION_MATRICES/

EXPLAINABILITY
    ├── Attention Diagnostics CSV: Research Paper/08_EXPLAINABILITY/01_RESULTS/current_final_attention_diagnostics.csv
    └── Sample Explanation JSON: Research Paper/08_EXPLAINABILITY/04_EXAMPLES/

LIMITATIONS & FUTURE WORK
    ├── Documented Limitations: Research Paper/02_RESEARCH_DESIGN/08_ETHICS_LIMITATIONS/
    └── Final Audit Report: Research Paper/ARCHIVE_FINAL_REPORT.md
```
