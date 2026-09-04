# Comprehensive Forensic Research Archive Audit Report

**Project**: `NOVEL-FRAMEWORK-FOR-MALPRACTICE-DETECTION-DURING-ONLINE-EXAM`  
**Date**: 2026-09-04 16:29:23  

---

## 1. Executive Forensic Summary

A complete, non-destructive forensic extraction, classification, and organization of every research artifact was performed across the repository.

- **Total Files Discovered & Processed**: 210
- **Tier A (Critical Paper Evidence)**: 101
- **Tier B (Supporting Evidence)**: 53
- **Tier C (Historical / Contextual)**: 56
- **Tier D (Minor Relevance)**: 0
- **Excluded System / Build Artifacts**: Virtualenvs (`venv_linux`), `.git` internals, `__pycache__`

---

## 2. Identified Dataset Versions

1. **Dataset 1 (Initial Synthetic Dataset)**:
   - Path: `data/dataset_1_initial/synthetic/dataset.csv`
   - Rows: 3,600 rows (2,700 honest : 900 cheating, 3:1 ratio)
   - Status: Initial prototype baseline.

2. **Dataset 2 (Large-Scale Unbalanced Dataset)**:
   - Path: `data/dataset_2_new/large_scale/large_scale_dataset.csv`
   - Rows: 94,190 rows (81,661 honest : 12,529 cheating, 6.5:1 ratio)
   - Status: Preserved immutable historical dataset.

3. **Dataset 3 (`dataset_final_balanced`)**:
   - Path: `data/dataset_final_balanced/dataset.csv`
   - Rows: 175,190 rows (93,391 honest : 81,799 cheating, ~52:48 ratio)
   - Status: **Authoritative Final Dataset for Research Paper**.

---

## 3. Results Traceability & Conflict Resolution

All metric values in `Research Paper/04_RESULTS/MASTER_RESULTS_TABLE.csv` are traced directly to exact source JSON metric files (`results/final/results.json` and `results/model_1_initial_dataset/run1/metrics/results.json`).

Conflicting AUC numbers between Exp 1 (0.8654) and Exp 2 (0.9924) have been reconciled and documented in `Research Paper/04_RESULTS/RESULT_CONFLICTS.md`.

---

## 4. Verification of Operational Rules

- **Rule 1 (Non-Destructive)**: 0 original files modified or deleted (`git status` clean).
- **Rule 2 (No Fabrication)**: All metrics imported from raw JSON output.
- **Rule 3 (Provenance)**: Every artifact registered with SHA256 in `SOURCE_INDEX.csv`.
- **Rule 4 (No Mixing)**: Exp 1 and Exp 2 separated into distinct directory hierarchies.
