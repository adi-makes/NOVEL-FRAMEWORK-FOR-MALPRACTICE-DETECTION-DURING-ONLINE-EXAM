# Reproduction Guide for Multimodal Proctoring Research Prototype

This document details the exact steps to reproduce all dataset generation, model training, evaluation, stress testing, and figure generation for the research paper:

> **"Explainable Three-Stream Fusion for Exam Malpractice Detection"**

---

## 1. Environment Setup

### Prerequisites
- Python 3.10+
- PyTorch 2.1+ (CUDA optional but recommended)
- `pip` or virtualenv

```bash
# Activate virtual environment
source venv_linux/bin/activate

# Install required packages
pip install -r requirements.txt
```

---

## 2. End-to-End Execution (Single Command)

To reproduce the entire pipeline (dataset verification, model training across all 9 configurations, test set evaluation, stress testing, hypothesis evaluation, and publication figure generation):

```bash
PYTHONPATH=. python run_experiments.py --force-retrain
```

---

## 3. Step-by-Step Execution

### Step 3.1: Generate Dataset 3 (FINAL BALANCED)
```bash
PYTHONPATH=. python -m simulators.generate_dataset_3
```
*Artifacts created*:
- `data/dataset_final_balanced/dataset.csv`
- `data/dataset_final_balanced/metadata.json`
- `data/dataset_final_balanced/data_quality_report.md`

### Step 3.2: Run Master Experiment Pipeline
```bash
PYTHONPATH=. python run_experiments.py --config configs/experiment_config.yaml --seed 42
```
*Artifacts created*:
- Checkpoints: `models/final/checkpoints/*.pt`, `models/final/checkpoints/scaler.joblib`
- Metrics & Tables: `results/final/main_results.csv`, `results/final/ablation_results.csv`, `results/final/results.json`
- Predictions: `results/final/predictions/*.csv`
- Figures: `figures/final/*.png`

---

## 4. Hardware & Seed Information
- **Random Seed**: `42`
- **Device**: PyTorch CUDA / CPU
- **Hardware**: NVIDIA GeForce RTX 4060 Laptop GPU (Tested)
- **Execution Time**: ~4 minutes for full 9-model GPU training on 175,190 rows
