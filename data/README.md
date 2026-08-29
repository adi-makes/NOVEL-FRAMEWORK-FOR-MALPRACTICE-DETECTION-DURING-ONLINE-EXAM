# Exam Proctoring Research Dataset

## Status: FROZEN

**DO NOT MODIFY OR REGENERATE THE CANONICAL DATASET.**

### Canonical Dataset
- **Location**: `data/synthetic/dataset.csv`
- **Rows**: 3,600 (synchronized 10-second windows)
- **Sessions**: 200 total sessions (100 honest, 100 cheating, 18 windows per session)
- **Features**: 19 domain feature fields + metadata & target labels (26 columns total)
- **Data Partition**: Session-level zero-leakage split (`train`, `val`, `test`)
- **Random Seed**: 42

### Stress Test Datasets
- **Location**: `data/stress_tests/`
- **Files**:
  - `test_a_noisy_gaze.csv`
  - `test_b_mouse_noise.csv`
  - `test_c_environment_failure.csv`
  - `test_d_single_modality.csv`
  - `test_e_silent_cheating.csv`
- **Rows per file**: 720 (40 sessions per stress test)

### Verification
Dataset integrity can be verified at any time using:
```bash
python3 -m pytest tests/data/test_simulators.py -v
```
