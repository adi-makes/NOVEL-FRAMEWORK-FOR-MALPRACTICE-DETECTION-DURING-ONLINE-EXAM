# Day 4 Robustness Stress-Test Suites

This directory contains evaluation sets designed for model robustness and perturbation testing.

## Test Set Descriptions

| Dataset File | Target Failure Mode | Description |
|---|---|---|
| `test_a_noisy_gaze.csv` | Sensor Degradation | Degraded gaze confidence (30–60% reduction), increased gaze deviation noise, and head yaw perturbations. Simulates poor webcams and lighting. |
| `test_b_mouse_noise.csv` | Input Jitter | Honest candidate erratic movements, elevated velocity standard deviations, and artificial velocity spikes. |
| `test_c_environment_failure.csv` | Occlusion / Camera Drop | Total environment camera detection failure (0% recall across phones, notes, and extra persons). |
| `test_d_single_modality.csv` | Isolated Cheating | Cheating manifests exclusively in the environment camera (phone detection) while gaze and mouse remain strictly normal. |
| `test_e_silent_cheating.csv` | Covert Collusion | Minimal subtle gaze deviation with zero tab switches and completely normal mouse kinematics. |

## Schema & Format
* **Format:** CSV with identical columns as `data/synthetic/dataset.csv`.
* **Cohort Size:** 40 sessions (720 10-second windows) per test suite.
* **Split Flag:** Set to `test` for automated ingestion into Person 4 evaluation scripts.