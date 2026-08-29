# Repository Cleanup and Restructuring Report

## 1. Overview
This report documents the reorganization, cleaning, and standardization of the research repository:
`NOVEL-FRAMEWORK-FOR-MALPRACTICE-DETECTION-DURING-ONLINE-EXAM`.

The multi-person development structure (`P2/`, `p3/`, `exam-proctoring/`, `code/gaze/`) has been consolidated into a single research-grade structure based on **function** rather than **person**.

## 2. Directory Mappings & Action Log

| Old Path | New Path | Action | Rationale |
|----------|----------|--------|-----------|
| `code/gaze/gaze_estimator.py` | `src/exam_proctoring/gaze/gaze_estimator.py` | MOVE | Core gaze estimation source module |
| `code/gaze/calibration.py` | `src/exam_proctoring/gaze/calibration.py` | MOVE | Core gaze calibration module |
| `code/gaze/feature_extractor.py` | `src/exam_proctoring/gaze/feature_extractor.py` | MOVE | Core 7D gaze temporal feature extractor |
| `code/gaze/demo.py` | `scripts/run_gaze_demo.py` | MOVE | Interactive demo script |
| `code/gaze/benchmark.py` | `scripts/benchmark_gaze.py` | MOVE | Performance benchmarking script |
| `p3/src/camera.py` | `src/exam_proctoring/environment/camera.py` | MOVE | OpenCV camera interface |
| `p3/src/face_landmarks.py` | `src/exam_proctoring/environment/face_landmarks.py` | MOVE | Environment face landmark module |
| `p3/src/object_detector.py` | `src/exam_proctoring/environment/object_detector.py` | MOVE | YOLO object detector module |
| `p3/src/environment_features.py` | `src/exam_proctoring/environment/environment_features.py` | MOVE | Environment feature extractor with temporal window |
| `P2/simulators/*.py` | `simulators/*.py` | MOVE | Data simulator tools for reproducibility |
| `P2/data/synthetic/*` | `data/synthetic/*` | MOVE | Canonical frozen synthetic dataset and quality reports |
| `P2/data/stress_tests/*` | `data/stress_tests/*` | MOVE | Canonical stress test CSV suites |
| `P2/data/synthetic/dataset_day1_sample.csv` | `archive/data/dataset_day1_sample.csv` | ARCHIVE | Obsolete 90-row initial dataset sample |
| `exam-proctoring/docs/data_schema.md` | `docs/data_schema.md` | MOVE | Primary data contract schema document |
| `P2/docs/data_schema.md` | N/A | DELETED | Identified duplicate of `data_schema.md` |
| `exam-proctoring/docs/fusion_architecture.md` | `docs/fusion_architecture.md` | MOVE | Fusion architecture documentation |
| `exam-proctoring/fusion/fusion_files.png` | `docs/assets/fusion_files.png` | MOVE | Architecture visual diagram asset |
| `exam-proctoring/fusion/attention_fusion.py` | `src/exam_proctoring/models/attention_fusion.py` | MOVE | Proposed Attention Fusion PyTorch model |
| `models/gaze/face_landmarker.task` | `models/pretrained/gaze/face_landmarker.task` | MOVE | Pretrained MediaPipe task asset |
| `tests/*` | `tests/unit/`, `tests/integration/` | MOVE | Reorganized into functional test structure |
| `P2/tests/test_simulators.py` | `tests/data/test_simulators.py` | MOVE | Data and simulator integrity test suite |
| `exam-proctoring/tests/test_attention_fusion.py` | `tests/integration/test_attention_fusion.py` | MOVE | Fusion model integration test |
| `p3/tests/*` | `scripts/run_environment_demo.py`, `tests/unit/test_environment.py` | RESTRUCTURE | Separated live hardware test scripts from unit tests |
| `Complete_Plan.md`, `ONE_WEEK_SPRINT_PLAN.md` | `docs/project/` | MOVE | Project execution planning documents |
| `Literature_Review_and_Novelty_Assessment.pdf`, etc. | `docs/research/` | MOVE | Literature review and research PDFs |
| `P2/README.md`, `p3/README.md` | `archive/README_P2.md`, `archive/README_p3.md` | ARCHIVE | Legacy team-folder README files |

## 3. Dataset Schema Contract Discrepancy Note

- **Discrepancy**: Some planning text mentioned 17 features, whereas counting feature lists yielded 19 features (7 Gaze + 7 Interaction + 5 Environment).
- **Authoritative Data Audit**: Inspection of `data/synthetic/dataset.csv` confirmed:
  - 3,600 rows, 26 total columns.
  - Exactly 19 domain feature columns:
    - Gaze (7): `fixation_duration_mean`, `fixation_count`, `saccade_velocity_mean`, `gaze_deviation`, `gaze_confidence`, `head_yaw`, `head_pitch`
    - Interaction (7): `cursor_velocity_mean`, `cursor_velocity_std`, `click_frequency`, `keystroke_frequency`, `idle_fraction`, `tab_switch_count`, `velocity_spike_ratio`
    - Environment (5): `phone_detected`, `phone_confidence`, `notes_detected`, `extra_person_count`, `suspicious_objects_count`
  - Metadata (4): `window_id`, `session_id`, `timestamp_start`, `timestamp_end`
  - Targets/Splits (3): `label`, `cheating_type`, `split`
- **SHA256 Checksum**: `d0367c9a199c5b93163b18c57e5ed357423398e4166ba61cfbccd9bbf1bbb2c8` (Unchanged before and after restructuring).

## 4. Test Suite Execution Summary

Execution Command:
```bash
python3 -m pytest tests/ -v
```

Output:
```
============================= test session starts ==============================
platform linux -- Python 3.13.9, pytest-8.2.2, pluggy-1.5.0
rootdir: /media/adi/New Volume1/CUSAT Academics/Research Paper/NOVEL-FRAMEWORK-FOR-MALPRACTICE-DETECTION-DURING-ONLINE-EXAM
configfile: pyproject.toml
plugins: anyio-4.10.0
collected 21 items

tests/data/test_simulators.py PASSED                                     [ 28%]
tests/integration/test_attention_fusion.py PASSED                       [ 33%]
tests/integration/test_gaze_fusion_integration.py PASSED                [ 38%]
tests/unit/test_environment.py PASSED                                   [ 42%]
tests/unit/test_gaze_calibration.py PASSED                            [ 57%]
tests/unit/test_gaze_estimator.py PASSED                              [ 80%]
tests/unit/test_gaze_features.py PASSED                               [100%]

============================== 21 passed in 3.43s ==============================
```

## 5. Unresolved Issues & Manual Review Items

None. All imports and test contracts pass without regression or missing dependencies.
