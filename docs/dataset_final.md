# Dataset 3 Documentation (`dataset_final_balanced`)

**Dataset Version**: `3.0-balanced-final`  
**Location**: `data/dataset_final_balanced/dataset.csv`  
**Date**: 2026-09-04  

---

## 1. Motivation & Overview

Dataset 3 was created specifically for the final research experiments of the *Explainable Three-Stream Fusion for Exam Malpractice Detection* framework. 

Historical Dataset 2 (`data/dataset_2_new/large_scale/large_scale_dataset.csv`, 94,190 rows) contained a severe class imbalance (~6.5:1 honest:cheating ratio, with 81,661 honest windows vs 12,529 cheating windows). While preserved as an immutable historical artifact (Rule 2), running final evaluation solely on Dataset 2 would obscure positive-class precision/recall trade-offs.

Dataset 3 combines **all 94,190 historical rows from Dataset 2** with **new realistic, multi-scenario synthetic cheating sessions** to achieve an approximate **50:50 window-level class balance** (93,391 honest vs 81,799 cheating windows).

---

## 2. Dataset Statistics

| Metric | Value |
|---|---|
| **Total Rows (10s Windows)** | **175,190** |
| **Total Sessions** | **650** |
| **Total Unique Participants** | **425** |
| **Honest Windows (label=0)** | 93,391 (53.3%) |
| **Cheating Windows (label=1)** | 81,799 (46.7%) |
| **Class Ratio** | ~1.14 : 1 (Approximately 50:50) |

---

## 3. Participant-Isolated Group Split (60 / 20 / 20)

Splitting was performed strictly at the **Participant level** (Rule 5). No participant or session appears in more than one split (`TRAIN ∩ VAL = ∅`, `TRAIN ∩ TEST = ∅`, `VAL ∩ TEST = ∅`).

| Split | Participants | Sessions | Total Rows | Honest Rows | Cheating Rows |
|---|---|---|---|---|---|
| **Train (60%)** | 255 | 390 | 106,494 | 56,712 | 49,782 |
| **Validation (20%)** | 85 | 130 | 35,024 | 18,740 | 16,284 |
| **Test (20%)** | 85 | 130 | 33,672 | 17,939 | 15,733 |

---

## 4. Scenario Library & Hard Negatives

The new cheating sessions incorporate a diverse library of temporal behavior:
1. **Phone Lookup**: Gaze deviation + head yaw + intermittent phone detection + mouse idle.
2. **Notes / Cheat Sheet**: Gaze deviation + head pitch + notes detection + low mouse velocity.
3. **Copy / Paste**: Rapid tab switching + high cursor velocity spikes + keystroke bursts + gaze centered.
4. **Web Search**: Tab switching + high cursor velocity + typing bursts + normal camera.
5. **External Assistance**: Extra person detection + gaze deviation + intermittent pauses.
6. **Textbook / Secondary Material**: Prolonged gaze off-screen + zero mouse activity + zero camera detection.
7. **Silent Cheating**: Subtle gaze shift + normal mouse movement + zero camera detection (hard positive).
8. **Mixed Cheating**: Blended multi-modality evidence.
9. **Intermittent Cheating**: Randomized active cheating windows interspersed within honest behavior.

### Benign Hard Negatives (Honest Class)
- System popups causing occasional benign tab switches (4%).
- Prolonged thinking/reading pauses (12% high idle fraction).
- Holding permitted objects (calculators, wallets) triggering camera false alerts (7%).
- Permitted scratch paper consultation (5%).
