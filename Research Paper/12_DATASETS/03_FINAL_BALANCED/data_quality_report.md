# Dataset 3 (FINAL BALANCED) Quality Report

## 1. Summary Statistics
* **Dataset Version:** `3.0-balanced-final`
* **Total Rows (Windows):** 175190
* **Total Sessions:** 650
* **Total Participants:** 425
* **Class Balance:** Honest = 93391 (53.3%), Cheating = 81799 (46.7%)

## 2. Participant-Level Isolation Splits (60/20/20)
* **Train:** 255 participants, 106494 rows
* **Val:** 85 participants, 35024 rows
* **Test:** 85 participants, 33672 rows
* **Leakage Verification:** 0 participant overlap across splits.

## 3. Data Integrity Audit
* **NaNs:** 0
* **Infinities:** 0
* **Metadata Leakage:** Session/Participant IDs stripped during modeling.
