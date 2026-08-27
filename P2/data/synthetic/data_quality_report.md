# Synthetic Dataset Quality Report (Day 2 Deliverable)

## 1. Summary Statistics
* **Total Sessions:** 200
* **Total 10-Second Windows:** 3600
* **Session Splits:** Train (120 sessions), Val (40 sessions), Test (40 sessions)
* **Missing / NaN Values:** 0

## 2. Window Class Balance
* **Honest Windows (label=0):** 2700
* **Cheating Windows (label=1):** 900

## 3. Scenario Distribution
| Modality Scenario | Session Count | Window Count |
|---|---|---|
| Honest (`none`) | 100 | 2700 |
| Phone | 25 | 225 |
| Notes | 25 | 225 |
| Copy/Paste | 25 | 225 |
| External Assistance | 25 | 225 |

## 4. Leakage Verification
* Distinct Train-Val Session Overlap: 0
* Distinct Train-Test Session Overlap: 0
* Distinct Val-Test Session Overlap: 0
