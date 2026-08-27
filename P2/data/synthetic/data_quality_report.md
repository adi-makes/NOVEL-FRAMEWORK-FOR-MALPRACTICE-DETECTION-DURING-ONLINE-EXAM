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

## 5. Feature-Label Correlation Audit (Day 3 QA)
* **Shortcut Learning Check:** PASS (All individual feature correlations $|\rho| < 0.85$)
* **Top Contributing Signals:**
  * `gaze_deviation`: $r = 0.7790$
  * `idle_fraction`: $r = 0.7185$
  * `head_yaw`: $r = 0.7177$
  * `saccade_velocity_mean`: $r = 0.6954$
  * `suspicious_objects_count`: $r = 0.6699$
* **Non-Separable Single Cues:**
  * `phone_detected`: $r = 0.4145$ (requires multimodal context)
  * `tab_switch_count`: $r = 0.4041$
  * `click_frequency`: $r = 0.0338$