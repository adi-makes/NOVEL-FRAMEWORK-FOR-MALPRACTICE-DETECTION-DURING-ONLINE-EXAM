# Three-Stream Exam Proctoring Data Schema

## 1. Purpose

Each exam observation is represented as a synchronized
10-second window.

The system contains three independent sensing streams:

1. Gaze
2. Computer interaction
3. Environment camera

All streams must produce features using the schema below.

---

## 2. Window Definition

Each record represents one 10-second window.

Required fields:

- window_id
- session_id
- timestamp_start
- timestamp_end

---

## 3. Gaze Features

| Feature | Type | Description |
|---|---|---|
| fixation_duration_mean | float | Mean fixation duration in seconds |
| fixation_count | float | Number of fixations |
| saccade_velocity_mean | float | Mean saccade velocity |
| gaze_deviation | float | Deviation from expected exam-screen gaze |
| gaze_confidence | float | Gaze estimation confidence [0,1] |
| head_yaw | float | Head yaw in degrees |
| head_pitch | float | Head pitch in degrees |

---

## 4. Interaction Features

| Feature | Type | Description |
|---|---|---|
| cursor_velocity_mean | float | Mean cursor velocity |
| cursor_velocity_std | float | Cursor velocity variability |
| click_frequency | float | Clicks per second |
| keystroke_frequency | float | Keystrokes per second |
| idle_fraction | float | Fraction of window spent idle [0,1] |
| tab_switch_count | int | Number of focus/tab changes |
| velocity_spike_ratio | float | Fraction of unusually fast cursor movements |

---

## 5. Environment Features

| Feature | Type | Description |
|---|---|---|
| phone_detected | int | 1 if phone detected, otherwise 0 |
| phone_confidence | float | Phone detection confidence [0,1] |
| notes_detected | int | 1 if notes detected, otherwise 0 |
| extra_person_count | int | Number of additional people |
| suspicious_objects_count | int | Number of suspicious objects |

---

## 6. Labels

| Field | Type | Description |
|---|---|---|
| label | int | 0 = honest, 1 = cheating/suspicious |
| cheating_type | string | none, phone, notes, copy_paste, external_assistance |

---

## 7. Complete Record

Example:

```json
{
  "window_id": "session_001_window_0001",
  "session_id": "session_001",
  "timestamp_start": 0.0,
  "timestamp_end": 10.0,

  "fixation_duration_mean": 1.2,
  "fixation_count": 5,
  "saccade_velocity_mean": 300.0,
  "gaze_deviation": 0.15,
  "gaze_confidence": 0.94,
  "head_yaw": -4.2,
  "head_pitch": 2.1,

  "cursor_velocity_mean": 150.0,
  "cursor_velocity_std": 45.0,
  "click_frequency": 0.2,
  "keystroke_frequency": 2.4,
  "idle_fraction": 0.30,
  "tab_switch_count": 1,
  "velocity_spike_ratio": 0.05,

  "phone_detected": 0,
  "phone_confidence": 0.02,
  "notes_detected": 0,
  "extra_person_count": 0,
  "suspicious_objects_count": 0,

  "label": 0,
  "cheating_type": "none"


### Why we're doing this

This file is **the contract between all four people**.

Person 2 will generate data matching this.

Person 3 will extract CV features matching this.

Person 4 will evaluate models expecting this.

You build the fusion model against this.

Nobody needs to wait for anybody else. This exact interface is the key dependency-breaking mechanism in the plan. 
