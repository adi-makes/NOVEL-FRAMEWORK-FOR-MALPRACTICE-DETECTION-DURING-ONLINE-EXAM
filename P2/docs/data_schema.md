# 10-Second Window Synthetic Dataset Schema

## Window Structure
Each record represents an aggregated 10-second exam interval.

### 1. Metadata & Identifiers
* `window_id`: Unique window identifier (`<session_id>_w<index>`)
* `session_id`: Exam session identifier
* `timestamp_start`: Start of interval in seconds
* `timestamp_end`: End of interval in seconds
* `split`: Data partition (`train`, `val`, `test` split by session)
* `cheating_type`: Ground-truth behavior pattern (`none`, `phone`, `notes`, `copy_paste`, `external_assistance`)
* `label`: Binary classification target (`0` = honest, `1` = active cheating)

### 2. Stream 1: Gaze Features
* `fixation_duration_mean`: Mean fixation duration (seconds)
* `fixation_count`: Number of distinct fixations
* `saccade_velocity_mean`: Average saccade velocity (deg/s)
* `gaze_deviation`: Normalized eye deviation from screen center (0.0 to 1.0)
* `gaze_confidence`: Model confidence score (0.0 to 1.0)
* `head_yaw`: Head rotation left/right in degrees
* `head_pitch`: Head tilt up/down in degrees

### 3. Stream 2: Mouse & Interaction Features
* `cursor_velocity_mean`: Mean pointer speed (px/s)
* `cursor_velocity_std`: Standard deviation of pointer speed
* `click_frequency`: Clicks per second
* `keystroke_frequency`: Keystrokes per second
* `idle_fraction`: Fraction of the 10s window without interaction (0.0 to 1.0)
* `tab_switch_count`: Count of active window/focus losses
* `velocity_spike_ratio`: Fraction of movement intervals exceeding rapid velocity threshold

### 4. Stream 3: Environment Camera Features
* `phone_detected`: Binary indicator (0 or 1)
* `phone_confidence`: Detection probability score
* `notes_detected`: Binary indicator (0 or 1)
* `extra_person_count`: Secondary persons detected in frame
* `suspicious_objects_count`: Total count of flagged objects