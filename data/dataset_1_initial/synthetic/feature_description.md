# Multimodal Feature Schema & Value Ranges (v1.0 Frozen)

This specification defines the 17 behavioral features, window metadata, and ground-truth labels across all 10-second exam intervals.

## 1. Identifiers & Metadata
* `session_id` (str): Unique candidate session token (`sess_000` to `sess_199`).
* `window_id` (str): Synchronized interval token (`sess_XXX_wYYY`).
* `timestamp_start` (float): Window opening timestamp (seconds).
* `timestamp_end` (float): Window closing timestamp (seconds).

## 2. Gaze Stream Features
* `fixation_duration_mean` (float, range [0.2, 8.0] s): Mean length of ocular fixations.
* `fixation_count` (int, range [0, 15]): Total discrete gaze fixations during the window.
* `saccade_velocity_mean` (float, range [50, 600] deg/s): Mean speed of rapid eye shifts.
* `gaze_deviation` (float, range [0.0, 1.0]): Normalized gaze offset from on-screen exam zone.
* `gaze_confidence` (float, range [0.1, 1.0]): Gaze tracking sensor confidence score.
* `head_yaw` (float, range [-60.0, 60.0] deg): Horizontal head rotation angle.
* `head_pitch` (float, range [-60.0, 60.0] deg): Vertical head tilt angle.

## 3. Mouse & Interaction Stream Features
* `cursor_velocity_mean` (float, range [0.0, 800.0] px/s): Average cursor travel velocity.
* `cursor_velocity_std` (float, range [0.0, 300.0] px/s): Standard deviation of cursor velocity.
* `click_frequency` (float, range [0.0, 4.0] clicks/s): Mouse click rate per second.
* `keystroke_frequency` (float, range [0.0, 8.0] keys/s): Keyboard input rate per second.
* `idle_fraction` (float, range [0.0, 1.0]): Fraction of window without cursor/keyboard activity.
* `tab_switch_count` (int, range [0, 6]): Count of OS application/tab switch events.
* `velocity_spike_ratio` (float, range [0.0, 1.0]): Ratio of abrupt cursor acceleration bursts.

## 4. Environment Stream Features
* `phone_detected` (int, {0, 1}): Binary indicator of smartphone presence.
* `phone_confidence` (float, range [0.0, 1.0]): Object detector confidence for phone cue.
* `notes_detected` (int, {0, 1}): Binary indicator of unauthorized notes or cheat sheets.
* `extra_person_count` (int, range [0, 3]): Count of secondary persons in room camera frame.
* `suspicious_objects_count` (int, range [0, 5]): Aggregate count of anomalous items.

## 5. Ground Truth & Partitions
* `label` (int, {0, 1}): `0` for honest baseline, `1` for active malpractice.
* `cheating_type` (str): Category label (`none`, `phone`, `notes`, `copy_paste`, `external_assistance`).
* `split` (str): Session partition token (`train`, `val`, `test`).