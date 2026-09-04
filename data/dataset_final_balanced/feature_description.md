# Dataset 3 Feature Description (19 Features)

## Gaze Stream (7 Features)
1. `fixation_duration_mean`: Mean fixation duration (seconds)
2. `fixation_count`: Number of gaze fixations in 10s window
3. `saccade_velocity_mean`: Mean saccade angular velocity (deg/s)
4. `gaze_deviation`: Normalized gaze distance off center [0, 1]
5. `gaze_confidence`: Eye tracker tracking confidence [0, 1]
6. `head_yaw`: Head horizontal rotation (degrees)
7. `head_pitch`: Head vertical rotation (degrees)

## Interaction Stream (7 Features)
8. `cursor_velocity_mean`: Mean cursor velocity (pixels/s)
9. `cursor_velocity_std`: Standard deviation of cursor velocity
10. `click_frequency`: Clicks per second
11. `keystroke_frequency`: Keystrokes per second
12. `idle_fraction`: Fraction of window with no user input [0, 1]
13. `tab_switch_count`: Count of window/tab switches
14. `velocity_spike_ratio`: Ratio of sudden rapid cursor movements

## Environment Stream (5 Features)
15. `phone_detected`: Binary indicator of mobile phone detection
16. `phone_confidence`: Object detector confidence score for phone [0, 1]
17. `notes_detected`: Binary indicator of notes/cheat sheet detection
18. `extra_person_count`: Count of secondary people in camera view
19. `suspicious_objects_count`: Total suspicious objects detected
