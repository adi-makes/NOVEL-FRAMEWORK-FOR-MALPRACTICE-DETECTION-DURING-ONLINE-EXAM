# Three-Stream Fusion Architecture

## 1. Objective

The fusion model combines three synchronized sources of evidence:

1. Gaze behavior
2. Computer interaction behavior
3. Independently positioned environment camera

The model produces a calibrated risk score for each
10-second exam window.

The score is intended to support human review rather
than automatically establish academic misconduct.

---

## 2. Input Representation

Each 10-second window contains three feature vectors.

### Gaze

7 features:

[gaze_features]

1. fixation_duration_mean
2. fixation_count
3. saccade_velocity_mean
4. gaze_deviation
5. gaze_confidence
6. head_yaw
7. head_pitch

Dimension:

7

---

### Interaction

7 features:

1. cursor_velocity_mean
2. cursor_velocity_std
3. click_frequency
4. keystroke_frequency
5. idle_fraction
6. tab_switch_count
7. velocity_spike_ratio

Dimension:

7

---

### Environment

5 features:

1. phone_detected
2. phone_confidence
3. notes_detected
4. extra_person_count
5. suspicious_objects_count

Dimension:

5

---

# 3. Stream Encoders

Each stream is projected into the same latent dimension.

Gaze:

7 → 128

Interaction:

7 → 128

Environment:

5 → 128

The common representation allows the streams to be compared
and fused.

---

# 4. Fusion Strategies

Three fusion strategies will be evaluated.

## 4.1 Early Fusion

Concatenate all raw feature vectors:

[gaze | interaction | environment]

Dimension:

7 + 7 + 5 = 19

Then:

19 → 128 → 64 → 1

Advantages:

- Simple
- Fast
- Easy to reproduce

Limitation:

- Does not explicitly model modality-specific importance
- Does not explicitly model interactions between modalities

---

## 4.2 Late Fusion

Each modality receives an independent prediction head.

Gaze:

7 → 64 → 1

Interaction:

7 → 64 → 1

Environment:

5 → 64 → 1

The three predictions are combined:

final_score =
    weighted_average(
        gaze_score,
        interaction_score,
        environment_score
    )

Advantages:

- Highly interpretable
- Each modality can be evaluated independently
- Missing modalities can potentially be handled

Limitation:

- Cross-modal relationships are not explicitly modeled

---

## 4.3 Attention Fusion — Proposed Model

Each modality is encoded:

Gaze → 128
Interaction → 128
Environment → 128

The three embeddings are treated as modality tokens:

[Gaze, Interaction, Environment]

Self-attention is applied across the three tokens.

The resulting representations are flattened and passed
through a classification head.

Architecture:

7 → 128
7 → 128
5 → 128

          ↓

[3 × 128 modality tokens]

          ↓

Multi-Head Self Attention

          ↓

3 × 128

          ↓

384 → 64 → 1

          ↓

Sigmoid

          ↓

Risk score [0,1]

---

# 5. Proposed Model

The attention model is the primary proposed architecture.

The early and late fusion models serve as baselines.

---

# 6. Output

The model produces:

risk_score ∈ [0,1]

Example:

{
    "risk_score": 0.87,
    "prediction": "suspicious"
}

The model should additionally expose modality-level
evidence for explainability.

Example:

{
    "risk_score": 0.87,
    "prediction": "suspicious",
    "evidence": {
        "gaze": 0.71,
        "interaction": 0.42,
        "environment": 0.95
    },
    "reasons": [
        "phone detected by environment camera",
        "gaze deviation increased",
        "unusual cursor activity"
    ]
}

The score is an alerting mechanism and should not be
interpreted as automatic proof of cheating.

---

# 7. Required Ablations

The following configurations must eventually be evaluated:

1. Gaze only
2. Interaction only
3. Environment only
4. Gaze + Interaction
5. Gaze + Environment
6. Interaction + Environment
7. Gaze + Interaction + Environment

These experiments test whether combining the streams
actually provides additional information.

---

# 8. Evaluation Metrics

Primary:

- ROC-AUC
- PR-AUC
- F1 score
- Precision
- Recall

Operational:

- False-positive rate
- False-negative rate
- Inference latency
- FPS

Calibration:

- Reliability curve
- Brier score
- Expected Calibration Error

---

# 9. Important Experimental Rule

Data must be split by SESSION rather than individual
10-second windows.

Windows from the same session must never appear in both
training and testing.

This prevents temporal/session leakage.