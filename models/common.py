from typing import Sequence

import torch
import torch.nn as nn


GAZE_FEATURES: Sequence[str] = (
    "fixation_duration_mean",
    "fixation_count",
    "saccade_velocity_mean",
    "gaze_deviation",
    "gaze_confidence",
    "head_yaw",
    "head_pitch",
)

INTERACTION_FEATURES: Sequence[str] = (
    "cursor_velocity_mean",
    "cursor_velocity_std",
    "click_frequency",
    "keystroke_frequency",
    "idle_fraction",
    "tab_switch_count",
    "velocity_spike_ratio",
)

ENVIRONMENT_FEATURES: Sequence[str] = (
    "phone_detected",
    "phone_confidence",
    "notes_detected",
    "extra_person_count",
    "suspicious_objects_count",
)

ALL_FEATURES: Sequence[str] = (
    *GAZE_FEATURES,
    *INTERACTION_FEATURES,
    *ENVIRONMENT_FEATURES,
)

GAZE_DIM = len(GAZE_FEATURES)          # 7
INTERACTION_DIM = len(INTERACTION_FEATURES)  # 7
ENVIRONMENT_DIM = len(ENVIRONMENT_FEATURES)  # 5
TOTAL_DIM = len(ALL_FEATURES)          # 17


class MLPBlock(nn.Module):
    """
    Reusable MLP classifier.

    Returns raw logits.
    Sigmoid is intentionally NOT applied here.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims=(32, 16),
        dropout: float = 0.2,
    ):
        super().__init__()

        layers = []
        previous_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(previous_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            previous_dim = hidden_dim

        layers.append(nn.Linear(previous_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)