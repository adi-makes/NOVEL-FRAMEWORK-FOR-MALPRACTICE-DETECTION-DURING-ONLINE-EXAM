import torch

from .common import (
    GAZE_DIM,
    INTERACTION_DIM,
    ENVIRONMENT_DIM,
    MLPBlock,
)


class GazeOnlyModel(MLPBlock):
    """
    Gaze-only baseline.

    Input:
        (B, 7)

    Output:
        raw logit (B,)
    """

    def __init__(self, dropout: float = 0.2):
        super().__init__(
            input_dim=GAZE_DIM,
            hidden_dims=(32, 16),
            dropout=dropout,
        )


class InteractionOnlyModel(MLPBlock):
    """
    Interaction-only baseline.

    Input:
        (B, 7)
    """

    def __init__(self, dropout: float = 0.2):
        super().__init__(
            input_dim=INTERACTION_DIM,
            hidden_dims=(32, 16),
            dropout=dropout,
        )


class EnvironmentOnlyModel(MLPBlock):
    """
    Environment-only baseline.

    Input:
        (B, 5)
    """

    def __init__(self, dropout: float = 0.2):
        super().__init__(
            input_dim=ENVIRONMENT_DIM,
            hidden_dims=(32, 16),
            dropout=dropout,
        )