import torch
import torch.nn as nn

from .common import (
    GAZE_DIM,
    INTERACTION_DIM,
    ENVIRONMENT_DIM,
)


class PairwiseMLP(nn.Module):
    """
    Generic pairwise MLP.

    Returns raw binary-classification logits.
    """

    def __init__(
        self,
        input_dim: int,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)


class GazeInteractionModel(PairwiseMLP):

    def __init__(self, dropout=0.2):
        super().__init__(
            input_dim=GAZE_DIM + INTERACTION_DIM,
            dropout=dropout,
        )

    def forward(self, gaze, interaction):
        x = torch.cat([gaze, interaction], dim=1)
        return super().forward(x)


class GazeEnvironmentModel(PairwiseMLP):

    def __init__(self, dropout=0.2):
        super().__init__(
            input_dim=GAZE_DIM + ENVIRONMENT_DIM,
            dropout=dropout,
        )

    def forward(self, gaze, environment):
        x = torch.cat([gaze, environment], dim=1)
        return super().forward(x)


class InteractionEnvironmentModel(PairwiseMLP):

    def __init__(self, dropout=0.2):
        super().__init__(
            input_dim=INTERACTION_DIM + ENVIRONMENT_DIM,
            dropout=dropout,
        )

    def forward(self, interaction, environment):
        x = torch.cat([interaction, environment], dim=1)
        return super().forward(x)