import torch
import torch.nn as nn

from .common import TOTAL_DIM


class EarlyFusionModel(nn.Module):
    """
    Early-fusion baseline.

    All 17 features are concatenated first.

    17 -> 128 -> 64 -> 1

    Returns raw logits.
    """

    def __init__(self, dropout: float = 0.2):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(TOTAL_DIM, 128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(64, 1),
        )

    def forward(
        self,
        gaze,
        interaction,
        environment,
    ):
        x = torch.cat(
            [gaze, interaction, environment],
            dim=1,
        )

        return self.classifier(x).squeeze(-1)