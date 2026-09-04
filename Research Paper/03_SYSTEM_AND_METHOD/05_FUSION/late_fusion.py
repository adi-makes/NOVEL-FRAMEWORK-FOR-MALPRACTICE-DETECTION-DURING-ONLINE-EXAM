import torch
import torch.nn as nn
import torch.nn.functional as F


class StreamHead(nn.Module):
    """
    Independent classifier for one stream.
    """

    def __init__(
        self,
        input_dim: int,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)


class LateFusionModel(nn.Module):
    """
    Learnable late-fusion baseline.

    Each modality produces an independent logit.
    The logits are combined using learned softmax-normalized
    stream weights.
    """

    def __init__(self, dropout: float = 0.2):
        super().__init__()

        self.gaze_head = StreamHead(
            input_dim=7,
            dropout=dropout,
        )

        self.interaction_head = StreamHead(
            input_dim=7,
            dropout=dropout,
        )

        self.environment_head = StreamHead(
            input_dim=5,
            dropout=dropout,
        )

        # Learnable fusion weights.
        self.stream_logits = nn.Parameter(
            torch.zeros(3)
        )

    def forward(
        self,
        gaze,
        interaction,
        environment,
    ):
        gaze_logit = self.gaze_head(gaze)

        interaction_logit = self.interaction_head(
            interaction
        )

        environment_logit = self.environment_head(
            environment
        )

        weights = F.softmax(
            self.stream_logits,
            dim=0,
        )

        fused_logit = (
            weights[0] * gaze_logit
            + weights[1] * interaction_logit
            + weights[2] * environment_logit
        )

        return fused_logit