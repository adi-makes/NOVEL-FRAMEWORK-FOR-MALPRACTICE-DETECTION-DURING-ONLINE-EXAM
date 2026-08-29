import torch
import torch.nn as nn


class AttentionFusionModel(nn.Module):
    """
    Three-stream attention fusion model.

    Inputs:
        gaze:        (batch, 7)
        interaction: (batch, 7)
        environment: (batch, 5)

    Output:
        risk_score:  (batch, 1)
    """

    def __init__(
        self,
        gaze_dim=7,
        interaction_dim=7,
        environment_dim=5,
        hidden_dim=128,
        num_heads=4,
    ):
        super().__init__()

        # Stream-specific encoders
        self.gaze_encoder = nn.Sequential(
            nn.Linear(gaze_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.interaction_encoder = nn.Sequential(
            nn.Linear(interaction_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.environment_encoder = nn.Sequential(
            nn.Linear(environment_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # Cross-modal attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, gaze, interaction, environment):

        # Encode individual streams
        gaze_embedding = self.gaze_encoder(gaze)

        interaction_embedding = self.interaction_encoder(
            interaction
        )

        environment_embedding = self.environment_encoder(
            environment
        )

        # Create modality tokens
        tokens = torch.stack(
            [
                gaze_embedding,
                interaction_embedding,
                environment_embedding,
            ],
            dim=1,
        )

        # Attention across modalities
        attended, attention_weights = self.attention(
            tokens,
            tokens,
            tokens,
        )

        # Flatten
        fused = attended.flatten(start_dim=1)

        # Classification
        logits = self.classifier(fused)

        # Risk score
        risk_score = torch.sigmoid(logits)

        return risk_score, attention_weights