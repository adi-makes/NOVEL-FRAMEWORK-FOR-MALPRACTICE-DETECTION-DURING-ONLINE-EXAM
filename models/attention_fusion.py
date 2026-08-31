import torch
import torch.nn as nn


class AttentionFusionModel(nn.Module):
    """
    Proposed three-stream attention fusion model.

    Streams:
        gaze:        (B, 7)
        interaction: (B, 7)
        environment: (B, 5)

    Each stream is independently projected into a shared
    128-dimensional representation.

    Three stream representations become three modality tokens.

    Multi-head self-attention allows information from one
    modality to influence the representation of another.

    Final classifier:
        384 -> 64 -> 1

    Output:
        raw binary classification logit
    """

    def __init__(
        self,
        gaze_dim: int = 7,
        interaction_dim: int = 7,
        environment_dim: int = 5,
        hidden_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.gaze_encoder = nn.Sequential(
            nn.Linear(gaze_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.interaction_encoder = nn.Sequential(
            nn.Linear(interaction_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.environment_encoder = nn.Sequential(
            nn.Linear(environment_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        gaze,
        interaction,
        environment,
        return_attention=False,
    ):

        # --------------------------------------------------
        # 1. Encode each stream
        # --------------------------------------------------

        gaze_embedding = self.gaze_encoder(gaze)
        interaction_embedding = self.interaction_encoder(
            interaction
        )
        environment_embedding = self.environment_encoder(
            environment
        )

        # --------------------------------------------------
        # 2. Create three modality tokens
        # --------------------------------------------------

        tokens = torch.stack(
            [
                gaze_embedding,
                interaction_embedding,
                environment_embedding,
            ],
            dim=1,
        )

        # Shape:
        # (batch, 3, 128)

        # --------------------------------------------------
        # 3. Cross-modal self-attention
        # --------------------------------------------------

        attended, attention_weights = self.attention(
            tokens,
            tokens,
            tokens,
            need_weights=True,
        )

        attended = self.norm(
            attended + tokens
        )

        # --------------------------------------------------
        # 4. Flatten modality representations
        # --------------------------------------------------

        fused = attended.reshape(
            attended.size(0),
            -1,
        )

        # (batch, 384)

        # --------------------------------------------------
        # 5. Classification
        # --------------------------------------------------

        logit = self.classifier(
            fused
        ).squeeze(-1)

        if return_attention:
            return logit, attention_weights

        return logit