import torch
import torch.nn as nn
from models.common import GAZE_FEATURES, INTERACTION_FEATURES, ENVIRONMENT_FEATURES


class ModelAdapter(nn.Module):
    """
    Standard adapter ensuring all models accept (gaze, interaction, environment)
    and return raw logits of shape (B,).
    """

    def __init__(self, model: nn.Module, required_streams: tuple):
        super().__init__()
        self.model = model
        self.required_streams = required_streams

    def forward(self, gaze, interaction, environment, return_attention=False):
        if self.required_streams == ("gaze",):
            out = self.model(gaze)
        elif self.required_streams == ("interaction",):
            out = self.model(interaction)
        elif self.required_streams == ("environment",):
            out = self.model(environment)
        elif self.required_streams == ("gaze", "interaction"):
            out = self.model(gaze, interaction)
        elif self.required_streams == ("gaze", "environment"):
            out = self.model(gaze, environment)
        elif self.required_streams == ("interaction", "environment"):
            out = self.model(interaction, environment)
        else:
            if hasattr(self.model, "forward") and "return_attention" in self.model.forward.__code__.co_varnames:
                return self.model(gaze, interaction, environment, return_attention=return_attention)
            out = self.model(gaze, interaction, environment)

        if isinstance(out, tuple):
            logits = out[0]
            attn = out[1] if len(out) > 1 else None
            if return_attention:
                return logits.squeeze(-1) if logits.ndim > 1 else logits, attn
            return logits.squeeze(-1) if logits.ndim > 1 else logits

        return out.squeeze(-1) if out.ndim > 1 else out
