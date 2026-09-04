from typing import Dict, Any, Callable
import torch.nn as nn

from models.single_stream import GazeOnlyModel, InteractionOnlyModel, EnvironmentOnlyModel
from models.pairwise import GazeInteractionModel, GazeEnvironmentModel, InteractionEnvironmentModel
from models.early_fusion import EarlyFusionModel
from models.late_fusion import LateFusionModel
from models.attention_fusion import AttentionFusionModel
from src.exam_proctoring.models.adapter import ModelAdapter

MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "gaze_only": {
        "name": "Gaze-only",
        "constructor": GazeOnlyModel,
        "required_streams": ("gaze",),
        "checkpoint": "models/model_1_initial_dataset/checkpoints/gaze_only.pt",
        "prediction_file": "results/model_1_initial_dataset/run1/predictions/gaze_only.csv",
    },
    "interaction_only": {
        "name": "Interaction-only",
        "constructor": InteractionOnlyModel,
        "required_streams": ("interaction",),
        "checkpoint": "models/model_1_initial_dataset/checkpoints/interaction_only.pt",
        "prediction_file": "results/model_1_initial_dataset/run1/predictions/interaction_only.csv",
    },
    "environment_only": {
        "name": "Environment-only",
        "constructor": EnvironmentOnlyModel,
        "required_streams": ("environment",),
        "checkpoint": "models/model_1_initial_dataset/checkpoints/environment_only.pt",
        "prediction_file": "results/model_1_initial_dataset/run1/predictions/environment_only.csv",
    },
    "gaze_interaction": {
        "name": "Gaze + Interaction",
        "constructor": GazeInteractionModel,
        "required_streams": ("gaze", "interaction"),
        "checkpoint": "models/model_1_initial_dataset/checkpoints/gaze_interaction.pt",
        "prediction_file": "results/model_1_initial_dataset/run1/predictions/gaze_interaction.csv",
    },
    "gaze_environment": {
        "name": "Gaze + Environment",
        "constructor": GazeEnvironmentModel,
        "required_streams": ("gaze", "environment"),
        "checkpoint": "models/model_1_initial_dataset/checkpoints/gaze_environment.pt",
        "prediction_file": "results/model_1_initial_dataset/run1/predictions/gaze_environment.csv",
    },
    "interaction_environment": {
        "name": "Interaction + Environment",
        "constructor": InteractionEnvironmentModel,
        "required_streams": ("interaction", "environment"),
        "checkpoint": "models/model_1_initial_dataset/checkpoints/interaction_environment.pt",
        "prediction_file": "results/model_1_initial_dataset/run1/predictions/interaction_environment.csv",
    },
    "early_fusion": {
        "name": "Early Fusion",
        "constructor": EarlyFusionModel,
        "required_streams": ("gaze", "interaction", "environment"),
        "checkpoint": "models/model_1_initial_dataset/checkpoints/early_fusion.pt",
        "prediction_file": "results/model_1_initial_dataset/run1/predictions/early_fusion.csv",
    },
    "late_fusion": {
        "name": "Late Fusion",
        "constructor": LateFusionModel,
        "required_streams": ("gaze", "interaction", "environment"),
        "checkpoint": "models/model_1_initial_dataset/checkpoints/late_fusion.pt",
        "prediction_file": "results/model_1_initial_dataset/run1/predictions/late_fusion.csv",
    },
    "attention_fusion": {
        "name": "Three-stream Attention Fusion",
        "constructor": AttentionFusionModel,
        "required_streams": ("gaze", "interaction", "environment"),
        "checkpoint": "models/model_1_initial_dataset/checkpoints/attention_fusion.pt",
        "prediction_file": "results/model_1_initial_dataset/run1/predictions/attention_fusion.csv",
    },
}


def build_model(model_key: str, dropout: float = 0.2) -> nn.Module:
    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model key: {model_key}. Available: {list(MODEL_REGISTRY.keys())}")

    config = MODEL_REGISTRY[model_key]
    raw_model = config["constructor"](dropout=dropout)
    adapted_model = ModelAdapter(raw_model, config["required_streams"])
    return adapted_model
