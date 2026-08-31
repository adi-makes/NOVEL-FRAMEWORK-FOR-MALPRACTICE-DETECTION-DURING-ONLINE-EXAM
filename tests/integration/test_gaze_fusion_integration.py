"""
Integration test for connecting GazeFeatureExtractor output with existing AttentionFusionModel.
"""

import sys
import os
import pytest
import numpy as np

from exam_proctoring.gaze.feature_extractor import GazeFeatureExtractor, TORCH_AVAILABLE
from exam_proctoring.models.attention_fusion import AttentionFusionModel


def test_gaze_features_attention_fusion_contract():
    """Verify that gaze feature extractor produces exactly 7 features matching AttentionFusionModel signature."""
    extractor = GazeFeatureExtractor()

    # Feed synthetic frames
    for i in range(10):
        extractor.update({
            "timestamp": i * 0.1,
            "gaze_x": 0.5,
            "gaze_y": 0.5,
            "gaze_confidence": 0.9,
            "head_yaw": 2.0,
            "head_pitch": -1.0,
            "face_detected": True
        })

    vector = extractor.get_feature_vector()
    assert len(vector) == 7, f"Expected 7 gaze features, got {len(vector)}"
    assert all(isinstance(v, (float, int, np.floating, np.integer)) for v in vector)

    if TORCH_AVAILABLE:
        import torch

        model = AttentionFusionModel(gaze_dim=7, interaction_dim=7, environment_dim=5)
        model.eval()

        gaze_tensor = extractor.to_tensor()  # shape (1, 7)
        interaction_tensor = torch.zeros((1, 7), dtype=torch.float32)
        environment_tensor = torch.zeros((1, 5), dtype=torch.float32)

        with torch.no_grad():
            risk_score, attn_weights = model(gaze_tensor, interaction_tensor, environment_tensor)

        assert risk_score.shape == (1, 1)
        assert 0.0 <= risk_score.item() <= 1.0
