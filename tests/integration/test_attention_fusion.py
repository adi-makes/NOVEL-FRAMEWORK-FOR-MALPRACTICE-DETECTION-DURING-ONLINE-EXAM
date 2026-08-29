import torch

from exam_proctoring.models.attention_fusion import AttentionFusionModel


def test_attention_fusion_forward():

    model = AttentionFusionModel()

    batch_size = 4

    gaze = torch.randn(batch_size, 7)
    interaction = torch.randn(batch_size, 7)
    environment = torch.randn(batch_size, 5)

    risk_score, attention_weights = model(
        gaze,
        interaction,
        environment,
    )

    assert risk_score.shape == (batch_size, 1)

    assert attention_weights.shape[0] == batch_size

    assert torch.all(risk_score >= 0)
    assert torch.all(risk_score <= 1)