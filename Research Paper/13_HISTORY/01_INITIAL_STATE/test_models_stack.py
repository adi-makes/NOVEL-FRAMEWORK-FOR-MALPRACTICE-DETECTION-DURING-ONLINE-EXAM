import torch
import pytest
from src.exam_proctoring.models.registry import MODEL_REGISTRY, build_model

@pytest.mark.parametrize("model_key", list(MODEL_REGISTRY.keys()))
def test_all_models_forward_and_shapes(model_key):
    gaze = torch.randn(4, 7)
    interaction = torch.randn(4, 7)
    environment = torch.randn(4, 5)

    model = build_model(model_key)
    model.eval()

    with torch.no_grad():
        out = model(gaze, interaction, environment)

    assert out.shape == (4,), f"Model {model_key} returned shape {out.shape}, expected (4,)"
    assert torch.isfinite(out).all(), f"Model {model_key} returned non-finite values"

def test_attention_fusion_weights():
    gaze = torch.randn(4, 7)
    interaction = torch.randn(4, 7)
    environment = torch.randn(4, 5)

    model = build_model("attention_fusion")
    model.eval()

    with torch.no_grad():
        logits, attn = model(gaze, interaction, environment, return_attention=True)

    assert logits.shape == (4,)
    assert attn is not None
    assert attn.shape == (4, 3, 3)
    assert torch.isfinite(attn).all()
