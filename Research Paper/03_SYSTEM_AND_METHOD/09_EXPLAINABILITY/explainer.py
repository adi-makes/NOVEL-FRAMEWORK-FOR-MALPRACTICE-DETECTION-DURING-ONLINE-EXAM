import torch
import numpy as np
from typing import Dict, Any, List

class MultimodalProctoringExplainer:
    """
    Provides explainable risk scoring and stream/feature evidence attribution 
    using feature perturbation and occlusion analysis (rather than assuming 
    raw attention weights are causal).
    """

    def __init__(self, model, scaler=None):
        self.model = model
        self.model.eval()
        self.scaler = scaler

    def explain_instance(
        self, 
        gaze: torch.Tensor, 
        interaction: torch.Tensor, 
        environment: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Calculates risk score, stream attributions via occlusion/zeroing, 
        and extracts human-readable evidence strings.
        """
        with torch.no_grad():
            # Original forward pass
            orig_logit = self.model(gaze, interaction, environment)
            risk_score = float(torch.sigmoid(orig_logit).item())

            # Occlude Gaze (zero out gaze tensor)
            gaze_zero = torch.zeros_like(gaze)
            gaze_occ_logit = self.model(gaze_zero, interaction, environment)
            gaze_impact = max(0.0, float((torch.sigmoid(orig_logit) - torch.sigmoid(gaze_occ_logit)).item()))

            # Occlude Interaction
            inter_zero = torch.zeros_like(interaction)
            inter_occ_logit = self.model(gaze, inter_zero, environment)
            inter_impact = max(0.0, float((torch.sigmoid(orig_logit) - torch.sigmoid(inter_occ_logit)).item()))

            # Occlude Environment
            env_zero = torch.zeros_like(environment)
            env_occ_logit = self.model(gaze, interaction, env_zero)
            env_impact = max(0.0, float((torch.sigmoid(orig_logit) - torch.sigmoid(env_occ_logit)).item()))

        # Normalize stream contributions
        tot = gaze_impact + inter_impact + env_impact + 1e-6
        stream_evidence = {
            "gaze": round(gaze_impact / tot, 4),
            "interaction": round(inter_impact / tot, 4),
            "environment": round(env_impact / tot, 4),
        }

        # Generate rule-based diagnostic reasons from unscaled input if available
        reasons: List[str] = []
        if risk_score > 0.5:
            if stream_evidence["gaze"] > 0.3:
                reasons.append("Elevated gaze deviation / off-screen fixation detected")
            if stream_evidence["interaction"] > 0.3:
                reasons.append("Unusual cursor velocity spikes or rapid tab-switching observed")
            if stream_evidence["environment"] > 0.2:
                reasons.append("Secondary object or potential phone/notes detection in camera stream")

        if not reasons:
            reasons.append("Behavior consistent with standard exam activity")

        return {
            "risk_score": round(risk_score, 4),
            "stream_evidence": stream_evidence,
            "reasons": reasons,
        }
