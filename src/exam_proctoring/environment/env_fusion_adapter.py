"""
Person 3 — Environment → Fusion Adapter
========================================

Converts the TemporalWindowAggregator feature dict into the exact 5-element
tensor and ordered list required by AttentionFusionModel.

Fusion vector order (must match AttentionFusionModel(environment_dim=5)):
    index 0 : phone_detected          (int   : 0 or 1)
    index 1 : phone_confidence        (float : [0, 1])
    index 2 : notes_detected          (int   : 0 or 1)
    index 3 : extra_person_count      (int   : >= 0)
    index 4 : suspicious_objects_count (int  : >= 0)

This order matches the data_schema.md contract:
    exam-proctoring/docs/data_schema.md  §5 Environment Features

Usage
-----
    from p3.src.env_fusion_adapter import EnvironmentFusionAdapter

    adapter = EnvironmentFusionAdapter()
    tensor = adapter.to_tensor(env_feature_dict)   # shape (1, 5)
    vector = adapter.to_vector(env_feature_dict)   # list of 5 values
"""

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# Canonical ordered feature names — defines fusion vector position.
ENVIRONMENT_FEATURE_ORDER = [
    "phone_detected",
    "phone_confidence",
    "notes_detected",
    "extra_person_count",
    "suspicious_objects_count",
]


class EnvironmentFusionAdapter:
    """
    Thin adapter between TemporalWindowAggregator and AttentionFusionModel.

    Ensures the environment feature vector always has the correct dimension
    and field ordering, regardless of what extra diagnostic keys are present
    in the raw feature dict.
    """

    def to_vector(self, env_features: dict) -> list:
        """
        Extract the ordered 5-element list from an environment feature dict.

        Parameters
        ----------
        env_features : dict
            Output of TemporalWindowAggregator.get_features() or .update().

        Returns
        -------
        list of 5 numeric values in ENVIRONMENT_FEATURE_ORDER.
        """
        return [float(env_features.get(key, 0.0)) for key in ENVIRONMENT_FEATURE_ORDER]

    def to_tensor(self, env_features: dict):
        """
        Convert environment feature dict to a PyTorch tensor of shape (1, 5).

        Parameters
        ----------
        env_features : dict
            Output of TemporalWindowAggregator.get_features() or .update().

        Returns
        -------
        torch.FloatTensor of shape (1, 5)

        Raises
        ------
        RuntimeError if PyTorch is not installed.
        """
        if not _TORCH_AVAILABLE:
            raise RuntimeError(
                "PyTorch is not installed. Cannot produce fusion tensor."
            )
        vec = self.to_vector(env_features)
        return torch.tensor([vec], dtype=torch.float32)

    def verify_shape(self, tensor) -> bool:
        """
        Confirm the tensor has shape (batch, 5) as expected by the fusion model.
        Returns True if shape is correct.
        """
        if not _TORCH_AVAILABLE:
            return False
        return (
            isinstance(tensor, torch.Tensor)
            and tensor.ndim == 2
            and tensor.shape[1] == len(ENVIRONMENT_FEATURE_ORDER)
        )

    @staticmethod
    def feature_names() -> list:
        """Return the canonical list of feature names in fusion vector order."""
        return list(ENVIRONMENT_FEATURE_ORDER)
