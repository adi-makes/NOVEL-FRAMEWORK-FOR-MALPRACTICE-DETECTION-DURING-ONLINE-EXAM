"""
baselines.py
Person 4 — defines the fixed list of models/experiments (E1-E7 from the sprint plan).
This is a CONFIG file, not a trainer. Person 1 will eventually produce real predictions
for each of these model names; until then, we generate mock predictions for the same names
so the rest of the pipeline (evaluator, ablation, visualize) never has to change.
"""

# E1 — Single modality models
SINGLE_MODALITY_MODELS = ["gaze_only", "mouse_only", "environment_only"]

# E2 — Fusion comparison
FUSION_MODELS = ["early_fusion", "late_fusion", "attention_fusion"]

# E3 — Leave-one-stream-out / pairwise ablation
ABLATION_MODELS = ["gaze_mouse", "gaze_environment", "mouse_environment", "all_three"]

# Full set of models the evaluator must be able to score
ALL_MODELS = SINGLE_MODALITY_MODELS + FUSION_MODELS + ABLATION_MODELS

# E4: stream importance -> handled via attention/gating weights (Person 1 output, not here)
# E5: leave-one-stream-out -> same as ABLATION_MODELS
# E6: error analysis -> handled in evaluator.py (false positive / false negative extraction)
# E7: fairness/sensitivity -> handled by re-running evaluator on stress-test datasets from Person 2

EXPERIMENTS = {
    "E1_single_modality": SINGLE_MODALITY_MODELS,
    "E2_fusion_comparison": FUSION_MODELS,
    "E3_ablation": ABLATION_MODELS,
    "E4_stream_importance": ["attention_fusion"],  # weights inspected separately
    "E5_leave_one_out": ABLATION_MODELS,
    "E6_error_analysis": ALL_MODELS,
    "E7_fairness_sensitivity": ALL_MODELS,  # re-run against stress-test CSVs from Person 2
}

# Research hypotheses to check against real results (do NOT hardcode expected numbers)
HYPOTHESES = {
    "H1": "Three-stream attention fusion will outperform single-modality models (AUC).",
    "H2": "Attention fusion will outperform early and late fusion baselines (AUC).",
    "H3": "Removing any one modality will reduce performance vs all_three (ablation).",
}

if __name__ == "__main__":
    print("Registered models:", ALL_MODELS)
    print("Experiments:", list(EXPERIMENTS.keys()))
