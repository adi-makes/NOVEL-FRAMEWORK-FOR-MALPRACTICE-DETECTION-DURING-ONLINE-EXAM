"""
generate_fake_predictions.py
Person 4 — Day 1 mock data generator.
Produces results/fake_predictions.csv with schema: model,label,prediction
Each model has a DIFFERENT quality level baked in, so the evaluator/visualizer
produce visibly different results per model, similar to how real experiments should look
(single-stream weaker, fusion stronger, attention best) WITHOUT claiming these are real numbers.
This is purely a plumbing-test fixture. Swap for Person 1/2's real output on Day 3.
"""

import numpy as np
import pandas as pd
from baselines import ALL_MODELS

# quality[model] = (signal_strength, noise_level) — arbitrary, just to differentiate models visually
QUALITY = {
    "gaze_only":            (0.35, 0.35),
    "mouse_only":            (0.40, 0.35),
    "environment_only":      (0.30, 0.40),
    "gaze_mouse":            (0.50, 0.30),
    "gaze_environment":      (0.48, 0.30),
    "mouse_environment":     (0.52, 0.28),
    "all_three":             (0.60, 0.22),
    "early_fusion":          (0.55, 0.25),
    "late_fusion":           (0.57, 0.24),
    "attention_fusion":      (0.65, 0.20),
}


def generate_predictions(n_samples=400, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    labels = rng.integers(0, 2, n_samples)  # 0 = honest, 1 = cheating, session-level mock

    for model in ALL_MODELS:
        strength, noise = QUALITY.get(model, (0.4, 0.35))
        scores = labels * strength + rng.normal(0.25, noise, n_samples)
        scores = np.clip(scores, 0, 1)
        for lab, score in zip(labels, scores):
            rows.append({"model": model, "label": int(lab), "prediction": round(float(score), 4)})

    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    df = generate_predictions()
    out_path = "results/fake_predictions.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows across {df['model'].nunique()} models to {out_path}")
    print(df.groupby("model")["label"].count())
