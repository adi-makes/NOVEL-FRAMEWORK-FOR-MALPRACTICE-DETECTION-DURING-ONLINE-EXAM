import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import yaml

from src.exam_proctoring.data.dataset import (
    create_dataloaders,
    get_project_root,
    load_stress_dataset,
)
from src.exam_proctoring.evaluation.metrics import (
    calculate_metrics,
    find_optimal_threshold,
)
from src.exam_proctoring.evaluation.utils import (
    count_parameters,
    evaluate_hypotheses,
    generate_visualizations,
    get_predictions,
    set_seed,
)
from src.exam_proctoring.models.registry import MODEL_REGISTRY, build_model
from src.exam_proctoring.training.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Master Orchestrator for Proctoring ML Framework Experiments"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_REGISTRY.keys()),
        help="Models to train/evaluate (default: all 9 models)",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Force retrain models even if checkpoints exist",
    )
    parser.add_argument(
        "--skip-stress-tests", action="store_true", help="Skip stress test evaluation"
    )
    parser.add_argument("--skip-figures", action="store_true", help="Skip figure generation")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiment_config.yaml",
        help="Path to experiment configuration YAML",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = get_project_root()

    # Load configuration
    config_path = project_root / args.config
    if config_path.exists():
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    else:
        config = {
            "seed": 42,
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 50,
            "patience": 8,
            "dropout": 0.2,
            "dataset_path": "data/dataset_1_initial/synthetic/dataset.csv",
            "scaler_path": "models/model_1_initial_dataset/checkpoints/scaler.joblib",
        }

    seed = args.seed if args.seed is not None else config.get("seed", 42)
    epochs = args.epochs if args.epochs is not None else config.get("epochs", 50)
    lr = config.get("learning_rate", 0.001)
    batch_size = config.get("batch_size", 32)
    patience = config.get("patience", 8)
    dropout = config.get("dropout", 0.2)

    set_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("============================================================")
    print("PROCTORING FRAMEWORK EXPERIMENT PIPELINE")
    print("============================================================")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Random Seed: {seed}")
    print("============================================================")

    # 1. Dataset & DataLoaders Verification
    dataset_path = project_root / config.get(
        "dataset_path", "data/dataset_1_initial/synthetic/dataset.csv"
    )
    scaler_path = project_root / config.get("scaler_path", "models/model_1_initial_dataset/checkpoints/scaler.joblib")

    print("\n[1/6] Loading & Validating Canonical Dataset...")
    try:
        train_loader, val_loader, test_loader, scaler, integrity_info = create_dataloaders(
            data_path=str(dataset_path),
            batch_size=batch_size,
            scaler_path=str(scaler_path),
            fit_scaler=True,
        )
    except Exception as e:
        print(f"CRITICAL ERROR in dataset validation: {e}")
        sys.exit(1)

    print(
        f"Dataset validation passed! Total rows: {integrity_info['total_rows']}, "
        f"Train: {integrity_info['train_rows']} ({integrity_info['train_sessions']} sess), "
        f"Val: {integrity_info['val_rows']} ({integrity_info['val_sessions']} sess), "
        f"Test: {integrity_info['test_rows']} ({integrity_info['test_sessions']} sess)"
    )
    pos_weight = integrity_info["pos_weight"]
    print(f"Calculated Train pos_weight for BCEWithLogitsLoss: {pos_weight:.4f}")

    # Results & Checkpoint Directories
    checkpoints_dir = project_root / "models" / "final" / "checkpoints"
    results_dir = project_root / "results" / "final"
    metrics_dir = results_dir
    predictions_dir = results_dir / "predictions"
    stress_dir = results_dir / "stress_tests"
    explanations_dir = results_dir / "explanations"
    figures_dir = project_root / "figures" / "final"

    for d in [
        checkpoints_dir,
        metrics_dir,
        predictions_dir,
        stress_dir,
        explanations_dir,
        figures_dir,
    ]:
        d.mkdir(parents=True, exist_ok=True)

    # 2. Build & Train Models
    print("\n[2/6] Training / Loading Models...")
    trained_models = {}
    optimal_thresholds = {}

    for model_key in args.models:
        if model_key not in MODEL_REGISTRY:
            print(f"Warning: Skipping unknown model key {model_key}")
            continue

        meta = MODEL_REGISTRY[model_key]
        cp_path = project_root / meta["checkpoint"]

        model = build_model(model_key, dropout=dropout)

        if cp_path.exists() and not args.force_retrain:
            print(f" -> Loading saved checkpoint for [{meta['name']}] from {cp_path}")
            ckpt = torch.load(cp_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            print(f" -> Training [{meta['name']}]...")
            trainer = Trainer(
                model=model,
                device=device,
                learning_rate=lr,
                pos_weight=pos_weight,
            )
            fit_res = trainer.fit(
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=epochs,
                patience=patience,
                checkpoint_path=str(cp_path),
                model_name=meta["name"],
                config_meta={"seed": seed, "lr": lr, "batch_size": batch_size},
            )
            # Load best weights back
            ckpt = torch.load(cp_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])

        model.to(device)
        trained_models[model_key] = model

        # Tune threshold on VAL set
        val_probs, val_labels, _ = get_predictions(model, val_loader, device)
        thresh = find_optimal_threshold(val_labels, val_probs, metric="f1")
        optimal_thresholds[model_key] = thresh
        print(f"    Optimal Val Threshold (max F1) for [{meta['name']}]: {thresh:.4f}")

    # 3. Test Evaluation
    print("\n[3/6] Running Test Set Inference & Evaluation...")
    test_results_list = []
    test_predictions_dict = {}
    attention_diagnostics = []

    # Get underlying raw test dataframe for session/window identifiers
    raw_test_df = load_stress_dataset(dataset_path, scaler)[1]
    raw_test_df = raw_test_df[raw_test_df["split"] == "test"].reset_index(drop=True)

    for model_key, model in trained_models.items():
        meta = MODEL_REGISTRY[model_key]
        thresh = optimal_thresholds[model_key]

        probs, labels, attns = get_predictions(model, test_loader, device)
        preds = (probs >= thresh).astype(int)

        # Save predictions CSV
        pred_df = pd.DataFrame(
            {
                "model": model_key,
                "session_id": raw_test_df["session_id"].values,
                "window_id": raw_test_df["window_id"].values,
                "label": labels.astype(int),
                "probability": probs,
                "prediction": preds,
                "threshold": thresh,
            }
        )
        pred_csv_path = project_root / meta["prediction_file"]
        pred_df.to_csv(pred_csv_path, index=False)
        test_predictions_dict[model_key] = pred_df

        # Calculate metrics
        m = calculate_metrics(labels, probs, threshold=thresh)
        m["model_key"] = model_key
        m["model"] = meta["name"]
        m["parameter_count"] = count_parameters(model)
        test_results_list.append(m)

        # Record attention diagnostics if available
        if attns is not None:
            # attns shape: (N, 3, 3)
            avg_attn = np.mean(attns, axis=0)  # average attention matrix across test samples
            for r_idx, src in enumerate(["gaze", "interaction", "environment"]):
                for c_idx, tgt in enumerate(["gaze", "interaction", "environment"]):
                    attention_diagnostics.append(
                        {
                            "source_modality": src,
                            "target_modality": tgt,
                            "mean_attention_weight": float(avg_attn[r_idx, c_idx]),
                        }
                    )

    test_results_df = pd.DataFrame(test_results_list)
    test_results_df.to_csv(metrics_dir / "main_results.csv", index=False)
    test_results_df.to_csv(metrics_dir / "results.csv", index=False)

    # Save ablation_results.csv (comparing full attention vs pairwise/ablated models)
    ablation_keys = ["attention_fusion", "gaze_interaction", "gaze_environment", "interaction_environment"]
    ablation_df = test_results_df[test_results_df["model_key"].isin(ablation_keys)].copy()
    ablation_df.to_csv(metrics_dir / "ablation_results.csv", index=False)

    results_json_dict = {row["model_key"]: row for row in test_results_list}
    with open(metrics_dir / "results.json", "w") as f:
        json.dump(results_json_dict, f, indent=2)

    if attention_diagnostics:
        pd.DataFrame(attention_diagnostics).to_csv(
            explanations_dir / "attention_diagnostics.csv", index=False
        )

    # Error analysis for Attention Fusion (or best model)
    if "attention_fusion" in test_predictions_dict:
        att_preds = test_predictions_dict["attention_fusion"]
        # False Positives (label=0, pred=1) sorted by highest probability
        fps = (
            att_preds[(att_preds["label"] == 0) & (att_preds["prediction"] == 1)]
            .sort_values(by="probability", ascending=False)
            .head(5)
        )
        # False Negatives (label=1, pred=0) sorted by lowest probability
        fns = (
            att_preds[(att_preds["label"] == 1) & (att_preds["prediction"] == 0)]
            .sort_values(by="probability", ascending=True)
            .head(5)
        )
        error_analysis_df = pd.concat([fps, fns], axis=0)
        error_analysis_df.to_csv(explanations_dir / "error_analysis.csv", index=False)

    # 4. Hypothesis Evaluation
    print("\n[4/6] Evaluating Hypotheses H1, H2, H3...")
    hypotheses_res = evaluate_hypotheses(results_json_dict)
    with open(metrics_dir / "hypothesis_results.json", "w") as f:
        json.dump(hypotheses_res, f, indent=2)

    # 5. Stress Testing
    stress_results_list = []
    if not args.skip_stress_tests:
        print("\n[5/6] Running Stress Tests...")
        stress_dir_path = project_root / "data" / "stress_tests"
        stress_files = {
            "A": stress_dir_path / "test_a_noisy_gaze.csv",
            "B": stress_dir_path / "test_b_mouse_noise.csv",
            "C": stress_dir_path / "test_c_environment_failure.csv",
            "D": stress_dir_path / "test_d_single_modality.csv",
            "E": stress_dir_path / "test_e_silent_cheating.csv",
        }

        for scenario_code, s_path in stress_files.items():
            if not s_path.exists():
                print(f"Warning: Stress file {s_path} not found.")
                continue

            stress_loader, _ = load_stress_dataset(s_path, scaler, batch_size=batch_size)

            for model_key, model in trained_models.items():
                meta = MODEL_REGISTRY[model_key]
                thresh = optimal_thresholds[model_key]

                probs, labels, _ = get_predictions(model, stress_loader, device)
                sm = calculate_metrics(labels, probs, threshold=thresh)
                clean_auc = results_json_dict[model_key]["auc_roc"]
                auc_drop = (
                    clean_auc - sm["auc_roc"]
                    if not (np.isnan(clean_auc) or np.isnan(sm["auc_roc"]))
                    else float("nan")
                )

                sm["scenario"] = scenario_code
                sm["model"] = meta["name"]
                sm["model_key"] = model_key
                sm["auc_degradation"] = float(auc_drop)
                stress_results_list.append(sm)

    stress_df = pd.DataFrame(stress_results_list)
    if not stress_df.empty:
        stress_df.to_csv(stress_dir / "stress_results.csv", index=False)
        with open(stress_dir / "stress_results.json", "w") as f:
            json.dump(stress_df.to_dict(orient="records"), f, indent=2)

    # 6. Figures & Visualization
    if not args.skip_figures:
        print("\n[6/6] Generating Publication-Quality Figures...")
        generate_visualizations(
            test_results_df, test_predictions_dict, stress_df, figures_dir
        )

    # Terminal Output Summary Table
    print("\n" + "=" * 70)
    print(f"{'Model':<30} {'AUC':<10} {'F1':<10} {'Precision':<10} {'Recall':<10}")
    print("-" * 70)
    for row in test_results_list:
        auc_str = f"{row['auc_roc']:.4f}" if not np.isnan(row['auc_roc']) else "N/A"
        print(
            f"{row['model']:<30} {auc_str:<10} {row['f1']:<10.4f} {row['precision']:<10.4f} {row['recall']:<10.4f}"
        )
    print("=" * 70)

    print("\nHYPOTHESIS EVALUATION RESULTS:")
    print(
        f"H1 (Attention vs Single-stream): {hypotheses_res['H1']['status']} (Diff: {hypotheses_res['H1']['difference']:+.4f})"
    )
    print(
        f"H2 (Attention vs Early/Late):    {hypotheses_res['H2']['status']} (Diff: {hypotheses_res['H2']['difference']:+.4f})"
    )
    print(
        f"H3 (Attention vs Pairwise):      {hypotheses_res['H3']['status']} (Diff: {hypotheses_res['H3']['difference']:+.4f})"
    )

    if not stress_df.empty:
        print("\nSTRESS TEST SUMMARY (AUC Degradation):")
        for sc in ["A", "B", "C", "D", "E"]:
            sc_df = stress_df[
                (stress_df["scenario"] == sc)
                & (stress_df["model_key"] == "attention_fusion")
            ]
            if not sc_df.empty:
                r = sc_df.iloc[0]
                print(
                    f"Scenario {sc}: AUC={r['auc_roc']:.4f} (Degradation: {r['auc_degradation']:+.4f})"
                )

    print("\nARTIFACTS CREATED:")
    print(f" - Checkpoints: {checkpoints_dir}")
    print(f" - Results:     {results_dir}")
    print(f" - Figures:     {figures_dir}")
    print("\nExecution Completed Successfully!")


if __name__ == "__main__":
    main()
