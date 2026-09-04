import argparse
import random

import numpy as np
import torch

from exam_proctoring.data.dataset import (
    create_dataloaders,
)

from exam_proctoring.models import (
    GazeOnlyModel,
    InteractionOnlyModel,
    EnvironmentOnlyModel,
    GazeInteractionModel,
    GazeEnvironmentModel,
    InteractionEnvironmentModel,
    EarlyFusionModel,
    LateFusionModel,
    AttentionFusionModel,
)

from exam_proctoring.training.trainer import Trainer


MODEL_REGISTRY = {

    "gaze": GazeOnlyModel,

    "interaction": InteractionOnlyModel,

    "environment": EnvironmentOnlyModel,

    "gaze_interaction": GazeInteractionModel,

    "gaze_environment": GazeEnvironmentModel,

    "interaction_environment":
        InteractionEnvironmentModel,

    "early_fusion": EarlyFusionModel,

    "late_fusion": LateFusionModel,

    "attention_fusion": AttentionFusionModel,
}


def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_model(
    model_name,
    data_path="data/dataset_1_initial/synthetic/dataset.csv",
    epochs=50,
    batch_size=32,
    patience=8,
    seed=42,
    checkpoint_dir="models/model_1_initial_dataset/checkpoints",
):

    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {model_name}"
        )

    set_seed(seed)

    print("=" * 70)
    print(f"Training model: {model_name}")
    print("=" * 70)

    # --------------------------------------------------
    # Data
    # --------------------------------------------------

    (
        train_loader,
        val_loader,
        test_loader,
        scaler,
    ) = create_dataloaders(
        data_path=data_path,
        batch_size=batch_size,
    )

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    model_class = MODEL_REGISTRY[
        model_name
    ]

    model = model_class()

    print(
        f"Parameters: "
        f"{sum(p.numel() for p in model.parameters()):,}"
    )

    # --------------------------------------------------
    # Trainer
    # --------------------------------------------------

    trainer = Trainer(
        model=model,
        learning_rate=1e-3,
        pos_weight=3.0,
    )

    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    checkpoint_path = (
        f"{checkpoint_dir}/{model_name}.pt"
    )

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        patience=patience,
        checkpoint_path=checkpoint_path,
    )

    print()
    print("=" * 70)
    print(f"Finished: {model_name}")
    print(f"Checkpoint: {checkpoint_path}")
    print("=" * 70)

    return model, history


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
        choices=MODEL_REGISTRY.keys(),
    )

    parser.add_argument(
        "--data",
        default="data/dataset_1_initial/synthetic/dataset.csv",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    train_model(
        model_name=args.model,
        data_path=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()