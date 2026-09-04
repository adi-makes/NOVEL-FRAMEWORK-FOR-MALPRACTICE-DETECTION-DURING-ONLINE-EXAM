from pathlib import Path
from typing import Tuple, Dict, Any

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

from models.common import (
    GAZE_FEATURES,
    INTERACTION_FEATURES,
    ENVIRONMENT_FEATURES,
    ALL_FEATURES,
)


class ExamDataset(Dataset):
    """
    Dataset returning three separate feature streams (Gaze, Interaction, Environment) and binary label.
    """

    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe.reset_index(drop=True)

        self.gaze = torch.tensor(
            self.dataframe[list(GAZE_FEATURES)].values,
            dtype=torch.float32,
        )

        self.interaction = torch.tensor(
            self.dataframe[list(INTERACTION_FEATURES)].values,
            dtype=torch.float32,
        )

        self.environment = torch.tensor(
            self.dataframe[list(ENVIRONMENT_FEATURES)].values,
            dtype=torch.float32,
        )

        self.labels = torch.tensor(
            self.dataframe["label"].values,
            dtype=torch.float32,
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return (
            self.gaze[index],
            self.interaction[index],
            self.environment[index],
            self.labels[index],
        )


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent.parent


def load_dataframe(data_path: Path) -> pd.DataFrame:
    data_path = Path(data_path)
    if not data_path.is_absolute():
        data_path = get_project_root() / data_path

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {data_path}")

    df = pd.read_csv(data_path)

    required_features = list(ALL_FEATURES) + ["label"]
    missing = [c for c in required_features if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {data_path}: {missing}")

    return df


def verify_dataset_integrity(df: pd.DataFrame) -> Dict[str, Any]:
    required_cols = list(ALL_FEATURES) + ["label", "split", "session_id"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column for integrity check: {c}")

    train_sessions = set(df[df["split"] == "train"]["session_id"].unique())
    val_sessions = set(df[df["split"] == "val"]["session_id"].unique())
    test_sessions = set(df[df["split"] == "test"]["session_id"].unique())

    if train_sessions.intersection(val_sessions):
        raise ValueError("Session leakage detected between train and val splits!")
    if train_sessions.intersection(test_sessions):
        raise ValueError("Session leakage detected between train and test splits!")
    if val_sessions.intersection(test_sessions):
        raise ValueError("Session leakage detected between val and test splits!")

    feature_df = df[list(ALL_FEATURES)]
    if feature_df.isna().sum().sum() > 0:
        raise ValueError("NaN detected in model features!")
    if np.isinf(feature_df.values).sum() > 0:
        raise ValueError("Inf detected in model features!")

    train_df = df[df["split"] == "train"]
    pos_count = int((train_df["label"] == 1).sum())
    neg_count = int((train_df["label"] == 0).sum())
    pos_weight = neg_count / max(1, pos_count)

    return {
        "total_rows": len(df),
        "train_rows": len(train_df),
        "val_rows": len(df[df["split"] == "val"]),
        "test_rows": len(df[df["split"] == "test"]),
        "train_sessions": len(train_sessions),
        "val_sessions": len(val_sessions),
        "test_sessions": len(test_sessions),
        "pos_count": pos_count,
        "neg_count": neg_count,
        "pos_weight": pos_weight,
    }


def create_dataloaders(
    data_path: str = "data/dataset_1_initial/synthetic/dataset.csv",
    batch_size: int = 32,
    scaler_path: str = "models/model_1_initial_dataset/checkpoints/scaler.joblib",
    fit_scaler: bool = True,
    scaler_to_use: StandardScaler = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, StandardScaler, Dict[str, Any]]:
    project_root = get_project_root()
    df = load_dataframe(Path(data_path))
    integrity_info = verify_dataset_integrity(df)

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    feature_cols = list(ALL_FEATURES)

    if fit_scaler:
        scaler = StandardScaler()
        scaler.fit(train_df[feature_cols])
        if scaler_path:
            sp = Path(scaler_path)
            if not sp.is_absolute():
                sp = project_root / sp
            sp.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(scaler, sp)
    else:
        if scaler_to_use is not None:
            scaler = scaler_to_use
        elif scaler_path:
            sp = Path(scaler_path)
            if not sp.is_absolute():
                sp = project_root / sp
            if sp.exists():
                scaler = joblib.load(sp)
            else:
                raise FileNotFoundError(f"Scaler path not found: {sp}")
        else:
            raise ValueError("No scaler provided or found.")

    train_df[feature_cols] = train_df[feature_cols].astype(np.float32)
    val_df[feature_cols] = val_df[feature_cols].astype(np.float32)
    test_df[feature_cols] = test_df[feature_cols].astype(np.float32)

    train_df.loc[:, feature_cols] = scaler.transform(train_df[feature_cols])
    val_df.loc[:, feature_cols] = scaler.transform(val_df[feature_cols])
    test_df.loc[:, feature_cols] = scaler.transform(test_df[feature_cols])

    train_ds = ExamDataset(train_df)
    val_ds = ExamDataset(val_df)
    test_ds = ExamDataset(test_df)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader, scaler, integrity_info


def load_stress_dataset(
    stress_csv_path: Path,
    scaler: StandardScaler,
    batch_size: int = 32,
) -> Tuple[DataLoader, pd.DataFrame]:
    df = load_dataframe(stress_csv_path)
    feature_cols = list(ALL_FEATURES)
    
    scaled_df = df.copy()
    scaled_df[feature_cols] = scaled_df[feature_cols].astype(np.float32)
    scaled_df.loc[:, feature_cols] = scaler.transform(df[feature_cols])

    dataset = ExamDataset(scaled_df)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )
    return loader, df
